"""Training contracts: honest masks, causal conditioning, and flow direction."""

import numpy as np
import pytest
import torch

from motion_scene_pretrain.common import (
    collate, fit_normalizer, masked_mean, prepare, sample_flow,
)
from motion_scene_pretrain.model import ModelConfig, MotionSceneModel
from motion_scene_pretrain.train import objective, validate


class TinyDataset:
    def __init__(self, rows, split="train"):
        self.rows = rows
        self.split = split

    def __iter__(self):
        return iter(self.rows)


def make_batch():
    torch.manual_seed(19)
    return {
        "history": torch.randn(2, 6, 65),
        "future": torch.randn(2, 10, 65),
        "scene": torch.randn(2, 3, 9),
        "scene_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "goal": torch.randn(2, 4),
        "future_mask": torch.tensor([[True] * 8 + [False] * 2, [True] * 10]),
    }


@pytest.mark.parametrize("mode", ["masked", "forecast", "flow"])
def test_each_objective_produces_finite_shared_encoder_gradients(mode):
    model = MotionSceneModel(ModelConfig(width=32, heads=4, layers=1))
    loss, metrics = objective(model, make_batch(), mode)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in metrics.values())
    assert model.history_projection.weight.grad.abs().sum() > 0
    assert model.scene_projection[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if mode == "flow":
        assert model.velocity_head.weight.grad.abs().sum() > 0
    elif mode == "forecast":
        assert model.forecast_head.weight.grad.abs().sum() > 0
        assert model.logvar_head.weight.grad.abs().sum() > 0


def test_padding_values_do_not_change_loss_or_gradient():
    batch = make_batch()
    perturbed = {key: value.clone() for key, value in batch.items()}
    perturbed["future"][~batch["future_mask"]] = 1e6
    model = MotionSceneModel(ModelConfig(width=32, heads=4, layers=1))
    torch.manual_seed(41)
    first_loss, _ = objective(model, batch, "flow")
    first_loss.backward()
    first_grad = model.history_projection.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(41)
    second_loss, _ = objective(model, perturbed, "flow")
    second_loss.backward()
    torch.testing.assert_close(first_loss, second_loss)
    torch.testing.assert_close(first_grad, model.history_projection.weight.grad)


def test_clean_future_is_not_a_conditioning_input():
    class Recorder(MotionSceneModel):
        def __init__(self):
            super().__init__(ModelConfig(width=32, heads=4, layers=1))
            self.calls = []

        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            self.calls.append({
                "future_input": kwargs.get("future_input"),
                "flow_time": kwargs.get("flow_time"),
                "observed": kwargs.get("history_observed"),
                "representation": result["representation"].detach().clone(),
                "forecast_mean": result["forecast_mean"].detach().clone(),
            })
            return result

    model = Recorder().eval()
    batch = make_batch()
    torch.manual_seed(71)
    objective(model, batch, "flow")
    first = model.calls.copy()
    model.calls.clear()
    other_target = dict(batch, future=batch["future"] + 20)
    torch.manual_seed(71)
    objective(model, other_target, "flow")
    second = model.calls
    assert len(first) == 3  # masked past, noised future, fully observed forecast
    assert first[0]["future_input"] is None and first[2]["future_input"] is None
    assert first[1]["future_input"] is not None
    assert first[0]["observed"][:, -1].all()
    for observed in first[0]["observed"]:
        hidden = torch.nonzero(~observed, as_tuple=False).flatten()
        assert len(hidden) >= 1
        assert hidden[-1] < len(observed) - 1
        assert torch.equal(hidden, torch.arange(hidden[0], hidden[-1] + 1))
    assert first[1]["observed"] is None and first[2]["observed"] is None
    for i in range(3):
        torch.testing.assert_close(first[i]["representation"], second[i]["representation"])
    torch.testing.assert_close(first[2]["forecast_mean"], second[2]["forecast_mean"])
    assert not torch.equal(first[1]["future_input"], second[1]["future_input"])


def test_cfm_target_and_sampler_share_data_at_zero_convention(monkeypatch):
    batch = make_batch()
    target = batch["future"]
    noise = torch.full_like(target, 1.7)
    velocity = noise - target
    captured = []

    class PerfectStraightFlow:
        def __call__(self, history, scene, goal, **kwargs):
            if "future_input" in kwargs:
                captured.append((kwargs["future_input"].clone(), kwargs["flow_time"].clone()))
            return {
                "history_reconstruction": torch.zeros_like(history),
                "forecast_mean": torch.zeros_like(target),
                "logvar": torch.zeros_like(target),
                "flow_velocity": velocity,
            }

    def fixed_rand(*size, **kwargs):
        shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, torch.Size)) else size
        return torch.full(shape, .25, device=kwargs.get("device"))

    with monkeypatch.context() as patch:
        patch.setattr(torch, "rand", fixed_rand)
        patch.setattr(torch, "randn_like", lambda value: noise.clone())
        loss, parts = objective(PerfectStraightFlow(), batch, "flow",
                                reconstruction_weight=0, forecast_weight=0)
    assert parts["flow_mse"] == 0
    torch.testing.assert_close(loss, torch.tensor(0.))
    torch.testing.assert_close(captured[0][0], .75 * target + .25 * noise)
    captured.clear()
    sampled = sample_flow(PerfectStraightFlow(), batch, steps=4, noise=noise)
    torch.testing.assert_close(sampled, target, atol=5e-7, rtol=1e-5)
    torch.testing.assert_close(torch.stack([t[0] for _, t in captured]),
                               torch.tensor([1., .75, .5, .25]))
    # Label contents supply a shape only; an explicit initial noise fixes sampling.
    changed_labels = dict(batch, future=target + 1000)
    sampled_again = sample_flow(PerfectStraightFlow(), changed_labels, steps=4, noise=noise)
    torch.testing.assert_close(sampled_again, sampled)


def test_train_only_normalization_ignores_invalid_future():
    row = {
        "history": np.array([[1., 2.], [3., 4.]], dtype=np.float32),
        "future": np.array([[5., 6.], [1e8, -1e8]], dtype=np.float32),
        "future_mask": np.array([True, False]),
    }
    result = fit_normalizer(TinyDataset([row], "train"))
    np.testing.assert_allclose(result["mean"], [3., 4.])
    np.testing.assert_allclose(result["std"], np.sqrt([8 / 3, 8 / 3]))
    assert result["frames"] == 3 and result["fit_split"] == "train"
    for split in ("val", "calibration", "test"):
        with pytest.raises(ValueError, match="training split"):
            fit_normalizer(TinyDataset([row], split))
    with pytest.raises(ValueError, match="Empty"):
        fit_normalizer(TinyDataset([], "train"))


def test_normalization_preserves_metric_scene_and_does_not_mutate_batch():
    batch = make_batch()
    original_history = batch["history"].clone()
    original_scene = batch["scene"].clone()
    normalizer = {"mean": [2.] * 65, "std": [3.] * 65}
    normalized = prepare(batch, normalizer, torch.device("cpu"))
    torch.testing.assert_close(normalized["history"], (original_history - 2) / 3)
    torch.testing.assert_close(normalized["scene"], original_scene)
    torch.testing.assert_close(batch["history"], original_history)
    assert normalized["scene_mask"].dtype == torch.bool


def test_validation_is_reproducible_and_preserves_training_rng():
    model = MotionSceneModel(ModelConfig(width=32, heads=4, layers=1))
    batch = make_batch()
    normalizer = {"mean": [0.] * 65, "std": [1.] * 65}
    state = torch.random.get_rng_state().clone()
    first = validate(model, [batch], normalizer, torch.device("cpu"), "flow")
    assert torch.equal(state, torch.random.get_rng_state())
    second = validate(model, [batch], normalizer, torch.device("cpu"), "flow")
    assert first == second
    assert model.training


def test_masked_reduction_handles_empty_masks():
    values = torch.tensor([[[1., 3.], [7., 9.]]], requires_grad=True)
    kept = masked_mean(values, torch.tensor([[True, False]]))
    torch.testing.assert_close(kept, torch.tensor(2.))
    kept.backward()
    torch.testing.assert_close(values.grad[:, 1], torch.zeros(1, 2))
    empty = masked_mean(values, torch.zeros((1, 2), dtype=torch.bool))
    torch.testing.assert_close(empty, torch.tensor(0.))


def test_masked_reduction_discards_nonfinite_leaf_values():
    values = torch.tensor([[[1., 3.], [float("nan"), float("inf")]]], requires_grad=True)
    kept = masked_mean(values, torch.tensor([[True, False]]))
    torch.testing.assert_close(kept, torch.tensor(2.))
    kept.backward()
    assert torch.isfinite(values.grad).all()
    torch.testing.assert_close(values.grad[:, 1], torch.zeros(1, 2))


def test_masked_objective_requires_at_least_two_past_states():
    model = MotionSceneModel(ModelConfig(width=32, heads=4, layers=1))
    batch = make_batch()
    batch["history"] = batch["history"][:, -1:]
    with pytest.raises(ValueError, match="history|states|frames"):
        objective(model, batch, "masked")


def test_normalizer_requires_explicit_training_split_metadata():
    row = {"history": np.zeros((2, 65)), "future": np.ones((3, 65))}
    with pytest.raises(ValueError, match="training split"):
        fit_normalizer([row])
