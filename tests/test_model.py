"""Mechanism tests for causal inputs, set geometry, masking, and trainability."""

import pytest
import torch

from motion_scene_pretrain.model import ModelConfig, MotionSceneModel


@pytest.fixture
def setup_model():
    torch.manual_seed(13)
    model = MotionSceneModel(ModelConfig(width=24, heads=4, layers=2)).eval()
    history = torch.randn(2, 5, 65)
    scene = torch.randn(2, 4, 9)
    goal = torch.randn(2, 4)
    scene_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    return model, history, scene, goal, scene_mask


def test_forecast_and_flow_have_finite_gradients(setup_model):
    model, history, scene, goal, scene_mask = setup_model
    model.train()
    target = torch.randn(2, 8, 65)
    eps = torch.randn_like(target)
    time = torch.rand(2, 8, 1)
    masked = torch.tensor([[True, False, True, False, True]]).expand(2, -1)
    forecast = model(history, scene, goal, horizon=8, scene_mask=scene_mask,
                     history_observed=masked, return_attention=True)
    flow = model(history, scene, goal, horizon=8, scene_mask=scene_mask,
                 future_input=(1 - time) * target + time * eps, flow_time=time)
    residual = target - forecast["forecast_mean"]
    gaussian_nll = 0.5 * (forecast["logvar"] + residual.square() * (-forecast["logvar"]).exp())
    loss = (gaussian_nll.mean()
            + (flow["flow_velocity"] - (eps - target)).square().mean()
            + (forecast["history_reconstruction"][~masked] - history[~masked]).square().mean())
    loss.backward()
    assert torch.isfinite(loss)
    assert forecast["forecast_mean"].shape == (2, 8, 65)
    assert forecast["scene_attention"].shape == (2, 8, 5)
    assert torch.all(forecast["logvar"] >= model.config.logvar_min)
    assert torch.all(forecast["logvar"] <= model.config.logvar_max)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert model.history_projection.weight.grad.abs().sum() > 0
    assert model.scene_projection[0].weight.grad.abs().sum() > 0
    assert model.velocity_head.weight.grad.abs().sum() > 0
    assert model.mask_token.grad.abs().sum() > 0


@torch.no_grad()
def test_scene_permutation_and_padding_invariance(setup_model):
    model, history, scene, goal, mask = setup_model
    output = model(history, scene, goal, horizon=8, scene_mask=mask, return_attention=True)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model(history, scene[:, permutation], goal, horizon=8,
                     scene_mask=mask[:, permutation], return_attention=True)
    for key in ("forecast_mean", "logvar", "representation", "history_reconstruction"):
        torch.testing.assert_close(output[key], permuted[key], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(output["scene_attention"][..., permutation],
                               permuted["scene_attention"][..., :-1], atol=1e-6, rtol=1e-5)
    padded = torch.cat((scene, torch.full((2, 3, 9), float("nan"))), dim=1)
    padded_mask = torch.cat((mask, torch.zeros((2, 3), dtype=torch.bool)), dim=1)
    with_padding = model(history, padded, goal, horizon=8, scene_mask=padded_mask,
                         return_attention=True)
    torch.testing.assert_close(output["forecast_mean"], with_padding["forecast_mean"],
                               atol=2e-6, rtol=2e-5)
    assert (with_padding["scene_attention"][..., 4:7] == 0).all()


@torch.no_grad()
def test_masked_history_values_cannot_leak(setup_model):
    model, history, scene, goal, mask = setup_model
    observed = torch.tensor([[True, False, True, False, True]]).expand(2, -1)
    corrupted = history.clone()
    corrupted[~observed] = float("nan")
    kwargs = dict(horizon=8, scene_mask=mask, history_observed=observed)
    original_output = model(history, scene, goal, **kwargs)
    corrupted_output = model(corrupted, scene, goal, **kwargs)
    for key in original_output:
        torch.testing.assert_close(original_output[key], corrupted_output[key])


@torch.no_grad()
def test_left_padded_history_has_identical_anchor_relative_timing(setup_model):
    model, history, scene, goal, mask = setup_model
    original = model(history, scene, goal, horizon=8, scene_mask=mask)
    padded_history = torch.cat((torch.full((2, 3, 65), float("nan")), history), dim=1)
    history_mask = torch.tensor([[False, False, False, True, True, True, True, True]]).expand(2, -1)
    padded = model(padded_history, scene, goal, horizon=8, scene_mask=mask,
                   history_mask=history_mask)
    torch.testing.assert_close(original["forecast_mean"], padded["forecast_mean"],
                               atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(original["representation"], padded["representation"],
                               atol=2e-6, rtol=2e-5)


@torch.no_grad()
def test_noisy_future_cannot_change_conditioning_representation(setup_model):
    model, history, scene, goal, mask = setup_model
    kwargs = dict(horizon=8, scene_mask=mask, flow_time=0.6)
    a = model(history, scene, goal, future_input=torch.randn(2, 8, 65), **kwargs)
    b = model(history, scene, goal, future_input=torch.randn(2, 8, 65) * 100, **kwargs)
    torch.testing.assert_close(a["representation"], b["representation"])
    torch.testing.assert_close(a["history_reconstruction"], b["history_reconstruction"])
    assert not torch.allclose(a["flow_velocity"], b["flow_velocity"])


@torch.no_grad()
def test_empty_scene_and_fully_padded_sequences_are_finite(setup_model):
    model, history, scene, goal, mask = setup_model
    output = model(torch.full_like(history, float("nan")), scene[:, :0], goal,
                   horizon=8, history_mask=torch.zeros((2, 5), dtype=torch.bool),
                   future_mask=torch.zeros((2, 8), dtype=torch.bool), return_attention=True)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert (output["forecast_mean"] == 0).all()
    assert (output["history_reconstruction"] == 0).all()
    empty = model(history, scene[:, :0], goal, horizon=8, return_attention=True)
    padded = model(history, scene, goal, horizon=8, scene_mask=torch.zeros_like(mask))
    torch.testing.assert_close(empty["forecast_mean"], padded["forecast_mean"],
                               atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(empty["scene_attention"], torch.ones(2, 8, 1))


@torch.no_grad()
def test_no_scene_baseline_is_independent_of_obstacles(setup_model):
    _, history, scene, goal, mask = setup_model
    model = MotionSceneModel(ModelConfig(width=24, heads=4, layers=1, scene_enabled=False)).eval()
    a = model(history, scene, goal, horizon=8, scene_mask=mask, return_attention=True)
    b = model(history, torch.full_like(scene, float("nan")), goal, horizon=8,
               scene_mask=~mask, return_attention=True)
    for key in a:
        torch.testing.assert_close(a[key], b[key])
    assert (a["scene_attention"][..., :-1] == 0).all()
    assert (a["scene_attention"][..., -1] == 1).all()


def test_rejects_ambiguous_masks_and_partial_flow_inputs(setup_model):
    model, history, scene, goal, mask = setup_model
    with pytest.raises(ValueError, match="bool"):
        model(history, scene, goal, horizon=8, scene_mask=mask.float())
    with pytest.raises(ValueError, match="together"):
        model(history, scene, goal, horizon=8, flow_time=0.5)
    with pytest.raises(ValueError, match="future_input"):
        model(history, scene, goal, horizon=8, future_input=torch.randn(2, 7, 65), flow_time=0.5)
