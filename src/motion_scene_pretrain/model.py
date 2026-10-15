"""Portable motion/scene transformer; no robot controller or simulator imports.

The conditioning encoder consumes only measured past motion, the current scene,
and the supplied goal. The decoder either predicts a future from temporal
queries, or predicts a conditional flow velocity from an explicitly supplied
noisy future. A noisy future never enters the conditioning representation.

All boolean masks use True=valid/observed, unlike PyTorch key-padding masks.
Scene order is arbitrary: there are deliberately no obstacle-index embeddings.
The uncertainty head is a diagonal conditional residual baseline, not collision
probability, epistemic uncertainty, or a safety certificate.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass
class ModelConfig:
    state_dim: int = 65
    scene_dim: int = 9
    goal_dim: int = 4
    width: int = 128
    heads: int = 4
    layers: int = 3
    dropout: float = 0.0
    fps: float = 50.0
    scene_enabled: bool = True
    logvar_min: float = -6.0
    logvar_max: float = 4.0

    def __post_init__(self):
        if self.width < 4 or self.width % self.heads:
            raise ValueError("width must be >=4 and divisible by heads")
        if min(self.state_dim, self.scene_dim, self.goal_dim, self.layers) < 1:
            raise ValueError("input dimensions and layers must be positive")
        if self.fps <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("fps must be positive and dropout in [0,1)")
        if self.logvar_min >= self.logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")


def _time_features(time: Tensor, width: int, max_frequency: float) -> Tensor:
    """Fixed Fourier features; motion time is in seconds relative to the anchor."""
    half = width // 2
    frequencies = torch.exp(
        torch.linspace(math.log(0.25), math.log(max_frequency), half,
                       device=time.device, dtype=time.dtype)
    )
    phase = 2 * math.pi * time[..., None] * frequencies
    features = torch.cat((phase.sin(), phase.cos()), dim=-1)
    if width % 2:
        features = torch.nn.functional.pad(features, (0, 1))
    return features


def _valid_mask(mask: Tensor | None, shape: tuple[int, int], reference: Tensor,
                name: str) -> Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=reference.device)
    if mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a bool tensor of shape {shape}")
    if mask.device != reference.device:
        raise ValueError(f"{name} must be on the input device")
    return mask


class _ContextBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        w = config.width
        self.self_norm = nn.LayerNorm(w)
        self.self_attention = nn.MultiheadAttention(
            w, config.heads, dropout=config.dropout, batch_first=True
        )
        self.scene_norm = nn.LayerNorm(w)
        self.scene_attention = nn.MultiheadAttention(
            w, config.heads, dropout=config.dropout, batch_first=True
        ) if config.scene_enabled else None
        self.feedforward_norm = nn.LayerNorm(w)
        self.feedforward = nn.Sequential(
            nn.Linear(w, 4 * w), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(4 * w, w), nn.Dropout(config.dropout),
        )

    def forward(self, context: Tensor, valid: Tensor, scene: Tensor,
                scene_valid: Tensor) -> Tensor:
        x = self.self_norm(context)
        context = context + self.self_attention(
            x, x, x, key_padding_mask=~valid, need_weights=False
        )[0]
        if self.scene_attention is not None:
            context = context + self.scene_attention(
                self.scene_norm(context), scene, scene,
                key_padding_mask=~scene_valid, need_weights=False,
            )[0]
        context = context + self.feedforward(self.feedforward_norm(context))
        return context.masked_fill(~valid[..., None], 0)


class _FutureBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        w = config.width
        self.self_norm = nn.LayerNorm(w)
        self.self_attention = nn.MultiheadAttention(
            w, config.heads, dropout=config.dropout, batch_first=True
        )
        self.context_norm = nn.LayerNorm(w)
        self.context_attention = nn.MultiheadAttention(
            w, config.heads, dropout=config.dropout, batch_first=True
        )
        self.scene_norm = nn.LayerNorm(w)
        self.scene_attention = nn.MultiheadAttention(
            w, config.heads, dropout=config.dropout, batch_first=True
        ) if config.scene_enabled else None
        self.feedforward_norm = nn.LayerNorm(w)
        self.feedforward = nn.Sequential(
            nn.Linear(w, 4 * w), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(4 * w, w), nn.Dropout(config.dropout),
        )

    def forward(self, future: Tensor, valid: Tensor, context: Tensor,
                context_valid: Tensor, scene: Tensor, scene_valid: Tensor,
                return_attention: bool) -> tuple[Tensor, Tensor | None]:
        x = self.self_norm(future)
        future = future + self.self_attention(
            x, x, x, key_padding_mask=~valid, need_weights=False
        )[0]
        future = future + self.context_attention(
            self.context_norm(future), context, context,
            key_padding_mask=~context_valid, need_weights=False,
        )[0]
        attention = None
        if self.scene_attention is not None:
            update, attention = self.scene_attention(
                self.scene_norm(future), scene, scene,
                key_padding_mask=~scene_valid, need_weights=return_attention,
                average_attn_weights=True,
            )
            future = future + update
        future = future + self.feedforward(self.feedforward_norm(future))
        return future.masked_fill(~valid[..., None], 0), attention


class MotionSceneModel(nn.Module):
    """Joint non-RL representation, forecasting, and conditional flow model.

    ``history`` is [B,P,state_dim], ``scene`` is [B,M,scene_dim], and ``goal``
    is [B,goal_dim], all float tensors on the same device. An empty M is legal.
    Left-padding history preserves its anchor-relative timestamps. The newest
    history state is time 0; future queries start at 1/fps seconds.

    ``history_observed`` marks states visible during masked pretraining. Hidden
    states are replaced *before projection*; their reconstruction targets must
    remain outside this module. ``history_mask`` instead marks actual padding.

    With ``future_input=None``, the future decoder sees only learned temporal
    queries. Supplying ``future_input`` [B,H,D] and ``flow_time`` selects flow
    mode. Time may be scalar, [B], [B,1], [B,H], or [B,H,1]. The intended trainer
    convention is x_t=(1-t)y+t*eps, target velocity eps-y, sampling t=1 to 0.

    ``representation`` is pooled conditioning context, invariant to noised
    future inputs. ``scene_attention`` (if requested) is the last future layer's
    head-averaged direct scene attention [B,H,M+1]; the final column is a learned
    null token. Attention is a diagnostic, not a causal explanation of the model.
    Padded query outputs are zero. No-scene models report attention only on null.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        w = cfg.width
        self.history_projection = nn.Linear(cfg.state_dim, w)
        self.goal_projection = nn.Linear(cfg.goal_dim, w)
        self.scene_projection = nn.Sequential(
            nn.Linear(cfg.scene_dim, w), nn.GELU(), nn.Linear(w, w), nn.LayerNorm(w)
        ) if cfg.scene_enabled else None
        self.future_projection = nn.Linear(cfg.state_dim, w)
        self.flow_time_projection = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, w))
        self.null_scene_token = nn.Parameter(torch.zeros(1, 1, w))
        self.null_future_token = nn.Parameter(torch.zeros(1, 1, w))
        self.history_type = nn.Parameter(torch.zeros(1, 1, w))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, w))
        self.forecast_type = nn.Parameter(torch.zeros(1, 1, w))
        self.flow_type = nn.Parameter(torch.zeros(1, 1, w))
        self.context_blocks = nn.ModuleList(_ContextBlock(cfg) for _ in range(cfg.layers))
        self.future_blocks = nn.ModuleList(_FutureBlock(cfg) for _ in range(cfg.layers))
        self.context_norm = nn.LayerNorm(w)
        self.future_norm = nn.LayerNorm(w)
        self.forecast_head = nn.Linear(w, cfg.state_dim)
        self.logvar_head = nn.Linear(w, cfg.state_dim)
        self.velocity_head = nn.Linear(w, cfg.state_dim)
        self.reconstruction_head = nn.Linear(w, cfg.state_dim)
        for parameter in (self.mask_token, self.null_scene_token, self.null_future_token,
                          self.history_type, self.goal_type, self.forecast_type, self.flow_type):
            nn.init.normal_(parameter, std=0.02)

    def _flow_times(self, time: Tensor | float, batch: int, horizon: int,
                    reference: Tensor) -> Tensor:
        time = torch.as_tensor(time, device=reference.device, dtype=reference.dtype)
        if time.ndim == 0:
            time = time.expand(batch, horizon)
        elif time.shape == (batch,):
            time = time[:, None].expand(-1, horizon)
        elif time.shape == (batch, 1):
            time = time.expand(-1, horizon)
        elif time.shape == (batch, horizon, 1):
            time = time[..., 0]
        elif time.shape != (batch, horizon):
            raise ValueError("flow_time must be scalar, [B], [B,1], [B,H], or [B,H,1]")
        return time

    def forward(
        self, history: Tensor, scene: Tensor, goal: Tensor, *, horizon: int,
        history_mask: Tensor | None = None, scene_mask: Tensor | None = None,
        future_mask: Tensor | None = None, history_observed: Tensor | None = None,
        future_input: Tensor | None = None, flow_time: Tensor | float | None = None,
        return_attention: bool = False,
    ) -> dict[str, Tensor]:
        cfg = self.config
        if history.ndim != 3 or history.shape[-1] != cfg.state_dim or history.shape[1] < 1:
            raise ValueError("history must be [B,P,state_dim] with P>=1")
        b, p, _ = history.shape
        if scene.ndim != 3 or scene.shape[0] != b or scene.shape[-1] != cfg.scene_dim:
            raise ValueError("scene must be [B,M,scene_dim]")
        if goal.shape != (b, cfg.goal_dim) or horizon < 1:
            raise ValueError("goal must be [B,goal_dim] and horizon must be positive")
        if (future_input is None) != (flow_time is None):
            raise ValueError("future_input and flow_time must be supplied together")
        m = scene.shape[1]
        history_valid = _valid_mask(history_mask, (b, p), history, "history_mask")
        observed = _valid_mask(history_observed, (b, p), history, "history_observed") & history_valid
        scene_valid = _valid_mask(scene_mask, (b, m), history, "scene_mask")
        future_valid = _valid_mask(future_mask, (b, horizon), history, "future_mask")

        # Sanitize before projection: even NaNs in hidden/padded data cannot leak.
        safe_history = history.masked_fill(~observed[..., None], 0)
        past = self.history_projection(safe_history)
        past = torch.where(observed[..., None], past, self.mask_token)
        past_time = torch.arange(1 - p, 1, device=history.device, dtype=history.dtype) / cfg.fps
        past = past + _time_features(past_time, cfg.width, cfg.fps / 4)[None] + self.history_type
        goal_token = self.goal_projection(goal)[:, None] + self.goal_type
        context = torch.cat((past, goal_token), dim=1)
        always_valid = torch.ones((b, 1), dtype=torch.bool, device=history.device)
        context_valid = torch.cat((history_valid, always_valid), dim=1)

        if self.scene_projection is not None:
            safe_scene = scene.masked_fill(~scene_valid[..., None], 0)
            scene_tokens = self.scene_projection(safe_scene)
        else:
            scene_tokens = history.new_zeros((b, m, cfg.width))
            scene_valid = torch.zeros_like(scene_valid)
        scene_tokens = torch.cat((scene_tokens, self.null_scene_token.expand(b, -1, -1)), dim=1)
        scene_valid = torch.cat((scene_valid, always_valid), dim=1)
        for block in self.context_blocks:
            context = block(context, context_valid, scene_tokens, scene_valid)
        context = self.context_norm(context).masked_fill(~context_valid[..., None], 0)
        representation = context.sum(dim=1) / context_valid.sum(dim=1, keepdim=True).to(context.dtype)
        reconstruction = self.reconstruction_head(context[:, :p]).masked_fill(~history_valid[..., None], 0)

        future_time = torch.arange(1, horizon + 1, device=history.device, dtype=history.dtype) / cfg.fps
        future = _time_features(future_time, cfg.width, cfg.fps / 4)[None].expand(b, -1, -1)
        if future_input is None:
            future = future + self.forecast_type
        else:
            if future_input.shape != (b, horizon, cfg.state_dim):
                raise ValueError("future_input must be [B,horizon,state_dim]")
            time = self._flow_times(flow_time, b, horizon, history)
            time = time.masked_fill(~future_valid, 0)
            safe_future = future_input.masked_fill(~future_valid[..., None], 0)
            future = (future + self.future_projection(safe_future) + self.flow_type
                      + self.flow_time_projection(_time_features(time, cfg.width, 16.0)))
        # A sentinel avoids all-masked self-attention when an example has no valid future.
        future = torch.cat((future, self.null_future_token.expand(b, -1, -1)), dim=1)
        decoder_valid = torch.cat((future_valid, always_valid), dim=1)
        attention = None
        for i, block in enumerate(self.future_blocks):
            future, attention = block(
                future, decoder_valid, context, context_valid, scene_tokens, scene_valid,
                return_attention and i == len(self.future_blocks) - 1,
            )
        future = self.future_norm(future[:, :horizon])
        mean = self.forecast_head(future).masked_fill(~future_valid[..., None], 0)
        # Smooth bounded log variance avoids dead gradients from hard clipping.
        raw_logvar = self.logvar_head(future)
        logvar = (cfg.logvar_min + (cfg.logvar_max - cfg.logvar_min) * raw_logvar.sigmoid())
        logvar = logvar.masked_fill(~future_valid[..., None], 0)
        outputs = {
            "forecast_mean": mean, "logvar": logvar,
            "history_reconstruction": reconstruction, "representation": representation,
        }
        if future_input is not None:
            outputs["flow_velocity"] = self.velocity_head(future).masked_fill(~future_valid[..., None], 0)
        if return_attention:
            if attention is None:
                attention = history.new_zeros((b, horizon, m + 1))
                attention[..., -1] = 1
            else:
                attention = attention[:, :horizon]
            outputs["scene_attention"] = attention.masked_fill(~future_valid[..., None], 0)
        return outputs
