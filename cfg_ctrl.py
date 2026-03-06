from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


SwitchMode = Literal["sign", "sat", "tanh", "vector_norm"]
EpsMode = Literal["absolute", "relative"]
DTypeMode = Literal["match", "float16", "bfloat16", "float32"]

_SIGMA_STEP_RTOL = 1e-5
_SIGMA_STEP_ATOL = 1e-8


def _to_torch_dtype(mode: DTypeMode, ref: torch.Tensor) -> torch.dtype:
    if mode == "match":
        return ref.dtype
    if mode == "float16":
        return torch.float16
    if mode == "bfloat16":
        return torch.bfloat16
    if mode == "float32":
        return torch.float32
    return ref.dtype


def _sigma_to_float(sigma: Optional[torch.Tensor | float | int]) -> Optional[float]:
    if sigma is None:
        return None
    if isinstance(sigma, (float, int)):
        return float(sigma)
    if not torch.is_tensor(sigma):
        return float(sigma)
    if sigma.numel() == 0:
        return None
    return float(sigma.detach().reshape(-1)[0].item())


@dataclass(frozen=True)
class CFGCtrlConfig:
    """
    Implements CFG-Ctrl / SMC-CFG as described in the paper and official code.

    SMC-CFG (discrete):
      e_t = cond - uncond
      s_t = (e_t - e_{t-1}) + lambda * e_{t-1}
      u_sw = -K * sign(s_t)
      e'_t = e_t + u_sw
      guided = uncond + cfg_scale * e'_t

    Warmup:
      during the first N logical sampling steps, output the conditional prediction
      directly (no CFG mixing).
    """

    enable_smc: bool = True

    # SMC parameters
    smc_lambda: float = 6.0
    smc_k: float = 0.3

    # Warmup: first N logical steps, use pure conditional (matches repo behavior)
    no_cfg_warmup_steps: int = 0

    # Logical-step window (sigma-change based, not raw model-call based)
    active_start_step: int = 0
    active_end_step: int = 10**9

    # Optional sigma-percent window for sampler-independent gating
    active_start_percent: float = 0.0
    active_end_percent: float = 1.0
    active_sigma_start: Optional[float] = None
    active_sigma_end: Optional[float] = None

    # Stability controls (to reduce chattering)
    switch_mode: SwitchMode = "sign"
    boundary_epsilon: float = 0.0
    epsilon_mode: EpsMode = "absolute"

    # Numerics / memory
    math_dtype: DTypeMode = "float32"  # dtype for computing s/u_sw
    state_dtype: DTypeMode = "match"  # dtype used to store prev guidance
    reset_on_shape_change: bool = True

    # Behavior
    apply_only_if_cfg_gt_1: bool = False
    detach_prev: bool = True  # store prev as detached tensor

    def needs_custom_cfg_for_run(self, cfg_scale: float) -> bool:
        if self.no_cfg_warmup_steps > 0:
            return True
        if not self.enable_smc:
            return False
        if self.apply_only_if_cfg_gt_1 and float(cfg_scale) <= 1.0:
            return False
        return True


@dataclass
class CFGCtrlState:
    logical_step_index: int = 0
    raw_call_index: int = 0
    last_sigma_value: Optional[float] = None
    prev_guidance: Optional[torch.Tensor] = None


class CFGCtrlController:
    """
    Stateful controller that can be called during sampling.

    Logical steps are tracked by changes in sigma/timestep rather than by raw
    model-call count. This aligns warmup / step windows more closely with the
    actual denoising schedule across samplers.
    """

    def __init__(self, config: CFGCtrlConfig):
        self.config = config
        self.state = CFGCtrlState()

    def reset(self) -> None:
        self.state = CFGCtrlState()

    @torch.no_grad()
    def apply_guided(
        self,
        *,
        cond_pred: torch.Tensor,
        uncond_pred: torch.Tensor,
        cfg_scale: float,
        sigma: Optional[torch.Tensor | float | int] = None,
    ) -> torch.Tensor:
        mode, corrected_guidance = self._compute_guidance(
            cond_pred=cond_pred,
            uncond_pred=uncond_pred,
            cfg_scale=cfg_scale,
            sigma=sigma,
        )
        if mode == "warmup_conditional":
            return cond_pred
        return uncond_pred + (float(cfg_scale) * corrected_guidance)

    @torch.no_grad()
    def apply_pre_cfg(
        self,
        *,
        cond_pred: torch.Tensor,
        uncond_pred: torch.Tensor,
        cfg_scale: float,
        sigma: Optional[torch.Tensor | float | int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mode, corrected_guidance = self._compute_guidance(
            cond_pred=cond_pred,
            uncond_pred=uncond_pred,
            cfg_scale=cfg_scale,
            sigma=sigma,
        )
        if mode == "warmup_conditional":
            return cond_pred, cond_pred
        if mode == "passthrough":
            return cond_pred, uncond_pred
        cond_pred_mod = uncond_pred + corrected_guidance
        return cond_pred_mod, uncond_pred

    @torch.no_grad()
    def _compute_guidance(
        self,
        *,
        cond_pred: torch.Tensor,
        uncond_pred: torch.Tensor,
        cfg_scale: float,
        sigma: Optional[torch.Tensor | float | int],
    ) -> Tuple[Literal["warmup_conditional", "passthrough", "smc"], torch.Tensor]:
        cfg_scale_f = float(cfg_scale)
        cfg = self.config
        st = self.state

        step = self._logical_step_index(sigma)
        guidance = cond_pred - uncond_pred

        # Warmup: pure conditional output; also clear controller state so the first
        # active step initializes from its own guidance.
        if cfg.no_cfg_warmup_steps > 0 and step < cfg.no_cfg_warmup_steps:
            st.prev_guidance = None
            return "warmup_conditional", guidance

        active = self._step_window_active(step) and self._sigma_window_active(sigma)

        if cfg.apply_only_if_cfg_gt_1 and cfg_scale_f <= 1.0:
            return "passthrough", guidance

        if not (cfg.enable_smc and active):
            return "passthrough", guidance

        if cfg.reset_on_shape_change and st.prev_guidance is not None:
            if (
                st.prev_guidance.shape != guidance.shape
                or st.prev_guidance.device != guidance.device
            ):
                st.prev_guidance = None

        math_dtype = _to_torch_dtype(cfg.math_dtype, guidance)
        store_dtype = _to_torch_dtype(cfg.state_dtype, guidance)

        g = guidance.to(dtype=math_dtype)

        if st.prev_guidance is None:
            prev = g.detach() if cfg.detach_prev else g
        else:
            prev = st.prev_guidance.to(dtype=math_dtype)

        s = (g - prev) + (float(cfg.smc_lambda) * prev)
        u_sw = self._switching_control(s, float(cfg.smc_k), cfg)
        g2 = g + u_sw

        st.prev_guidance = (g2.detach() if cfg.detach_prev else g2).to(dtype=store_dtype)
        return "smc", g2.to(dtype=guidance.dtype)

    def _logical_step_index(self, sigma: Optional[torch.Tensor | float | int]) -> int:
        sigma_value = _sigma_to_float(sigma)
        st = self.state

        if sigma_value is None:
            step = st.raw_call_index
            st.raw_call_index += 1
            st.logical_step_index = step
            return step

        if st.last_sigma_value is None:
            st.last_sigma_value = sigma_value
            return st.logical_step_index

        if not math.isclose(
            sigma_value,
            st.last_sigma_value,
            rel_tol=_SIGMA_STEP_RTOL,
            abs_tol=_SIGMA_STEP_ATOL,
        ):
            st.logical_step_index += 1
            st.last_sigma_value = sigma_value

        return st.logical_step_index

    def _step_window_active(self, step: int) -> bool:
        start = int(self.config.active_start_step)
        end = int(self.config.active_end_step)
        if end < start:
            start, end = end, start
        return start <= step <= end

    def _sigma_window_active(self, sigma: Optional[torch.Tensor | float | int]) -> bool:
        sigma_value = _sigma_to_float(sigma)
        start = self.config.active_sigma_start
        end = self.config.active_sigma_end

        if sigma_value is None or start is None or end is None:
            return True

        sigma_hi = max(float(start), float(end))
        sigma_lo = min(float(start), float(end))
        return sigma_lo <= sigma_value <= sigma_hi

    @staticmethod
    @torch.no_grad()
    def _switching_control(s: torch.Tensor, k: float, cfg: CFGCtrlConfig) -> torch.Tensor:
        mode = cfg.switch_mode
        eps = float(cfg.boundary_epsilon)

        if eps > 0.0 and cfg.epsilon_mode == "relative":
            if s.ndim <= 1:
                scale = s.abs().mean().clamp(min=1e-12)
            else:
                reduce_dims = tuple(range(1, s.ndim))
                scale = s.abs().mean(dim=reduce_dims, keepdim=True).clamp(min=1e-12)
            eps_tensor = scale * eps
        else:
            eps_tensor = torch.as_tensor(eps, device=s.device, dtype=s.dtype)

        if mode == "sign" or eps <= 0.0:
            return (-k) * torch.sign(s)

        if mode == "sat":
            return (-k) * (s / (s.abs() + eps_tensor))

        if mode == "tanh":
            return (-k) * torch.tanh(s / eps_tensor)

        if mode == "vector_norm":
            b = s.shape[0] if s.ndim > 0 else 1
            s_flat = s.reshape(b, -1)
            n = torch.linalg.vector_norm(s_flat, dim=1, keepdim=True).clamp(min=1e-12)
            n = n.view([b] + [1] * (s.ndim - 1))
            return (-k) * (s / (n + eps_tensor))

        return (-k) * torch.sign(s)
