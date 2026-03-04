from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


SwitchMode = Literal["sign", "sat", "tanh", "vector_norm"]
EpsMode = Literal["absolute", "relative"]
DTypeMode = Literal["match", "float16", "bfloat16", "float32"]


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

    Warmup "no CFG":
      during the first N steps, output the conditional prediction (no CFG).
    """

    enable_smc: bool = True

    # SMC parameters
    smc_lambda: float = 6.0
    smc_k: float = 0.2

    # Warmup: first N steps, use pure conditional (matches repo behavior)
    no_cfg_warmup_steps: int = 0

    # Optional active window (step index in the sampling loop)
    active_start_step: int = 0
    active_end_step: int = 10**9

    # Stability controls (to reduce chattering)
    switch_mode: SwitchMode = "sign"
    boundary_epsilon: float = 0.0
    epsilon_mode: EpsMode = "absolute"

    # Numerics / memory
    math_dtype: DTypeMode = "float32"  # dtype for computing s/u_sw
    state_dtype: DTypeMode = "match"  # dtype used to store prev guidance
    reset_on_shape_change: bool = True

    # Behavior
    apply_only_if_cfg_gt_1: bool = True
    detach_prev: bool = True  # store prev as detached tensor


@dataclass
class CFGCtrlState:
    step_index: int = 0
    prev_guidance: Optional[torch.Tensor] = None


class CFGCtrlController:
    """
    Stateful controller that can be called once per sampling step.

    You apply it to (cond_pred, uncond_pred) BEFORE CFG mixing, i.e.:
      cond' = cond + u_sw    (equivalently: guidance' = (cond-uncond)+u_sw)
      uncond stays the same
    """

    def __init__(self, config: CFGCtrlConfig):
        self.config = config
        self.state = CFGCtrlState()

    def reset(self) -> None:
        self.state = CFGCtrlState()

    @torch.no_grad()
    def apply(
        self,
        cond_pred: torch.Tensor,
        uncond_pred: torch.Tensor,
        cfg_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg_scale_f = float(cfg_scale)
        cfg = self.config
        st = self.state

        # Step gating
        step = st.step_index
        st.step_index += 1

        # Warmup: no CFG (return pure conditional). Matches repo.
        if cfg.no_cfg_warmup_steps > 0 and step < cfg.no_cfg_warmup_steps:
            # Make CFG output equal to conditional for any cfg by setting uncond=cond.
            st.prev_guidance = None
            return cond_pred, cond_pred

        # Optional window
        active = cfg.active_start_step <= step <= cfg.active_end_step

        # If cfg <= 1 and user wants to skip, do nothing
        if cfg.apply_only_if_cfg_gt_1 and cfg_scale_f <= 1.0:
            return cond_pred, uncond_pred

        if not (cfg.enable_smc and active):
            return cond_pred, uncond_pred

        # Compute guidance e_t
        guidance = cond_pred - uncond_pred

        # Reset if shape changes (multi-batch / resolution change / etc.)
        if cfg.reset_on_shape_change and st.prev_guidance is not None:
            if st.prev_guidance.shape != guidance.shape or st.prev_guidance.device != guidance.device:
                st.prev_guidance = None

        # Choose compute dtype
        math_dtype = _to_torch_dtype(cfg.math_dtype, guidance)
        store_dtype = _to_torch_dtype(cfg.state_dtype, guidance)

        g = guidance.to(dtype=math_dtype)

        # Initialize prev as current guidance (repo behavior)
        if st.prev_guidance is None:
            prev = g.detach() if cfg.detach_prev else g
        else:
            prev = st.prev_guidance.to(dtype=math_dtype)

        # Sliding surface: s_t = (e_t - e_{t-1}) + lambda * e_{t-1}
        s = (g - prev) + (float(cfg.smc_lambda) * prev)

        # Switching control u_sw
        u_sw = self._switching_control(s, float(cfg.smc_k), cfg)

        # Apply correction
        g2 = g + u_sw

        # Store prev AFTER correction (repo behavior)
        st.prev_guidance = (g2.detach() if cfg.detach_prev else g2).to(dtype=store_dtype)

        # Convert back to original dtype
        g2 = g2.to(dtype=guidance.dtype)
        cond_pred_mod = uncond_pred + g2  # equals cond_pred + u_sw (in guided space)
        return cond_pred_mod, uncond_pred

    @staticmethod
    @torch.no_grad()
    def _switching_control(s: torch.Tensor, k: float, cfg: CFGCtrlConfig) -> torch.Tensor:
        mode = cfg.switch_mode
        eps = float(cfg.boundary_epsilon)

        # Optional relative epsilon (scaled by mean |s|)
        if eps > 0.0 and cfg.epsilon_mode == "relative":
            # scalar mean over all elements (cheap, stable)
            scale = float(s.abs().mean().clamp(min=1e-12).item())
            eps = eps * scale

        if mode == "sign" or eps <= 0.0:
            # Paper + repo: elementwise sign()
            return (-k) * torch.sign(s)

        if mode == "sat":
            # Boundary-layer / saturation: s / (|s|+eps)
            return (-k) * (s / (s.abs() + eps))

        if mode == "tanh":
            # Smooth sign approximation
            return (-k) * torch.tanh(s / eps)

        if mode == "vector_norm":
            # Vector normalization per-batch element:
            # u = -k * s / (||s|| + eps)
            b = s.shape[0] if s.ndim > 0 else 1
            s_flat = s.reshape(b, -1)
            n = torch.linalg.vector_norm(s_flat, dim=1, keepdim=True).clamp(min=1e-12)
            n = n.view([b] + [1] * (s.ndim - 1))
            return (-k) * (s / (n + eps))

        return (-k) * torch.sign(s)
