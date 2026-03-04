from __future__ import annotations

from typing import Any, Dict, Tuple

import comfy.model_patcher
import comfy.samplers

from .cfg_ctrl import CFGCtrlConfig, CFGCtrlController


CATEGORY = "sampling/custom_sampling/guiders"


class Guider_CFGCtrl(comfy.samplers.CFGGuider):
    """
    ComfyUI CFGGuider subclass that injects an SMC-CFG pre_cfg hook.
    """

    def __init__(self, model_patcher):
        super().__init__(model_patcher)
        self._cfg_ctrl_controller: CFGCtrlController | None = None

    def set_cfg_ctrl(self, config: CFGCtrlConfig) -> None:
        self._cfg_ctrl_controller = CFGCtrlController(config)

    def _cfg_ctrl_reset(self) -> None:
        if self._cfg_ctrl_controller is not None:
            self._cfg_ctrl_controller.reset()

    def outer_sample(self, *args, **kwargs):
        # Reset state for every sampling run (most robust)
        self._cfg_ctrl_reset()
        return super().outer_sample(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}

        # Clone options so we don't mutate upstream dicts
        mo = comfy.model_patcher.create_model_options_clone(model_options)

        if self._cfg_ctrl_controller is not None:
            # Append our pre_cfg hook at the end (plays nicer with other pre_cfg modifiers)
            mo = comfy.model_patcher.set_model_options_pre_cfg_function(
                mo,
                self._cfg_ctrl_pre_cfg_hook,
                disable_cfg1_optimization=True,
            )

        return comfy.samplers.sampling_function(
            self.inner_model,
            x,
            timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            self.cfg,
            model_options=mo,
            seed=seed,
        )

    def _cfg_ctrl_pre_cfg_hook(self, args: Dict[str, Any]):
        out = args.get("conds_out", None)
        if out is None or len(out) < 2:
            return out

        cond_pred = out[0]
        uncond_pred = out[1]

        # If uncond is missing for some reason, do nothing safely
        if uncond_pred is None:
            return out

        cfg_scale = float(args.get("cond_scale", 1.0))

        cond_mod, uncond_mod = self._cfg_ctrl_controller.apply(
            cond_pred=cond_pred,
            uncond_pred=uncond_pred,
            cfg_scale=cfg_scale,
        )
        return [cond_mod, uncond_mod]


class CFGCtrlSMCGuiderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "enable_smc": ("BOOLEAN", {"default": True}),
                "smc_lambda": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "smc_k": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.01}),
                "no_cfg_warmup_steps": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "active_start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "active_end_step": (
                    "INT",
                    {"default": 1000000000, "min": 0, "max": 1000000000, "step": 1},
                ),
                "switch_mode": (("sign", "sat", "tanh", "vector_norm"), {"default": "sign"}),
                "boundary_epsilon": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "epsilon_mode": (("absolute", "relative"), {"default": "absolute"}),
                "math_dtype": (("float32", "match", "float16", "bfloat16"), {"default": "float32"}),
                "state_dtype": (("match", "float16", "bfloat16", "float32"), {"default": "match"}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = CATEGORY

    def get_guider(
        self,
        model,
        positive,
        negative,
        cfg: float,
        enable_smc: bool,
        smc_lambda: float,
        smc_k: float,
        no_cfg_warmup_steps: int,
        active_start_step: int,
        active_end_step: int,
        switch_mode: str,
        boundary_epsilon: float,
        epsilon_mode: str,
        math_dtype: str,
        state_dtype: str,
    ) -> Tuple[Any]:
        guider = Guider_CFGCtrl(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        config = CFGCtrlConfig(
            enable_smc=bool(enable_smc),
            smc_lambda=float(smc_lambda),
            smc_k=float(smc_k),
            no_cfg_warmup_steps=int(no_cfg_warmup_steps),
            active_start_step=int(active_start_step),
            active_end_step=int(active_end_step),
            switch_mode=switch_mode,  # type: ignore[arg-type]
            boundary_epsilon=float(boundary_epsilon),
            epsilon_mode=epsilon_mode,  # type: ignore[arg-type]
            math_dtype=math_dtype,  # type: ignore[arg-type]
            state_dtype=state_dtype,  # type: ignore[arg-type]
        )
        guider.set_cfg_ctrl(config)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "CFGCtrlSMCGuider": CFGCtrlSMCGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFGCtrlSMCGuider": "CFG-Ctrl / SMC-CFG Guider",
}
