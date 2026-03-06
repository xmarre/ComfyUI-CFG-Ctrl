from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import comfy.model_patcher
import comfy.samplers

from .cfg_ctrl import CFGCtrlConfig, CFGCtrlController


CATEGORY = "sampling/custom_sampling/guiders"


class Guider_CFGCtrl(comfy.samplers.CFGGuider):
    """
    ComfyUI CFGGuider subclass that injects CFG-Ctrl / SMC-CFG into sampling.

    Preferred path:
      - sampler_cfg_function for exact final guided denoised output
        (avoids an extra cond/uncond encode-decode round trip).

    Fallback path:
      - sampler_pre_cfg_function if a sampler_cfg_function already exists in the
        incoming model options and we do not want to clobber it.
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
        self._cfg_ctrl_reset()
        return super().outer_sample(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}

        mo = comfy.model_patcher.create_model_options_clone(model_options)

        controller = self._cfg_ctrl_controller
        if controller is not None and controller.config.needs_custom_cfg_for_run(float(self.cfg)):
            if "sampler_cfg_function" in mo:
                mo = comfy.model_patcher.set_model_options_pre_cfg_function(
                    mo,
                    self._cfg_ctrl_pre_cfg_hook,
                    disable_cfg1_optimization=True,
                )
            else:
                mo["sampler_cfg_function"] = self._cfg_ctrl_sampler_cfg_hook
                mo["disable_cfg1_optimization"] = True

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

    def _cfg_ctrl_sampler_cfg_hook(self, args: Dict[str, Any]):
        controller = self._cfg_ctrl_controller
        if controller is None:
            return args["uncond"] + (args["cond"] - args["uncond"]) * float(args.get("cond_scale", 1.0))

        cond_pred = args["cond_denoised"]
        uncond_pred = args["uncond_denoised"]
        sigma = args.get("sigma", None)
        cfg_scale = float(args.get("cond_scale", 1.0))
        x = args["input"]

        guided = controller.apply_guided(
            cond_pred=cond_pred,
            uncond_pred=uncond_pred,
            cfg_scale=cfg_scale,
            sigma=sigma,
        )
        return x - guided

    def _cfg_ctrl_pre_cfg_hook(self, args: Dict[str, Any]):
        out = args.get("conds_out", None)
        if out is None or len(out) < 2:
            return out

        cond_pred = out[0]
        uncond_pred = out[1]
        if uncond_pred is None:
            return out

        controller = self._cfg_ctrl_controller
        if controller is None:
            return out

        cfg_scale = float(args.get("cond_scale", 1.0))
        sigma = args.get("sigma", None)

        cond_mod, uncond_mod = controller.apply_pre_cfg(
            cond_pred=cond_pred,
            uncond_pred=uncond_pred,
            cfg_scale=cfg_scale,
            sigma=sigma,
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
                "smc_lambda": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "smc_k": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.01}),
                "no_cfg_warmup_steps": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "active_start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "active_end_step": ("INT", {"default": 1000000000, "min": 0, "max": 1000000000, "step": 1}),
                "active_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "active_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "switch_mode": (("sign", "sat", "tanh", "vector_norm"), {"default": "sign"}),
                "boundary_epsilon": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                "epsilon_mode": (("absolute", "relative"), {"default": "absolute"}),
                "math_dtype": (("float32", "match", "float16", "bfloat16"), {"default": "float32"}),
                "state_dtype": (("match", "float16", "bfloat16", "float32"), {"default": "match"}),
                "apply_only_if_cfg_gt_1": ("BOOLEAN", {"default": False}),
                "reset_on_shape_change": ("BOOLEAN", {"default": True}),
                "detach_prev": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = CATEGORY

    @staticmethod
    def _percent_to_sigma(model, percent: float) -> Optional[float]:
        try:
            model_sampling = model.get_model_object("model_sampling")
            sigma = model_sampling.percent_to_sigma(float(percent))
            return float(sigma)
        except Exception:
            return None

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
        active_start_percent: float,
        active_end_percent: float,
        switch_mode: str,
        boundary_epsilon: float,
        epsilon_mode: str,
        math_dtype: str,
        state_dtype: str,
        apply_only_if_cfg_gt_1: bool,
        reset_on_shape_change: bool,
        detach_prev: bool,
    ) -> Tuple[Any]:
        guider = Guider_CFGCtrl(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        step_start = int(active_start_step)
        step_end = int(active_end_step)
        if step_end < step_start:
            step_start, step_end = step_end, step_start

        pct_start = max(0.0, min(1.0, float(active_start_percent)))
        pct_end = max(0.0, min(1.0, float(active_end_percent)))
        if pct_end < pct_start:
            pct_start, pct_end = pct_end, pct_start

        config = CFGCtrlConfig(
            enable_smc=bool(enable_smc),
            smc_lambda=float(smc_lambda),
            smc_k=float(smc_k),
            no_cfg_warmup_steps=int(no_cfg_warmup_steps),
            active_start_step=step_start,
            active_end_step=step_end,
            active_start_percent=pct_start,
            active_end_percent=pct_end,
            active_sigma_start=self._percent_to_sigma(model, pct_start),
            active_sigma_end=self._percent_to_sigma(model, pct_end),
            switch_mode=switch_mode,  # type: ignore[arg-type]
            boundary_epsilon=float(boundary_epsilon),
            epsilon_mode=epsilon_mode,  # type: ignore[arg-type]
            math_dtype=math_dtype,  # type: ignore[arg-type]
            state_dtype=state_dtype,  # type: ignore[arg-type]
            apply_only_if_cfg_gt_1=bool(apply_only_if_cfg_gt_1),
            reset_on_shape_change=bool(reset_on_shape_change),
            detach_prev=bool(detach_prev),
        )
        guider.set_cfg_ctrl(config)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "CFGCtrlSMCGuider": CFGCtrlSMCGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFGCtrlSMCGuider": "CFG-Ctrl / SMC-CFG Guider",
}
