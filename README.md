# ComfyUI-CFG-Ctrl (SMC-CFG)

Implements CFG-Ctrl / SMC-CFG (Sliding Mode Control CFG) as a ComfyUI GUIDER.

References:
- Paper: ["CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance"](https://arxiv.org/pdf/2603.03281)
- Official repo: [hanyang-21/CFG-Ctrl](https://github.com/hanyang-21/CFG-Ctrl)

## Install
Place this folder into:

`ComfyUI/custom_nodes/ComfyUI-CFG-Ctrl/`

Restart ComfyUI.

## How to use
This node outputs a GUIDER. Use it with ComfyUI custom sampling nodes that accept `GUIDER` input
(e.g. SamplerCustom / SamplerCustomAdvanced workflows).

Steps:
1. Create your model + conditioning (positive/negative) as usual.
2. Add `CFG-Ctrl / SMC-CFG Guider` and connect model/positive/negative.
3. Feed the GUIDER into your custom sampler node.

## Parameters
- `cfg`:
  Standard CFG scale. SMC-CFG stabilizes behavior at higher cfg.

- `smc_lambda`:
  Sliding-surface decay shaping. Paper commonly uses values around 4–7 (often 6).

- `smc_k`:
  Switching gain. Controls how hard the controller pulls toward the sliding manifold.
  Too low => weak effect. Too high => chattering/instability.

- `no_cfg_warmup_steps`:
  First N steps: use pure conditional prediction (no CFG). Helps avoid early noise blowups.

- `switch_mode`:
  - `sign` (paper/repo): exact switching
  - `sat` / `tanh`: boundary layer to reduce chattering
  - `vector_norm`: normalizes by ||s|| (per-batch), smoother but slightly more compute

- `boundary_epsilon`:
  Only used for sat/tanh/vector_norm. Start with `1e-3` to `1e-2` if you see chattering.

## Practical defaults (starting points)
- FLUX-like flow models:
  cfg 3–6, lambda 5–7, k 0.2–0.8, warmup 1–3

- If you increase cfg above your usual:
  keep lambda ~6, increase k slightly, add warmup 1–3, or set `switch_mode=tanh` with
  `epsilon ~1e-3..1e-2`.
