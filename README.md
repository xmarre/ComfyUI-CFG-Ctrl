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
This node outputs a `GUIDER`. Use it with ComfyUI custom sampling nodes that accept `GUIDER` input
(for example `SamplerCustom` / `SamplerCustomAdvanced`).

Steps:
1. Create your model + conditioning (positive/negative) as usual.
2. Add `CFG-Ctrl / SMC-CFG Guider` and connect model/positive/negative.
3. Feed the `GUIDER` into your custom sampler node.

## Implementation notes
- The guider prefers a `sampler_cfg_function` hook so it can compute the final guided denoised tensor directly.
  This avoids the extra numeric round trip of encoding the corrected guidance back into `cond` and letting ComfyUI
  subtract `uncond` again.
- If another node has already installed a `sampler_cfg_function`, this node falls back to a `sampler_pre_cfg_function`
  so it does not clobber the existing hook.
- Logical step counting is based on sigma changes, not raw model-call count. This makes warmup and step windows track
  the denoising schedule more closely across different samplers.
- `active_start_percent` / `active_end_percent` are converted through `model_sampling.percent_to_sigma(...)` when
  available, so you can gate the controller by denoising progress in a sampler-independent way.

## Parameters
- `cfg`:
  Standard CFG scale.

- `smc_lambda`:
  Sliding-surface shaping. The paper reports values around `6` for its selected settings.

- `smc_k`:
  Switching gain. This is strongly model-dependent in the paper and examples. The node default (`0.3`) is a generic
  starter value, not a claimed paper-optimal default for every model.

- `no_cfg_warmup_steps`:
  First `N` logical sampling steps output the pure conditional prediction directly. This is not "unconditional"; it is
  simply conditional-only warmup with no CFG mixing.

- `active_start_step` / `active_end_step`:
  Optional logical-step window. Logical steps are tracked from sigma changes, so this is more stable than raw denoiser
  call counts.

- `active_start_percent` / `active_end_percent`:
  Optional denoising-progress window in `[0, 1]`. This is usually the more portable way to gate the controller across
  samplers and schedules.

- `switch_mode`:
  - `sign`: paper / official repo behavior
  - `sat` / `tanh`: smooth boundary-layer variants to reduce chattering
  - `vector_norm`: normalized vector controller per batch item

- `boundary_epsilon`:
  Used by `sat`, `tanh`, and `vector_norm`.

- `epsilon_mode`:
  - `absolute`: fixed epsilon
  - `relative`: epsilon scales with mean `|s|` per batch item instead of one global batch-wide scalar

- `apply_only_if_cfg_gt_1`:
  Optional compatibility toggle. Off by default so the node does not silently disable itself at `cfg <= 1`.

- `reset_on_shape_change`:
  Resets controller memory if latent shape or device changes mid-run.

- `detach_prev`:
  Stores the previous corrected guidance as a detached tensor.

## Practical defaults (starting points)
- Generic default: `lambda=6.0`, `k=0.3`, `switch_mode=sign`
- FLUX-like flow models often tolerate higher `k`
- If you see chattering or overshoot, try `tanh` or `sat` with `boundary_epsilon` around `1e-3 .. 1e-2`
- If you want sampler-independent activation windows, prefer `active_start_percent` / `active_end_percent`
