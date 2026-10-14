"""Teacher probe helper node (h3_record/teacher_probe.py; plans_20260926/design_dmd-rollout.md section 8, risk 1): load a
saved H3 AV latent with the noise mask of one of the probe's views.

H3ProbeLoadLatent reads a safetensors file with fp32 `video` (1,24,T,H,W) and `audio` (1,32,2,Ta) tensors (the layout
H3DashboardSaveLatent writes) from the R+ work tree or the Story latent folder only (rplus_nodes.allowed_path), and
returns a copy of the optional template latent (the target latent of MiniMaxH3ReferenceToVideo, whose stream shapes must
match) with samples = NestedTensor((video, audio)). The sampler noises it at the first sigma of its schedule with the
graph's RandomNoise (the same seed gives every view the same noise), so one Euler step from sigma to 0 returns the
model's x0 prediction. H3's noise masks are 0 = preserve / 1 = generate:
  mask "none": no noise mask (decode graphs).
  mask "video_free_audio_pinned": every video step re-noised, the audio exactly as loaded (a fresh text-to-video window
      with nothing of the video pinned: the probe's real-score views; rplus_nodes.pinned_mask).
  mask "context_pinned": the first `context_steps` video steps (12 = the 39-frame carried context) and the whole audio
      as loaded, the rest of the video re-noised (the masked continuation the generator runs: the fake-score proxy).
This file is NOT imported by the pack's __init__.py (read-only for the probe): teacher_probe.py installs a small loader
pack, ComfyUI-H3-TeacherProbe, holding copies of this file and rplus_nodes.py, whose __init__.py registers the node.
"""
import os

from . import rplus_nodes as RN

MASKS = ("none", "video_free_audio_pinned", "context_pinned")
CONTEXT_STEPS = 12  # rplus_layout.CTX_STEPS: the 39 pinned pixel frames of a continuation


def probe_masks(video, audio, mask, context_steps=CONTEXT_STEPS):
    """(video mask (1,1,T,H,W), audio mask (1,1,2,Ta)) float32 for `mask`, None for "none"."""
    if mask not in MASKS:
        raise ValueError(f"H3ProbeLoadLatent: mask {mask!r} is not one of {MASKS}")
    if mask == "none":
        return None
    video_mask, audio_mask = RN.pinned_mask(video, audio)
    if mask == "context_pinned":
        steps = int(context_steps)
        if not 0 < steps < int(video.shape[2]):
            raise ValueError(f"H3ProbeLoadLatent: context_steps {steps} must be between 1 and {int(video.shape[2]) - 1}")
        video_mask[:, :, :steps] = 0.0
    return video_mask, audio_mask


class H3ProbeLoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "", "tooltip": f"A safetensors file under {RN.RP} or "
                                                                           f"{RN.LATENTS}"}),
                             "mask": (list(MASKS), {"default": "none"}),
                             "context_steps": ("INT", {"default": CONTEXT_STEPS, "min": 1, "max": 4096,
                                                       "tooltip": "context_pinned: the pinned opening, in latent steps"})},
                "optional": {"latent": ("LATENT", {"tooltip": "Template: its other keys are kept, its shapes checked"})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3Turbo/record"

    @classmethod
    def IS_CHANGED(cls, path, **kwargs):
        return os.stat(RN.allowed_path(path)).st_mtime_ns

    def load(self, path, mask, context_steps=CONTEXT_STEPS, latent=None):
        import comfy.nested_tensor
        if mask not in MASKS:
            raise ValueError(f"H3ProbeLoadLatent: mask {mask!r} is not one of {MASKS}")
        video, audio = RN.load_streams(RN.allowed_path(path), RN.template_streams(latent))
        out = dict(latent) if latent else {}
        out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
        out.pop("noise_mask", None)
        masks = probe_masks(video, audio, mask, context_steps)
        if masks is not None:
            out["noise_mask"] = comfy.nested_tensor.NestedTensor(masks)
        return (out,)


NODE_CLASS_MAPPINGS = {"H3ProbeLoadLatent": H3ProbeLoadLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ProbeLoadLatent": "MiniMax H3 Teacher Probe Load Latent"}
