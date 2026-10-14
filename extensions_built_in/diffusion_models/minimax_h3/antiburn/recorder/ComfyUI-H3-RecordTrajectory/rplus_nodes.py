"""R+ (repaired-context self-distillation) helper node: load a saved H3 AV latent file as a LATENT.

H3RPlusLoadLatent reads a safetensors file with fp32 `video` (1,24,T,H,W) and `audio` (1,32,2,Ta) tensors, the layout
H3DashboardSaveLatent writes, from the R+ work tree or the Story latent folder only. It returns a copy of the optional
template latent (or a bare one) with samples = NestedTensor((video, audio)).
  mask "none": no noise mask (decode graphs).
  mask "video_free_audio_pinned": noise_mask = NestedTensor((ones(1,1,T,H,W), zeros(1,1,2,Ta))), H3's 0 = preserve /
  1 = generate: the video is re-sampled (the SDEdit repair), the audio stays exactly as loaded.
A template (the target latent of MiniMaxH3ReferenceToVideo) must have the same stream shapes; a mismatch raises.
"""
import os

RP = "/workspace/h3eval/rplus"                 # rplus_layout.RP
LATENTS = "/workspace/h3_story_latents"         # rplus_layout.LATENTS
ROOTS = (RP, LATENTS)
MASKS = ("none", "video_free_audio_pinned")


def allowed_path(path, roots=ROOTS):
    """The real path of `path` if it is a file under one of `roots`; ValueError otherwise."""
    real = os.path.realpath(str(path or ""))
    for root in roots:
        root = os.path.realpath(root)
        if real.startswith(root + os.sep) and os.path.isfile(real):
            return real
    raise ValueError(f"H3RPlusLoadLatent: {path!r} is not a file under {' or '.join(roots)}")


def template_streams(latent):
    samples = (latent or {}).get("samples")
    if samples is None:
        return None
    parts = list(samples.unbind()) if hasattr(samples, "unbind") else list(samples)
    if len(parts) < 2:
        raise ValueError("H3RPlusLoadLatent: the template is not an H3 AV latent (video + audio)")
    return parts[0], parts[1]


def load_streams(path, template=None):
    """(video, audio) from the file, shape-checked, on the template's device and dtype when a template is given."""
    from safetensors.torch import load_file
    saved = load_file(path)
    video, audio = saved["video"].float(), saved["audio"].float()
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError(f"H3RPlusLoadLatent: video must be (1,24,T,H,W), got {tuple(video.shape)}")
    if audio.ndim != 4 or audio.shape[0] != 1 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(f"H3RPlusLoadLatent: audio must be (1,32,2,Ta), got {tuple(audio.shape)}")
    if template is not None:
        tv, ta = template
        if tuple(tv.shape) != tuple(video.shape) or tuple(ta.shape) != tuple(audio.shape):
            raise ValueError(f"H3RPlusLoadLatent: file {tuple(video.shape)} / {tuple(audio.shape)} does not match the "
                             f"template {tuple(tv.shape)} / {tuple(ta.shape)}")
        video = video.to(device=tv.device, dtype=tv.dtype)
        audio = audio.to(device=ta.device, dtype=ta.dtype)
    return video, audio


def pinned_mask(video, audio):
    import torch
    t, h, w = (int(s) for s in video.shape[2:])
    return (torch.ones((1, 1, t, h, w), dtype=torch.float32, device=video.device),
            torch.zeros((1, 1, 2, int(audio.shape[-1])), dtype=torch.float32, device=audio.device))


class H3RPlusLoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "", "tooltip": f"A safetensors file under {RP} or {LATENTS}"}),
                             "mask": (list(MASKS), {"default": "none"})},
                "optional": {"latent": ("LATENT", {"tooltip": "Template: its other keys are kept, its shapes checked"})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3Turbo/record"

    @classmethod
    def IS_CHANGED(cls, path, **kwargs):
        return os.stat(allowed_path(path)).st_mtime_ns

    def load(self, path, mask, latent=None):
        import comfy.nested_tensor
        if mask not in MASKS:
            raise ValueError(f"H3RPlusLoadLatent: mask {mask!r} is not one of {MASKS}")
        video, audio = load_streams(allowed_path(path), template_streams(latent))
        out = dict(latent) if latent else {}
        out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
        out.pop("noise_mask", None)
        if mask == "video_free_audio_pinned":
            out["noise_mask"] = comfy.nested_tensor.NestedTensor(pinned_mask(video, audio))
        return (out,)


NODE_CLASS_MAPPINGS = {"H3RPlusLoadLatent": H3RPlusLoadLatent}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RPlusLoadLatent": "MiniMax H3 R+ Load Latent"}
