"""Continuation training for MiniMax H3: learn to continue CLEANLY from a burned, protected context.

Long H3 videos are made by chaining clips: each clip is generated with the previous clip's final 39 frames copied in
as a protected prefix (ComfyUI masked AV continuation). A turbo model adds a little crunch, saturation and darkening
every clip, copies it from the prefix and adds more, so long chains "burn". This trains a LoRA on clean clips whose
first 39 frames are replaced by a synthetically burned copy, protected exactly the way inference protects the prefix,
with the loss on the frames after it only: the model learns not to carry the burn forward.

Enabled per run with ``model_kwargs``:
  continuation_context_frames: 39     # the protected prefix (a 17n+5 run; 39 = 12 latent steps)
  continuation_burn_variants: 3       # burned copies cached per clip, mild -> strong; one is drawn per step
"""
import math

import torch
import torch.nn.functional as F

CONTEXT_T = 0.999  # inference injects protected rows at 0.999 * clean + 0.001 * noise and labels them t = 0.999
EXTRA_PREFIX = "ctx_burn"


def latent_steps(frames: int) -> int:
    """Latent steps of a 17n+5 frame run (5 -> 2, 22 -> 7, 39 -> 12, ...)."""
    if frames < 5 or (frames - 5) % 17:
        raise ValueError(f"continuation context must be a 17n+5 frame run, got {frames}")
    return 2 + 5 * ((frames - 5) // 17)


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over the last two dims of (N, C, H, W)."""
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = (kernel / kernel.sum()).to(x.dtype)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def burn_frames(frames: torch.Tensor, level: float) -> torch.Tensor:
    """The chain drift measured on real Stories, at strength ``level`` in [0, 1] (1 = a long, badly burned chain):
    harder facial lines and fine texture (two-scale unsharp mask), mottled grain, a little saturation, a slightly
    darker frame with crushed blacks. Checked by eye against a real burned Story frame (2026-09-22). ``frames`` is
    (T, C, H, W) in [-1, 1]; returns the same shape and range."""
    level = float(max(0.0, min(1.0, level)))
    x = (frames.float().clamp(-1, 1) + 1) / 2  # [0, 1]
    # the "etched" face (texture measured +40-60% on faces over long chains): fine grain plus harder lines and
    # folds, i.e. contrast at two scales relative to the frame height (fine ~0.3%, lines ~1%)
    scale = frames.shape[-2] / 480.0
    x = x + (0.15 + 0.30 * level) * (x - _blur(x, max(0.8, 1.2 * scale)))
    x = x + (0.20 + 0.50 * level) * (x - _blur(x, max(1.5, 4.0 * scale)))
    # mottled, blotchy skin grain (slightly blurred luminance noise, the same pattern on every frame of the run)
    generator = torch.Generator().manual_seed(1234)
    grain = torch.randn((1, 1) + tuple(x.shape[-2:]), generator=generator).to(x.device, x.dtype)
    x = x + (0.008 + 0.022 * level) * _blur(grain, max(0.7, 1.0 * scale))
    # a little extra saturation (+3..7%); real chains drift in colour both ways, so this stays mild
    grey = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = grey + (1.03 + 0.04 * level) * (x - grey)
    # somewhat darker with crushed blacks (the frame's own darkening; kept mild: part of the measured -20% is the
    # scene, and a dark frame alone is not what reads as burned)
    x = x.clamp(0, 1) ** (1.02 + 0.10 * level)
    black = 0.005 + 0.02 * level
    x = ((x - black) / (1 - black)).clamp(0, 1)
    return (x * 2 - 1).to(frames.dtype)


def variant_levels(count: int):
    """Burn strengths of the cached variants, mild to strong."""
    return [(i + 1) / count for i in range(count)]


def pick_context(extras: dict, batch_size: int, variants: int, generator=None):
    """One cached burned context per sample, drawn uniformly from the variants: (B, 24, s, h, w) or None."""
    options = [extras.get(f"{EXTRA_PREFIX}{i}") for i in range(variants)]
    options = [o for o in options if o is not None]
    if not options:
        return None
    choice = torch.randint(len(options), (batch_size,), generator=generator)
    return torch.stack([options[int(choice[b])][b] for b in range(batch_size)])


def inject_context(noisy: torch.Tensor, context: torch.Tensor, generator=None) -> torch.Tensor:
    """Replace the first s latent steps of the noisy target with the protected context, noised exactly as inference
    injects it (CONTEXT_T * clean + (1 - CONTEXT_T) * noise)."""
    steps = context.shape[2]
    out = noisy.clone()
    context = context.to(out.device, out.dtype)
    noise = torch.randn(context.shape, generator=generator, dtype=torch.float32).to(out.device, out.dtype)
    out[:, :, :steps] = CONTEXT_T * context + (1.0 - CONTEXT_T) * noise
    return out


def context_row_count(steps: int, latent_height: int, latent_width: int, patch=(1, 2, 2)) -> int:
    """Packed video rows covering the first ``steps`` latent steps (rows are frame-major)."""
    return (steps // patch[0]) * (latent_height // patch[1]) * (latent_width // patch[2])


def neutralize_prefix(pred: torch.Tensor, noisy: torch.Tensor, clean: torch.Tensor, sigma: torch.Tensor,
                      steps: int) -> torch.Tensor:
    """Set the prediction over the protected prefix to the trainer's exact target (noise - clean), so the prefix
    adds no loss and no gradient: only the frames after it teach. Flow matching: noisy = (1 - s) clean + s noise,
    so noise - clean = (noisy - clean) / s. ``noisy`` must be the trainer's input BEFORE the context replaced it."""
    out = pred.clone()
    s = sigma.to(out.device, torch.float32).view(-1, 1, 1, 1, 1).clamp(min=1e-6)
    target = (noisy[:, :, :steps].float() - clean[:, :, :steps].to(noisy.device).float()) / s
    out[:, :, :steps] = target.to(out.dtype).detach()
    return out
