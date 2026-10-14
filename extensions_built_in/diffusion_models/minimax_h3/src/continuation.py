"""Continuation training for MiniMax H3: learn to continue CLEANLY from a burned, protected context.

Long H3 videos are made by chaining clips: each clip is generated with the previous clip's final 39 frames copied in
as a protected prefix (ComfyUI masked AV continuation). A turbo model adds a little crunch, saturation and darkening
every clip, copies it from the prefix and adds more, so long chains "burn". This trains a LoRA on clean clips whose
first 39 frames are replaced by a synthetically burned copy, protected exactly the way inference protects the prefix,
with the loss on the frames after it only: the model learns not to carry the burn forward.

Enabled per run with ``model_kwargs``:
  continuation_context_frames: 39              # the protected prefix (a 17n+5 run; 39 = 12 latent steps)
  continuation_burn_levels: [0, 0.1, 0.2, 0.4] # contexts cached per clip; level 0 is the clean clip itself
  continuation_clean_share: 0.5                # share of steps that draw the level-0 (clean) context
  continuation_no_context_share: 0.15         # share of steps with no context at all: plain generation
  continuation_protect_audio: true             # protect the context's audio too, with inference's own mask
(older runs: ``continuation_burn_variants: N`` = levels 1/N .. 1, no clean context.)
The dataset must keep clips at real speed (``shrink_video_to_frames: false``); check_dataset_config refuses otherwise.

v1 (2026-09-22) had no clean context: every training context was burned, so the LoRA learned "always make the new
frames softer than the context" and applied it to clean contexts too; a real chain lost about 55% of its fine detail
over 11 clips. A clean context (level 0, encoded exactly like the burned ones, so encoding is no cue) teaches that
detail is kept when nothing is wrong, and the burned levels are near what one real clip adds.

v3 (2026-09-24), after v2 dulled eyes and made the audio quiet and noise-like:
- ai-toolkit's ``shrink_video_to_frames`` defaults to true, and it squeezed every clip longer than ``num_frames`` into
  it: v1/v2 trained on video sped up 1.55x (median, up to 2.9x) with the audio linearly resampled to match
  (pitched up and aliased). Continuation training now requires real-speed clips.
- Inference protects the previous clip's AUDIO as well as its frames. Training used to protect only the frames and
  trained the audio from scratch while the picture had a context. The audio prefix is now protected with the same
  mask inference builds: exactly clean, labelled t = 1, the last ticks released on a half-cosine.
- A share of steps has no context at all, so the LoRA also learns to leave a Story's first clip (plain generation,
  where it never saw a context) as it was.
"""
import math
from typing import List, NamedTuple

import torch
import torch.nn.functional as F

CONTEXT_T = 0.999  # inference injects protected rows at 0.999 * clean + 0.001 * noise and labels them t = 0.999
EXTRA_PREFIX = "ctx_burn"

# The audio side of the masked AV continuation (ComfyUI-H3-MaskedExtension, existing_video_extension.py, and the H3
# model's process_denoise_mask / scale_latent_inpaint): 40 Hz audio latents at 24 fps video; protected audio rows are
# exactly clean and labelled t = max(t_a, 1.0); the last AUDIO_FEATHER_TICKS protected ticks release on a half-cosine;
# mask values are quantised to 1/256, <= 0.05 -> 0 and >= 0.995 -> 1. Mask value m puts a row at sigma = m * sigma_a.
FPS = 24
AUDIO_HZ = 40
AUDIO_FEATHER_TICKS = 8
AUDIO_CONTEXT_T = 1.0
AUDIO_CHANNELS = 2  # packed audio rows are channel-major: all ticks of channel 0, then channel 1


class Settings(NamedTuple):
    frames: int = 0
    levels: List[float] = []
    clean_share: float = 0.0
    no_context_share: float = 0.0
    protect_audio: bool = False


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
    if level == 0.0:
        return frames.clone()  # the clean context: identical frames, encoded the same way as the burned ones
    x = (frames.float().clamp(-1, 1) + 1) / 2  # [0, 1]
    # the "etched" face (texture measured +40-60% on faces over long chains): fine grain plus harder lines and
    # folds, i.e. contrast at two scales relative to the frame height (fine ~0.3%, lines ~1%)
    scale = frames.shape[-2] / 480.0
    # every effect scales from nothing at level 0 (v1 had a floor, so even its mildest context was strongly burned);
    # level 1 equals the strongest v1 context
    x = x + 0.45 * level * (x - _blur(x, max(0.8, 1.2 * scale)))
    x = x + 0.70 * level * (x - _blur(x, max(1.5, 4.0 * scale)))
    # mottled, blotchy skin grain (slightly blurred luminance noise, the same pattern on every frame of the run)
    generator = torch.Generator().manual_seed(1234)
    grain = torch.randn((1, 1) + tuple(x.shape[-2:]), generator=generator).to(x.device, x.dtype)
    x = x + 0.030 * level * _blur(grain, max(0.7, 1.0 * scale))
    # a little extra saturation (+3..7%); real chains drift in colour both ways, so this stays mild
    grey = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = grey + (1.0 + 0.07 * level) * (x - grey)
    # somewhat darker with crushed blacks (the frame's own darkening; kept mild: part of the measured -20% is the
    # scene, and a dark frame alone is not what reads as burned)
    x = x.clamp(0, 1) ** (1.0 + 0.12 * level)
    black = 0.025 * level
    x = ((x - black) / (1 - black)).clamp(0, 1)
    return (x * 2 - 1).to(frames.dtype)


def variant_levels(count: int):
    """Burn strengths of the cached variants, mild to strong (the v1 form: no clean context)."""
    return [(i + 1) / count for i in range(count)]


def settings(kw: dict) -> Settings:
    """Continuation settings from model_kwargs; Settings() (frames 0) when continuation is off."""
    frames = int(kw.get("continuation_context_frames", 0) or 0)
    if not frames:
        return Settings()
    latent_steps(frames)  # validates the 17n+5 run
    levels = kw.get("continuation_burn_levels")
    levels = [float(v) for v in levels] if levels else variant_levels(max(1, int(kw.get("continuation_burn_variants", 3))))
    if any(not 0.0 <= v <= 1.0 for v in levels):
        raise ValueError(f"continuation_burn_levels must be numbers in [0, 1], got {levels}")
    share = float(kw.get("continuation_clean_share", 0.0) or 0.0)
    if not 0.0 <= share <= 1.0 or (share and 0.0 not in levels):
        raise ValueError("continuation_clean_share needs a value in [0, 1] and a level 0 in continuation_burn_levels")
    no_context = float(kw.get("continuation_no_context_share", 0.0) or 0.0)
    if not 0.0 <= no_context < 1.0:
        raise ValueError(f"continuation_no_context_share must be in [0, 1), got {no_context}")
    protect_audio = bool(kw.get("continuation_protect_audio", False))
    if protect_audio:
        audio_context_steps(frames)  # the prefix must end on the shared 24 fps / 40 Hz boundary
    return Settings(frames, levels, share, no_context, protect_audio)


def draw_no_context(no_context_share: float, generator=None) -> bool:
    """True for a step that trains plain generation (no context at all), with probability ``no_context_share``."""
    return bool(no_context_share) and torch.rand((), generator=generator).item() < no_context_share


def step_draws(batch, no_context_share: float, pick):
    """One draw per training step: ``(plain_step, context)``. Every pass of a step (the audio teacher, the primary)
    must see the same input, so the draw is kept on the batch; ``pick()`` chooses the context of a continuation step."""
    draws = getattr(batch, "continuation_draws", None)
    if draws is None:
        plain = draw_no_context(no_context_share)
        draws = (plain, None if plain else pick())
        batch.continuation_draws = draws
    return draws


def check_real_speed(dataset_config) -> None:
    """Refuse a video dataset that would squeeze clips longer than num_frames into num_frames (ai-toolkit's default
    shrink_video_to_frames): the context and the frames after it must run at the speed inference sees, and the
    squeeze also resamples the audio (pitch and aliasing). Real speed is a random num_frames window at the dataset
    fps (shrink_video_to_frames: false), or auto_frame_count with trim_auto_frame_count_tail."""
    if dataset_config is None or int(getattr(dataset_config, "num_frames", 1) or 1) <= 1:
        return
    if getattr(dataset_config, "auto_frame_count", False) and getattr(dataset_config, "trim_auto_frame_count_tail", True):
        return
    if getattr(dataset_config, "shrink_video_to_frames", True):
        raise ValueError(
            "MiniMax-H3 continuation training needs clips at real speed, but this dataset has shrink_video_to_frames "
            "on (the ai-toolkit default), which squeezes every longer clip into num_frames and resamples its audio. "
            "Set shrink_video_to_frames: false in the dataset, and delete any _latent_cache made with it on.")


def audio_context_steps(frames: int) -> int:
    """40 Hz audio latents covering a ``frames`` prefix at 24 fps (39 -> 65). Inference only protects AV prefixes that
    end exactly on both grids (39, 90, 141, ... frames), so anything else is refused."""
    if (frames * AUDIO_HZ) % FPS:
        raise ValueError(f"an audio-protected context must end on the shared 24 fps / 40 Hz grid (39, 90, 141, ... "
                         f"frames), got {frames}")
    return frames * AUDIO_HZ // FPS


def audio_context_mask(total_steps: int, context_steps: int, feather: int = AUDIO_FEATHER_TICKS) -> torch.Tensor:
    """(total_steps,) inference audio mask, 0 = preserve and 1 = generate: the context ticks preserved, the last
    ``feather`` of them released on a half-cosine, then quantised the way the H3 model's process_denoise_mask does."""
    mask = torch.ones(total_steps, dtype=torch.float32)
    feather = max(0, min(int(feather), context_steps))
    hard = context_steps - feather
    mask[:hard] = 0.0
    if feather:
        i = torch.arange(1, feather + 1, dtype=torch.float32)
        mask[hard:context_steps] = 0.5 - 0.5 * torch.cos(math.pi * i / float(feather))
    mask = torch.round(mask * 256.0) / 256.0
    return mask.masked_fill(mask >= 0.995, 1.0).masked_fill(mask <= 0.05, 0.0)


def engine_audio_context_mask(total_steps: int, context_steps: int, feather: int = AUDIO_FEATHER_TICKS) -> torch.Tensor:
    """(total_steps,) audio mask as the pod engine's DiT receives it (ComfyUI-H3-test's native token-grid path,
    MiniMaxH3._token_grid_masks): the same half-cosine release rounded up to 1/256, with no snapping, so the first
    feathered tick is 10/256 rather than 0. What an R+ recording carries; audio_context_mask stays the mask of the
    legacy samples."""
    mask = torch.ones(total_steps, dtype=torch.float32)
    feather = max(0, min(int(feather), context_steps))
    hard = context_steps - feather
    mask[:hard] = 0.0
    if feather:
        i = torch.arange(1, feather + 1, dtype=torch.float32)
        mask[hard:context_steps] = 0.5 - 0.5 * torch.cos(torch.pi * i / float(feather))
    return torch.ceil(mask * 256.0) / 256.0


def audio_row_mask(mask: torch.Tensor) -> torch.Tensor:
    """A per-tick audio mask -> per packed row (channel-major rows: the ticks of channel 0, then of channel 1)."""
    return mask.repeat(AUDIO_CHANNELS)


def protect_audio_rows(clean: torch.Tensor, noise: torch.Tensor, sigma_a: torch.Tensor, row_mask: torch.Tensor):
    """Audio rows (B, rows, C) and their timesteps (B, rows) as inference sets them: a row with mask m sits at
    sigma = m * sigma_a (so fully preserved rows are exactly clean) and is labelled 1 - m * sigma_a, capped at the
    preserved-row timestep max(t_a, 1.0). Rows with m = 1 match the ordinary noisy audio."""
    m = row_mask.to(clean.device, torch.float32).view(1, -1)
    sigma_rows = m * sigma_a.to(clean.device, torch.float32).view(-1, 1)  # (B, rows)
    s = sigma_rows.unsqueeze(-1)
    rows = (1.0 - s) * clean.float() + s * noise.float()
    t_pin = torch.clamp(1.0 - sigma_a.to(clean.device, torch.float32).view(-1, 1), min=AUDIO_CONTEXT_T)
    return rows.to(clean.dtype), torch.minimum(1.0 - sigma_rows, t_pin)


def blend_protected_audio(pred: torch.Tensor, target: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    """Only generated audio teaches, at the weight inference gives it: the prediction becomes target + m * (pred -
    target), so preserved rows (m = 0) add no loss and no gradient and a released row's error is scaled by m."""
    m = row_mask.to(pred.device, pred.dtype).view(1, -1, 1)
    return target.to(pred.dtype).detach() + m * (pred - target.to(pred.dtype).detach())


def pick_context(extras: dict, batch_size: int, levels, clean_share: float = 0.0, generator=None):
    """One cached context per sample: the clean one (level 0) with probability ``clean_share``, else one of the
    burned levels uniformly. With clean_share 0 every cached level is equally likely. (B, 24, s, h, w) or None."""
    if isinstance(levels, int):
        levels = variant_levels(levels)
    found = [(level, extras.get(f"{EXTRA_PREFIX}{i}")) for i, level in enumerate(levels)]
    found = [(level, o) for level, o in found if o is not None]
    if not found:
        return None
    clean = [o for level, o in found if level == 0.0]
    burned = [o for level, o in found if level != 0.0]
    picks = []
    for b in range(batch_size):
        if not clean_share:
            pool = [o for _, o in found]
        elif not burned or torch.rand((), generator=generator).item() < clean_share:
            pool = clean or burned
        else:
            pool = burned
        picks.append(pool[int(torch.randint(len(pool), (1,), generator=generator))][b])
    return torch.stack(picks)


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
