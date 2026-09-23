# H3 continuation training (anti-burn)

Long H3 videos are usually made by chaining clips: each new clip is generated with the previous clip's last 39 frames
copied in as a protected prefix. Over a chain, quality drifts. Faces get harsh and etched, skin gets mottled, and colour
and brightness drift. Each clip copies the drift from its context and adds a little more. Across about a minute of chained
video, measured face texture roughly doubles.

This is an opt-in training mode that teaches a LoRA to continue **cleanly** from a slightly degraded context, in the
spirit of error-recycling approaches such as Stable Video Infinity. The goal is prevention: keep each new clip at the
quality of a fresh generation, rather than repairing video that is already badly burned.

## How it works

- `src/continuation.py` applies a synthetic burn to a clip's opening frames. The burn has harder facial lines and
  fine texture (a two-scale unsharp mask), mottled grain, slight saturation, and a darker frame with crushed blacks.
  Every effect scales from zero, so level 0 is the clean clip itself.
- At latent-caching time, each clip's first N frames are re-encoded once per burn level and cached as latent extras.
  The clean context is encoded the same standalone way as the burned ones, so encoding gives the model no cue.
- At each training step, one context is injected as the protected prefix. It is pinned at the same timestep inference
  uses for protected rows (t = 0.999), and the loss on the prefix is neutralised, so only the frames after it teach.

## Config

```yaml
model:
  arch: minimax_h3
  model_kwargs:
    continuation_context_frames: 39              # protected prefix, a 17n+5 frame run (39 = 12 latent steps)
    continuation_burn_levels: [0, 0.05, 0.1, 0.2] # contexts cached per clip; 0 is the clean clip
    continuation_clean_share: 0.5                # share of steps that draw the clean context
```

Clips must be longer than the context. If you turn this on for a dataset that already has a latent cache, delete the
cache first. The trainer prints a warning if a batch has no burned context.

Calibration: at 512 px, levels 0.05, 0.1 and 0.2 add about +5%, +9% and +17% face texture. One real chained generation
adds about +19%.

## Status (experimental)

- v1 (every context burned, strong levels) stopped the drift but softened every clip. A chain lost about 55% of its
  fine detail, because the LoRA learned "always be softer than the context".
- v2 (half clean contexts, the calibrated levels above) fixed the softening, but dulled fine eye detail and degraded
  audio. The suspects are a train/inference mismatch in the audio schedule, no protection of the audio prefix, and
  training at 512 px.
- Next: train and evaluate on one matched setup, protect the audio prefix, and try a higher-resolution run.

## Tests

`test_h3_continuation.py` in the repo root has CPU contracts with stand-in models. It stubs the package `__init__`,
because another model needs a GPU at import time:

```bash
CUDA_VISIBLE_DEVICES= python -m unittest test_h3_continuation
```
