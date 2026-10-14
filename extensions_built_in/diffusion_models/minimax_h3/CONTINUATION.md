# H3 continuation training (anti-burn)

> The released LoRA (universal v1), its training config, the recorder node, the scorer and a write-up of what
> worked and what failed are in [antiburn/](antiburn/README.md).

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
- With `continuation_protect_audio`, the context's soundtrack is protected too, using the mask the ComfyUI masked AV
  continuation builds: the 65 audio ticks under a 39-frame context are exactly clean and labelled t = 1, the last ones
  release on a half-cosine (the mask is quantised the same way the H3 model quantises it), and only generated audio
  counts towards the loss.
- With `continuation_no_context_share`, that share of steps has no context at all, so the LoRA also learns to leave
  plain generation (a Story's first clip) as it was.
- With `audio_teacher` (a train setting), the audio target is not the clip's soundtrack but the frozen model's own
  audio prediction on the same noisy input: the trained LoRA is switched off for that pass, the assistant LoRA stays on.
  The LoRA then learns its video task and leaves the audio as the base model makes it. Each step's random draws
  (plain step or not, which context) are shared by both passes. It costs one extra forward pass per step.

## The recipe (training and use share it)

- Clip 1: plain text-to-video on fl2va, **without** the LoRA.
- Every continuation: the ref2va model with the LoRA, the previous clip's last 39 frames (and their audio) as the
  protected masked context, **no** reference video, and this wording before the prompt (the seconds are
  (frames - 39) / 24, so 3.542 for a 124-frame clip):

  > Continue the existing scene from the previous generated clip with no cut, reset, or re-establishment. The incoming
  > protected audiovisual opening is authoritative for current pose, motion, camera trajectory, facial state,
  > lighting, colour, environment, object state, voice, ambience and timing. The visible characters retain their
  > identity and facial appearance. After this protected 1.625-second opening, the remaining 3.542 seconds continue
  > directly:

  followed by a newline and the prompt.
- Training matches it: `partition: ref2va_pruned`, `continuation_no_context_share: 0`, and every caption file starts
  with the same wording (for 124-frame training windows).

Why: on the same Story with no LoRA, continuations on fl2va with the plain prompt burned more (fine detail +94% over
4 clips) than on ref2va (+57% with the wording, +62% with the tail also given as a reference video). A prompt beat that
turns the face away ("glances out at the fields") made H3 cut to a new shot at the seam in every recipe, so test
Stories keep the speaker facing the camera.

## Config

```yaml
model:
  arch: minimax_h3
  model_kwargs:
    continuation_context_frames: 39              # protected prefix, a 17n+5 frame run (39 = 12 latent steps)
    continuation_burn_levels: [0, 0.05, 0.1, 0.2] # contexts cached per clip; 0 is the clean clip
    continuation_clean_share: 0.5                # share of steps that draw the clean context
    continuation_no_context_share: 0.15          # share of steps with no context (plain generation)
    continuation_protect_audio: true             # protect the context's audio like inference does
train:
  audio_teacher: true                            # audio target = the frozen model's audio on the same input
datasets:
  - num_frames: 124
    fps: 24
    shrink_video_to_frames: false                # REQUIRED: keep clips at real speed
    do_audio: true
```

Clips must be longer than the context. If you turn this on for a dataset that already has a latent cache, delete the
cache first. The trainer prints a warning if a batch has no burned context.

**Keep clips at real speed.** ai-toolkit's `shrink_video_to_frames` defaults to true, which squeezes any clip longer than
`num_frames` into `num_frames` and linearly resamples its audio to match (faster motion, higher pitch, aliasing).
Continuation training refuses to cache or train with it on. With it off, each clip contributes a random real-speed
window of `num_frames`. An audio-protected context must end on the shared 24 fps / 40 Hz grid (39, 90, 141, ... frames),
because those are the only prefixes inference can protect exactly.

Calibration: at 512 px, levels 0.05, 0.1 and 0.2 add about +5%, +9% and +17% face texture. One real chained generation
adds about +19%.

## Status (experimental)

- v1 (every context burned, strong levels) stopped the drift but softened every clip. A chain lost about 55% of its
  fine detail, because the LoRA learned "always be softer than the context".
- v2 (half clean contexts, the calibrated levels above) fixed the softening, but dulled fine eye detail and degraded
  audio. The suspects are a train/inference mismatch in the audio schedule, no protection of the audio prefix, and
  training at 512 px.
- Found afterwards: v1 and v2 were trained with ai-toolkit's default `shrink_video_to_frames`, so most clips were
  squeezed into 124 frames (1.55x faster at the median, up to 2.9x) with their audio resampled to match. This is the
  prime suspect for the degraded audio, and possibly for the eyes.
- v3 (real-speed clips, the audio prefix protected like inference, a share of plain-generation steps; 1000 steps at
  512 px on 100 clips) cut chain drift sharply: over 4 chained clips face texture went from +76% without the LoRA to
  -11% at step 500. But it made audio quieter (57-63% of the base loudness already in a single text-to-video clip)
  and progressively more noise-like through a chain, and at step 1000 it over-corrected the video (-21%). The loader
  and the cached audio are faithful (decoded level 1.02x, aligned with the video window); the LoRA simply learns audio
  from 100 clips and pulls every soundtrack toward their loudness (mean RMS 0.10, against 0.18 for the base model on
  the test prompt).
- On a clip it was trained on (same prompt and seed, which the engine reproduces pixel for pixel without the LoRA),
  v3 step 500 still lowered the audio to 72% and fine detail by 19%. So it is not the data: fitting a flow-matching
  LoRA on top of the distilled 8-step assistant pulls predictions toward an average (quieter audio, softer video).
- v4: `audio_teacher`, so the LoRA stops learning audio from the dataset; trained on the recipe above; checkpoints
  every 100 steps to find the point with no drift in either direction.

## v6: joint-trajectory self-distillation (`continuation_trajectory_dir`)

A flow-matching loss with fresh noise is minimised by the posterior mean, so training on top of the distilled turbo
un-distils it (softer video, quieter audio), and the synthetic burn teaches "soften whatever looks sharp". v6 drops
both. It records the turbo's own 8-step sampling runs and trains the LoRA to reproduce them in continuation mode:

- **Recording.** A ComfyUI node, `H3RecordTrajectory` (kept outside this repo, with its own README), wraps the model
  like any patch node. For every model call of a plain text-to-video run it saves the state the H3 DiT received, its
  timestep and what it returned. It also saves the schedule, seed, initial noise, final latent and the Qwen3-VL
  conditioning. Record runs exactly like inference: the ref2va checkpoint, the turbo LoRA, shifts 12/3, the turbo
  sampler, 8 `simple` steps, cfg 1 (BasicGuider), 124 frames at the training resolution, and no cache node.
- **Samples.** Each training step draws, per batch item, one recorded call and a kind:
  - *continuation*: the first 39 frames (12 latent steps) are pinned from the run's own final latent, exactly as the
    masked AV continuation pins a context: `0.999 * clean + 0.001 * noise`, labelled `max(t_v, 0.999)`. The 65 audio
    ticks under it use the continuation's own audio mask `m`. Each row is `m * recorded state + (1 - m) * clean`,
    labelled `1 - m * sigma_a`, so the protected ticks are exactly clean and labelled 1. The rest of the input is the
    recorded joint state. Only the new tokens teach: pinned video rows add no loss, and audio rows count by `m`.
  - *anchor*: the recorded input unchanged, with the recorded output as the target everywhere. This keeps plain
    generation as it was.
  - *noised*: a continuation sample whose video context is `(1 - s) * clean + s * noise`, labelled `1 - s`.
  The target is always the recorded output. No sample has fresh noise or synthetic burn, and no teacher pass runs.

### Conventions (checked in the tests)

Recorded per call, as the DiT sees it, after ComfyUI's `MiniMaxH3Model.forward` has undone the audio carry:

| Tensor | Shape | Meaning |
|---|---|---|
| `video.x` | (1, 24, T, H, W) | The video state the DiT received. |
| `audio.x` | (1, 32, 2, Ta) | The audio state in its **own** flow space. The sampler carries `(sigma_v / sigma_a) * x_a`. |
| `timestep` | (1,) | `1000 * sigma_v`. The DiT derives `sigma_a = remap(sigma_v, 12 -> 3)` and labels `t = 1 - sigma`. |
| `video.out`, `audio.out` | as above | `-(raw H3 prediction)`, that is `noise - clean` in each stream's own space: `x0 = x - sigma * out`. |

ai-toolkit uses the same convention: `get_noise_prediction` returns `-(raw prediction)`, the target is
`noise - clean`, and timesteps are `1000 * sigma`. So the targets are the recorded outputs as they are. The only
conversion is the audio layout: ComfyUI's `(B, 32, 2, Ta)` becomes the trainer's channel-major rows `(B, 2 * Ta, 32)`
through `permute(0, 2, 3, 1)`. The recorder's README spells out the sampler-side relations: the carried audio, its
velocity, and the Euler step it replays to verify a recording.

### Config

```yaml
model:
  arch: minimax_h3
  assistant_lora_path: ".../turbo_lora.safetensors"   # the turbo LoRA stays the frozen assistant
  model_kwargs:
    partition: ref2va_pruned                        # the checkpoint the runs were recorded on
    continuation_trajectory_dir: /path/to/recordings   # <stem>.safetensors per dataset item
    continuation_context_frames: 39                 # pinned prefix (default 39: 12 latent steps, 65 audio ticks)
    continuation_protect_audio: true                # default true in v6
    continuation_trajectory_anchor_share: 0.2       # plain generation, target = recorded output everywhere
    continuation_trajectory_noised_share: 0.15      # noised contexts; the rest (0.65) are clean-context continuations
    continuation_trajectory_context_noise: [0.02, 0.05]  # noise level s ~ U[low, high]
    continuation_trajectory_context_noise_mode: fixed    # fixed: s as drawn; mask: a mask value m, s = m * sigma_v
    continuation_trajectory_noise_audio: false      # also noise the protected audio ticks of noised samples
    continuation_trajectory_step_weights: [1, 1, 1, 1, 1, 2, 2, 2]  # draw probability per sampler step (uniform)
    continuation_trajectory_text: recorded          # recorded: the engine's own conditioning; dataset: the captions
    continuation_trajectory_sweep: false            # cycle every recording through every kind and step (diagnostics)
    continuation_trajectory_log_every: 50           # print the running error table every N samples
train:
  batch_size: 1
  # no audio_teacher (the audio target is recorded); MSE loss; no prior / guidance / preservation losses
datasets:
  - folder_path: /path/to/dataset                   # <stem>.mp4 (+ <stem>.txt): the recorded clips, same stems
    num_frames: 124
    do_audio: true
```

- The dataset only names the recordings. Each item's file stem selects `<stem>.safetensors` in
  `continuation_trajectory_dir`. The clip's own latents, the trainer's noise and its timestep are replaced by the
  recording, and the trainer then runs at the recorded timestep. The recorder's `make_dataset.py` links the decoded
  clips and writes captions. The latent cache is still built, but nothing uses it.
- `continuation_trajectory_text: recorded` (the default) feeds the recorded Qwen3-VL states. Anchors get the
  conditioning the run was sampled with. Continuation samples get the node's optional `student_positive` recording
  when there is one: record the continuation wording plus the prompt there, to train with the text inference uses.
  With `dataset`, every sample uses the caption's embedding from this trainer's text encoder.
- Refused with an error: the burned-context keys (`continuation_burn_levels`, `continuation_clean_share`,
  `continuation_no_context_share`, ...), `audio_teacher`, non-MSE losses, prior, guidance and preservation losses,
  `next_sample` timesteps, `single_item_batching`, and a dataset item without a recording (checked before caching).
- The trainer's timestep settings (`timestep_type`, `content_or_style`) only shape its own draws, and v6 replaces
  those. Timestep loss weightings (`timestep_type: weighted`, `linear_timesteps`, the SNR gammas) would weight by the
  recorded timestep, so leave them off. `continuation_trajectory_step_weights` weights the sampler steps instead.
- The mask noise mode mirrors an inference context mask value `m` (rows at `m * sigma_v`, labelled
  `1 - m * sigma_v`). Values at or below 0.05 snap to a clean pin at inference, so the range must stay above 0.05.
  The pod's tapered variant is not modelled.
- The research suggests rank 16-32, lr 5e-5 to 1e-4, 500-1500 steps, and checkpoints every 200 steps.

### Diagnostics

Every sample's error on the tokens that teach goes into the trainer logs (`traj/video_mse`, `traj/video_rel`,
`traj/audio_mse`, `traj/audio_rel`, `traj/step`). The first samples are printed, and every
`continuation_trajectory_log_every` samples a table of running means by kind and sampler step follows.
`*_rel` is the RMS error over the target's RMS.

- **Step 0 (LoRA at zero).** Run the training config with `lr: 0`, `continuation_trajectory_sweep: true`,
  `continuation_trajectory_log_every: 1`, and 24 to 100 steps. The LoRA stays exactly zero, and the table covers
  every kind at every step.
  - The *anchor* rows measure how far this trainer's forward is from the engine's. They must sit at the numeric noise
    floor: bf16, different attention and matmul kernels. If they do not, the LoRA would learn to paper over the gap.
    Suspects, in order:
    1. The turbo LoRA's adaLN deltas. The ComfyUI MiniMax-H3-Turbo node re-injects them on pruned checkpoints; the trainer's
       assistant file drops them. Set `assistant_adaln_lora_path` and `assistant_adaln_grid_path` (see below) so the
       trainer applies them the same way.
    2. The engine's numerics (see "Engine numerics" below; on by default since v7).
    3. Sage attention in the engine: record the trajectories with sage off. The trainer cannot run sage (no backward).
    4. The text encoder, if you use `continuation_trajectory_text: dataset`.
  - The anchor rows cannot be flat. The turbo amplifies any one-ulp difference, and the engine does it to itself.
    Measured on an RTX 5090 (sage off, 3 recordings, 2026-09-25): the engine replays its own recorded calls bit for
    bit, but nudging a fifth of its bf16 video input elements by one step moves its video prediction by the "engine
    floor" below. Swapping its attention kernel for sage moves it by the same amounts. The v7 trainer on the same calls
    sits on that floor:

    | step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
    |---|---|---|---|---|---|---|---|---|
    | sigma | 1.000 | 0.988 | 0.973 | 0.952 | 0.923 | 0.878 | 0.800 | 0.632 |
    | engine floor (nudge) | 4.0% | 12.9% | 16.6% | 15.2% | 11.6% | 9.5% | 6.2% | 3.9% |
    | engine, sage vs SDPA | 4.5% | 14.1% | 20.4% | 18.8% | 14.0% | 11.2% | 7.3% | 4.8% |
    | trainer vs engine (v7 anchors) | 4.9% | 15.7% | 18.1% | 16.8% | 12.3% | 10.4% | 6.2% | 4.3% |

    The difference appears in block 0 (0.5-0.8% of the hidden stream for any nudge, even one of 0.2% of the elements).
    It stays at that level to block ~25, then grows in blocks ~28-40 to 10-36% before the output. Audio floors are
    1.7-5%. Set `continuation_trajectory_floor` (below) to print the model's own floor next to every row. An anchor
    row at its floor is as close to the engine as a forward that is not bit-identical can get.
  - The same floor applies to training: at steps with a high floor, part of every recorded target is detail that
    only a bit-identical network reproduces. The loss there cannot go below the floor, and chasing that detail pulls
    the prediction toward its average. Read the continuation rows against the floor of their step.
  - The *continuation* rows, per sigma, are the continuation offset that the LoRA is meant to remove.
- Verify each recording before training with the recorder's `verify_trajectory.py`. It replays the sampler from the
  recorded calls; relative errors of 0.2-0.5% are bf16 rounding.

### Status

Built and CPU-tested on 2026-09-24; nothing has been trained or recorded on a GPU yet.
- `test_h3_trajectory.py` records runs with a stand-in engine that mirrors ComfyUI's sampler, audio carry and DiT
  contract. It then trains on them through the real `get_noise_prediction`, `get_loss_target` and
  `SDTrainer.calculate_loss`, with a stand-in transformer that computes the same function. The anchor loss is
  exactly 0 at all 8 steps. Pinned tokens add no loss and no gradient.
- The recorder's own tests run it inside the real ComfyUI forward and turbo sampler code. They also feed its files
  into this trainer.

## The assistant's adaLN pieces on pruned checkpoints (`assistant_adaln_*`)

Pruned checkpoints feed the adaLN projections from an 8-wide curve table instead of the full model's 2688-wide
silu(t_emb). A full-model LoRA's adaLN pieces therefore cannot be loaded as weights, which is why the assistant file
used through v4 has them stripped. Inference still applies them: the ComfyUI MiniMax-H3-Turbo node re-injects the
turbo LoRA's 51 adaLN updates at run time, and its log line on pruned files ends with "+ 51 adaln injected at run
time". So v1-v4 trained against a turbo without those updates.

With both paths set, the trainer applies the same update in every `MiniMaxH3AdalnProj` forward
(`src/adaln_injection.py`):
- It adds `(B @ (A @ st.T)).T` to the projection's float32 output, before the split.
- `st` is the bundled grid row of the `adaln_t_table` row nearest to the projection's input.
- The pieces stay on the CPU and are cast per call. They never enter the state dict, quantization or offloading.
- The update follows the assistant LoRA: it is on in every training pass, the audio-teacher pass included, and off
  while the sampler has the assistant switched off.
- A one-time console line reports the number of projections.

The CPU tests check it bit for bit against the node's own function.

```yaml
model:
  assistant_lora_path: ".../turbo_lora_stripped.safetensors"      # backbone pieces, as before
  model_kwargs:
    assistant_adaln_lora_path: ".../turbo_lora_full.safetensors"   # the full turbo LoRA (only its adaln pieces are read)
    assistant_adaln_grid_path: ".../h3_silu_temb_grid.safetensors" # ships with the turbo node, 1025 x 2688
    assistant_adaln_strength: 1.0                                  # the node's strength (default 1.0)
```

## Turbos by name (`model_kwargs.turbo`): one LoRA across turbos

`model_kwargs.turbo` names the turbo (or turbos) the run trains against, by the names of the turbo registry that the
recorder and the eval driver use (a `turbos.py` with a `TURBOS` dict: files per DiT partition, shifts, step count,
sigma grid, strength, and the workflow patch that says what kind of turbo it is). It replaces the single
`assistant_lora_path` + `assistant_adaln_*` setup, and is refused next to those keys.

```yaml
model:
  arch: minimax_h3                   # no assistant_lora_path
  model_kwargs:
    partition: ref2va_pruned
    turbo: [larry, lightx2v8, hyperflow_lora]   # one name or a list
    turbo_registry: /path/to/turbos.py          # the recorder's registry (a copy)
    turbo_lora_dir: /path/to/loras              # each LoRA turbo's file under its registry file name
    turbo_adaln_grid_path: /path/to/h3_silu_temb_grid.safetensors   # Larry's adaLN pieces (the turbo node's grid)
    # optional:
    turbo_paths: {larry: /elsewhere/file.safetensors}   # a file somewhere else
    turbo_dit_dir: /path/to/diffusion_models    # checkpoint turbos (merged_turbo): the folder of their DiT
    turbo_sample: lightx2v8                     # the turbo previews sample with (default: the first listed)
    turbo_weights: {larry: 1, lightx2v8: 1}     # flow-matching runs only: the per-step draw weights (default uniform)
    turbo_keep_on_device: false                 # true: every turbo stays on the GPU once used (no per-switch copies)
    continuation_trajectory_step_floors: {larry: {video: [...8], audio: [...8]}, ...}   # see below
    continuation_trajectory_floor_ratio_cap: 4.0                # optional: the floor weights at most 4x apart
    continuation_trajectory_depth_weights: {"2": 2, "3": 2}     # optional: R+ recordings by info.rplus.depth
```

- **LoRA turbos** (the registry's `patch_lora_turbo`: lightx2v8, hyperflow_lora; and Larry, whose `patch` is None: the
  MiniMax-H3-Turbo node) are the frozen assistant, applied live and never merged (`src/runtime_lora.py`). The math is
  the ComfyUI H3RuntimeLoRA node's, which the recorder uses for them (`TURBO_APPLY=runtime`): each target linear's
  output gets `((x.float() @ A.T) @ B.T * alpha / rank * strength).to(out.dtype)` of that linear's own input, in
  2048-row chunks; fc1's delta lands before the SwiGLU; fc2's is taken on the float32 SwiGLU of fc1's output. The
  gradient flows through the deltas to the trained LoRA upstream; nothing extra is saved for the backward.
  The file for the run's partition comes from the registry (`files["fl2va"|"ref2va"]`); every tensor in it must map
  onto a trainer module, and anything that does not is an error. Larry's file must carry its 51 adaLN pieces (its node
  always re-injects them; they run through `turbo_adaln_grid_path`); a plain LoRA turbo may not carry any (H3RuntimeLoRA
  refuses them). Larry's recordings come from its own node (bypassed backbone, fc2 merged into int8), which the
  trainer applies with the same live math: the difference sits at the numeric floor, and the step-0 check measures it.
- **Checkpoint turbos** (`patch_checkpoint_turbo`: merged_turbo) are the DiT itself (`turbo_dit_dir`), with no assistant.
  Files that declare their int8 layers in the metadata (`_quantization_metadata`) load with the per-layer markers
  ComfyUI builds from it; the unused `adaln_basis` / `adaln_mean` are dropped. A checkpoint turbo runs alone.
- Refused: the HyperFlow node (`patch_hyperflow`: it rewrites the time conditioning) and `blocked` entries.
- **Shifts and sigmas.** The live turbo's shifts drive the audio clock (`sigma_a = remap(sigma_v)`) in every pass; the
  flow-matching scheduler starts at the first turbo's video shift; previews use the sample turbo's shifts and its grid
  (a fixed grid such as hyperflow_lora's replaces the shifted linspace).
- **Several turbos, one base DiT.** Every LoRA turbo is loaded (CPU) and hooked once; one is live at a time, and only
  its tensors sit on the GPU (a switch copies about 0.6 GB for Larry, 2 GB for lightx2v8, 4 GB for hyperflow_lora, in bf16).
  - Trajectory training: each sample's assistant is the turbo its recording was sampled with, live for that sample's
    passes (the floor probe included). The recording names it in `info["turbo"]` (a name or a block with `name`), else
    its `.json` sidecar, else its stem (`A_<turbo>_<idx>`). Each recording is checked against the registry when it is
    opened and before caching: the listed turbo, the schedule's shifts, and, from the block, its file for this
    partition, strength, step count and sigma grid. An untagged recording counts as the only turbo of a one-turbo run
    and is refused in a mix. A batch of recordings from different turbos is refused (use batch_size 1). The error table
    gains a turbo column.
  - Flow-matching training: each step draws a turbo (`turbo_weights`), live for every pass of the step (prior, audio
    teacher, continuation draws). A timestep drawn on the scheduler's shifted grid (`timestep_type: shift`) moves to
    the same schedule position on the step's turbo's shift, and the noisy latents are rebuilt to match.
- **Per-step floor weighting** (`continuation_trajectory_step_floors`, off by default). The step-0 run measures each
  turbo's floor per sampler step and prints it ready to paste
  (`measured floors (continuation_trajectory_step_floors): {...}`). With it set, each sample's loss is weighted by
  `1 / floor^2` of its turbo and step, normalised to mean 1 over the turbo's steps (separately for video and audio),
  so no step's irreducible mismatch dominates. The logged errors stay unweighted. A `default` entry covers recordings
  without a turbo. `continuation_trajectory_floor_ratio_cap: 4.0` caps each weight at 4x the turbo's smallest before
  the normalisation (off when absent).
- **R+ recordings** (repaired-context self-distillation: a file with `run.context.video`). The turbo continued a clip
  from a repaired context R (a masked AV continuation, recorded with the context, its masks and the burned context C as
  `run.context.student.*`). All its conditional calls are points, masked or not, and every draw from it is a
  continuation sample, whatever the shares (the other files keep theirs): the recorded state with the pinned video
  steps moved from R to C (`video.x[:, :, :S] += 0.999 * (C - R)`, labelled `max(t_v, 0.999)`), the audio as recorded
  under the recorded mask m (it must be the continuation mask as the engine quantises it, ceil to 1/256 with no snap
  (`engine_audio_context_mask`), or the compat layer's rounded and snapped one: labels `min(1 - m * sigma_a, 1)`,
  weight m),
  the recorded outputs as targets and the joint conditioning. Without a student context the sampled one is kept. The
  recording's `info.context.video_steps` must be the trainer's context steps.
  `continuation_trajectory_depth_weights: {"2": 2, ...}` multiplies both loss weights of a sample by its recording's
  `info.rplus.depth` entry (1 otherwise; not renormalised); the residual scaling runs whenever a weight is not 1.
- **Checking a config before a run.** `src/turbos.py` imports nothing from the package, so tooling can load it by path:
  `check_model_kwargs(model_kwargs, partition, dataset_folder)` checks the registry names, every turbo file (header
  only: each tensor's module and shape against the pruned H3), the adaLN grid, and every dataset item's recording and
  its turbo. `python src/turbos.py config.yaml` does the same for a config.

## Engine numerics (`match_engine_precision`)

On a pre-quantized (`comfy_quant`) checkpoint the trainer runs with the weights ComfyUI runs, not the stored ones
(`match_engine_precision: dynamic`, the default; it only acts on such checkpoints). ComfyUI loads these files with
MixedPrecisionOps, which converts every unquantized linear weight to the compute dtype (bf16), whatever dtype the model
code declares. The engine therefore runs the fp16 adaLN projections and the fp32 patch projections and output heads
with bf16-rounded weights, still computed in float32. `legacy` is for an engine started with `--disable-dynamic-vram`,
whose loader also rounds the biases; `off` keeps the stored precision.

Always on (transformer code): the final layer modulates in float32 like the engine (there `1 + scale` sits near
0.02-0.05 for most channels, where a bf16 round trip moves it by up to 10%). Gated residuals use one rounding
(`addcmul`, like the engine's `_mod_gate`), and the adaLN curve uses the engine's exact lerp.

With these, a tiny CPU model loaded both ways matches the engine's bf16 output bit for bit at all 8 turbo steps
(`test_h3_engine_match.py`). On a GPU the forward still differs from the engine by kernels, about 0.5-0.9% per op on
block 0 with identical inputs:
- comfy-kitchen's int8 linear: its fused ConvRot quantizer's activation codes are not a float32 or bf16 rotation
  rounded to nearest (7-8% of codes differ from either);
- its fused RMSNorm+RoPE kernel and the attention kernel;
- the turbo LoRA's fc2 pieces, which the node merges into the int8 weights and ComfyUI requantizes with seeded
  stochastic rounding (0.67% on block 0's fc2 output).

Because of the floor above, closing these without becoming bit-identical would not move the anchor rows.

A LoRA loaded with `LoraLoaderModelOnly` on this int8 checkpoint, which is how the dashboard's optional H3 LoRA stack
loads, is requantized the same way at load: every patched int8 weight. Small deltas survive only on average.

## The model's own floor (`continuation_trajectory_floor`)

`continuation_trajectory_floor: 1e-3` (default 0 = off) makes every trajectory sample run a second, gradient-free
forward. That forward sees the same call with the video state nudged by Gaussian noise of that relative size, rounded
to bf16 (about a fifth of the elements move by one bf16 step). The table and the per-sample lines then show
`video floor` / `audio floor`: how far the model's own prediction moves, on the teaching tokens, relative to the
target's RMS. On a 5090 the trainer's floor tracks the engine's (sigma 0.988: 14.8% vs 12.9%; sigma 0.632: 4.0%
vs 3.9%). It costs one extra forward per sample, so use it for diagnostic runs.

## Tests

`test_h3_continuation.py` (v1-v4), `test_h3_trajectory.py` (v6), `test_h3_adaln.py` (the assistant's adaLN
pieces) and `test_h3_turbos.py` (turbos by name) in the repo root have CPU contracts with stand-in models. They stub
the package `__init__`, because another model needs a GPU at import time:

```bash
CUDA_VISIBLE_DEVICES= python -m unittest test_h3_continuation test_h3_trajectory test_h3_adaln test_h3_engine_match test_h3_turbos
```

`test_h3_turbos.py` holds the H3RuntimeLoRA node's functions verbatim and checks the live application against them bit
for bit. Optional: `H3_RUNTIME_LORA_DIR` (the node's folder) checks the copy against the node's source;
`H3_TURBO_REGISTRY` (the registry file) with `H3_TURBO_LORA_DIR` (the turbo files, or their safetensors headers as
`<file>.json`) maps every tensor of the real files onto the pruned H3, header only.

`test_h3_engine_match.py` pins the trainer's forward to the engine's: timestep labels and per-row modulation rows,
the adaLN curve, the block and final-layer numerics and the engine-precision weights. With
`H3_ENGINE_DIR` set to the ComfyUI checkout that runs H3, it also reads the engine's own functions from its source.
Run with that checkout's Python, it also compares whole tiny models:

```bash
H3_ENGINE_DIR=/path/to/ComfyUI CUDA_VISIBLE_DEVICES= /path/to/comfy/python -m unittest \
    test_h3_engine_match.EngineSourceContracts test_h3_engine_match.WholeModelContracts
```
