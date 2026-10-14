# H3 Anti-Burn LoRA (universal, v1): how it was made and what didn't work

The LoRA itself, how to use it and its measured results are on its model card:
[Imagineer13/H3-Anti-Burn-LoRA](https://huggingface.co/Imagineer13/H3-Anti-Burn-LoRA).
This folder holds what it took to make it:
- the training config;
- the turbo registry;
- the recorder node that makes the training data;
- the scoring script;
- a record of the attempts that failed.

The trainer itself is this branch's MiniMax H3 extension (`../CONTINUATION.md` documents every option).

## The problem

Long MiniMax H3 videos are made by chaining clips. Each clip continues from the previous clip's last 39 frames (and
their audio), which are copied in as a protected context. The model copies that context's small flaws and adds a few
of its own, so quality drifts clip by clip. We call it burn: faces get crunchy and etched, fine texture sharpens, and
sometimes the colour gets redder.

How much it burns depends on the turbo. Face texture, clip 1 to clip 10:

| Turbo | Face burn |
|---|---|
| HyperFlow | +25% |
| lightx2v | +44% |
| An unseen model (never trained on) | +61% |
| Larry Turbo v4 | +261% |

## How v1 was made (design A: trajectory self-distillation)

1. **Record.** `recorder/ComfyUI-H3-RecordTrajectory` is a ComfyUI node that wraps the model. For every model call of
   a normal 8-step text-to-video run, it saves what the DiT received, the timestep and what the DiT returned. It also
   saves the schedule, the noise, the final latent and the text conditioning.
   - We recorded 150 full-size runs (640×1152, 1152×640, 864×864, 124 frames): 50 each with Larry, HyperFlow and
     lightx2v.
   - The LoRA turbos were applied live (unmerged); Larry used its own turbo node.
   - `recorder/make_dataset.py` turns the recordings into the trainer's dataset folder.
2. **Train.** Each step draws one recorded call and turns it into one of three kinds of sample:
   - **continuation** (65%): the start of the same video is pinned as the protected context, exactly as inference
     pins it.
   - **anchor** (20%): the recorded call unchanged, so plain generation stays as it was.
   - **noised** (15%): a continuation whose context is lightly noised.

   The target is always what the turbo itself returned. So in continuation mode the LoRA learns to make what the
   turbo makes in a fresh generation, instead of amplifying its context.
   - The turbo is applied during training, one of the three per step (`model_kwargs.turbo`).
   - Settings: 300 steps, rank 32, lr 1e-4, network weight 0.6, about 40 s/step on one RTX PRO 6000 96 GB.
   - The exact config is `config_universal_v1.yaml`. Its paths are placeholders.
3. **Test.** Chains of 10 clips (about 37 s), 4 scenes per setting, scored with `metrics/h3_recipe_score.py`.

## How burn was measured

`metrics/h3_recipe_score.py DIR NAME[,NAME...]` reads `DIR/NAME_clip<k>.mp4`, the last clip of a chain, which holds
the whole joined video. The main number is `face_texture_sized_change_pct`:
- It is the face's fine texture (Laplacian energy on a YuNet face box resized to 128 px), clip 1 against clip 10.
- Only frames where the face is 0.8–1.25× its clip-1 size count, so a camera push-in doesn't look like burn.

It also reports fine detail, saturation, skin redness, audio level and hiss, lip sync, skipped frames and seams.

Setup:
- Put `face_detection_yunet_2023mar.onnx` (from opencv_zoo) next to the script.
- Set `FFMPEG` if ffmpeg isn't on your PATH.

One chain moves about ±15 points on its own, so compare at least 4.

## What didn't work

In order. Every step taught us something the next step used.

1. **Synthetic burn (v1, v2).** We made "burned" copies of clean clips (unsharp mask, grain, saturation, crushed
   blacks) and trained the LoRA to continue clean from them.
   - It stopped burn, but it softened every clip: fine detail fell 55% over 11 clips. It learned "soften whatever
     looks sharp".
   - v2's audio also turned noisy, most likely because a dataset setting (`shrink_video_to_frames`, on by default)
     squeezed clips into 124 frames, which sped them up and pitched up their audio. The trainer now refuses it.
2. **Real-speed clips, protected audio (v3).** Burn was fixed, but the audio got quieter and hissier, and lip sync got
   worse. The LoRA learned the audio of the 100 training clips.
3. **An audio teacher (v4).** The audio target became the frozen model's own audio, so the LoRA leaves the sound
   alone. This was usable, but the flow-matching loss with fresh noise still pulled the distilled turbo towards soft,
   averaged video.
4. **Training on the base model (v5).** We tried training on the model without the turbo, then using the LoRA with
   it. The evidence was weak; parked.
5. **A judge model (DMD).** Before spending on a distribution-matching setup, we checked whether the model, used as a
   judge, can tell burned frames from clean ones. It sees texture burn a little, but not colour drift: a redder face
   is a plausible picture, and only the chain's own first clip says it's drift. Dropped before training.
6. **A colour lock.** We matched each continuation's latent colour statistics to clip 1. That fixed redness, but faces
   turned darker and harsher when the camera moved in. Rejected by eye.
7. **Design A with one turbo, then two.** Trajectory self-distillation (above) was the first approach that cut burn
   without softening.
   - Trained on HyperFlow and lightx2v, it fixed both, but did much less on a model it never saw: +61% → +40%.
   - A version trained on that model's own clips fixed it (+61% → −1%), but a LoRA per turbo breaks with the next turbo.
8. **Three turbos, including the worst burner (v1 universal).** Adding Larry, which burns the hardest, to the
   training turbos is what made it general: the unseen model went from +61% to +4%.
9. **More Larry data.** 150 more steps weighted 4:1 towards Larry did not fix Larry (+159% → +148%) and made lightx2v
   and the unseen model softer. We kept v1.

## Things worth knowing

- **One strength doesn't fit every turbo.** The right strength roughly follows how hard the turbo burns.
- **How the LoRA is loaded changes its strength.** Merging it into an 8-bit (int8) checkpoint, which is what we
  tested with, weakens it a lot. Loaded live (unmerged), 0.6 fixed Larry (+261% → −5%), but it was 2–3× too strong
  for the other turbos (−27% to −57%, soft faces). Merged at the same 0.6, Larry still burned +159%.
- **Larry's burn scales with Larry's own turbo strength.** At 0.8 instead of 1.0, burn by clip 4 dropped from +97%
  to +56%. The sampler and the step shift were not the cause.
- **Never put the LoRA on clip 1.** It is a continuation fix. Clip 1 has nothing to burn from.
- **Test with several chains.** One chain per scene swings too much to rank anything.

## Files

| Path | What |
|---|---|
| `config_universal_v1.yaml` | the exact training config of v1 (paths are placeholders) |
| `turbos.py` | the turbo registry the recorder, trainer and tests share: files, shifts, sigmas, how each turbo enters a ComfyUI graph |
| `recorder/ComfyUI-H3-RecordTrajectory/` | the recording node (ComfyUI custom node pack) |
| `recorder/make_dataset.py` | recordings + videos → the trainer's dataset folder |
| `metrics/h3_recipe_score.py` | the burn, detail, colour, audio and lip-sync scorer |

The scripts that rented GPUs, moved files and drove the recording and test runs are left out: they are tied to our
machines. The pieces above are what those scripts ran.

Training was paid for by a community compute grant.
