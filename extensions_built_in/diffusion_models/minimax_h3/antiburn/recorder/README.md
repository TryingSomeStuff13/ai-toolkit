# ComfyUI-H3-RecordTrajectory

A ComfyUI custom node pack that records MiniMax H3 sampling runs as training data for trajectory self-distillation.

**MiniMax H3 Record Trajectory** (`H3RecordTrajectory`): MODEL in, MODEL out. Put it after your turbo / LoRA nodes,
before the guider. Each sampling run made with the patched model writes `<output_dir>/<output_name>.safetensors`
plus a `.json` sidecar. A failed run writes nothing.

- **Per model call:** the state the H3 DiT received (video, and audio in its own flow space), the timestep, and what
  the DiT returned.
- **Per run:** the schedule, seed, initial noise, final latent, and the Qwen3-VL conditioning with its token tags.

Record exactly like inference:
- the ref2va checkpoint and your turbo;
- its shifts, sampler and steps;
- cfg 1 (BasicGuider);
- 124 frames;
- no cache node.

The other nodes in the pack (`probe_nodes.py`, `rplus_nodes.py`) were used for diagnostics and an experiment that
v1 did not use.

Install: copy or symlink `ComfyUI-H3-RecordTrajectory` into `custom_nodes` and restart ComfyUI. It needs only `torch`
and `safetensors`.

`make_dataset.py RECS VIDEOS DATASET [--wording] [--copy]` builds the trainer's dataset folder (`<name>.mp4` +
`<name>.txt`) from the recordings and their videos. Point `continuation_trajectory_dir` in the config at the
recordings.
