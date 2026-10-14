"""v6 continuation training for MiniMax H3: joint-trajectory self-distillation.

Why: v1-v4 trained the anti-burn LoRA with flow matching on dataset clips and fresh noise. On top of a step-distilled
turbo that loss is minimised by the posterior mean, so it un-distils the turbo: softer video, quieter audio. Here the
LoRA is trained on the turbo's OWN sampling trajectories instead. A ComfyUI node (H3RecordTrajectory, kept outside this
repo) records every model call of plain text-to-video runs made exactly like inference (same checkpoint, the turbo LoRA,
shifts 12/3, the turbo sampler, 8 "simple" steps, cfg 1): the state the DiT saw and what it returned. A training sample
takes one recorded call and asks the network to return the recorded output:

  - continuation: the first ``continuation_context_frames`` frames and their audio ticks are pinned clean from the
    run's own final latent, the way the masked AV continuation pins a previous clip; the rest of the input is the
    recorded joint state; only the new tokens count.
  - anchor: the recorded input unchanged, target = the recorded output everywhere. With the LoRA at zero this loss is
    the trainer-vs-engine forward mismatch (the step-0 diagnostic: it must sit at the numeric noise floor). That floor
    is not flat: the turbo amplifies a one-ulp nudge of its input to ~4% (sigma 1), 12-17% (sigma 0.99-0.95) and 4%
    (sigma 0.63) in the engine itself. ``continuation_trajectory_floor`` measures it next to every sample.
  - noised: a continuation sample whose context carries a little Gaussian noise with a matching label.

Every target is a fixed function of its input (no fresh noise, no synthetic burn), so the MSE has nothing to average.

R+ recordings (repaired-context self-distillation; a file with run.context.video): the turbo continued a clip from a
repaired context R (a masked AV continuation), and the recording also holds the burned context C the student is pinned
on (run.context.student.*, else the sampled context itself). Every conditional call is a point, masked or not, and
every sample is a continuation sample: the recorded state with its pinned video steps moved from R to C
(video.x[:, :, :S] + 0.999 * (C - R)), labelled max(t_v, 0.999); the audio exactly as recorded, with the recorded
audio mask m (it must be the trainer's own continuation mask), labels min(1 - m * sigma_a, max(t_a, 1)) and weight m;
the recorded outputs as targets; the recorded joint conditioning. info["rplus"]["depth"] (the clip the context was
carried out of) can weight the loss (continuation_trajectory_depth_weights).

Recording convention, per model call, as the H3 DiT sees it (ComfyUI MiniMaxH3Model.forward -> _forward, recorded by a
DIFFUSION_MODEL wrapper, i.e. after forward() has undone the sampler's audio carry):
  call.NNN.video.x    (1, 24, T, H, W)  the video latent the DiT received (sampler state cast to the compute dtype)
  call.NNN.audio.x    (1, 32, 2, Ta)    the audio latent in its OWN flow space. The sampler carries audio scaled onto
                                        the video schedule, y = (sigma_v / sigma_a) * x_a; forward() divides it back out.
  call.NNN.timestep   (1,) fp32         1000 * sigma_v, the value handed to the DiT. sigma_v = timestep / 1000,
                                        sigma_a = remap(sigma_v, 12 -> 3); the labels are t = 1 - sigma per stream
  call.NNN.video.out  / audio.out       what _forward returned: -(raw H3 prediction) = noise - clean in each stream's
                                        own flow space, so x0 = x - sigma * out (sigma_a for the audio)
  run.final.video / run.final.audio     the sampler's final latent (audio divided back by the carry scale)
  run.noise.video / run.noise.audio     the initial noise; run.sigmas the schedule
  cond.joint.embeds / cond.joint.tags   the Qwen3-VL conditioning the run was sampled with (and its token tags)
  cond.student.*                        optional second conditioning recorded for continuation samples
That is exactly ai-toolkit's convention: get_noise_prediction returns -(raw prediction) and the loss target is
noise - clean. So the video target is video.out as is, and the audio target is audio.out in the trainer's packed
channel-major rows (B, 2 * Ta, 32): no sign flip, no rescale.
"""
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from . import continuation as h3_continuation
from . import turbos as h3_turbos
from .packing import AUDIO_SIGMA_SHIFT, VIDEO_SIGMA_SHIFT, remap_sigma

FORMAT = "h3_trajectory"
FORMAT_VERSION = 1
KINDS = ("anchor", "continuation", "noised")
TEXT_SOURCES = ("recorded", "dataset")
NOISE_MODES = ("fixed", "mask")
MEDIA_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".flv", ".wmv", ".png", ".jpg", ".jpeg", ".webp")
# the burned-context mode's settings (v1-v4) mean nothing here; refused so a copied config cannot half-apply them
BURN_MODE_KEYS = ("continuation_burn_levels", "continuation_burn_variants", "continuation_clean_share",
                  "continuation_no_context_share")
MASK_SNAP = 0.05  # inference snaps mask values <= 0.05 to 0 (fully protected), so a mask-mode noise must exceed it


def call_key(index: int, name: str) -> str:
    """Tensor key of one recorded model call: call_key(3, "video.x") -> "call.003.video.x"."""
    return f"call.{int(index):03d}.{name}"


def audio_rows(latent: torch.Tensor) -> torch.Tensor:
    """ComfyUI audio latent (B, 32, 2, T) -> the trainer's packed channel-major rows (B, 2 * T, 32)."""
    b, c, ch, t = latent.shape
    return latent.permute(0, 2, 3, 1).reshape(b, ch * t, c)


def audio_latent(rows: torch.Tensor, channels: int = h3_continuation.AUDIO_CHANNELS) -> torch.Tensor:
    """Inverse of audio_rows: (B, 2 * T, 32) -> (B, 32, 2, T)."""
    b, n, c = rows.shape
    return rows.reshape(b, channels, n // channels, c).permute(0, 3, 1, 2)


def stream_sigmas(timestep: torch.Tensor, shift_video: float = VIDEO_SIGMA_SHIFT,
                  shift_audio: float = AUDIO_SIGMA_SHIFT) -> Tuple[torch.Tensor, torch.Tensor]:
    """(sigma_v, sigma_a) as the H3 model derives them from its timestep (same fp32 ops as ComfyUI's _forward and
    get_noise_prediction): sigma_v = timestep / 1000 clamped at 1e-6, sigma_a by the closed-form shift remap."""
    sigma_v = (timestep.to(torch.float32) / 1000.0).clamp(1e-6, 1.0)
    return sigma_v, remap_sigma(sigma_v, shift_video, shift_audio)


def enabled(kw: dict) -> bool:
    return bool((kw or {}).get("continuation_trajectory_dir"))


@dataclass(frozen=True)
class Settings:
    directory: str
    context_frames: int = 39
    protect_audio: bool = True
    anchor_share: float = 0.2
    noised_share: float = 0.15
    context_noise: Tuple[float, float] = (0.02, 0.05)
    context_noise_mode: str = "fixed"
    noise_audio_context: bool = False
    step_weights: Optional[Tuple[float, ...]] = None
    text: str = "recorded"
    sweep: bool = False
    log_every: int = 50
    floor: float = 0.0  # > 0: also measure the model's own floor (a nudge of this relative size), diagnostics only
    # measured per-step floors by turbo: ((turbo, video floors, audio floors), ...); the loss weights derive from them
    step_floors: Optional[Tuple[Tuple[str, Tuple[float, ...], Tuple[float, ...]], ...]] = None
    floor_ratio_cap: Optional[float] = None  # no floor weight above this many times the turbo's smallest one
    depth_weights: Optional[Tuple[Tuple[int, float], ...]] = None  # ((R+ depth, loss weight), ...); others 1

    def floors_for(self, turbo: Optional[str]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """(video floors, audio floors) per sampler step of ``turbo`` (its entry, else the "default" one)."""
        floors = {name: (video, audio) for name, video, audio in self.step_floors or ()}
        entry = floors.get(turbo) if turbo is not None else None
        entry = entry if entry is not None else floors.get("default")
        if entry is None:
            raise ValueError(f"continuation_trajectory_step_floors has no entry for turbo {turbo} (nor a default)")
        return entry

    def floor_weight(self, turbo: Optional[str], step: int) -> Tuple[float, float]:
        """(video, audio) loss weight of a sample at sampler step ``step`` of ``turbo``: inverse-variance in the
        measured floor, 1 / floor^2, normalised to mean 1 over the turbo's steps (1, 1 without step_floors)."""
        if not self.step_floors:
            return 1.0, 1.0
        video, audio = self.floors_for(turbo)
        cap = self.floor_ratio_cap
        return inverse_variance_weights(video, cap)[step], inverse_variance_weights(audio, cap)[step]

    def depth_weight(self, depth: Optional[int]) -> float:
        """The loss weight of an R+ sample whose context was carried out of clip ``depth`` (1 without an entry)."""
        return dict(self.depth_weights or ()).get(depth, 1.0) if depth is not None else 1.0

    def sample_weight(self, turbo: Optional[str], step: int, depth: Optional[int] = None) -> Tuple[float, float]:
        """(video, audio) loss weight of a sample: its floor weight times its depth weight (not renormalised)."""
        video, audio = self.floor_weight(turbo, step)
        d = self.depth_weight(depth)
        return video * d, audio * d

    @property
    def context_steps(self) -> int:
        return h3_continuation.latent_steps(self.context_frames)

    @property
    def audio_context_ticks(self) -> int:
        return h3_continuation.audio_context_steps(self.context_frames) if self.protect_audio else 0

    @property
    def shares(self) -> Dict[str, float]:
        return {"anchor": self.anchor_share, "continuation": 1.0 - self.anchor_share - self.noised_share,
                "noised": self.noised_share}


def inverse_variance_weights(floors, cap: Optional[float] = None) -> Tuple[float, ...]:
    """1 / floor^2 per step, normalised to mean 1: every step's irreducible (floor) error counts the same. With ``cap``
    no step weighs more than cap times the lightest one before the normalisation."""
    inverse = [1.0 / (float(f) ** 2) for f in floors]
    if cap is not None:
        ceiling = cap * min(inverse)
        inverse = [min(v, ceiling) for v in inverse]
    mean = sum(inverse) / len(inverse)
    return tuple(v / mean for v in inverse)


def parse_step_floors(value) -> Optional[Tuple[Tuple[str, Tuple[float, ...], Tuple[float, ...]], ...]]:
    """continuation_trajectory_step_floors: {turbo (or "default"): {video: [floor per step], audio: [...]}}, the
    relative floors the step-0 run measures (continuation_trajectory_floor), e.g. 0.129 for 12.9%."""
    if not value:
        return None
    if not isinstance(value, dict):
        raise ValueError("continuation_trajectory_step_floors maps a turbo name to {video: [...], audio: [...]}")
    out = []
    for name, entry in value.items():
        if not isinstance(entry, dict) or set(entry) != {"video", "audio"}:
            raise ValueError(f"continuation_trajectory_step_floors.{name} needs exactly video: [...] and audio: [...]")
        video, audio = (tuple(float(f) for f in entry[k]) for k in ("video", "audio"))
        if not video or len(video) != len(audio) or any(not 0.0 < f < 10.0 for f in video + audio):
            raise ValueError(f"continuation_trajectory_step_floors.{name}: one positive relative floor per sampler "
                             f"step for video and audio (e.g. 0.129), got {entry}")
        out.append((str(name), video, audio))
    return tuple(out)


def parse_depth_weights(value) -> Optional[Tuple[Tuple[int, float], ...]]:
    """continuation_trajectory_depth_weights: {depth: weight}, e.g. {"2": 2, "3": 2}; depths are R+ context depths."""
    if not value:
        return None
    if not isinstance(value, dict):
        raise ValueError("continuation_trajectory_depth_weights maps an R+ context depth to a loss weight")
    out = []
    for depth, weight in value.items():
        try:
            depth, weight = int(depth), float(weight)
        except (TypeError, ValueError):
            raise ValueError(f"continuation_trajectory_depth_weights: {depth!r}: {weight!r} is not depth: weight")
        if depth < 1 or not 0.0 <= weight < 100.0:
            raise ValueError(f"continuation_trajectory_depth_weights: depth {depth} >= 1, weight {weight} in [0, 100)")
        out.append((depth, weight))
    return tuple(sorted(out))


def settings(kw: dict) -> Optional[Settings]:
    """Trajectory settings from model_kwargs; None when ``continuation_trajectory_dir`` is not set."""
    kw = kw or {}
    if not enabled(kw):
        return None
    stale = [k for k in BURN_MODE_KEYS if k in kw]
    if stale:
        raise ValueError(f"continuation_trajectory_dir replaces the burned-context mode; remove {stale} "
                         "(continuation_trajectory_anchor_share takes the place of continuation_no_context_share)")
    frames = kw.get("continuation_context_frames")
    frames = 39 if frames is None else int(frames)
    h3_continuation.latent_steps(frames)  # a 17n+5 run
    protect_audio = bool(kw.get("continuation_protect_audio", True))
    if protect_audio:
        h3_continuation.audio_context_steps(frames)  # the prefix must end on the shared 24 fps / 40 Hz grid
    anchor = float(kw.get("continuation_trajectory_anchor_share", 0.2))
    noised = float(kw.get("continuation_trajectory_noised_share", 0.15))
    if not (0.0 <= anchor <= 1.0 and 0.0 <= noised <= 1.0 and anchor + noised <= 1.0 + 1e-9):
        raise ValueError(f"continuation_trajectory_anchor_share ({anchor}) and _noised_share ({noised}) must be in "
                         "[0, 1] and sum to at most 1 (the rest are clean-context continuation samples)")
    mode = str(kw.get("continuation_trajectory_context_noise_mode", "fixed")).lower()
    if mode not in NOISE_MODES:
        raise ValueError(f"continuation_trajectory_context_noise_mode must be one of {NOISE_MODES}, got {mode!r}")
    noise = kw.get("continuation_trajectory_context_noise", (0.02, 0.05))
    if isinstance(noise, (int, float)):
        noise = (noise, noise)
    if len(noise) != 2:
        raise ValueError(f"continuation_trajectory_context_noise must be [low, high], got {noise}")
    low, high = float(noise[0]), float(noise[1])
    if not 0.0 <= low <= high < 1.0:
        raise ValueError(f"continuation_trajectory_context_noise must satisfy 0 <= low <= high < 1, got {noise}")
    if mode == "mask" and noised and low <= MASK_SNAP:
        raise ValueError(f"mask-mode context noise values <= {MASK_SNAP} snap to 0 at inference (fully protected); "
                         f"use a range above it, got {noise}")
    weights = kw.get("continuation_trajectory_step_weights")
    if weights is not None:
        weights = tuple(float(w) for w in weights)
        if not weights or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(f"continuation_trajectory_step_weights must be non-negative with a positive sum, "
                             f"got {list(weights)}")
    text = str(kw.get("continuation_trajectory_text", "recorded")).lower()
    if text not in TEXT_SOURCES:
        raise ValueError(f"continuation_trajectory_text must be one of {TEXT_SOURCES}, got {text!r}")
    floor = kw.get("continuation_trajectory_floor", 0.0)
    floor = 1e-3 if floor is True else float(floor or 0.0)
    if not 0.0 <= floor < 0.1:
        raise ValueError(f"continuation_trajectory_floor is the relative size of the nudge (e.g. 1e-3; 0 = off), got {floor}")
    cap = kw.get("continuation_trajectory_floor_ratio_cap")
    cap = None if cap is None else float(cap)
    if cap is not None and not cap >= 1.0:
        raise ValueError(f"continuation_trajectory_floor_ratio_cap must be at least 1 (e.g. 4.0), got {cap}")
    return Settings(directory=str(kw["continuation_trajectory_dir"]), context_frames=frames,
                    protect_audio=protect_audio, anchor_share=anchor, noised_share=noised, context_noise=(low, high),
                    context_noise_mode=mode, noise_audio_context=bool(kw.get("continuation_trajectory_noise_audio",
                                                                              False)),
                    step_weights=weights, text=text, sweep=bool(kw.get("continuation_trajectory_sweep", False)),
                    log_every=max(1, int(kw.get("continuation_trajectory_log_every", 50))), floor=floor,
                    step_floors=parse_step_floors(kw.get("continuation_trajectory_step_floors")),
                    floor_ratio_cap=cap,
                    depth_weights=parse_depth_weights(kw.get("continuation_trajectory_depth_weights")))


def check_train_config(train_config) -> None:
    """Refuse trainer settings that would change the recorded target or the prediction it is compared with."""
    problems = []

    def refuse(name, bad, why):
        value = getattr(train_config, name, None)
        if value is not None and bad(value):
            problems.append(f"{name}: {value!r} ({why})")

    refuse("audio_teacher", bool, "the audio target is recorded")
    refuse("loss_type", lambda v: v != "mse", "recorded targets are matched with MSE")
    refuse("loss_target", lambda v: v != "noise", "the target comes from the recording")
    refuse("timestep_type", lambda v: v == "next_sample", "it replaces the recorded input")
    refuse("pred_scaler", lambda v: float(v) != 1.0, "it rescales the prediction")
    refuse("diffusion_feature_extractor_path", bool, "it compares against the dataset latents")
    for name in ("do_prior_divergence", "inverted_mask_prior", "diff_output_preservation", "blank_prompt_preservation",
                 "do_guidance_loss", "do_differential_guidance", "t0_loss_target", "do_fft_loss", "train_turbo",
                 "correct_pred_norm", "match_noise_norm"):
        refuse(name, bool, "it changes the target or the prediction")
    for name in ("short_and_long_captions", "single_item_batching"):
        refuse(name, bool, "the recorded samples are drawn per batch item")
    if problems:
        raise ValueError("MiniMax-H3 trajectory training cannot run with: " + "; ".join(problems))


RPLUS_KEYS = ("run.context.video", "run.context.audio", "run.context.mask.video", "run.context.mask.audio")


class TrajectoryFile:
    """One recorded sampling run, read lazily: only the tensors a sample needs are loaded. ``turbo`` is the turbo it
    was sampled with (its info's "turbo"; set by the store, checked against the run's turbos). ``rplus``: an R+
    teacher run (it has run.context.video), whose masked calls are points too."""

    def __init__(self, path: str):
        from safetensors import safe_open

        self.path = path
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
            self.keys = set(f.keys())
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not an H3 trajectory recording (format {meta.get('format')!r})")
        if int(meta.get("version", 0)) != FORMAT_VERSION:
            raise ValueError(f"{path}: trajectory format version {meta.get('version')}, this trainer reads "
                             f"{FORMAT_VERSION}")
        self.info = json.loads(meta["info"])
        self.turbo: Optional[str] = None
        schedule = self.info.get("schedule", {})
        self.shift_video = float(schedule.get("shift_video", VIDEO_SIGMA_SHIFT))
        self.shift_audio = float(schedule.get("shift_audio", AUDIO_SIGMA_SHIFT))
        self.calls = {int(c["index"]): c for c in self.info["calls"]}
        self.rplus = "run.context.video" in self.keys
        # the training points: conditional, unmasked calls on the sampler's schedule (a cfg-1 run makes one per step);
        # an R+ run's calls are all masked (its context), and all of them count
        self.points = [c for c in self.info["calls"]
                       if list(c.get("cond_or_uncond", [0])) == [0] and int(c.get("step", -1)) >= 0
                       and (self.rplus or not c.get("masked", False))]
        if not self.points:
            raise ValueError(f"{path}: no conditional, unmasked calls on the schedule to train on")
        self.steps = sorted({int(c["step"]) for c in self.points})
        self.context_steps: Optional[int] = None
        self.depth: Optional[int] = None
        self.student = {}
        if self.rplus:
            context = self.info.get("context") or {}
            self.context_steps = context.get("video_steps")
            depth = (self.info.get("rplus") or {}).get("depth")
            self.depth = None if depth is None else int(depth)
            # the context the student is pinned on: the recorded student context, else the sampled one
            self.student = {n: f"run.context.student.{n}" if f"run.context.student.{n}" in self.keys
                            else f"run.context.{n}" for n in ("video", "audio")}
        missing = [k for k in ("run.final.video", "run.final.audio") + (RPLUS_KEYS if self.rplus else ())
                   if k not in self.keys]
        for c in self.points:
            missing += [call_key(c["index"], n) for n in ("video.x", "audio.x", "video.out", "audio.out", "timestep")
                        if call_key(c["index"], n) not in self.keys]
        if missing:
            raise ValueError(f"{path}: missing tensors {missing[:6]}")

    def load(self, names) -> Dict[str, torch.Tensor]:
        from safetensors import safe_open

        with safe_open(self.path, framework="pt") as f:
            return {n: f.get_tensor(n) for n in names}

    def has_text(self, which: str) -> bool:
        return f"cond.{which}.embeds" in self.keys and f"cond.{which}.tags" in self.keys

    def calls_at(self, step: int) -> List[int]:
        return [int(c["index"]) for c in self.points if int(c["step"]) == step]

    def check_context(self, settings: "Settings") -> None:
        """An R+ recording's pinned latent steps must be the continuation context the trainer pins."""
        if self.rplus and self.context_steps != settings.context_steps:
            raise ValueError(f"{self.path}: the recording pins {self.context_steps} latent steps "
                             f"(info.context.video_steps), the trainer's {settings.context_frames}-frame context "
                             f"{settings.context_steps}")

    def audio_mask_key(self, index: int) -> str:
        """The audio generate mask of an R+ call: its own when it has one, else the run's."""
        own = call_key(index, "audio.mask")
        return own if own in self.keys else "run.context.mask.audio"


@dataclass
class Sample:
    """One training sample drawn from a recording: the student's input, its labels, the target and the loss weights.
    Tensors carry a batch axis of 1; video (1, 24, T, H, W), audio rows (1, 2 * Ta, 32), all fp32."""
    kind: str
    path: str
    call: int
    step: int
    timestep: torch.Tensor  # (1,) the recorded DiT timestep, 1000 * sigma_v
    sigma_v: torch.Tensor  # (1,)
    sigma_a: torch.Tensor  # (1,)
    video_in: torch.Tensor
    video_target: torch.Tensor
    context_steps: int  # pinned latent steps at the start of the clip (0: anchor)
    context_label: Optional[float]  # the label of the pinned video rows
    context_sigma: float  # noise level of the pinned context (0.001 for the clean pin, like inference)
    audio_in: torch.Tensor
    audio_target: torch.Tensor
    audio_labels: torch.Tensor  # (2 * Ta,) per-row labels
    audio_weight: torch.Tensor  # (2 * Ta,) 0 = protected (no loss), m = released tick, 1 = generated
    text_embeds: Optional[torch.Tensor] = None  # (L, 5120) recorded conditioning; None = the dataset caption's
    text_tags: Optional[torch.Tensor] = None
    logged: bool = False  # its errors went into the diagnostics already
    turbo: Optional[str] = None  # the turbo the recording was sampled with: the frozen assistant of this sample
    depth: Optional[int] = None  # R+: the clip its context was carried out of (continuation_trajectory_depth_weights)


def text_source(settings: Settings, kind: str, traj: TrajectoryFile) -> Optional[str]:
    """Which recorded conditioning a sample uses: anchors the one the run was sampled with, continuation samples the
    recorded student conditioning when there is one. None with ``continuation_trajectory_text: dataset``."""
    if settings.text == "dataset":
        return None
    which = "joint" if kind == "anchor" or traj.rplus or not traj.has_text("student") else "student"
    if not traj.has_text(which):
        raise ValueError(f"{traj.path} has no recorded conditioning; record it with the H3RecordTrajectory node or "
                         "set continuation_trajectory_text: dataset")
    return which


def build_sample(traj: TrajectoryFile, settings: Settings, kind: str, call_index: int,
                 noise_level: Optional[float] = None, generator: Optional[torch.Generator] = None) -> Sample:
    """The student's view of one recorded call.

    anchor: the recorded state as is; labels t_v / t_a; target the recorded output everywhere.
    continuation: video latent steps [0, S) replaced by the run's own final latent pinned like inference
      (0.999 * clean + 0.001 * noise, labelled max(t_v, 0.999)); with protect_audio the audio rows under the context use
      the continuation's audio mask m: m * recorded state + (1 - m) * final clean audio, labelled 1 - m * sigma_a
      (protected rows exactly clean at t = 1), and only m > 0 rows teach, weighted by m.
    noised: like continuation, the video context at noise level s: (1 - s) * clean + s * noise, labelled 1 - s (fixed
      mode: s = noise_level; mask mode: s = noise_level * sigma_v, what an inference mask value does to the rows).
    An R+ recording gives continuation samples only (see the module docstring)."""
    if kind not in KINDS:
        raise ValueError(f"unknown sample kind {kind!r}")
    if traj.rplus:
        return build_rplus_sample(traj, settings, kind, call_index)
    call = traj.calls[int(call_index)]
    index = int(call["index"])
    which = text_source(settings, kind, traj)
    names = [call_key(index, n) for n in ("video.x", "audio.x", "video.out", "audio.out", "timestep")]
    if kind != "anchor":
        names += ["run.final.video", "run.final.audio"]
    if which is not None:
        names += [f"cond.{which}.embeds", f"cond.{which}.tags"]
    t = traj.load(names)

    timestep = t[call_key(index, "timestep")].to(torch.float32).reshape(-1)[:1].clone()
    sigma_v, sigma_a = stream_sigmas(timestep, traj.shift_video, traj.shift_audio)
    t_v, t_a = 1.0 - sigma_v, 1.0 - sigma_a
    video_state = t[call_key(index, "video.x")].to(torch.float32)
    video_target = t[call_key(index, "video.out")].to(torch.float32)
    audio_state = audio_rows(t[call_key(index, "audio.x")].to(torch.float32))
    audio_target = audio_rows(t[call_key(index, "audio.out")].to(torch.float32))
    n_rows = audio_state.shape[1]

    video_in, audio_in = video_state, audio_state
    audio_labels = t_a.expand(n_rows).clone()
    audio_weight = torch.ones(n_rows)
    context_steps, context_label, context_sigma = 0, None, 0.0
    if kind != "anchor":
        context_steps = settings.context_steps
        if context_steps >= video_state.shape[2]:
            raise ValueError(f"{traj.path}: a {settings.context_frames}-frame context leaves no new latent steps "
                             f"(the recording has {video_state.shape[2]})")
        clean = t["run.final.video"].to(torch.float32)[:, :, :context_steps]
        eps = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
        pin = max(float(t_v), h3_continuation.CONTEXT_T)  # inference's cap for protected rows: max(t_v, 0.999)
        audio_context_sigma = 0.0
        if kind == "continuation":
            context_sigma = 1.0 - h3_continuation.CONTEXT_T
            pinned = h3_continuation.CONTEXT_T * clean + (1.0 - h3_continuation.CONTEXT_T) * eps
            context_label = pin
        else:
            if noise_level is None:
                raise ValueError("a noised sample needs a noise level")
            mask_mode = settings.context_noise_mode == "mask"
            # a mask value m puts a row at m * sigma of its own stream
            context_sigma = float(noise_level) * (float(sigma_v) if mask_mode else 1.0)
            audio_context_sigma = float(noise_level) * (float(sigma_a) if mask_mode else 1.0)
            pinned = (1.0 - context_sigma) * clean + context_sigma * eps
            context_label = min(1.0 - context_sigma, pin)
        video_in = video_state.clone()
        video_in[:, :, :context_steps] = pinned

        if settings.protect_audio:
            ticks = n_rows // h3_continuation.AUDIO_CHANNELS
            if settings.audio_context_ticks >= ticks:
                raise ValueError(f"{traj.path}: the audio context covers the whole soundtrack")
            m = h3_continuation.audio_row_mask(h3_continuation.audio_context_mask(ticks, settings.audio_context_ticks))
            clean_audio = audio_rows(t["run.final.audio"].to(torch.float32))
            hard = m == 0
            noise_audio = kind == "noised" and settings.noise_audio_context
            if noise_audio:
                # only the fully protected ticks carry the noise; the released ones keep inference's blend
                a_eps = torch.randn(clean_audio.shape, generator=generator, dtype=torch.float32)
                noisy = (1.0 - audio_context_sigma) * clean_audio + audio_context_sigma * a_eps
                clean_audio = torch.where(hard.view(1, -1, 1), noisy, clean_audio)
            mm = m.view(1, -1, 1)
            audio_in = mm * audio_state + (1.0 - mm) * clean_audio
            # inference's label: 1 - m * sigma_a, capped at the preserved-row timestep max(t_a, 1.0)
            audio_labels = torch.minimum(1.0 - m * sigma_a, torch.clamp(t_a, min=h3_continuation.AUDIO_CONTEXT_T))
            if noise_audio:
                audio_labels = torch.where(hard, torch.full_like(audio_labels, 1.0 - audio_context_sigma),
                                           audio_labels)
            audio_weight = m.clone()

    return Sample(kind=kind, path=traj.path, call=index, step=int(call.get("step", -1)), timestep=timestep,
                  sigma_v=sigma_v, sigma_a=sigma_a, video_in=video_in, video_target=video_target,
                  context_steps=context_steps, context_label=context_label, context_sigma=context_sigma,
                  audio_in=audio_in, audio_target=audio_target, audio_labels=audio_labels, audio_weight=audio_weight,
                  text_embeds=None if which is None else t[f"cond.{which}.embeds"].reshape(
                      -1, t[f"cond.{which}.embeds"].shape[-1]),
                  text_tags=None if which is None else t[f"cond.{which}.tags"].reshape(-1).to(torch.long),
                  turbo=traj.turbo)


def build_rplus_sample(traj: TrajectoryFile, settings: Settings, kind: str, call_index: int) -> Sample:
    """The student's view of one call of an R+ teacher run: the recorded state with the pinned video steps moved from
    the sampled (repaired) context to the student's (burned) one, delta-substituted so the engine's own pin (clean or
    0.999 * context + 0.001 * noise) carries over, labelled max(t_v, 0.999); the audio as recorded under the recorded
    mask m, labels min(1 - m * sigma_a, max(t_a, 1)), weight m; targets the recorded outputs; joint conditioning."""
    if kind != "continuation":
        raise ValueError(f"{traj.path}: an R+ recording gives continuation samples only, not {kind!r}")
    traj.check_context(settings)
    call = traj.calls[int(call_index)]
    index = int(call["index"])
    which = text_source(settings, kind, traj)
    mask_key = traj.audio_mask_key(index)
    names = [call_key(index, n) for n in ("video.x", "audio.x", "video.out", "audio.out", "timestep")]
    names += sorted({"run.context.video", traj.student["video"], mask_key})
    if which is not None:
        names += [f"cond.{which}.embeds", f"cond.{which}.tags"]
    t = traj.load(names)

    timestep = t[call_key(index, "timestep")].to(torch.float32).reshape(-1)[:1].clone()
    sigma_v, sigma_a = stream_sigmas(timestep, traj.shift_video, traj.shift_audio)
    t_v, t_a = 1.0 - sigma_v, 1.0 - sigma_a
    video_state = t[call_key(index, "video.x")].to(torch.float32)
    steps = settings.context_steps
    if steps >= video_state.shape[2]:
        raise ValueError(f"{traj.path}: a {settings.context_frames}-frame context leaves no new latent steps "
                         f"(the recording has {video_state.shape[2]})")
    context = t["run.context.video"].to(torch.float32)[:, :, :steps]
    student = t[traj.student["video"]].to(torch.float32)[:, :, :steps]
    if student.shape != context.shape:
        raise ValueError(f"{traj.path}: the student context {list(student.shape)} does not cover the pinned "
                         f"{list(context.shape)}")
    video_in = video_state.clone()
    video_in[:, :, :steps] += h3_continuation.CONTEXT_T * (student - context)

    audio_in = audio_rows(t[call_key(index, "audio.x")].to(torch.float32))
    n_rows = audio_in.shape[1]
    m = audio_rows(t[mask_key].to(torch.float32)[:, :1])[0, :, 0]  # (1, 1, 2, Ta) -> one value per packed row
    ticks = n_rows // h3_continuation.AUDIO_CHANNELS
    # the engine's native quantisation (ceil to 1/256), or the MaskedExtension compat one (round, snapped)
    diffs = [(m - h3_continuation.audio_row_mask(fn(ticks, settings.audio_context_ticks))).abs().max().item()
             for fn in (h3_continuation.engine_audio_context_mask, h3_continuation.audio_context_mask)
             if m.shape == (h3_continuation.AUDIO_CHANNELS * ticks,)]
    if not diffs or min(diffs) > 1e-4:
        raise ValueError(f"{traj.path}: the recorded audio mask is not the {settings.context_frames}-frame "
                         f"continuation mask (max difference {min(diffs) if diffs else 'shape'})")
    audio_labels = torch.minimum(1.0 - m * sigma_a, torch.clamp(t_a, min=h3_continuation.AUDIO_CONTEXT_T))

    return Sample(kind=kind, path=traj.path, call=index, step=int(call.get("step", -1)), timestep=timestep,
                  sigma_v=sigma_v, sigma_a=sigma_a, video_in=video_in,
                  video_target=t[call_key(index, "video.out")].to(torch.float32), context_steps=steps,
                  context_label=max(float(t_v), h3_continuation.CONTEXT_T),
                  context_sigma=1.0 - h3_continuation.CONTEXT_T, audio_in=audio_in,
                  audio_target=audio_rows(t[call_key(index, "audio.out")].to(torch.float32)),
                  audio_labels=audio_labels, audio_weight=m.clone(),
                  text_embeds=None if which is None else t[f"cond.{which}.embeds"].reshape(
                      -1, t[f"cond.{which}.embeds"].shape[-1]),
                  text_tags=None if which is None else t[f"cond.{which}.tags"].reshape(-1).to(torch.long),
                  turbo=traj.turbo, depth=traj.depth)


class TrajectoryStore:
    """Finds, opens (cached) and draws from the recordings behind a dataset."""

    def __init__(self, settings: Settings, turbos: Optional[Dict[str, "h3_turbos.Turbo"]] = None):
        self.settings = settings
        self.turbos = turbos  # the run's turbos (model_kwargs.turbo); None: the single-assistant setup
        self._files: Dict[str, TrajectoryFile] = {}
        self.counter = 0  # samples drawn so far
        self._swept: Dict[str, int] = {}  # sweep position per recording
        self.stats = Diagnostics()
        self.announced = False

    def path_for(self, item_path: str) -> str:
        stem = os.path.splitext(os.path.basename(str(item_path)))[0]
        path = os.path.join(self.settings.directory, stem + ".safetensors")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no trajectory recording for dataset item {item_path}: expected {path} "
                                    "(recordings and dataset files must share the file stem)")
        return path

    def open(self, path: str) -> TrajectoryFile:
        traj = self._files.get(path)
        if traj is None:
            traj = TrajectoryFile(path)
            if self.turbos:
                traj.turbo = h3_turbos.check_recording(path, traj.info, self.turbos)
            else:
                traj.turbo = h3_turbos.recording_turbo(path, traj.info, ())
            traj.check_context(self.settings)
            if self.settings.step_weights is not None and len(self.settings.step_weights) != len(traj.steps):
                raise ValueError(f"continuation_trajectory_step_weights has {len(self.settings.step_weights)} "
                                 f"entries but {path} has {len(traj.steps)} sampler steps")
            if self.settings.step_floors:
                video, _ = self.settings.floors_for(traj.turbo)
                if len(video) != len(traj.steps) or traj.steps != list(range(len(traj.steps))):
                    raise ValueError(f"continuation_trajectory_step_floors for turbo {traj.turbo} has {len(video)} "
                                     f"steps but {path} has sampler steps {traj.steps}")
            self._files[path] = traj
        return traj

    def turbo_counts(self, dataset_config) -> Dict[str, int]:
        """Recordings per turbo behind a dataset, each checked against the run's turbos (header only); {} without
        turbos."""
        folder = getattr(dataset_config, "folder_path", None)
        if not self.turbos or not folder or not os.path.isdir(folder):
            return {}
        counts = dict.fromkeys(self.turbos, 0)
        for name in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(name)
            if ext.lower() in MEDIA_EXTS:
                counts[self.open(os.path.join(self.settings.directory, stem + ".safetensors")).turbo] += 1
        return counts

    def draw(self, traj: TrajectoryFile, generator: Optional[torch.Generator] = None) -> Tuple[str, int, Optional[float]]:
        """(kind, call index, noise level) for one sample: kinds by their shares, steps by step_weights (uniform by
        default), or with ``sweep`` a fixed cycle through every kind and step of each recording (for the step-0
        diagnostic: an epoch over the dataset draws every recording once, so every recording covers every kind and
        step in len(kinds) * steps epochs). An R+ recording always gives a continuation sample."""
        s = self.settings
        if s.sweep:
            kinds = ["continuation"] if traj.rplus else [k for k in KINDS if s.shares[k] > 1e-9]
            n = self._swept.get(traj.path, 0)
            self._swept[traj.path] = n + 1
            kind = kinds[n % len(kinds)]
            step = traj.steps[(n // len(kinds)) % len(traj.steps)]
            call = traj.calls_at(step)[0]
        else:
            u = torch.rand((), generator=generator).item()
            kind = "anchor" if u < s.anchor_share else ("noised" if u < s.anchor_share + s.noised_share
                                                         else "continuation")
            if traj.rplus:
                kind = "continuation"
            weights = torch.tensor(s.step_weights if s.step_weights is not None else [1.0] * len(traj.steps),
                                   dtype=torch.float64)
            step = traj.steps[int(torch.multinomial(weights, 1, generator=generator))]
            candidates = traj.calls_at(step)
            call = candidates[int(torch.randint(len(candidates), (1,), generator=generator))]
        self.counter += 1
        level = None
        if kind == "noised":
            low, high = s.context_noise
            level = low + (high - low) * torch.rand((), generator=generator).item()
        return kind, call, level

    def check_dataset(self, dataset_config) -> int:
        """Every media file of a trajectory dataset needs a recording with the same stem; checked before caching."""
        folder = getattr(dataset_config, "folder_path", None)
        if not folder or not os.path.isdir(folder):
            return 0
        stems = sorted({os.path.splitext(n)[0] for n in os.listdir(folder)
                        if os.path.splitext(n)[1].lower() in MEDIA_EXTS})
        missing = [n for n in stems if not os.path.exists(os.path.join(self.settings.directory, n + ".safetensors"))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} of {len(stems)} items in {folder} have no recording in "
                                    f"{self.settings.directory}, e.g. {missing[:5]}")
        return len(stems)


def batch_samples(batch, settings: Settings, store: TrajectoryStore,
                  generator: Optional[torch.Generator] = None) -> List[Sample]:
    """The step's samples, one per batch item. Drawn once and kept on the batch, so every pass of a step (the trainer's
    input hook, the prediction, the loss target) sees the same sample."""
    samples = getattr(batch, "trajectory_samples", None)
    if samples is None:
        samples = []
        for item in batch.file_items:
            traj = store.open(store.path_for(item.path))
            kind, call, level = store.draw(traj, generator)
            samples.append(build_sample(traj, settings, kind, call, level, generator))
        batch.trajectory_samples = samples
    return samples


@torch.no_grad()
def sample_errors(sample: Sample, pred_video: torch.Tensor, pred_audio: torch.Tensor) -> Dict[str, float]:
    """Error of one item's prediction on the tokens that teach: video after the pinned context, fully generated audio
    rows. ``rel`` is the RMS error over the target's RMS."""
    s = sample.context_steps
    target_v = sample.video_target[:, :, s:].to(pred_video.device, torch.float32)
    dv = pred_video[:, :, s:].float() - target_v
    keep = (sample.audio_weight >= 1.0).to(pred_audio.device)
    target_a = sample.audio_target.to(pred_audio.device, torch.float32)[:, keep]
    da = pred_audio.float()[:, keep] - target_a
    out = {"video_mse": dv.pow(2).mean().item(), "audio_mse": da.pow(2).mean().item() if da.numel() else 0.0}
    out["video_rel"] = math.sqrt(out["video_mse"]) / max(target_v.pow(2).mean().sqrt().item(), 1e-12)
    out["audio_rel"] = (math.sqrt(out["audio_mse"]) / max(target_a.pow(2).mean().sqrt().item(), 1e-12)
                        if da.numel() else 0.0)
    return out


def nudge(video: torch.Tensor, eps: float, seed: int = 4321) -> torch.Tensor:
    """The floor probe's input: the video state plus Gaussian noise of relative RMS ``eps``, rounded to bf16 like the
    engine's DiT input (at eps 1e-3 about a fifth of the elements move by one bf16 step)."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(video.shape, generator=g) * (eps * float(video.float().pow(2).mean().sqrt()))
    return (video.float().cpu() + noise).to(torch.bfloat16).to(video.device, torch.float32)


@torch.no_grad()
def sample_floor(sample: Sample, pred_video: torch.Tensor, pred_audio: torch.Tensor, nudged_video: torch.Tensor,
                 nudged_audio: torch.Tensor) -> Dict[str, float]:
    """How far the model's own prediction moves under the nudge, on the tokens that teach, relative to the target's RMS
    (the scale of ``*_rel``): the floor any network that is not bit-identical to the engine sits at."""
    s = sample.context_steps
    target_v = sample.video_target[:, :, s:].to(pred_video.device, torch.float32)
    dv = nudged_video[:, :, s:].float() - pred_video[:, :, s:].float()
    keep = (sample.audio_weight >= 1.0).to(pred_audio.device)
    target_a = sample.audio_target.to(pred_audio.device, torch.float32)[:, keep]
    da = nudged_audio.float()[:, keep] - pred_audio.float()[:, keep]
    return {"video_floor": dv.pow(2).mean().sqrt().item() / max(target_v.pow(2).mean().sqrt().item(), 1e-12),
            "audio_floor": (da.pow(2).mean().sqrt().item() / max(target_a.pow(2).mean().sqrt().item(), 1e-12)
                            if da.numel() else 0.0)}


class Diagnostics:
    """Running mean of the per-sample errors by (kind, step), and by turbo when the recordings say which turbo sampled
    them: with the LoRA at zero (step 0, or an lr 0 run with ``continuation_trajectory_sweep``) the anchor rows are the
    trainer-vs-engine mismatch and the continuation rows the continuation offset, per sigma. With
    ``continuation_trajectory_floor`` the table also shows the model's own floor at each step: an anchor row at its
    floor is as close to the engine as a non-identical forward gets."""

    FIELDS = ("video_mse", "video_rel", "audio_mse", "audio_rel")
    FLOOR_FIELDS = ("video_floor", "audio_floor")

    def __init__(self):
        self.rows: Dict[tuple, Dict[str, float]] = {}  # (kind, step) or (kind, step, turbo)
        self.total = 0

    def add(self, kind: str, step: int, sigma_v: float, errors: Dict[str, float], turbo: Optional[str] = None):
        self.total += 1
        key = (kind, step) if turbo is None else (kind, step, turbo)
        row = self.rows.setdefault(key, {"n": 0, "sigma_v": sigma_v, **{f: 0.0 for f in self.FIELDS}})
        row["n"] += 1
        for f in self.FIELDS:
            row[f] += (errors[f] - row[f]) / row["n"]
        if all(f in errors for f in self.FLOOR_FIELDS):
            row["n_floor"] = row.get("n_floor", 0) + 1
            for f in self.FLOOR_FIELDS:
                row[f] = row.get(f, 0.0) + (errors[f] - row.get(f, 0.0)) / row["n_floor"]

    def report(self) -> str:
        floor = any("n_floor" in r for r in self.rows.values())
        turbos = any(len(key) > 2 for key in self.rows)
        lines = [(f"{'turbo':<16}" if turbos else "") + f"{'kind':<13}{'step':>5}{'sigma_v':>9}{'n':>6}{'video mse':>12}"
                 f"{'video rel':>11}{'audio mse':>12}{'audio rel':>11}"
                 + (f"{'video floor':>13}{'audio floor':>13}" if floor else "")]
        order = lambda kv: (kv[0][2] if len(kv[0]) > 2 else "", KINDS.index(kv[0][0]), kv[0][1])
        for key, r in sorted(self.rows.items(), key=order):
            kind, step = key[0], key[1]
            line = ((f"{key[2] if len(key) > 2 else '-':<16}" if turbos else "")
                    + f"{kind:<13}{step:>5}{r['sigma_v']:>9.4f}{r['n']:>6}{r['video_mse']:>12.3e}"
                    f"{r['video_rel']:>10.2%} {r['audio_mse']:>12.3e}{r['audio_rel']:>10.2%}")
            if floor:
                line += (f"{r['video_floor']:>12.2%} {r['audio_floor']:>12.2%}" if "n_floor" in r
                         else f"{'-':>12} {'-':>12}")
            lines.append(line)
        floors = self.step_floors()
        if floors:
            lines.append("measured floors (continuation_trajectory_step_floors): " + json.dumps(floors))
        return "\n".join(lines)

    def step_floors(self) -> Dict[str, Dict[str, List[float]]]:
        """{turbo ("default" without one): {"video": [...], "audio": [...]}}, the floor per sampler step pooled over
        the sample kinds, for every turbo whose steps 0..n-1 all have a measurement."""
        pooled: Dict[str, Dict[int, List[float]]] = {}
        for key, r in self.rows.items():
            if "n_floor" not in r:
                continue
            acc = pooled.setdefault(key[2] if len(key) > 2 else "default", {}).setdefault(key[1], [0.0, 0.0, 0])
            acc[0] += r["video_floor"] * r["n_floor"]
            acc[1] += r["audio_floor"] * r["n_floor"]
            acc[2] += r["n_floor"]
        out = {}
        for turbo, steps in sorted(pooled.items()):
            if sorted(steps) != list(range(len(steps))):
                continue
            out[turbo] = {"video": [round(steps[k][0] / steps[k][2], 5) for k in sorted(steps)],
                          "audio": [round(steps[k][1] / steps[k][2], 5) for k in sorted(steps)]}
        return out
