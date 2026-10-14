"""Core of the MiniMax H3 trajectory recorder. No ComfyUI imports, so it can be tested and reused without ComfyUI.

The node (__init__.py) puts two wrappers on a cloned MODEL:
  OUTER_SAMPLE     around CFGGuider.outer_sample: one sampling run. Starts a record (seed, schedule, initial noise, the
                   guider's positive conditioning); when the sampler returns, saves it with the final latent. A run
                   that fails saves nothing.
  DIFFUSION_MODEL  around MiniMaxH3Model._forward: every model call of that run, exactly as the H3 DiT sees it.

What ComfyUI hands those wrappers for MiniMax H3 (read from comfy/samplers.py, comfy/model_base.py,
comfy/ldm/minimax/model.py and the MiniMax-H3-Turbo sampler, 2026-09-24):
  * The sampler walks one PACKED latent x (B, 1, Nv + Na): the flattened video latent (B, 24, T, H, W) followed by
    the flattened audio latent (B, 32, 2, Ta). The audio part is CARRIED on the video schedule (ModelSamplingAV):
    y_a = (sigma_v / sigma_a) * x_a, whose clean end is audio_scale * audio with audio_scale = shift_video /
    shift_audio (12 / 3 = 4). With that carry the pack is an ordinary single-schedule flow latent, and the turbo
    sampler's native path is plain Euler, one model call per step:
        denoised = model(x, sigma);  d = (x - denoised) / sigma;  x = x + (sigma_next - sigma) * d
  * OUTER_SAMPLE args: (executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed,
    latent_shapes=[video shape, audio shape]). noise / latent_image are packed fp32; executor.class_obj is the guider,
    whose original_conds["positive"][i] holds "cross_attn" (the raw Qwen3-VL states [1, L, 5120]) and
    "minimax_token_tags" ((L,), 1 text / 0 vision). The executor returns the sampler output after process_latent_out:
    packed fp32, the audio part divided by audio_scale again (the audio latent in its own space).
  * model_base._apply_model casts x to the compute dtype (bf16), passes timestep = 1000 * sigma (fp32) and unpacks
    it into [video, audio]. MiniMaxH3Model.forward derives sigma_v = timestep / 1000 and
    sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a) and multiplies the audio by sigma_a / sigma_v BEFORE the
    DIFFUSION_MODEL wrappers run. So a DIFFUSION_MODEL wrapper gets (executor, x, timestep, context,
    transformer_options, minimax_payload=..., denoise_mask=..., audio_denoise_mask=..., ...) with
    x = [video (B, 24, T, H, W), audio in its OWN flow space (B, 32, 2, Ta)], both in the compute dtype;
    executor.class_obj is the diffusion model; transformer_options["sigmas"] is the sampler's sigma and
    ["cond_or_uncond"] says which conditioning each batch row is.
  * _forward returns [-video_raw, -audio_raw]. H3 predicts clean - noise, so these are the flow velocities
    noise - clean of each stream in its own space (x0 = x - sigma * out, sigma_a for the audio), in the input dtype.
    Afterwards forward() multiplies by the inpaint masks (if any) and converts the audio back to the carried variable,
        out_c = (1 - audio_scale) * x_a + (1 + (audio_scale - 1) * sigma_a) * out_a,
    and CONST's calculate_denoised hands the sampler denoised = x - sigma_v * [out_v, out_c] (fp32 x).
A DIFFUSION_MODEL record is therefore the network's own (input, timestep, output) in each stream's own flow space.
That is also ai-toolkit's training convention (prediction and target are noise - clean on the 1000 * sigma scale), so
the trainer uses the recorded outputs as targets with no conversion beyond the audio row layout.

Masked continuation runs (a denoise mask, or a non-empty latent_image: the pinned context of a masked AV continuation)
also store the context as the sampler received it, before process_latent_in (comfy/samplers.py inner_sample), i.e. in
the space of run.final.*: run.context.video / .audio (fp32, full length, the pinned prefix and zeros), and the
generate masks once, from the first model call that carries them: run.context.mask.video (1, 1, T, H, W) and
run.context.mask.audio (1, 1, 2, Ta), 1 = generate. A call whose mask differs from those also gets its own
call.NNN.video.mask / audio.mask. An optional student context (student_context_path: a safetensors file with video
(1, 24, S, H, W) and audio (1, 32, 2, Ta') for the S pinned latent steps) is stored as run.context.student.*: the
context the trainer pins instead of the one this run was sampled from. info["context"] = {video_steps, audio_ticks,
student: {path, sha256 of the file} or null}.
"""
import datetime
import hashlib
import json
import math
import os

import torch

FORMAT = "h3_trajectory"
FORMAT_VERSION = 1
RECORDER_VERSION = "1"
CHAIN_TOLERANCE = 0.02  # relative RMS: bf16 rounding of the recorded states costs ~0.2-0.5%
MASK_ONE = 1.0 - 1e-3  # a masked run's check replays only the elements the sampler generates in full (mask >= this)
MASK_ZERO = 1e-3  # a pinned (context) latent step: every mask value <= this

CONVENTIONS = (
    "Per call (DiT level, after MiniMaxH3Model.forward undid the audio carry): video.x (1,24,T,H,W) and audio.x "
    "(1,32,2,Ta, the audio in its own flow space) are the DiT inputs in the compute dtype; timestep = 1000*sigma_v "
    "(fp32); sigma_a = time_shift_sigma(sigma_v, shift_video, shift_audio); labels t = 1 - sigma per stream; "
    "video.out / audio.out = what _forward returned = -(raw H3 prediction) = noise - clean in each stream's own space "
    "(x0 = x - sigma*out). The sampler's carried audio is y = (sigma_v/sigma_a)*x_a and its velocity "
    "(1-r)*x_a + (1+(r-1)*sigma_a)*out_a with r = audio_scale. run.final.* is the sampler output (audio in its own "
    "space); run.noise.* the initial noise; cond.* the raw Qwen3-VL conditioning and token tags."
)


def call_key(index, name):
    """Tensor key of one recorded model call: call_key(3, "video.x") -> "call.003.video.x"."""
    return f"call.{int(index):03d}.{name}"


def time_shift_sigma(sigma, from_shift, to_shift):
    """comfy/ldm/minimax/model.py: invert the exponential shift to the base grid, re-apply the other one."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def unpack(packed, shapes):
    """comfy.utils.unpack_latents: (B, 1, N) -> one tensor per shape."""
    out, rest = [], packed
    for shape in shapes:
        n = math.prod(shape[1:])
        out.append(rest[:, :, :n].reshape([rest.shape[0]] + list(shape)[1:]))
        rest = rest[:, :, n:]
    return out


def carried_audio(x_a, sigma_v, sigma_a):
    """The sampler's carried audio variable for a state in the audio's own space."""
    return (sigma_v / sigma_a) * x_a


def carried_audio_velocity(out_a, x_a, sigma_a, audio_scale):
    """d(carried audio)/d(sigma_v) for the DiT's own-space audio output (MiniMaxH3Model.forward)."""
    return (1.0 - audio_scale) * x_a + (1.0 + (audio_scale - 1.0) * sigma_a) * out_a


def _cpu(t):
    return t.detach().to("cpu", copy=True).contiguous()


def _sha256(t):
    t = _cpu(t)
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _streams(x):
    if getattr(x, "is_nested", False):
        return list(x.unbind())
    return list(x)


def conditioning_tensors(conditioning):
    """(states (L, C), tags (L,)) from a ComfyUI CONDITIONING ([[tensor, {"minimax_token_tags": ...}], ...]) or from
    one of the guider's converted cond dicts ({"cross_attn": tensor, "minimax_token_tags": ...}); None if absent."""
    if conditioning is None:
        return None
    if isinstance(conditioning, dict):
        states, extra = conditioning.get("cross_attn"), conditioning
    else:
        if not conditioning:
            return None
        states, extra = conditioning[0][0], conditioning[0][1]
    if states is None:
        return None
    states = _cpu(states)
    states = states.reshape(-1, states.shape[-1])
    tags = extra.get("minimax_token_tags")
    tags = _cpu(tags).reshape(-1).to(torch.long) if tags is not None else torch.ones(states.shape[0], dtype=torch.long)
    if tags.shape[0] != states.shape[0]:
        raise ValueError(f"conditioning has {states.shape[0]} states but {tags.shape[0]} token tags")
    return states, tags


def write_trajectory(path, tensors, info):
    """Atomically write the recording (safetensors, the info JSON in its metadata) and a readable JSON sidecar."""
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    save_file({k: v.contiguous() for k, v in tensors.items()}, tmp,
              metadata={"format": FORMAT, "version": str(FORMAT_VERSION), "info": json.dumps(info)})
    os.replace(tmp, path)
    sidecar = os.path.splitext(path)[0] + ".json"
    with open(sidecar + ".tmp", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(sidecar + ".tmp", sidecar)
    return sidecar


def read_trajectory(path, names=None):
    """(info, tensors): every tensor, or only ``names``."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not an H3 trajectory recording")
        keys = f.keys() if names is None else names
        tensors = {k: f.get_tensor(k) for k in keys}
    return json.loads(meta["info"]), tensors


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def load_student_context(path):
    """{"path", "sha256", "video", "audio"} from a student context file (absolute path, keys video and audio); None
    for an empty path."""
    from safetensors import safe_open

    path = (path or "").strip()
    if not path:
        return None
    if not os.path.isabs(path) or not os.path.isfile(path):
        raise ValueError(f"student_context_path must be an absolute path to a safetensors file, got {path!r}")
    with safe_open(path, framework="pt") as f:
        missing = [k for k in ("video", "audio") if k not in f.keys()]
        if missing:
            raise ValueError(f"student context {path} has no {missing} (keys video and audio)")
        video, audio = f.get_tensor("video").float(), f.get_tensor("audio").float()
    if video.dim() != 5 or audio.dim() != 4:
        raise ValueError(f"student context {path}: video {list(video.shape)} / audio {list(audio.shape)}, expected "
                         "(1, 24, S, H, W) / (1, 32, 2, Ta)")
    return {"path": path, "sha256": file_sha256(path), "video": video.contiguous(), "audio": audio.contiguous()}


def pinned_steps(video_mask):
    """Latent steps whose every mask value is 0 (pinned context), from a (1, C, T, H, W) mask."""
    m = video_mask.float()
    return int((m.transpose(0, 2).reshape(m.shape[2], -1) <= MASK_ZERO).all(dim=1).sum())


def context_audio_ticks(video_steps):
    """The 40 Hz audio ticks under a pinned video prefix of that many latent steps (12 -> 39 frames -> 65), as the
    AV continuation protects them; None when the prefix is not on the shared 24 fps / 40 Hz grid."""
    if video_steps <= 0:
        return 0
    if (video_steps - 2) % 5:
        return None
    frames = 17 * ((video_steps - 2) // 5) + 5
    return frames * 40 // 24 if (frames * 40) % 24 == 0 else None


def check_student_context(student, video_shape, audio_shape, video_mask):
    """The student context must fit the run: its channels, height and width, and exactly the mask-0 latent steps."""
    sv, sa = student["video"], student["audio"]
    steps = pinned_steps(video_mask)
    if (sv.shape[1], sv.shape[3], sv.shape[4]) != (video_shape[1], video_shape[3], video_shape[4]):
        raise ValueError(f"student context {student['path']}: video {list(sv.shape)} does not fit the run's "
                         f"{list(video_shape)} (channels, height, width)")
    if sv.shape[2] != steps:
        raise ValueError(f"student context {student['path']}: {sv.shape[2]} latent steps, the run pins {steps}")
    if (sa.shape[1], sa.shape[2]) != (audio_shape[1], audio_shape[2]) or sa.shape[3] > audio_shape[3]:
        raise ValueError(f"student context {student['path']}: audio {list(sa.shape)} does not fit the run's "
                         f"{list(audio_shape)}")


class TrajectoryRecorder:
    """Collects one sampling run at a time and writes it as <directory>/<name>.safetensors (+ .json)."""

    def __init__(self, name, directory, overwrite=False, prompt="", student=None, save_noise=True,
                 student_context=None):
        name = (name or "").strip()
        if not name or os.path.basename(name) != name or name in (".", ".."):
            raise ValueError(f"output_name must be a plain file stem (it may contain {{seed}} or {{run}}), got {name!r}")
        self.name_template = name
        self.directory = directory
        self.overwrite = bool(overwrite)
        self.prompt = prompt or ""
        self.student = student  # (states (L, C), tags (L,)) or None
        self.save_noise = bool(save_noise)
        self.student_context = load_student_context(student_context)  # {"path", "sha256", "video", "audio"} or None
        self.runs = 0
        self.run = None
        self.saved = []

    @property
    def active(self):
        return self.run is not None

    def target(self, seed, run):
        name = self.name_template
        if "{" in name:
            name = name.format(seed="noseed" if seed is None else seed, run=run)
        if not name or os.path.basename(name) != name:
            raise ValueError(f"output_name {self.name_template!r} gives an invalid file name {name!r}")
        return os.path.join(self.directory, name + ".safetensors"), name

    def begin_run(self, noise, latent_image, sigmas, seed, latent_shapes, denoise_mask=None, guider=None):
        if self.run is not None:
            raise RuntimeError("H3RecordTrajectory: a sampling run started inside another one")
        if latent_shapes is None or len(latent_shapes) != 2:
            raise ValueError("H3RecordTrajectory expects the MiniMax H3 audio+video latent (two packed streams), got "
                             f"latent_shapes {latent_shapes}")
        run_index = self.runs + 1
        path, name = self.target(seed, run_index)
        if os.path.exists(path) and not self.overwrite:
            raise FileExistsError(f"H3RecordTrajectory: {path} exists (set overwrite, or a new output_name)")
        os.makedirs(self.directory, exist_ok=True)
        shapes = [tuple(int(v) for v in s) for s in latent_shapes]
        notes = []
        noise_streams = None
        if self.save_noise and noise is not None:
            noise_streams = [_cpu(t.float()) for t in unpack(noise.float(), shapes)]
        cond, entries = None, 0
        positive = (getattr(guider, "original_conds", None) or {}).get("positive") or []
        if positive:
            entries = len(positive)
            cond = conditioning_tensors(positive[0])
            if entries > 1:
                notes.append(f"the positive conditioning has {entries} entries; the first is recorded")
        if cond is None:
            notes.append("no positive conditioning found on the guider: nothing recorded under cond.joint")
        if denoise_mask is not None:
            notes.append("the run has a denoise mask: not a plain text-to-video run (calls are marked masked)")
        has_image = latent_image is not None and bool(torch.count_nonzero(latent_image) > 0)
        if has_image:
            notes.append("the latent image is not empty (img2img or a masked context)")
        context = None
        if denoise_mask is not None or has_image:
            # the context as the sampler received it (before process_latent_in: the space of run.final.*)
            if latent_image is not None:
                video, audio = (_cpu(t.float()) for t in unpack(latent_image.float(), shapes))
            else:
                video, audio = torch.zeros(shapes[0]), torch.zeros(shapes[1])
            outer = None
            if denoise_mask is not None:
                outer = [_cpu(m.float())[:, :1] for m in unpack(denoise_mask.float(), shapes)]
            context = {"video": video, "audio": audio, "outer_mask": outer, "mask": {}}
        if self.student_context is not None:
            if context is None or context["outer_mask"] is None:
                raise ValueError("H3RecordTrajectory: student_context_path is set but the run has no masked context")
            check_student_context(self.student_context, shapes[0], shapes[1], context["outer_mask"][0])
        cfg = getattr(guider, "cfg", None)
        if cfg is not None and not math.isclose(float(cfg), 1.0):
            notes.append(f"cfg {cfg}: the recorded conditional outputs are not the guided velocity the sampler used")
        self.runs = run_index
        self.run = {"path": path, "name": name, "seed": seed, "sigmas": _cpu(sigmas.float().reshape(-1)),
                    "shapes": shapes, "noise": noise_streams, "masked": denoise_mask is not None, "cond": cond,
                    "cond_entries": entries, "guider": type(guider).__name__ if guider is not None else None,
                    "cfg": None if cfg is None else float(cfg), "calls": [], "tensors": {}, "notes": notes,
                    "shifts": set(), "audio_scales": set(), "context": context}

    def abort_run(self):
        self.run = None

    def step_of(self, sigma):
        """The schedule index whose sigma the sampler passed (the turbo sampler passes sigmas[i] exactly); -1 if none."""
        if sigma is None or self.run is None:
            return -1
        for i, s in enumerate(self.run["sigmas"][:-1].tolist()):
            if abs(s - sigma) <= 1e-6 * max(1.0, abs(sigma)):
                return i
        return -1

    def capture_inputs(self, x, timestep):
        """CPU copies of the DiT's inputs, taken before the forward runs."""
        return [_cpu(s) for s in _streams(x)], _cpu(timestep.float().reshape(-1))

    def record_call(self, inputs, out, transformer_options=None, payload=None, denoise_mask=None,
                    audio_denoise_mask=None, context=None, model=None):
        run = self.run
        if run is None:
            return
        xs, timestep = inputs
        outs = _streams(out)
        if len(xs) != 2 or len(outs) != 2:
            raise ValueError(f"H3RecordTrajectory expects [video, audio] in and out of the DiT, got {len(xs)} / "
                             f"{len(outs)} streams")
        to = transformer_options or {}
        payload = payload or {}
        shift_v = float(to.get("minimax_h3_sigma_shift_video", getattr(model, "sigma_shift_video", 12.0)))
        shift_a = float(to.get("minimax_h3_sigma_shift_audio", getattr(model, "sigma_shift_audio", 3.0)))
        # exactly the model's own derivation (fp32)
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
        sampler_sigma = to.get("sigmas")
        sampler_sigma = float(sampler_sigma.flatten()[0]) if torch.is_tensor(sampler_sigma) else None
        index = len(run["calls"])
        masked = denoise_mask is not None or audio_denoise_mask is not None or run["masked"]
        tensors = run["tensors"]
        tensors[call_key(index, "video.x")] = xs[0]
        tensors[call_key(index, "audio.x")] = xs[1]
        tensors[call_key(index, "video.out")] = _cpu(outs[0])
        tensors[call_key(index, "audio.out")] = _cpu(outs[1])
        tensors[call_key(index, "timestep")] = timestep
        for stream, mask in (("video", denoise_mask), ("audio", audio_denoise_mask)):
            if mask is None:
                continue
            if run["context"] is None:
                tensors[call_key(index, f"{stream}.mask")] = _cpu(mask.float())
                continue
            # a context run: the mask once (the first call that has it), a call's own only when it differs
            mask = _cpu(mask.float())[:, :1]
            first = run["context"]["mask"].setdefault(stream, mask)
            if first is not mask and not torch.equal(first, mask):
                tensors[call_key(index, f"{stream}.mask")] = mask
        audio_scale = float(payload.get("audio_scale", 1.0))
        run["shifts"].add((shift_v, shift_a))
        run["audio_scales"].add(audio_scale)
        run["calls"].append({
            "index": index,
            "step": self.step_of(sampler_sigma if sampler_sigma is not None else float(sigma_v)),
            "sigma": sampler_sigma,
            "timestep": float(timestep.flatten()[0]),
            "sigma_v": float(sigma_v),
            "sigma_a": float(sigma_a),
            "t_v": float(1.0 - sigma_v),
            "t_a": float(1.0 - sigma_a),
            "cond_or_uncond": [int(c) for c in to.get("cond_or_uncond", [0])],
            "masked": bool(masked),
            "dtype": {"video": str(xs[0].dtype).replace("torch.", ""), "audio": str(xs[1].dtype).replace("torch.", ""),
                      "out": str(outs[0].dtype).replace("torch.", "")},
            "audio_scale": audio_scale,
            "payload": {"refs": len(payload.get("refs") or ()), "keyframes": len(payload.get("keyframes") or ()),
                        "text_tokens": None if payload.get("text_token_tags") is None
                        else int(payload["text_token_tags"].numel())},
            "context_shape": None if context is None else list(context.shape),
        })

    def end_run(self, output):
        run, self.run = self.run, None
        if run is None:
            return None
        if not run["calls"]:
            raise RuntimeError("H3RecordTrajectory: the run made no H3 model call (is this a MiniMax H3 model, and "
                               "is the recorder the last patch before the guider?)")
        final_v, final_a = unpack(output.detach().float(), run["shapes"])
        tensors = run["tensors"]
        tensors["run.final.video"] = _cpu(final_v)
        tensors["run.final.audio"] = _cpu(final_a)
        tensors["run.sigmas"] = run["sigmas"]
        if run["noise"] is not None:
            tensors["run.noise.video"], tensors["run.noise.audio"] = run["noise"]
        text = {"joint": None, "student": None}
        for which, cond in (("joint", run["cond"]), ("student", self.student)):
            if cond is not None:
                tensors[f"cond.{which}.embeds"], tensors[f"cond.{which}.tags"] = cond
                text[which] = {"tokens": int(cond[0].shape[0]), "dim": int(cond[0].shape[1]),
                               "dtype": str(cond[0].dtype).replace("torch.", ""), "sha256": _sha256(cond[0])}
        if text["joint"] is not None:
            text["joint"]["entries"] = run["cond_entries"]
        context = None
        if run["context"] is not None:
            ctx = run["context"]
            outer = ctx["outer_mask"] or [None, None]
            masks = {}
            for i, stream in enumerate(("video", "audio")):
                mask = ctx["mask"].get(stream, outer[i])
                if mask is None:  # no mask anywhere (img2img): every element generated
                    mask = torch.ones((1, 1) + tuple(run["shapes"][i][2:]))
                masks[stream] = mask
            tensors["run.context.video"], tensors["run.context.audio"] = ctx["video"], ctx["audio"]
            tensors["run.context.mask.video"], tensors["run.context.mask.audio"] = masks["video"], masks["audio"]
            steps = pinned_steps(masks["video"])
            context = {"video_steps": steps, "audio_ticks": context_audio_ticks(steps), "student": None}
            student = self.student_context
            if student is not None:
                check_student_context(student, run["shapes"][0], run["shapes"][1], masks["video"])
                tensors["run.context.student.video"] = student["video"]
                tensors["run.context.student.audio"] = student["audio"]
                context["student"] = {"path": student["path"], "sha256": student["sha256"]}
        calls = run["calls"]
        steps = [c["step"] for c in calls if c["cond_or_uncond"] == [0]]
        schedule_steps = len(run["sigmas"]) - 1
        if sorted(s for s in steps if s >= 0) != list(range(schedule_steps)):
            run["notes"].append(f"conditional calls cover steps {steps}, the schedule has {schedule_steps}")
        if any(c["cond_or_uncond"] != [0] for c in calls):
            run["notes"].append("some calls are unconditional or batched (cfg > 1); the trainer uses only [0] calls")
        if len(run["shifts"]) != 1 or len(run["audio_scales"]) != 1:
            run["notes"].append(f"shifts {run['shifts']} / audio scales {run['audio_scales']} changed within the run")
        shift_v, shift_a = sorted(run["shifts"])[0]
        info = {
            "format": FORMAT, "version": FORMAT_VERSION,
            "recorder": {"node": "H3RecordTrajectory", "version": RECORDER_VERSION},
            "name": run["name"], "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "seed": run["seed"], "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest() if self.prompt else None,
            "schedule": {"sigmas": run["sigmas"].tolist(), "steps": schedule_steps, "shift_video": shift_v,
                         "shift_audio": shift_a, "audio_scale": sorted(run["audio_scales"])[0]},
            "shapes": {"video": list(run["shapes"][0]), "audio": list(run["shapes"][1])},
            "sampler": {"guider": run["guider"], "cfg": run["cfg"], "masked": run["masked"], "calls": len(calls)},
            "text": text,
            "calls": calls,
            "context": context,
            "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                        for k, v in sorted(tensors.items())},
            "conventions": CONVENTIONS,
            "notes": run["notes"],
        }
        write_trajectory(run["path"], tensors, info)
        self.saved.append(run["path"])
        return run["path"]


def make_outer_sample_wrapper(recorder):
    """OUTER_SAMPLE wrapper: one record per sampling run (see the module docstring for the arguments)."""

    def outer_sample(executor, *args, **kwargs):
        def arg(i, name):
            return args[i] if len(args) > i else kwargs.get(name)

        latent_shapes = kwargs["latent_shapes"] if "latent_shapes" in kwargs else (args[8] if len(args) > 8 else None)
        recorder.begin_run(noise=arg(0, "noise"), latent_image=arg(1, "latent_image"), sigmas=arg(3, "sigmas"),
                           seed=arg(7, "seed"), latent_shapes=latent_shapes, denoise_mask=arg(4, "denoise_mask"),
                           guider=getattr(executor, "class_obj", None))
        try:
            output = executor(*args, **kwargs)
        except BaseException:
            recorder.abort_run()
            raise
        recorder.end_run(output)
        return output

    return outer_sample


def make_diffusion_model_wrapper(recorder):
    """DIFFUSION_MODEL wrapper: records the DiT's inputs (copied before the forward) and outputs of every call made
    while a run is being recorded; calls outside a run pass straight through."""

    def diffusion_model(executor, *args, **kwargs):
        if not recorder.active:
            return executor(*args, **kwargs)
        x = args[0] if args else kwargs.get("x")
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        context = args[2] if len(args) > 2 else kwargs.get("context")
        transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        inputs = recorder.capture_inputs(x, timestep)
        out = executor(*args, **kwargs)
        recorder.record_call(inputs, out, transformer_options=transformer_options,
                             payload=kwargs.get("minimax_payload"), denoise_mask=kwargs.get("denoise_mask"),
                             audio_denoise_mask=kwargs.get("audio_denoise_mask"), context=context,
                             model=getattr(executor, "class_obj", None))
        return out

    return diffusion_model


def _rel(a, b, keep=None):
    if keep is not None:  # only these elements (a bool mask broadcast over the channels)
        keep = keep.expand_as(a)
        if not bool(keep.any()):
            return 0.0
        a, b = a[keep], b[keep]
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp(min=1e-12)).item()


def check_recording(info, tensors, tolerance=CHAIN_TOLERANCE):
    """Replay the sampler from the recorded calls and compare with the next recorded state: the conditional calls must
    form one plain Euler chain in the carried space, start at the recorded noise and end at the recorded final latent.
    A cache node, another sampler, cfg > 1 or a wrong reading of the convention shows up here. Relative RMS errors;
    bf16 rounding of the recorded states accounts for ~0.2-0.5%. A masked context run (run.context.mask.video) keeps
    its masked calls and compares only the elements its masks generate in full (mask >= 1 - 1e-3, per stream, over
    every channel); the pinned and feathered ones follow the context, not the chain."""
    sched = info["schedule"]
    sigmas = [float(s) for s in sched["sigmas"]]
    r = float(sched.get("audio_scale") or 1.0)
    carried = math.isclose(r, float(sched["shift_video"]) / float(sched["shift_audio"]))
    context = "run.context.mask.video" in tensors
    calls = sorted((c for c in info["calls"] if c["cond_or_uncond"] == [0] and c["step"] >= 0
                    and (context or not c["masked"])), key=lambda c: c["step"])
    get = lambda c, n: tensors[call_key(c["index"], n)].float()

    def keep(c, stream):
        """The fully generated elements of a call's stream (its own mask, else the run's); None: all of them."""
        if not context:
            return None
        own = call_key(c["index"], f"{stream}.mask")
        mask = tensors[own] if own in tensors else tensors[f"run.context.mask.{stream}"]
        return mask.float()[:, :1] >= MASK_ONE

    def both(a, b, stream):
        ka, kb = keep(a, stream), keep(b, stream)
        return None if ka is None else ka & kb

    def step(c, s_next):
        s = sigmas[c["step"]]
        xv, ov, xa, oa = get(c, "video.x"), get(c, "video.out"), get(c, "audio.x"), get(c, "audio.out")
        video = xv + (s_next - s) * ov
        y = carried_audio(xa, c["sigma_v"], c["sigma_a"]) if carried else xa
        vel = carried_audio_velocity(oa, xa, c["sigma_a"], r) if carried else oa
        return video, y + (s_next - s) * vel

    report = {"links": [], "carried_audio": carried, "context": context, "notes": []}
    if not carried:
        report["notes"].append("audio not carried (legacy sampler): audio chain checked on the video clock only")
    if context:
        report["notes"].append("masked context run: only the fully generated elements are replayed")
    for a, b in zip(calls, calls[1:]):
        if b["step"] != a["step"] + 1:
            continue
        video, y = step(a, sigmas[b["step"]])
        xa1 = get(b, "audio.x")
        audio = (b["sigma_a"] / b["sigma_v"]) * y if carried else y
        report["links"].append({"step": a["step"], "video": _rel(video, get(b, "video.x"), both(a, b, "video")),
                                "audio": _rel(audio, xa1, both(a, b, "audio"))})
    if calls and "run.final.video" in tensors:
        last = calls[-1]
        video, y = step(last, sigmas[last["step"] + 1])
        report["final"] = {"video": _rel(video, tensors["run.final.video"].float(), keep(last, "video")),
                           "audio": _rel(y / r if carried else y, tensors["run.final.audio"].float(),
                                         keep(last, "audio"))}
    if calls and "run.noise.video" in tensors and calls[0]["step"] == 0 and math.isclose(sigmas[0], 1.0):
        first = calls[0]
        report["start"] = {"video": _rel(get(first, "video.x"), tensors["run.noise.video"].float(),
                                         keep(first, "video")),
                           "audio":_rel(get(first, "audio.x"), tensors["run.noise.audio"].float(),
                                         keep(first, "audio"))}
    worst = [v for link in report["links"] for k, v in link.items() if k != "step"]
    worst += list(report.get("final", {}).values()) + list(report.get("start", {}).values())
    report["worst"] = max(worst) if worst else None
    report["ok"] = bool(worst) and max(worst) <= tolerance
    return report
