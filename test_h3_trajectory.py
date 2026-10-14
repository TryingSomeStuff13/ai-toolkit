"""CPU contracts for H3 v6 trajectory training.

A stand-in engine records turbo-style runs the way the ComfyUI H3RecordTrajectory node does (the H3 model's own view:
audio carry undone, output = noise - clean), and the real get_noise_prediction / get_loss_target / SDTrainer loss train
on them with a stand-in transformer that computes the same function. With the LoRA at zero every anchor sample must then
cost exactly nothing: the step-0 diagnostic of the v6 design, here on CPU.
"""
import json, math, os, sys, tempfile, types, unittest
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import torch
from safetensors.torch import save_file
from toolkit.dto import DTO
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import TrainConfig
# load only the H3 module: the package __init__ imports every model, and one needs a GPU at import time
for name, path in (("extensions_built_in", "extensions_built_in"), ("extensions_built_in.diffusion_models", "extensions_built_in/diffusion_models")):
    if name not in sys.modules:
        stub = types.ModuleType(name); stub.__path__ = [os.path.join(ROOT, path)]; sys.modules[name] = stub
from extensions_built_in.diffusion_models.minimax_h3 import minimax_h3 as h3m
from extensions_built_in.diffusion_models.minimax_h3.src import continuation as cont
from extensions_built_in.diffusion_models.minimax_h3.src import trajectory as traj
from extensions_built_in.diffusion_models.minimax_h3.src.packing import (pack_audio_latents, patchify_video_latents,
                                                                         remap_sigma, unpatchify_video_tokens)

VIDEO = (1, 24, 37, 4, 6)  # 124 frames at 64 x 96 px: 37 latent steps
AUDIO = (1, 32, 2, 207)  # the same 124 frames of 40 Hz stereo audio latents, ComfyUI layout
TEXT = (8, 16)  # stand-in Qwen3-VL states
SHIFT_V, SHIFT_A = 12.0, 3.0


def time_shift_sigma(sigma, from_shift, to_shift):
    """comfy/ldm/minimax/model.py"""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def turbo_sigmas(steps=8, shift=SHIFT_V):
    """ComfyUI's BasicScheduler "simple" over ModelSamplingDiscreteFlow(shift 12): the dashboard's 8-step schedule."""
    t = (torch.arange(1, 1001, 1) / 1000) * 1000 / 1000
    table = shift * t / (1 + (shift - 1) * t)
    ss = len(table) / steps
    return torch.FloatTensor([float(table[-(1 + int(x * ss))]) for x in range(steps)] + [0.0])


class StandIn:
    """The one function behind both sides, per position: velocity = tanh(x * A + t * B + (c + g) * C), with c from the
    text and its tags and g a global summary of both streams (mean input, highest label) standing in for H3's joint
    attention, so a pinned context moves the new tokens. Rounded to bf16 like the engine's DiT output. It depends on
    position, label, conditioning and nonlinearly on the input: any layout, label, conditioning, sign or scale slip
    between recorder and trainer costs loss."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.video = [torch.randn(VIDEO[1:], generator=g) * s for s in (0.8, 1.5, 0.7)]
        self.audio = [torch.randn(AUDIO[1:], generator=g) * s for s in (0.8, 1.5, 0.7)]

    @staticmethod
    def text_scalar(states, tags):
        return (states.float() * (1.0 + tags.float().view(-1, 1))).mean()

    def velocities(self, xv, tv, xa, ta, c):
        """xv / tv (1, 24, T, H, W), xa / ta (1, 32, 2, Ta): inputs and per-position labels, contiguous fp32."""
        g = xv.mean() + xa.mean() + tv.amax() + ta.amax()
        (av, bv, cv), (aa, ba, ca) = self.video, self.audio
        return (torch.tanh(xv * av + tv * bv + (c + g) * cv).to(torch.bfloat16),
                torch.tanh(xa * aa + ta * ba + (c + g) * ca).to(torch.bfloat16))

    def engine_forward(self, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
        """ComfyUI MiniMaxH3Model._forward's contract: x = [video, audio in its own flow space], labels t = 1 - sigma
        derived from the timestep, returns [-raw video, -raw audio] = noise - clean, in the input dtype."""
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - time_shift_sigma(sigma_v, SHIFT_V, SHIFT_A))
        xv, xa = x[0].float().contiguous(), x[1].float().contiguous()
        c = self.text_scalar(context[0], minimax_payload["text_token_tags"])
        v, a = self.velocities(xv, torch.full_like(xv, t_v), xa, torch.full_like(xa, t_a), c)
        return [v.to(x[0].dtype), a.to(x[1].dtype)]


class TrainerDiT(torch.nn.Module):
    """MiniMaxH3Transformer's contract with the same function: packed rows and per-row labels in, the data-ward
    velocity (clean - noise) out. ``lora`` is a zero-initialised trainable offset standing in for the LoRA."""

    def __init__(self, standin):
        super().__init__()
        self.standin = standin
        self.lora = torch.nn.Parameter(torch.zeros(()))
        self.seen = None
        self.out = None

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, row_timesteps, token_tags,
                position_ids, video_indices, audio_indices, text_indices, vsa_video_grid=None):
        self.seen = dict(hidden_states=hidden_states, audio_hidden_states=audio_hidden_states,
                         encoder_hidden_states=encoder_hidden_states, row_timesteps=row_timesteps, token_tags=token_tags,
                         video_indices=video_indices, audio_indices=audio_indices, text_indices=text_indices)
        t_lat, h, w = vsa_video_grid[0], vsa_video_grid[1] * 2, vsa_video_grid[2] * 2
        x = unpatchify_video_tokens(hidden_states.float(), t_lat, h, w)
        tv = unpatchify_video_tokens(row_timesteps[:, video_indices].unsqueeze(-1).expand(-1, -1, 96).float(), t_lat, h, w)
        xa = traj.audio_latent(audio_hidden_states.float()).contiguous()
        ta = traj.audio_latent(row_timesteps[:, audio_indices].unsqueeze(-1).expand(-1, -1, 32).float()).contiguous()
        vs, as_ = [], []
        for i in range(x.shape[0]):
            tags = token_tags[i, text_indices]
            live = tags >= 0
            c = StandIn.text_scalar(encoder_hidden_states[i][live], tags[live])
            v, a = self.standin.velocities(x[i:i + 1], tv[i:i + 1], xa[i:i + 1], ta[i:i + 1], c)
            vs.append(v.float())
            as_.append(a.float())
        v = torch.cat(vs) + self.lora * x
        a = torch.cat(as_) + self.lora * xa
        self.out = (v, a)
        return -patchify_video_latents(v), -traj.audio_rows(a)


def pack(tensors):
    return torch.cat([t.reshape(t.shape[0], 1, -1) for t in tensors], dim=-1), [t.shape for t in tensors]


def unpack(x, shapes):
    out, start = [], 0
    for s in shapes:
        n = int(torch.tensor(s[1:]).prod())
        out.append(x[:, :, start:start + n].reshape(s))
        start += n
    return out


def engine_run(standin, noise_v, noise_a, text, tags, sigmas, dtype=torch.bfloat16):
    """The turbo text-to-video run as ComfyUI executes it: the sampler walks a packed latent with the audio carried on
    the video schedule (y = sigma_v / sigma_a * x_a, clean end = audio_scale * audio), model_base casts it to the compute
    dtype and unpacks it, MiniMaxH3Model.forward undoes the carry, runs the DiT (recorded here, the DIFFUSION_MODEL
    level) and converts the audio velocity back, CONST turns it into denoised, and the turbo sampler takes an Euler step.
    Returns the calls and the final latent (audio back in its own space, as process_latent_out leaves it)."""
    scale = SHIFT_V / SHIFT_A
    noise, shapes = pack([noise_v, noise_a])
    x = sigmas[0] * noise  # CONST.noise_scaling over an empty latent
    calls = []
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i] * torch.ones(1)
        xc = x.to(dtype)
        timestep = (sigma * 1000).float()
        v, a = unpack(xc, shapes)
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        sigma_a = time_shift_sigma(sigma_v, SHIFT_V, SHIFT_A)
        a = a * (sigma_a / sigma_v).to(a.dtype)
        out = standin.engine_forward([v, a], timestep, text.unsqueeze(0), {}, minimax_payload={"text_token_tags": tags})
        calls.append({"x": [v.clone(), a.clone()], "out": [o.clone() for o in out], "timestep": timestep.clone(),
                      "sigma": float(sigmas[i]), "sigma_v": float(sigma_v), "sigma_a": float(sigma_a)})
        out_c = (1.0 - scale) * a + (1.0 + (scale - 1.0) * sigma_a).to(out[1].dtype) * out[1]
        model_output, _ = pack([out[0], out_c])
        denoised = x - model_output.float() * sigma.view(-1, 1, 1)
        d = (x - denoised) / sigmas[i]
        x = x + (float(sigmas[i + 1]) - float(sigmas[i])) * d
    final_v, final_a = unpack(x, shapes)
    return calls, final_v, final_a / scale


def write_recording(path, calls, final_v, final_a, noise_v, noise_a, sigmas, text, tags, student=None):
    """The H3RecordTrajectory file format (v1), as the recorder node writes it."""
    tensors, infos = {}, []
    for i, c in enumerate(calls):
        tensors.update({traj.call_key(i, "video.x"): c["x"][0], traj.call_key(i, "audio.x"): c["x"][1],
                        traj.call_key(i, "video.out"): c["out"][0], traj.call_key(i, "audio.out"): c["out"][1],
                        traj.call_key(i, "timestep"): c["timestep"]})
        infos.append({"index": i, "step": i, "sigma": c["sigma"], "timestep": float(c["timestep"][0]),
                      "sigma_v": c["sigma_v"], "sigma_a": c["sigma_a"], "t_v": 1.0 - c["sigma_v"],
                      "t_a": 1.0 - c["sigma_a"], "cond_or_uncond": [0], "masked": False})
    tensors.update({"run.final.video": final_v, "run.final.audio": final_a, "run.noise.video": noise_v,
                    "run.noise.audio": noise_a, "run.sigmas": sigmas, "cond.joint.embeds": text,
                    "cond.joint.tags": tags})
    if student is not None:
        tensors.update({"cond.student.embeds": student[0], "cond.student.tags": student[1]})
    info = {"format": traj.FORMAT, "version": traj.FORMAT_VERSION, "name": os.path.basename(path)[:-12],
            "schedule": {"sigmas": sigmas.tolist(), "steps": len(sigmas) - 1, "shift_video": SHIFT_V,
                         "shift_audio": SHIFT_A, "audio_scale": SHIFT_V / SHIFT_A},
            "shapes": {"video": list(final_v.shape), "audio": list(final_a.shape)}, "calls": infos}
    save_file({k: v.contiguous() for k, v in tensors.items()}, path,
              metadata={"format": traj.FORMAT, "version": str(traj.FORMAT_VERSION), "info": json.dumps(info)})


def record(directory, name="clip_0001", seed=0, standin=None, student=False):
    g = torch.Generator().manual_seed(seed + 100)
    noise_v, noise_a = torch.randn(VIDEO, generator=g), torch.randn(AUDIO, generator=g)
    text = torch.randn(TEXT, generator=g)
    tags = torch.ones(TEXT[0], dtype=torch.long)
    tags[2:4] = 0  # a vision block in the presentation
    sigmas = turbo_sigmas()
    calls, final_v, final_a = engine_run(standin or StandIn(), noise_v, noise_a, text, tags, sigmas)
    student_text = (torch.randn(TEXT[0] + 3, TEXT[1], generator=g), torch.ones(TEXT[0] + 3, dtype=torch.long)) if student else None
    path = os.path.join(directory, name + ".safetensors")
    write_recording(path, calls, final_v, final_a, noise_v, noise_a, sigmas, text, tags, student_text)
    return path, calls, final_v, final_a, (text, tags), student_text


def trainer_model(directory, standin=None, **kwargs):
    m = object.__new__(h3m.MinimaxH3Model)
    m.model_config = types.SimpleNamespace(model_kwargs={"continuation_trajectory_dir": directory,
                                                         "continuation_context_frames": 39, **kwargs})
    m.model = TrainerDiT(standin or StandIn())
    m.device_torch = torch.device("cpu"); m.torch_dtype = torch.float32; m.max_text_length = 512
    m.is_flow_matching = True; m.prediction_type = "epsilon"; m.x0_pred = False
    return m


class Batch:
    """The DataLoaderBatchDTO pieces trajectory training and SDTrainer.calculate_loss touch."""

    def __init__(self, *names):
        self.file_items = [types.SimpleNamespace(path=f"/data/clips/{n}.mp4", is_reg=False, prior_reg=False)
                           for n in names]
        self.dataset_config = types.SimpleNamespace(do_i2v=False, do_audio=True)
        self.num_frames = 124
        self.latents = torch.zeros((len(names),) + VIDEO[1:])
        self.mask_tensor = None
        self.loss_multiplier_list = [1.0] * len(names)

    def get_is_reg_list(self):
        return [False] * len(self.file_items)


def dataset_embeds(n=1, seed=7):
    g = torch.Generator().manual_seed(seed)
    return AdvancedPromptEmbeds(text_embeds=[torch.randn(5, TEXT[1], generator=g) for _ in range(n)],
                                text_token_tags=[torch.ones(5, dtype=torch.long) for _ in range(n)])


def predict(m, batch, n=1):
    noisy = torch.zeros((n,) + VIDEO[1:])
    return m.get_noise_prediction(noisy, torch.full((n,), 500.0), dataset_embeds(n), batch=batch)


def force_draws(m, kind, step, level=None):
    """Make the model's next draws a given kind and step (the store draws per batch item)."""
    s = m._trajectory_settings()
    store = m._trajectory_store_for(s)
    store.draw = lambda t, generator=None: (kind, t.calls_at(step)[0], level)
    return store


class TrajectoryContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.path, self.calls, self.final_v, self.final_a, (self.text, self.tags), _ = record(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_recording_convention_matches_the_trainer(self):
        """The target conversion: ComfyUI audio (B, 32, 2, T) -> the trainer's packed rows, no sign flip."""
        a = torch.randn(AUDIO)
        rows = traj.audio_rows(a)
        self.assertTrue(torch.equal(rows, pack_audio_latents(a.permute(0, 2, 1, 3))), "channel-major like encode_audio")
        self.assertTrue(torch.equal(traj.audio_latent(rows), a))
        # the recorded calls are one Euler chain of the engine (a sanity check of the stand-in engine itself)
        c0, c7 = self.calls[0], self.calls[7]
        self.assertTrue(torch.equal(c0["timestep"], torch.tensor([1000.0])))
        self.assertAlmostEqual(c7["sigma"], 12 * 0.125 / (1 + 11 * 0.125), places=5)
        x0_v = c7["x"][0].float() - c7["sigma_v"] * c7["out"][0].float()
        self.assertLess((x0_v - self.final_v).abs().max().item(), 0.05, "final video = the last x0 prediction")
        x0_a = c7["x"][1].float() - c7["sigma_a"] * c7["out"][1].float()
        self.assertLess((x0_a - self.final_a).abs().max().item(), 0.05, "final audio = x_a - sigma_a * out_a")
        sv, sa = traj.stream_sigmas(c7["timestep"])
        self.assertEqual(float(sv), c7["sigma_v"]); self.assertEqual(float(sa), c7["sigma_a"])
        self.assertTrue(torch.equal(remap_sigma(sv), time_shift_sigma(sv, SHIFT_V, SHIFT_A)))

    def test_anchor_loss_is_exactly_zero_with_a_zero_lora(self):
        """The step-0 diagnostic: the recorded input, the recorded labels and conditioning, the recorded output as the
        target -> with the network equal to the engine the loss is exactly 0, at every recorded step."""
        m = trainer_model(self.dir, continuation_trajectory_anchor_share=1.0, continuation_trajectory_noised_share=0.0)
        for step in range(8):
            force_draws(m, "anchor", step)
            batch = Batch("clip_0001")
            pred = predict(m, batch)
            target = m.get_loss_target(noise=torch.zeros((1,) + VIDEO[1:]), batch=batch)
            self.assertIsInstance(pred, DTO)
            self.assertTrue(torch.equal(pred.tensor, target), f"video, step {step}")
            self.assertTrue(torch.equal(pred.audio, pred.audio_target), f"audio, step {step}")
            self.assertTrue(torch.equal(target, self.calls[step]["out"][0].float()), "the target is the recorded output")
            seen = m.model.seen
            self.assertTrue(torch.equal(seen["hidden_states"], patchify_video_latents(self.calls[step]["x"][0].float())))
            self.assertTrue(torch.equal(seen["audio_hidden_states"], traj.audio_rows(self.calls[step]["x"][1].float())))
            t = seen["row_timesteps"][0]
            self.assertTrue(torch.equal(t[seen["video_indices"]], torch.full((222,), 1.0 - self.calls[step]["sigma_v"])))
            self.assertTrue(torch.equal(t[seen["audio_indices"]], torch.full((414,), 1.0 - self.calls[step]["sigma_a"])))
            self.assertTrue(torch.equal(seen["encoder_hidden_states"][0], self.text), "the recorded conditioning")
            self.assertTrue(torch.equal(seen["token_tags"][0, seen["text_indices"]], self.tags))
        errors = m._trajectory_store.stats.rows
        self.assertEqual(sorted(errors), [("anchor", k) for k in range(8)])
        self.assertTrue(all(r["video_mse"] == 0.0 and r["audio_mse"] == 0.0 for r in errors.values()))

    def test_floor_is_the_models_own_response_to_a_nudge(self):
        """continuation_trajectory_floor: a second, gradient-free call on the nudged video state. The prediction and the
        loss are unchanged; the table gains the model's own floor per step next to the error."""
        shares = dict(continuation_trajectory_anchor_share=1.0, continuation_trajectory_noised_share=0.0)
        plain = trainer_model(self.dir, **shares)
        m = trainer_model(self.dir, continuation_trajectory_floor=1e-3, **shares)
        for step in (1, 6):
            force_draws(plain, "anchor", step)
            force_draws(m, "anchor", step)
            a, b = predict(plain, Batch("clip_0001")), predict(m, Batch("clip_0001"))
            self.assertTrue(torch.equal(a.tensor, b.tensor) and torch.equal(a.audio, b.audio), "the floor pass changes nothing")
        for step in (1, 6):
            row = m._trajectory_store.stats.rows[("anchor", step)]
            self.assertEqual(row["video_rel"], 0.0)  # the stand-in is the engine
            self.assertGreater(row["video_floor"], 0.0)  # but it still moves when its input does
            self.assertLess(row["video_floor"], 0.05)
            self.assertGreaterEqual(row["audio_floor"], 0.0)
        self.assertIn("video floor", m._trajectory_store.stats.report())
        self.assertNotIn("video floor", plain._trajectory_store.stats.report())
        x = self.calls[1]["x"][0].float()
        nudged = traj.nudge(x, 1e-3)
        moved = float((nudged != x).float().mean())
        self.assertTrue(0.05 < moved < 0.5, f"about a fifth of the bf16 elements move by one step, got {moved:.1%}")
        self.assertTrue(torch.equal(nudged, traj.nudge(x, 1e-3)), "a fixed nudge")
        self.assertEqual(traj.settings({"continuation_trajectory_dir": self.dir}).floor, 0.0)
        self.assertEqual(traj.settings({"continuation_trajectory_dir": self.dir,
                                        "continuation_trajectory_floor": True}).floor, 1e-3)
        self.assertRaises(ValueError, traj.settings, {"continuation_trajectory_dir": self.dir,
                                                      "continuation_trajectory_floor": 0.5})

    def test_sdtrainer_loss_is_zero_for_anchors_and_positive_for_continuations(self):
        from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
        m = trainer_model(self.dir)
        trainer = types.SimpleNamespace(train_config=TrainConfig(noise_scheduler="flowmatch"), device_torch=torch.device("cpu"),
                                        sd=m, dfe=None, adapter=None, additional_logs={}, step_num=0)
        SDTrainer.check_model_train_config(trainer)  # the defaults are fine
        losses = {}
        for kind in ("anchor", "continuation"):
            force_draws(m, kind, 6)
            batch = Batch("clip_0001")
            noisy, noise, ts = SDTrainer.model_training_step_inputs(
                trainer, batch, torch.zeros((1,) + VIDEO[1:]), torch.randn((1,) + VIDEO[1:]), torch.tensor([321.0]))
            self.assertTrue(torch.equal(ts, self.calls[6]["timestep"]), "the trainer runs at the recorded timestep")
            pred = m.get_noise_prediction(noisy, ts, dataset_embeds(), batch=batch)
            losses[kind] = SDTrainer.calculate_loss(trainer, noise_pred=pred, noise=noise, noisy_latents=noisy,
                                                    timesteps=ts, batch=batch, mask_multiplier=torch.ones(1, 1, 1, 1))
        self.assertEqual(losses["anchor"].item(), 0.0)
        self.assertGreater(losses["continuation"].item(), 1e-4, "a pinned context changes the prediction")
        self.assertIn("traj/video_rel", trainer.additional_logs)
        bad = types.SimpleNamespace(train_config=TrainConfig(audio_teacher=True), sd=m)
        self.assertRaises(ValueError, SDTrainer.check_model_train_config, bad)
        off = trainer_model(self.dir); off.model_config.model_kwargs = {}
        plain = types.SimpleNamespace(sd=off, train_config=TrainConfig(audio_teacher=True))
        SDTrainer.check_model_train_config(plain)  # other modes keep every setting
        args = (Batch("clip_0001"), torch.ones(1), torch.ones(2), torch.ones(3))
        self.assertTrue(all(a is b for a, b in zip(SDTrainer.model_training_step_inputs(plain, *args), args[1:])),
                        "without trajectory training the trainer's own inputs are kept")

    def test_continuation_sample_pins_the_context_like_inference(self):
        m = trainer_model(self.dir)
        force_draws(m, "continuation", 5)
        batch = Batch("clip_0001")
        pred = predict(m, batch)
        sample = batch.trajectory_samples[0]
        call = self.calls[5]
        seen = m.model.seen
        rows = 12 * 2 * 3
        video_rows = seen["hidden_states"][0]
        # the context: the run's own final latent (not the recorded state), pinned at 0.999 like inference
        context = patchify_video_latents(self.final_v)[0, :rows]
        self.assertLess((video_rows[:rows] - 0.999 * context).abs().max().item(), 0.006)
        self.assertGreater((video_rows[:rows] - patchify_video_latents(call["x"][0].float())[0, :rows]).abs().max().item(), 0.1)
        self.assertTrue(torch.equal(video_rows[rows:], patchify_video_latents(call["x"][0].float())[0, rows:]),
                        "the new frames: the recorded joint state")
        t = seen["row_timesteps"][0]
        vidx, aidx = seen["video_indices"], seen["audio_indices"]
        self.assertTrue(torch.allclose(t[vidx[:rows]], torch.tensor(0.999)), "pinned rows labelled like inference")
        self.assertTrue(torch.equal(t[vidx[rows:]], torch.full((222 - rows,), 1.0 - call["sigma_v"])))
        # the audio under the context: inference's mask m, m * state + (1 - m) * clean, labelled 1 - m * sigma_a
        m_rows = cont.audio_row_mask(cont.audio_context_mask(207, 65))
        hard, released, free = m_rows == 0, (m_rows > 0) & (m_rows < 1), m_rows == 1
        audio_seen = seen["audio_hidden_states"][0]
        clean = traj.audio_rows(self.final_a)[0]
        state = traj.audio_rows(call["x"][1].float())[0]
        self.assertTrue(torch.equal(audio_seen[hard], clean[hard]), "protected audio: exactly the clip's own clean audio")
        self.assertTrue(torch.equal(audio_seen[free], state[free]))
        mm = m_rows[released].view(-1, 1)
        self.assertTrue(torch.allclose(audio_seen[released], mm * state[released] + (1 - mm) * clean[released]))
        self.assertTrue(torch.equal(t[aidx][hard], torch.ones(int(hard.sum()))), "protected audio labelled t = 1")
        self.assertTrue(torch.allclose(t[aidx][released], 1 - m_rows[released] * call["sigma_a"]))
        self.assertTrue(torch.allclose(t[aidx][free], torch.tensor(1 - call["sigma_a"])))
        # only the new tokens teach
        target = m.get_loss_target(noise=torch.zeros((1,) + VIDEO[1:]), batch=batch)
        self.assertTrue(torch.equal(pred.tensor[:, :, :12], target[:, :, :12]), "the context adds no loss")
        self.assertTrue(torch.equal(pred.tensor[:, :, 12:], m.model.out[0][:, :, 12:]), "after it: the model's prediction")
        self.assertTrue(torch.equal(target, call["out"][0].float()))
        out_a = pred.audio[0]
        self.assertTrue(torch.equal(out_a[hard], pred.audio_target[0][hard]), "protected audio adds no loss")
        self.assertTrue(torch.allclose(out_a[free], traj.audio_rows(m.model.out[1])[0][free], atol=1e-6))
        self.assertTrue(torch.equal(pred.audio_target[0], traj.audio_rows(call["out"][1].float())[0]))
        self.assertEqual((sample.kind, sample.step, sample.context_steps), ("continuation", 5, 12))

    def test_gradients_come_from_new_tokens_only(self):
        m = trainer_model(self.dir)
        force_draws(m, "continuation", 7)
        batch = Batch("clip_0001")
        pred = predict(m, batch)
        m.model.out[0].retain_grad(); m.model.out[1].retain_grad()
        target = m.get_loss_target(noise=torch.zeros((1,) + VIDEO[1:]), batch=batch)
        loss = torch.nn.functional.mse_loss(pred.tensor, target) + torch.nn.functional.mse_loss(pred.audio, pred.audio_target)
        loss.backward()
        self.assertNotEqual(m.model.lora.grad.item(), 0.0, "the LoRA learns")
        gv = m.model.out[0].grad
        self.assertEqual(gv[:, :, :12].abs().max().item(), 0.0, "no gradient through the pinned context")
        self.assertGreater(gv[:, :, 12:].abs().max().item(), 0.0)
        m_rows = cont.audio_row_mask(cont.audio_context_mask(207, 65))
        ga = traj.audio_rows(m.model.out[1].grad)[0]
        self.assertEqual(ga[m_rows == 0].abs().max().item(), 0.0, "no gradient through protected audio")
        self.assertGreater(ga[m_rows == 1].abs().max().item(), 0.0)

    def test_noised_context_fixed_and_mask_modes(self):
        f = traj.TrajectoryFile(self.path)
        s = traj.settings({"continuation_trajectory_dir": self.dir})
        g = torch.Generator().manual_seed(11)
        sample = traj.build_sample(f, s, "noised", 3, noise_level=0.04, generator=g)
        eps = torch.randn((1, 24, 12, 4, 6), generator=torch.Generator().manual_seed(11))
        self.assertTrue(torch.allclose(sample.video_in[:, :, :12], 0.96 * self.final_v[:, :, :12] + 0.04 * eps, atol=1e-6))
        self.assertAlmostEqual(sample.context_label, 0.96)
        self.assertTrue(torch.equal(sample.video_in[:, :, 12:], self.calls[3]["x"][0].float()[:, :, 12:]))
        self.assertTrue(torch.equal(sample.audio_in, traj.build_sample(f, s, "continuation", 3).audio_in),
                        "the audio context stays clean unless asked")
        sm = traj.settings({"continuation_trajectory_dir": self.dir, "continuation_trajectory_context_noise_mode": "mask",
                            "continuation_trajectory_context_noise": [0.1, 0.2]})
        masked = traj.build_sample(f, sm, "noised", 3, noise_level=0.1)
        sigma_v = self.calls[3]["sigma_v"]
        self.assertAlmostEqual(masked.context_sigma, 0.1 * sigma_v, places=6, msg="a mask value m sits at m * sigma_v")
        self.assertAlmostEqual(masked.context_label, 1 - 0.1 * sigma_v, places=6)
        sa = traj.settings({"continuation_trajectory_dir": self.dir, "continuation_trajectory_noise_audio": True})
        noisy_audio = traj.build_sample(f, sa, "noised", 3, noise_level=0.05)
        m_rows = cont.audio_row_mask(cont.audio_context_mask(207, 65))
        self.assertTrue(torch.allclose(noisy_audio.audio_labels[m_rows == 0], torch.tensor(0.95)))
        self.assertFalse(torch.equal(noisy_audio.audio_in[0, m_rows == 0], traj.audio_rows(self.final_a)[0, m_rows == 0]))
        clean = traj.build_sample(f, s, "continuation", 3)
        self.assertAlmostEqual(clean.context_sigma, 0.001, places=9)
        self.assertEqual(clean.context_label, 0.999)

    def test_draws_follow_shares_weights_and_sweep(self):
        s = traj.settings({"continuation_trajectory_dir": self.dir, "continuation_trajectory_anchor_share": 0.2,
                           "continuation_trajectory_noised_share": 0.3,
                           "continuation_trajectory_step_weights": [0, 0, 0, 1, 1, 1, 2, 2]})
        store = traj.TrajectoryStore(s)
        f = store.open(store.path_for("/x/clip_0001.png"))
        g = torch.Generator().manual_seed(4)
        draws = [store.draw(f, g) for _ in range(4000)]
        kinds = [d[0] for d in draws]
        self.assertTrue(0.17 < kinds.count("anchor") / 4000 < 0.23)
        self.assertTrue(0.27 < kinds.count("noised") / 4000 < 0.33)
        steps = [f.calls[d[1]]["step"] for d in draws]
        self.assertEqual(min(steps), 3, "a zero weight is never drawn")
        self.assertTrue(1.7 < steps.count(7) / steps.count(3) < 2.3, "weight 2 is drawn twice as often")
        levels = [d[2] for d in draws if d[0] == "noised"]
        self.assertTrue(all(0.02 <= v <= 0.05 for v in levels) and all(d[2] is None for d in draws if d[0] != "noised"))
        sweep = traj.TrajectoryStore(traj.settings({"continuation_trajectory_dir": self.dir, "continuation_trajectory_sweep": True}))
        seen = {(k, f.calls[c]["step"]) for k, c, _ in (sweep.draw(f) for _ in range(24))}
        self.assertEqual(seen, {(k, st) for k in traj.KINDS for st in range(8)}, "24 draws cover every kind and step")
        bad = traj.TrajectoryStore(traj.settings({"continuation_trajectory_dir": self.dir,
                                                  "continuation_trajectory_step_weights": [1, 1]}))
        self.assertRaises(ValueError, bad.open, self.path)

    def test_text_sources(self):
        _, _, _, _, _, student = record(self.dir, "clip_0002", seed=1, student=True)
        m = trainer_model(self.dir)
        for kind, name, expect in (("anchor", "clip_0002", None), ("continuation", "clip_0002", student[0]),
                                   ("continuation", "clip_0001", self.text)):
            force_draws(m, kind, 2)
            predict(m, Batch(name))
            got = m.model.seen["encoder_hidden_states"][0]
            if expect is None:
                self.assertEqual(got.shape[0], TEXT[0], "anchors: the conditioning the run was sampled with")
            else:
                self.assertTrue(torch.equal(got[: expect.shape[0]], expect))
        md = trainer_model(self.dir, continuation_trajectory_text="dataset")
        force_draws(md, "anchor", 2)
        predict(md, Batch("clip_0001"))
        self.assertTrue(torch.equal(md.model.seen["encoder_hidden_states"][0], dataset_embeds().text_embeds[0]),
                        "dataset text: the caption's embedding")

    def test_prepare_step_shares_one_draw_and_batches(self):
        record(self.dir, "clip_0002", seed=1)
        m = trainer_model(self.dir)
        batch = Batch("clip_0001", "clip_0002")
        noisy, noise, ts = m.prepare_training_step(batch, torch.zeros((2,) + VIDEO[1:]), torch.zeros(1), torch.zeros(2))
        samples = batch.trajectory_samples
        self.assertEqual(len(samples), 2)
        self.assertTrue(torch.equal(ts, torch.cat([s.timestep for s in samples])))
        self.assertTrue(torch.equal(noisy, torch.cat([s.video_in for s in samples])))
        pred = predict(m, batch, n=2)
        self.assertIs(batch.trajectory_samples, samples, "the prediction uses the same draw")
        self.assertEqual(tuple(pred.shape), (2,) + VIDEO[1:])
        self.assertEqual(tuple(pred.audio.shape), (2, 414, 32))
        target = m.get_loss_target(noise=noise, batch=batch)
        for i, s in enumerate(samples):
            self.assertTrue(torch.equal(target[i:i + 1], s.video_target))
        off = trainer_model(self.dir); off.model_config.model_kwargs = {}
        self.assertIsNone(off.prepare_training_step(Batch("clip_0001"), noisy, noise, ts))

    def test_settings_and_guards(self):
        s = traj.settings({"continuation_trajectory_dir": "/r"})
        self.assertEqual((s.context_frames, s.context_steps, s.audio_context_ticks, s.protect_audio), (39, 12, 65, True))
        self.assertEqual((s.anchor_share, s.noised_share, s.context_noise, s.text), (0.2, 0.15, (0.02, 0.05), "recorded"))
        self.assertAlmostEqual(s.shares["continuation"], 0.65)
        self.assertIsNone(traj.settings({}))
        for bad in ({"continuation_burn_levels": [0, 0.1]}, {"continuation_no_context_share": 0.1},
                    {"continuation_trajectory_anchor_share": 0.7, "continuation_trajectory_noised_share": 0.4},
                    {"continuation_trajectory_context_noise": [0.05, 0.02]},
                    {"continuation_trajectory_context_noise_mode": "mask"},  # 0.02 would snap to a clean pin
                    {"continuation_trajectory_step_weights": [0, 0]}, {"continuation_trajectory_text": "caption"},
                    {"continuation_context_frames": 40}, {"continuation_context_frames": 22}):
            self.assertRaises(ValueError, traj.settings, {"continuation_trajectory_dir": "/r", **bad})
        traj.settings({"continuation_trajectory_dir": "/r", "continuation_context_frames": 22,
                       "continuation_protect_audio": False})  # a video-only prefix may be any 17n+5 run
        m = trainer_model(self.dir)
        self.assertEqual(m._continuation_settings().frames, 0, "no burned contexts are cached in trajectory mode")
        traj.check_train_config(TrainConfig())
        for bad in ({"audio_teacher": True}, {"loss_type": "mae"}, {"diff_output_preservation": True},
                    {"do_guidance_loss": True}, {"timestep_type": "next_sample"}, {"single_item_batching": True}):
            self.assertRaises(ValueError, traj.check_train_config, TrainConfig(**bad))

    def test_missing_recordings_are_refused_before_training(self):
        m = trainer_model(self.dir)
        self.assertRaises(FileNotFoundError, predict, m, Batch("clip_9999"))
        with tempfile.TemporaryDirectory() as data:
            for n in ("clip_0001.mp4", "clip_0001.txt", "clip_0003.mp4"):
                open(os.path.join(data, n), "w").close()
            os.mkdir(os.path.join(data, "_latent_cache"))
            config = types.SimpleNamespace(folder_path=data)
            with self.assertRaises(FileNotFoundError) as err:
                m.check_dataset_config(config)
            self.assertIn("clip_0003", str(err.exception))
            os.remove(os.path.join(data, "clip_0003.mp4"))
            m.check_dataset_config(config)
        broken = os.path.join(self.dir, "broken.safetensors")
        save_file({"x": torch.zeros(1)}, broken, metadata={"format": "something_else"})
        self.assertRaises(ValueError, traj.TrajectoryFile, broken)


# ---------------------------------------------------------------- R+ (repaired-context self-distillation) recordings
S_CTX, A_CTX = 12, 65  # the 39-frame context: 12 latent steps, 65 audio ticks


def audio_mask_rows(ticks=AUDIO[3]):
    """The continuation's audio generate mask per packed row (what the engine's masked run carries: its native
    ceil-to-1/256 quantisation, not the trainer's snapped audio_context_mask)."""
    return cont.audio_row_mask(cont.engine_audio_context_mask(ticks, A_CTX))


def engine_run_masked(standin, noise_v, noise_a, context_v, context_a, text, tags, sigmas, pin_eps=0.001,
                      dtype=torch.bfloat16, shift_a=SHIFT_A):
    """A turbo masked AV continuation as the engine runs it (KSamplerX0Inpaint around the packed, carried latent): the
    masked-out elements enter the DiT at (1 - pin_eps) * context + pin_eps * noise and are handed back as the context;
    the DiT sees inference's per-row labels (pinned video rows max(t_v, 0.999), audio rows min(1 - m * sigma_a, 1)).
    Returns the calls and the final latent (audio in its own space). shift_a: the turbo's audio shift (merged_turbo: 7)."""
    scale = SHIFT_V / shift_a
    noise, shapes = pack([noise_v, noise_a])
    image, _ = pack([context_v, context_a * scale])  # process_latent_in: the audio carry
    mask_v = torch.ones(VIDEO); mask_v[:, :, :S_CTX] = 0.0
    m_rows = audio_mask_rows()
    mask_a = traj.audio_latent(m_rows.view(1, -1, 1).expand(1, -1, AUDIO[1]).contiguous())
    mask, _ = pack([mask_v, mask_a])
    pinned = (1.0 - pin_eps) * image + pin_eps * noise
    x = sigmas[0] * noise + (1.0 - sigmas[0]) * image
    calls = []
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i] * torch.ones(1)
        x_in = x * mask + pinned * (1.0 - mask)
        timestep = (sigma * 1000).float()
        v, a = unpack(x_in.to(dtype), shapes)
        sigma_v, sigma_a = traj.stream_sigmas(timestep, SHIFT_V, shift_a)
        t_v, t_a = 1.0 - sigma_v, 1.0 - sigma_a
        a = a * (sigma_a / sigma_v).to(a.dtype)
        tv = torch.full(VIDEO, float(t_v))
        tv[:, :, :S_CTX] = max(float(t_v), cont.CONTEXT_T)
        ta_rows = torch.minimum(1.0 - m_rows * sigma_a, torch.clamp(t_a, min=cont.AUDIO_CONTEXT_T)).view(1, -1)
        ta = traj.audio_latent(ta_rows.unsqueeze(-1).expand(-1, -1, 32).float()).contiguous()
        c = StandIn.text_scalar(text, tags)
        out_v, out_a = standin.velocities(v.float().contiguous(), tv, a.float().contiguous(), ta, c)
        out_v, out_a = out_v.to(dtype), out_a.to(dtype)
        calls.append({"x": [v.clone(), a.clone()], "out": [out_v.clone(), out_a.clone()], "timestep": timestep.clone(),
                      "sigma": float(sigmas[i]), "sigma_v": float(sigma_v), "sigma_a": float(sigma_a)})
        ov, oa = out_v * mask_v.to(dtype), out_a * mask_a.to(dtype)
        out_c = (1.0 - scale) * a + (1.0 + (scale - 1.0) * sigma_a).to(oa.dtype) * oa
        model_output, _ = pack([ov, out_c])
        denoised = x_in - model_output.float() * sigma.view(-1, 1, 1)
        denoised = denoised * mask + image * (1.0 - mask)
        x = x + (float(sigmas[i + 1]) - float(sigmas[i])) * (x - denoised) / sigmas[i]
    final_v, final_a = unpack(x, shapes)
    return calls, final_v, final_a / scale, (mask_v[:, :1].clone(), mask_a[:, :1].clone())


def record_rplus(directory, name="R_hyperflow_lora_r01_c3", seed=0, standin=None, student=True, depth=3,
                 video_steps=S_CTX, drop=(), audio_mask=None, student_text=False, shift_a=SHIFT_A):
    """An R+ teacher recording (the recorder's context layout): continued from a repaired context R, with the burned
    context C as run.context.student (student=False: none, the trainer falls back on R)."""
    g = torch.Generator().manual_seed(seed + 300)
    noise_v, noise_a = torch.randn(VIDEO, generator=g), torch.randn(AUDIO, generator=g)
    text = torch.randn(TEXT, generator=g)
    tags = torch.ones(TEXT[0], dtype=torch.long)
    context_v, context_a = torch.zeros(VIDEO), torch.zeros(AUDIO)
    context_v[:, :, :S_CTX] = torch.randn((1, 24, S_CTX) + VIDEO[3:], generator=g)
    context_a[..., :A_CTX] = torch.randn(AUDIO[:3] + (A_CTX,), generator=g)
    burned_v = context_v[:, :, :S_CTX] + 0.3 * torch.randn((1, 24, S_CTX) + VIDEO[3:], generator=g)
    sigmas = turbo_sigmas()
    calls, final_v, final_a, (mask_v, mask_a) = engine_run_masked(standin or StandIn(), noise_v, noise_a, context_v,
                                                                  context_a, text, tags, sigmas, shift_a=shift_a)
    path = os.path.join(directory, name + ".safetensors")
    tensors, infos = {}, []
    for i, c in enumerate(calls):
        tensors.update({traj.call_key(i, "video.x"): c["x"][0], traj.call_key(i, "audio.x"): c["x"][1],
                        traj.call_key(i, "video.out"): c["out"][0], traj.call_key(i, "audio.out"): c["out"][1],
                        traj.call_key(i, "timestep"): c["timestep"]})
        infos.append({"index": i, "step": i, "sigma": c["sigma"], "timestep": float(c["timestep"][0]),
                      "sigma_v": c["sigma_v"], "sigma_a": c["sigma_a"], "t_v": 1.0 - c["sigma_v"],
                      "t_a": 1.0 - c["sigma_a"], "cond_or_uncond": [0], "masked": True})
    tensors.update({"run.final.video": final_v, "run.final.audio": final_a, "run.noise.video": noise_v,
                    "run.noise.audio": noise_a, "run.sigmas": sigmas, "cond.joint.embeds": text, "cond.joint.tags": tags,
                    "run.context.video": context_v, "run.context.audio": context_a,
                    "run.context.mask.video": mask_v,
                    "run.context.mask.audio": mask_a if audio_mask is None else audio_mask})
    if student:
        tensors.update({"run.context.student.video": burned_v, "run.context.student.audio": context_a[..., :A_CTX]})
    if student_text:
        tensors.update({"cond.student.embeds": torch.randn(TEXT[0] + 2, TEXT[1], generator=g),
                        "cond.student.tags": torch.ones(TEXT[0] + 2, dtype=torch.long)})
    for k in drop:
        tensors.pop(k)
    info = {"format": traj.FORMAT, "version": traj.FORMAT_VERSION, "name": name,
            "schedule": {"sigmas": sigmas.tolist(), "steps": len(sigmas) - 1, "shift_video": SHIFT_V,
                         "shift_audio": shift_a, "audio_scale": SHIFT_V / shift_a},
            "shapes": {"video": list(final_v.shape), "audio": list(final_a.shape)}, "calls": infos,
            "context": {"video_steps": video_steps, "audio_ticks": A_CTX,
                        "student": {"path": "/x/C.safetensors", "sha256": "0" * 64} if student else None},
            "rplus": {"story": "r01", "depth": depth, "variant": "R050", "gamma": 0.5}}
    save_file({k: v.contiguous() for k, v in tensors.items()}, path,
              metadata={"format": traj.FORMAT, "version": str(traj.FORMAT_VERSION), "info": json.dumps(info)})
    return {"path": path, "calls": calls, "context": context_v, "student": burned_v, "text": text, "final": final_v,
            "noise": noise_v}


class RPlusContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.legacy_path = record(self.dir)[0]  # clip_0001, a design-A text-to-video recording

    def tearDown(self):
        self.tmp.cleanup()

    def test_rplus_file_trains_on_every_masked_call(self):
        """Without a student context the trainer continues from the sampled context itself: with the network equal to
        the engine the loss is exactly 0 at every step (the recorded masked run is reproduced, context and labels)."""
        r = record_rplus(self.dir, "R_self", student=False)
        f = traj.TrajectoryFile(r["path"])
        self.assertTrue(f.rplus)
        self.assertEqual((f.steps, f.context_steps, f.depth), (list(range(8)), 12, 3))
        self.assertEqual(f.student, {"video": "run.context.video", "audio": "run.context.audio"})
        m = trainer_model(self.dir)
        m_rows = audio_mask_rows()
        for step in range(8):
            force_draws(m, "continuation", step)
            batch = Batch("R_self")
            pred = predict(m, batch)
            target = m.get_loss_target(noise=torch.zeros((1,) + VIDEO[1:]), batch=batch)
            self.assertTrue(torch.equal(pred.tensor, target), f"video, step {step}")
            self.assertTrue(torch.equal(pred.audio, pred.audio_target), f"audio, step {step}")
            call = r["calls"][step]
            seen = m.model.seen
            self.assertTrue(torch.equal(seen["hidden_states"], patchify_video_latents(call["x"][0].float())),
                            "the recorded state, context included")
            self.assertTrue(torch.equal(seen["audio_hidden_states"], traj.audio_rows(call["x"][1].float())),
                            "the audio as recorded")
            t = seen["row_timesteps"][0]
            rows = 12 * 2 * 3
            self.assertTrue(torch.equal(t[seen["video_indices"][:rows]],
                                        torch.full((rows,), max(1.0 - call["sigma_v"], 0.999))))
            sample = batch.trajectory_samples[0]
            self.assertTrue(torch.equal(sample.audio_weight, m_rows))
            self.assertEqual((sample.kind, sample.context_steps, sample.depth), ("continuation", 12, 3))
            self.assertTrue(torch.equal(seen["encoder_hidden_states"][0], r["text"]), "the joint conditioning")

    def test_rplus_at_merged_turbo_shifts_uses_the_recorded_audio_clock(self):
        """An R+ teacher run of a turbo with its own audio shift (merged_turbo: shifts 12 / 7, a merged-turbo pilot's recordings)
        trains on the recorded audio clock, not the released 12 / 3: with the network equal to the engine the loss is
        exactly 0 at every step, and each sample's sigma_a / audio labels are the 12 -> 7 remap."""
        r = record_rplus(self.dir, "R_merged_turbo_r01_c3", student=False, shift_a=7.0)
        f = traj.TrajectoryFile(r["path"])
        self.assertEqual((f.shift_video, f.shift_audio), (12.0, 7.0))
        s = traj.settings({"continuation_trajectory_dir": self.dir})
        m = trainer_model(self.dir)
        for step in range(8):
            call = r["calls"][step]
            sample = traj.build_sample(f, s, "continuation", step)
            want = time_shift_sigma(torch.tensor(call["sigma_v"]), SHIFT_V, 7.0)
            self.assertAlmostEqual(float(sample.sigma_a), float(want), places=6)
            self.assertAlmostEqual(float(sample.sigma_a), call["sigma_a"], places=6)
            if 0.0 < call["sigma_v"] < 1.0:
                self.assertNotAlmostEqual(float(sample.sigma_a),
                                          float(time_shift_sigma(torch.tensor(call["sigma_v"]), SHIFT_V, SHIFT_A)), places=3)
            force_draws(m, "continuation", step)
            batch = Batch("R_merged_turbo_r01_c3")
            pred = predict(m, batch)
            target = m.get_loss_target(noise=torch.zeros((1,) + VIDEO[1:]), batch=batch)
            self.assertTrue(torch.equal(pred.tensor, target), f"video, step {step}")
            self.assertTrue(torch.equal(pred.audio, pred.audio_target), f"audio, step {step}")
            self.assertTrue(torch.equal(m.model.seen["audio_hidden_states"], traj.audio_rows(call["x"][1].float())),
                            "the audio as recorded")

    def test_student_context_is_pinned_by_delta(self):
        r = record_rplus(self.dir, "R_student", student_text=True)
        f = traj.TrajectoryFile(r["path"])
        self.assertEqual(f.student["video"], "run.context.student.video")
        s = traj.settings({"continuation_trajectory_dir": self.dir})
        call = r["calls"][4]
        sample = traj.build_sample(f, s, "continuation", 4)
        x = call["x"][0].float()
        self.assertTrue(torch.allclose(sample.video_in[:, :, :12], x[:, :, :12] + 0.999 * (r["student"] - r["context"][:, :, :12]),
                                       atol=1e-6))
        eps = r["noise"][:, :, :12]
        self.assertLess((sample.video_in[:, :, :12] - (0.999 * r["student"] + 0.001 * eps)).abs().max().item(), 0.03,
                        "the engine's pin, moved to the burned context (bf16 rounding of the recorded state)")
        self.assertTrue(torch.equal(sample.video_in[:, :, 12:], x[:, :, 12:]), "the new frames: the recorded state")
        self.assertTrue(torch.equal(sample.video_target, call["out"][0].float()), "the teacher's output")
        self.assertTrue(torch.equal(sample.text_embeds, r["text"]), "joint, not the recorded student conditioning")
        m = trainer_model(self.dir)
        force_draws(m, "continuation", 4)
        batch = Batch("R_student")
        pred = predict(m, batch)
        m.model.out[0].retain_grad()
        target = m.get_loss_target(noise=torch.zeros((1,) + VIDEO[1:]), batch=batch)
        self.assertGreater((pred.tensor - target)[:, :, 12:].abs().max().item(), 1e-3, "a burned context moves the output")
        loss = torch.nn.functional.mse_loss(pred.tensor, target) + torch.nn.functional.mse_loss(pred.audio, pred.audio_target)
        loss.backward()
        self.assertEqual(m.model.out[0].grad[:, :, :12].abs().max().item(), 0.0, "no gradient through the context")
        self.assertNotEqual(m.model.lora.grad.item(), 0.0)

    def test_draws_rplus_always_continuation_and_the_mix(self):
        """R+ files give continuation samples; the other files keep their shares. 72 burned items x 2 repeats and 32
        clean ones x 3 with anchor and noised shares of 0.1875: burned 60%, clean continuation 25%, 7.5% each."""
        rp = record_rplus(self.dir, "R_mix")["path"]
        s = traj.settings({"continuation_trajectory_dir": self.dir, "continuation_trajectory_anchor_share": 0.1875,
                           "continuation_trajectory_noised_share": 0.1875,
                           "continuation_trajectory_step_weights": [1, 1, 1, 1, 1, 2, 2, 2]})
        store = traj.TrajectoryStore(s)
        burned, clean = store.open(rp), store.open(self.legacy_path)
        g = torch.Generator().manual_seed(3)
        draws = [store.draw(burned if u < 144 else clean, g)[0] for u in torch.randint(240, (6000,), generator=g).tolist()]
        self.assertEqual({store.draw(burned, g)[0] for _ in range(300)}, {"continuation"})
        n = len(draws)
        shares = {k: draws.count(k) / n for k in traj.KINDS}
        self.assertAlmostEqual(shares["anchor"], 0.075, delta=0.015)
        self.assertAlmostEqual(shares["noised"], 0.075, delta=0.015)
        self.assertAlmostEqual(shares["continuation"], 0.85, delta=0.02)
        sweep = traj.TrajectoryStore(traj.settings({"continuation_trajectory_dir": self.dir,
                                                    "continuation_trajectory_sweep": True}))
        f = sweep.open(rp)
        self.assertEqual({(k, f.calls[c]["step"]) for k, c, _ in (sweep.draw(f) for _ in range(8))},
                         {("continuation", st) for st in range(8)})
        self.assertRaises(ValueError, traj.build_sample, burned, s, "anchor", 0)

    def test_rplus_guards(self):
        s = traj.settings({"continuation_trajectory_dir": self.dir})
        for k in ("run.context.audio", "run.context.mask.video", "run.context.mask.audio"):
            path = record_rplus(self.dir, f"R_no_{k.split('.')[-1]}_{k.count('mask')}", drop=(k,))["path"]
            self.assertRaises(ValueError, traj.TrajectoryFile, path)
        steps11 = record_rplus(self.dir, "R_steps11", video_steps=11)["path"]
        self.assertRaises(ValueError, traj.TrajectoryStore(s).open, steps11)
        self.assertRaises(ValueError, traj.build_sample, traj.TrajectoryFile(steps11), s, "continuation", 0)
        bad_mask = torch.ones(1, 1, 2, AUDIO[3])
        bad_mask[..., :60] = 0.0
        wrong = record_rplus(self.dir, "R_mask", audio_mask=bad_mask)["path"]
        self.assertRaises(ValueError, traj.build_sample, traj.TrajectoryFile(wrong), s, "continuation", 2)
        # the engine's mask (ceil to 1/256, no snap) and the compat layer's (round, snapped) are both accepted
        engine, compat = cont.engine_audio_context_mask(AUDIO[3], A_CTX), cont.audio_context_mask(AUDIO[3], A_CTX)
        self.assertEqual([round(v * 256) for v in engine[57:65].tolist()], [10, 38, 80, 128, 177, 219, 247, 256])
        self.assertAlmostEqual((engine - compat).abs().max().item(), 10 / 256, places=6)
        snapped = record_rplus(self.dir, "R_compat_mask",
                               audio_mask=compat.view(1, 1, 1, -1).expand(1, 1, 2, -1).contiguous())["path"]
        traj.build_sample(traj.TrajectoryFile(snapped), s, "continuation", 2)
        # a legacy masked recording (no run.context.*) is still refused: nothing to train on
        legacy = record_rplus(self.dir, "R_legacy", drop=("run.context.video",))["path"]
        with self.assertRaises(ValueError) as err:
            traj.TrajectoryFile(legacy)
        self.assertIn("no conditional, unmasked calls", str(err.exception))
        f = traj.TrajectoryFile(self.legacy_path)
        self.assertEqual((f.rplus, f.depth, f.context_steps), (False, None, None))

    def test_floor_ratio_cap_and_depth_weights(self):
        floors = [0.02258, 0.04703, 0.07297, 0.11047, 0.06681, 0.04596, 0.03245, 0.02485]
        plain = traj.inverse_variance_weights(floors)
        capped = traj.inverse_variance_weights(floors, 4.0)
        self.assertGreater(max(plain) / min(plain), 20)
        self.assertAlmostEqual(max(capped) / min(capped), 4.0, places=9)
        self.assertAlmostEqual(sum(capped) / 8, 1.0, places=9)
        raw = [min(1 / f ** 2, 4 * min(1 / g ** 2 for g in floors)) for f in floors]
        self.assertTrue(all(abs(c - v / (sum(raw) / 8)) < 1e-9 for c, v in zip(capped, raw)))
        self.assertEqual(traj.inverse_variance_weights(floors, 1000.0), plain, "a loose cap changes nothing")
        kw = {"continuation_trajectory_dir": self.dir, "continuation_trajectory_floor_ratio_cap": 4.0,
              "continuation_trajectory_step_floors": {"default": {"video": floors, "audio": [0.02] * 8}},
              "continuation_trajectory_depth_weights": {"2": 2, "3": 2, "4": 2, "5": 2}}
        s = traj.settings(kw)
        self.assertEqual(s.depth_weights, ((2, 2.0), (3, 2.0), (4, 2.0), (5, 2.0)))
        self.assertEqual(s.floor_weight(None, 3)[0], capped[3])
        self.assertEqual(s.sample_weight(None, 3, 3), (capped[3] * 2, 2.0))
        self.assertEqual(s.sample_weight(None, 3, 7), (capped[3], 1.0))
        self.assertEqual(s.sample_weight(None, 3, None), (capped[3], 1.0), "legacy files: depth weight 1")
        self.assertIsNone(traj.settings({"continuation_trajectory_dir": self.dir}).floor_ratio_cap)
        for bad in ({"continuation_trajectory_floor_ratio_cap": 0.5}, {"continuation_trajectory_depth_weights": [2]},
                    {"continuation_trajectory_depth_weights": {"x": 2}},
                    {"continuation_trajectory_depth_weights": {"0": 2}},
                    {"continuation_trajectory_depth_weights": {"2": -1}}):
            self.assertRaises(ValueError, traj.settings, {"continuation_trajectory_dir": self.dir, **bad})
        # the residual scaling runs for a depth weight alone (no floors), and not at all for weight-1 samples
        record_rplus(self.dir, "R_d3", depth=3)
        record_rplus(self.dir, "R_d7", depth=7)
        results = {}
        for key, extra in (("plain", {}), ("depth", {"continuation_trajectory_depth_weights": {"3": 2}})):
            m = trainer_model(self.dir, **extra)
            m.model.lora.data.fill_(0.01)
            for name in ("R_d3", "R_d7", "clip_0001"):
                force_draws(m, "continuation", 5)
                batch = Batch(name)
                torch.manual_seed(5)  # the legacy sample's pin noise
                pred = predict(m, batch)
                target = m.get_loss_target(noise=torch.zeros((1,) + VIDEO[1:]), batch=batch)
                results[key, name] = (pred.tensor - target, pred.audio - pred.audio_target)
        self.assertGreater(results["plain", "R_d3"][0].abs().max().item(), 1e-3)
        for i in (0, 1):
            self.assertTrue(torch.allclose(results["depth", "R_d3"][i], results["plain", "R_d3"][i] * math.sqrt(2.0),
                                           rtol=1e-4, atol=1e-6))
            self.assertTrue(torch.equal(results["depth", "R_d7"][i], results["plain", "R_d7"][i]))
            self.assertTrue(torch.equal(results["depth", "clip_0001"][i], results["plain", "clip_0001"][i]),
                            "a legacy recording trains exactly as before")


if __name__ == "__main__":
    unittest.main(verbosity=2)
