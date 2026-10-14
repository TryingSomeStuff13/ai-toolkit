"""CPU contracts for H3 continuation training: the real get_noise_prediction / encode_images with stand-in models."""
import os, sys, types, unittest
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from toolkit.dto import DTO
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
# load only the H3 module: the package __init__ imports every model, and one needs a GPU at import time
for name, path in (("extensions_built_in", "extensions_built_in"), ("extensions_built_in.diffusion_models", "extensions_built_in/diffusion_models")):
    stub = types.ModuleType(name); stub.__path__ = [os.path.join(ROOT, path)]; sys.modules[name] = stub
from extensions_built_in.diffusion_models.minimax_h3 import minimax_h3 as h3m
from extensions_built_in.diffusion_models.minimax_h3.src import continuation as cont
from extensions_built_in.diffusion_models.minimax_h3.src.packing import patchify_video_latents


class FakeTransformer:
    device = torch.device("cpu")
    def to(self, *a, **k): return self
    def __call__(self, **kw):
        self.seen = kw
        return torch.zeros_like(kw["hidden_states"]), torch.zeros_like(kw["audio_hidden_states"])


class FakeVideoVAE:
    dtype = torch.float32
    def encode(self, frames, sample=True, **kw):  # (B, C, T, H, W) -> (B, 24, steps, H/16, W/16), value = mean frame
        b, c, t, h, w = frames.shape
        steps = 1 if t == 1 else cont.latent_steps(t)
        pooled = torch.nn.functional.adaptive_avg_pool3d(frames.float().mean(1, keepdim=True), (steps, h // 16, w // 16))
        return pooled.expand(b, 24, steps, h // 16, w // 16).clone()


def model(context_frames=39, **kwargs):
    m = object.__new__(h3m.MinimaxH3Model)
    m.model_config = types.SimpleNamespace(model_kwargs={"continuation_context_frames": context_frames, **kwargs} if context_frames else {})
    m.model = FakeTransformer()
    m.device_torch = torch.device("cpu"); m.torch_dtype = torch.float32; m.max_text_length = 512
    m.vae_device_torch = torch.device("cpu"); m.vae_torch_dtype = torch.float32
    m.vae = types.SimpleNamespace(device=torch.device("cpu"), to=lambda *a, **k: None)
    m._warned_frame_trim = True
    m.__dict__["_vv"] = FakeVideoVAE()
    return m


class Contracts(unittest.TestCase):
    def setUp(self):
        self.orig = h3m.MinimaxH3Model.video_vae
        h3m.MinimaxH3Model.video_vae = property(lambda self: self.__dict__["_vv"])
    def tearDown(self):
        h3m.MinimaxH3Model.video_vae = self.orig

    def test_burn_moves_the_frame_the_way_chains_drift(self):
        torch.manual_seed(0)
        frames = (torch.rand(39, 3, 64, 64) * 0.6 - 0.3)
        grey = lambda f: (0.299 * f[:, 0] + 0.587 * f[:, 1] + 0.114 * f[:, 2])
        texture = lambda f: (grey(f) - cont._blur(grey(f).unsqueeze(1), 1.5).squeeze(1)).abs().mean().item()
        burned = [cont.burn_frames(frames, level) for level in (0.33, 1.0)]
        self.assertEqual(burned[0].shape, frames.shape); self.assertLessEqual(burned[1].abs().max().item(), 1.0)
        self.assertGreater(texture(burned[0]), texture(frames) * 1.1, "etched texture")
        self.assertGreater(texture(burned[1]), texture(burned[0]), "stronger at level 1")
        self.assertLess(grey(burned[1]).mean().item(), grey(frames).mean().item(), "darker")
        self.assertEqual(cont.latent_steps(39), 12); self.assertRaises(ValueError, cont.latent_steps, 40)
        self.assertTrue(torch.equal(cont.burn_frames(frames, 0.0), frames), "level 0 is the clean context, unchanged")
        small = cont.burn_frames(frames, 0.05)
        self.assertLess(texture(small), texture(frames) * 1.15, "the mildest v2 level is a small change")

    def test_settings_and_clean_share(self):
        self.assertEqual(cont.settings({}).frames, 0)
        self.assertEqual(cont.settings({"continuation_context_frames": 39}).levels, [1 / 3, 2 / 3, 1.0], "v1 default kept")
        s = cont.settings({"continuation_context_frames": 39, "continuation_burn_levels": [0, 0.05, 0.1, 0.2],
                           "continuation_clean_share": 0.5})
        frames, levels, share = s.frames, s.levels, s.clean_share
        self.assertEqual((frames, levels, share), (39, [0.0, 0.05, 0.1, 0.2], 0.5))
        self.assertEqual((s.no_context_share, s.protect_audio), (0.0, False), "v3 options are off unless asked for")
        self.assertRaises(ValueError, cont.settings, {"continuation_context_frames": 39, "continuation_burn_levels": [0.1],
                                                      "continuation_clean_share": 0.5})  # no clean level to draw
        self.assertRaises(ValueError, cont.settings, {"continuation_context_frames": 39, "continuation_burn_levels": [1.5]})
        extras = {f"ctx_burn{i}": torch.full((64, 1), float(i)) for i in range(4)}
        g = torch.Generator().manual_seed(3)
        picked = cont.pick_context(extras, 64, levels, 0.5, generator=g).view(-1)
        clean = (picked == 0).float().mean().item()
        self.assertTrue(0.3 < clean < 0.7, f"about half clean, got {clean}")
        self.assertTrue(set(picked.tolist()) == {0.0, 1.0, 2.0, 3.0}, "every burned level is drawn too")
        self.assertTrue(torch.equal(cont.pick_context(extras, 64, levels, 1.0, generator=g), extras["ctx_burn0"]), "share 1: always clean")
        never = cont.pick_context(extras, 64, levels, 0.0, generator=g).view(-1)
        self.assertIn(0.0, never.tolist(), "share 0 (v1 form): every cached level is drawn uniformly")

    def test_clean_context_is_encoded_exactly_like_the_burned_ones(self):
        m = model(continuation_burn_levels=[0, 0.1], continuation_clean_share=0.5)
        clip = torch.rand(124, 3, 64, 96) * 2 - 1
        out = m.encode_images([clip])
        self.assertEqual(sorted(out.extras), ["ctx_burn0", "ctx_burn1"])
        alone = FakeVideoVAE().encode(clip[:39].permute(1, 0, 2, 3).unsqueeze(0))
        self.assertTrue(torch.allclose(out.extras["ctx_burn0"], alone), "level 0: the clean opening, encoded on its own")
        self.assertFalse(torch.equal(out.extras["ctx_burn0"], out.extras["ctx_burn1"]))

    def test_encode_caches_burned_contexts_from_the_same_frames(self):
        m = model()
        clip = torch.rand(124, 3, 64, 96) * 2 - 1
        out = m.encode_images([clip, clip.flip(0)])
        self.assertIsInstance(out, DTO)
        self.assertEqual(tuple(out.shape), (2, 24, 37, 4, 6))
        self.assertEqual(sorted(out.extras), ["ctx_burn0", "ctx_burn1", "ctx_burn2"])
        for k in out.extras: self.assertEqual(tuple(out.extras[k].shape), (2, 24, 12, 4, 6))
        self.assertFalse(torch.equal(out.extras["ctx_burn0"], out.extras["ctx_burn2"]), "strengths differ")
        plain = model(context_frames=0).encode_images([clip])
        self.assertNotIsInstance(plain, DTO, "off by default: ordinary H3 training is unchanged")
        single = m.encode_images([clip[0]])  # an image / first frame: no context
        self.assertNotIsInstance(single, DTO)

    def batch(self, m, clean, extras):
        return types.SimpleNamespace(dataset_config=types.SimpleNamespace(do_i2v=False, do_audio=False),
                                     num_frames=124, latents=DTO(clean, **extras), audio_latents=None)

    def predict(self, m, noisy, sigma, batch):
        tags = torch.ones(8, dtype=torch.long)
        embeds = AdvancedPromptEmbeds(text_embeds=[torch.randn(8, 16)], text_token_tags=[tags])
        return m.get_noise_prediction(noisy, sigma * 1000, embeds, batch=batch)

    def test_training_step_protects_the_burned_prefix_like_inference(self):
        torch.manual_seed(1)
        m = model()
        clean = torch.randn(1, 24, 37, 4, 6); noise = torch.randn_like(clean); sigma = torch.tensor([0.6])
        noisy = (1 - sigma) * clean + sigma * noise
        ctx = {f"ctx_burn{i}": torch.full((1, 24, 12, 4, 6), float(i + 1)) for i in range(3)}
        pred = self.predict(m, noisy, sigma, self.batch(m, clean, ctx))
        seen = m.model.seen
        rows = 12 * 2 * 3  # 12 latent steps x (4/2) x (6/2)
        video_rows = seen["hidden_states"][0]
        injected = video_rows[:rows]
        level = injected.mean().item()
        self.assertTrue(any(abs(level - 0.999 * (i + 1)) < 0.05 for i in range(3)), "one burned variant, at 0.999")
        self.assertTrue(torch.allclose(video_rows[rows:], patchify_video_latents(noisy)[0, rows:]), "the rest is the noisy target")
        vidx = seen["video_indices"]; t = seen["row_timesteps"][0]
        self.assertTrue(torch.allclose(t[vidx[:rows]], torch.tensor(0.999)), "prefix pinned like inference")
        self.assertTrue(torch.allclose(t[vidx[rows:]], torch.tensor(0.4)), "target rows at t = 1 - sigma")
        target = noise - clean
        self.assertTrue(torch.allclose(pred[:, :, :12], target[:, :, :12], atol=1e-5), "prefix: zero loss")
        self.assertTrue(torch.equal(pred[:, :, 12:], torch.zeros_like(pred[:, :, 12:])), "after it: the model's own prediction")

    def test_without_contexts_training_is_ordinary_and_warns(self):
        m = model()
        clean = torch.randn(1, 24, 37, 4, 6); sigma = torch.tensor([0.5])
        noisy = (1 - sigma) * clean + sigma * torch.randn_like(clean)
        pred = self.predict(m, noisy, sigma, self.batch(m, clean, {}))
        self.assertTrue(torch.allclose(m.model.seen["hidden_states"][0], patchify_video_latents(noisy)[0]))
        self.assertTrue(m._warned_no_context)
        off = model(context_frames=0)
        self.predict(off, noisy, sigma, self.batch(off, clean, {f"ctx_burn0": torch.zeros(1, 24, 12, 4, 6)}))
        self.assertTrue(torch.allclose(off.model.seen["hidden_states"][0], patchify_video_latents(noisy)[0]), "off: contexts ignored")


    # --- v3 -----------------------------------------------------------------------------------------------------

    def test_v3_settings(self):
        s = cont.settings({"continuation_context_frames": 39, "continuation_burn_levels": [0, 0.05],
                           "continuation_clean_share": 0.5, "continuation_no_context_share": 0.15,
                           "continuation_protect_audio": True})
        self.assertEqual((s.no_context_share, s.protect_audio), (0.15, True))
        self.assertRaises(ValueError, cont.settings, {"continuation_context_frames": 39, "continuation_no_context_share": 1.0})
        # 22 frames at 24 fps is 36.67 audio ticks: inference cannot protect that AV prefix, so neither does training
        self.assertRaises(ValueError, cont.settings, {"continuation_context_frames": 22, "continuation_protect_audio": True})
        cont.settings({"continuation_context_frames": 22})  # video-only protection is still fine
        g = torch.Generator().manual_seed(5)
        self.assertFalse(any(cont.draw_no_context(0.0, g) for _ in range(200)))
        share = sum(cont.draw_no_context(0.3, g) for _ in range(4000)) / 4000
        self.assertTrue(0.26 < share < 0.34, f"about 30% plain steps, got {share}")

    def test_real_speed_guard(self):
        ds = lambda **kw: types.SimpleNamespace(num_frames=124, **kw)
        self.assertRaises(ValueError, cont.check_real_speed, ds(shrink_video_to_frames=True))
        self.assertRaises(ValueError, cont.check_real_speed, ds())  # ai-toolkit's default is to shrink
        cont.check_real_speed(ds(shrink_video_to_frames=False))
        cont.check_real_speed(ds(shrink_video_to_frames=True, auto_frame_count=True, trim_auto_frame_count_tail=True))
        cont.check_real_speed(types.SimpleNamespace(num_frames=1, shrink_video_to_frames=True))  # images
        cont.check_real_speed(None)
        model(context_frames=0).check_dataset_config(ds(shrink_video_to_frames=True))  # off: nothing to guard
        self.assertRaises(ValueError, model().check_dataset_config, ds(shrink_video_to_frames=True))
        m = model()  # an uncached run meets the same check at its first step
        clean = torch.randn(1, 24, 37, 4, 6); sigma = torch.tensor([0.5])
        batch = types.SimpleNamespace(dataset_config=types.SimpleNamespace(do_i2v=False, do_audio=False, num_frames=124,
                                                                           shrink_video_to_frames=True),
                                      num_frames=124, latents=DTO(clean), audio_latents=None)
        self.assertRaises(ValueError, self.predict, m, clean, sigma, batch)

    @staticmethod
    def inference_audio_mask(total, context, feather=8):
        """ComfyUI-H3-MaskedExtension _apply_audio_context_feather, then the H3 model's process_denoise_mask."""
        mask = torch.ones(total)
        hard = context - feather
        mask[:hard] = 0.0
        i = torch.arange(1, feather + 1, dtype=torch.float32)
        mask[hard:context] = 0.5 - 0.5 * torch.cos(torch.pi * i / float(feather))
        mask = torch.round(mask * 256.0) / 256.0
        return mask.masked_fill(mask >= 0.995, 1.0).masked_fill(mask <= 0.05, 0.0)

    def test_audio_mask_matches_inference(self):
        self.assertEqual(cont.audio_context_steps(39), 65)
        self.assertEqual(cont.audio_context_steps(90), 150)
        self.assertRaises(ValueError, cont.audio_context_steps, 73)
        m = cont.audio_context_mask(207, 65)
        self.assertTrue(torch.equal(m, self.inference_audio_mask(207, 65)))
        self.assertTrue(torch.equal(m[:58], torch.zeros(58)), "57 hard ticks plus the first ramp tick (rounds to 0)")
        self.assertTrue(bool(((m[58:64] > 0) & (m[58:64] < 1)).all()), "six ticks released gradually")
        self.assertTrue(torch.equal(m[64:], torch.ones(207 - 64)), "the rest is generated")
        rows = cont.audio_row_mask(m)
        self.assertEqual(rows.shape[0], 414)
        self.assertTrue(torch.equal(rows[:207], rows[207:]), "channel-major rows: both channels share the mask")

    def audio_batch(self, clean, extras, raw_audio, audio_noise):
        return types.SimpleNamespace(dataset_config=types.SimpleNamespace(do_i2v=False, do_audio=True),
                                     num_frames=124, latents=DTO(clean, audio_noise=audio_noise, **extras),
                                     audio_latents=raw_audio)

    def test_training_step_protects_the_audio_prefix_like_inference(self):
        torch.manual_seed(2)
        m = model(continuation_protect_audio=True)
        clean = torch.randn(1, 24, 37, 4, 6); sigma = torch.tensor([0.6])
        noisy = (1 - sigma) * clean + sigma * torch.randn_like(clean)
        raw = torch.randn(1, 414, 32); noise = torch.randn(1, 414, 32)
        ctx = {f"ctx_burn{i}": torch.full((1, 24, 12, 4, 6), float(i + 1)) for i in range(3)}
        pred = self.predict(m, noisy, sigma, self.audio_batch(clean, ctx, raw, noise))
        sa = h3m.remap_sigma(sigma).item()
        mask = cont.audio_row_mask(self.inference_audio_mask(207, 65))
        hard, released, free = mask == 0, (mask > 0) & (mask < 1), mask == 1
        self.assertEqual((int(hard.sum()), int(released.sum()), int(free.sum())), (116, 12, 286))
        seen = m.model.seen["audio_hidden_states"][0]
        self.assertTrue(torch.allclose(seen[hard], raw[0][hard]), "preserved audio is exactly clean")
        self.assertTrue(torch.allclose(seen[free], ((1 - sa) * raw + sa * noise)[0][free], atol=1e-6), "generated audio: the noisy target")
        s_rel = (mask[released] * sa).view(-1, 1)
        self.assertTrue(torch.allclose(seen[released], (1 - s_rel) * raw[0][released] + s_rel * noise[0][released], atol=1e-6),
                        "released ticks sit at sigma = m * sigma_a")
        t = m.model.seen["row_timesteps"][0][m.model.seen["audio_indices"]]
        self.assertTrue(torch.allclose(t[hard], torch.tensor(1.0)), "preserved audio labelled t = 1 (inference's audio pin)")
        self.assertTrue(torch.allclose(t[free], torch.tensor(1 - sa)), "generated audio at t_a")
        self.assertTrue(torch.allclose(t[released], 1 - mask[released] * sa))
        target = (noise - raw)[0]
        out = pred.audio[0]  # the fake transformer predicts 0
        self.assertTrue(torch.allclose(out[hard], target[hard]), "preserved audio: zero loss")
        self.assertTrue(torch.equal(out[free], torch.zeros_like(out[free])), "generated audio: the model's own prediction")
        self.assertTrue(torch.allclose(out[released], (1 - mask[released]).view(-1, 1) * target[released], atol=1e-6),
                        "released ticks: error scaled by m")
        self.assertTrue(torch.allclose(pred.audio_noisy, (1 - sa) * raw + sa * noise, atol=1e-6), "audio_noisy stays the plain noisy rows")

    def test_audio_is_left_alone_without_protection_or_context(self):
        torch.manual_seed(3)
        clean = torch.randn(1, 24, 37, 4, 6); sigma = torch.tensor([0.6])
        noisy = (1 - sigma) * clean + sigma * torch.randn_like(clean)
        raw = torch.randn(1, 414, 32); noise = torch.randn(1, 414, 32)
        ctx = {f"ctx_burn{i}": torch.full((1, 24, 12, 4, 6), float(i + 1)) for i in range(3)}
        sa = h3m.remap_sigma(sigma).item()
        plain_audio = ((1 - sa) * raw + sa * noise)[0]
        m = model()  # v2 form: video prefix only
        pred = self.predict(m, noisy, sigma, self.audio_batch(clean, ctx, raw, noise))
        self.assertTrue(torch.allclose(m.model.seen["audio_hidden_states"][0], plain_audio, atol=1e-6))
        self.assertTrue(torch.equal(pred.audio, torch.zeros_like(pred.audio)))

    def test_no_context_steps_train_plain_generation(self):
        torch.manual_seed(4)
        clean = torch.randn(1, 24, 37, 4, 6); sigma = torch.tensor([0.6])
        noisy = (1 - sigma) * clean + sigma * torch.randn_like(clean)
        raw = torch.randn(1, 414, 32); noise = torch.randn(1, 414, 32)
        ctx = {f"ctx_burn{i}": torch.full((1, 24, 12, 4, 6), float(i + 1)) for i in range(3)}
        m = model(continuation_protect_audio=True, continuation_no_context_share=0.2)
        original = cont.draw_no_context
        cont.draw_no_context = lambda share, generator=None: True
        try:
            pred = self.predict(m, noisy, sigma, self.audio_batch(clean, ctx, raw, noise))
        finally:
            cont.draw_no_context = original
        self.assertTrue(torch.allclose(m.model.seen["hidden_states"][0], patchify_video_latents(noisy)[0]), "no context injected")
        self.assertTrue(torch.equal(pred.tensor, torch.zeros_like(pred.tensor)), "every frame teaches")
        sa = h3m.remap_sigma(sigma).item()
        self.assertTrue(torch.allclose(m.model.seen["audio_hidden_states"][0], ((1 - sa) * raw + sa * noise)[0], atol=1e-6),
                        "and the whole soundtrack is generated")
        self.assertFalse(getattr(m, "_warned_no_context", False), "a planned plain step is not the missing-cache warning")


class StepDrawTest(unittest.TestCase):
    def test_every_pass_of_a_step_sees_the_same_draw(self):
        class Batch:
            pass

        picks = []

        def pick():
            picks.append(1)
            return torch.randn(1, 24, 12, 4, 4)

        batch = Batch()
        first = cont.step_draws(batch, 0.5, pick)
        for _ in range(5):
            again = cont.step_draws(batch, 0.5, pick)
            self.assertIs(again[0], first[0])
            self.assertIs(again[1], first[1])
        self.assertEqual(len(picks), 0 if first[0] else 1)
        self.assertEqual(first[1] is None, first[0])

    def test_a_new_step_draws_again(self):
        class Batch:
            pass

        plain = [cont.step_draws(Batch(), 0.5, lambda: torch.zeros(1))[0] for _ in range(200)]
        self.assertEqual(set(plain), {True, False})
        self.assertEqual({cont.step_draws(Batch(), 0.0, lambda: torch.zeros(1))[0] for _ in range(20)}, {False})


if __name__ == "__main__":
    unittest.main(verbosity=2)
