"""CPU contracts for H3 continuation training: the real get_noise_prediction / encode_images with stand-in models."""
import sys, types, unittest
sys.path.insert(0, "/home/jeff/ai/ai-toolkit-h3")
import torch
from toolkit.dto import DTO
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
# load only the H3 module: the package __init__ imports every model, and one needs a GPU at import time
for name, path in (("extensions_built_in", "extensions_built_in"), ("extensions_built_in.diffusion_models", "extensions_built_in/diffusion_models")):
    stub = types.ModuleType(name); stub.__path__ = ["/home/jeff/ai/ai-toolkit-h3/" + path]; sys.modules[name] = stub
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
        steps = cont.latent_steps(t)
        pooled = torch.nn.functional.adaptive_avg_pool3d(frames.float().mean(1, keepdim=True), (steps, h // 16, w // 16))
        return pooled.expand(b, 24, steps, h // 16, w // 16).clone()


def model(context_frames=39):
    m = object.__new__(h3m.MinimaxH3Model)
    m.model_config = types.SimpleNamespace(model_kwargs={"continuation_context_frames": context_frames} if context_frames else {})
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
