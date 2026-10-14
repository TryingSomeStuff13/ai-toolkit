"""CPU contracts: the trainer's MiniMax H3 forward reproduces the ComfyUI engine's.

The v6 step-0 diagnostic compares the trainer's prediction with the engine's recorded outputs on the same inputs
(ComfyUI's comfy/ldm/minimax/model.py driven by the MiniMax-H3-Turbo node, pruned int8 checkpoint). Pinned here:
everything the timestep touches (sigma -> per-stream labels, the pruned adaLN curve, the per-row modulation of the
blocks, the final layer, the turbo LoRA's run-time adaLN update), the packed positions and order, and the weights
ComfyUI actually runs for a pre-quantized checkpoint.

On a GPU the forward cannot be bit-identical anyway (comfy-kitchen's int8 and RMSNorm+RoPE kernels, the attention
kernel), and the turbo amplifies any one-ulp difference to its own floor per step (see continuation_trajectory_floor):
these contracts pin everything that is not a kernel.

References:
- always: the engine's arithmetic restated in plain torch;
- with H3_ENGINE_DIR (the ComfyUI checkout that runs H3): the engine's own functions, read from its source at test
  time, and a check that the restated lines are still in it;
- with H3_ENGINE_DIR and that checkout's packages importable (run this file with the engine's Python), the whole
  model: a tiny engine MiniMaxH3Model, loaded the way ComfyUI loads a pre-quantized checkpoint, against the trainer's
  transformer on the recorded T2V layout at the 8 turbo sigmas.
"""
import ast, contextlib, io, math, os, sys, types, unittest
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import torch
# load only the H3 module: the package __init__ imports every model, and one needs a GPU at import time
# (the minimax_h3 package too: its __init__ imports the trainer model and the toolkit's data stack, which the engine's
# Python, used for the whole-model comparison, does not have)
for name, path in (("extensions_built_in", "extensions_built_in"), ("extensions_built_in.diffusion_models", "extensions_built_in/diffusion_models"),
                   ("extensions_built_in.diffusion_models.minimax_h3", "extensions_built_in/diffusion_models/minimax_h3")):
    if name not in sys.modules:
        stub = types.ModuleType(name); stub.__path__ = [os.path.join(ROOT, path)]; sys.modules[name] = stub
from extensions_built_in.diffusion_models.minimax_h3.src import adaln_injection as inj
from extensions_built_in.diffusion_models.minimax_h3.src import packing
from extensions_built_in.diffusion_models.minimax_h3.src import trajectory as traj
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (MODALITY_NUM, MiniMaxH3Transformer,
                                                                             MiniMaxH3TransformerParams)

ENGINE_DIR = os.environ.get("H3_ENGINE_DIR")
BF16 = torch.bfloat16
SHIFT_V, SHIFT_A = 12.0, 3.0
# the recorded step-0 geometry: 124 frames at 384x672, 212 prompt tokens
TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T = 212, 37, 42, 24, 207


def turbo_sigmas(steps=8, shift=SHIFT_V):
    """ComfyUI's BasicScheduler "simple" over the shift-12 flow schedule: the dashboard's turbo steps (the last is 0)."""
    t = (torch.arange(1, 1001, 1) / 1000) * 1000 / 1000
    table = shift * t / (1 + (shift - 1) * t)
    ss = len(table) / steps
    return [float(table[-(1 + int(x * ss))]) for x in range(steps)]


SIGMAS = turbo_sigmas()


# ----------------------------------------------------------------------------------------------------------------
# The engine's arithmetic, restated (comfy/ldm/minimax/model.py MiniMaxH3Model._forward, AdalnProj, DiTBlock,
# FinalLayer; comfy/ops.py MixedPrecisionOps). Checked against the real source when H3_ENGINE_DIR is set.
# ----------------------------------------------------------------------------------------------------------------

def engine_time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def engine_labels(timestep):
    """_forward: the stream labels from the DiT timestep (1000 * sigma_v)."""
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    return float(1.0 - sigma_v), float(1.0 - engine_time_shift_sigma(sigma_v, SHIFT_V, SHIFT_A))


def engine_mod_rows(t_v, t_a, text_tags, n_audio, n_video):
    """_forward, plain text-to-video (no refs, keyframes or masks): the modulation row of every packed token, the
    final layer's row of the target streams and the sorted distinct labels (the t_emb rows)."""
    unique_t = sorted({t_v, t_a})
    t_row = {t: i for i, t in enumerate(unique_t)}
    rows = [t_row[t_v] * 3 + int(tag) for tag in text_tags.tolist()]
    rows += [t_row[t_a] * 3 + 2] * n_audio + [t_row[t_v] * 3 + 0] * n_video
    return torch.tensor(rows), t_row[t_v], t_row[t_a], unique_t


def engine_curve(table, t_vals):
    """_forward, curve-form checkpoints: the pruned adaLN coordinates of the distinct labels."""
    pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


def engine_block(block, x, t_emb, rows):
    """DiTBlock.forward on the unbatched (S, hidden) stream: the modulation cast to the stream dtype per row, the gated
    residuals accumulated in place with addcmul_ (_mod_scale_shift / _mod_gate). ``block`` supplies the layers."""
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    dt = x.dtype
    h = block.norm1(x)
    h.mul_(1.0 + scale_msa[rows].to(dt)).add_(shift_msa[rows].to(dt))
    x = x.clone()
    x.addcmul_(block.attn(h.unsqueeze(0), None).squeeze(0), gate_msa[rows].to(dt))
    h = block.norm2(x)
    h.mul_(1.0 + scale_mlp[rows].to(dt)).add_(shift_mlp[rows].to(dt))
    x.addcmul_(block.mlp(h), gate_mlp[rows].to(dt))
    return x


def engine_final(final, x, t_emb, rows):
    """FinalLayer.forward: float32 modulation of the (bf16) norm output, then the heads."""
    shift, scale = final.adaln_proj(t_emb)
    h = (final.norm(x) * (1.0 + scale[rows].to(scale.dtype)) + shift[rows].to(shift.dtype)).to(torch.float32)
    return final.video_out(h), final.audio_out(h)


class EngineMixedPrecisionLinear(torch.nn.Module):
    """How MixedPrecisionOps.Linear loads an unquantized layer of a pre-quantized checkpoint: its factory dtype is the
    compute dtype whatever the model code declares, the weight is converted to it, and the bias is a compute-dtype
    parameter that load_state_dict fills (legacy ModelPatcher) or replaces with the stored tensor (DynamicVRAM,
    assign=True)."""

    compute_dtype = BF16

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.factory_kwargs = {"device": None, "dtype": self.compute_dtype}
        self.in_features, self.out_features = in_features, out_features
        self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs)) if bias else None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        weight = state_dict.pop(f"{prefix}weight")
        self.weight = torch.nn.Parameter(weight.to(device=None, dtype=self.factory_kwargs["dtype"]),
                                         requires_grad=False)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)
        if f"{prefix}weight" in missing_keys:
            missing_keys.remove(f"{prefix}weight")


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------

def smooth_table(rows=1025, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.cumsum(torch.randn(rows, dim, generator=g) * 0.05, dim=0) + torch.randn(1, dim, generator=g)


def tiny_params(**kw):
    p = dict(hidden_size=32, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=32,
             ffn_hidden_size=64, text_dim=24, rope_inv_freq_len=4, time_embed_dim=8, adaln_t_table_size=1025)
    p.update(kw)
    return MiniMaxH3TransformerParams(**p)


def pruned_state_dict(params, seed=0, final_scale=-0.97):
    """Random weights in the pruned int8 checkpoint's stored dtypes (int8 linears as bf16 here): fp16 adaLN, fp32
    islands, bf16 rest, the table fp32. The final adaLN's scale half sits near ``final_scale`` like the release's
    (1 + scale ~ 0.02-0.05), the regime where the final layer's precision matters."""
    model = MiniMaxH3Transformer(params)
    g = torch.Generator().manual_seed(seed)
    hid = params.hidden_size
    sd = {}
    for k, v in model.state_dict().items():
        if k == "adaln_t_table":
            sd[k] = smooth_table(v.shape[0], v.shape[1], seed)
        elif k == "rope.inv_freq":
            sd[k] = v.clone()
        elif "norm" in k:
            sd[k] = (1.0 + 0.1 * torch.randn(v.shape, generator=g)).to(BF16)
        elif "adaln_proj.linear" in k:
            sd[k] = ((0.3 if k.endswith("weight") else 0.05) * torch.randn(v.shape, generator=g)).to(torch.float16)
        else:
            t = torch.randn(v.shape, generator=g) / (v.shape[-1] ** 0.5 if v.dim() > 1 else 10.0)
            fp32 = k.startswith(MiniMaxH3Transformer.FP32_KEY_PREFIXES)
            sd[k] = t.float() if fp32 else t.to(BF16)
    w, b = sd["final_layer.adaln_proj.linear.weight"].float(), sd["final_layer.adaln_proj.linear.bias"].float()
    w[hid:] *= 0.05
    b[hid:] = final_scale + 0.02 * torch.randn(hid, generator=g)
    sd["final_layer.adaln_proj.linear.weight"], sd["final_layer.adaln_proj.linear.bias"] = w.half(), b.half()
    return sd


def load_trainer(params, sd):
    model = MiniMaxH3Transformer(params)
    model.load_state_dict({k: v.clone() for k, v in sd.items()}, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model


def rel(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp(min=1e-12))


def t2v_rows(tags, n_audio, n_video, t_v, t_a):
    """The trainer's per-row labels for an anchor sample (minimax_h3._trajectory_prediction) and the indices its
    transformer derives from them (MiniMaxH3Transformer.forward)."""
    s = tags.shape[0] + n_audio + n_video
    row_t = torch.full((1, s), t_v, dtype=torch.float32)
    row_t[0, tags.shape[0]:tags.shape[0] + n_audio] = t_a
    token_tags = torch.cat([tags, torch.full((n_audio,), packing.AUDIO_TAG), torch.full((n_video,), packing.VIDEO_TAG)])
    unique_t, inverse = torch.unique(row_t.to(torch.float32), sorted=True, return_inverse=True)
    return (inverse * MODALITY_NUM + token_tags.clamp(min=0)).view(-1), inverse.view(-1), unique_t


def engine_source(relpath):
    with open(os.path.join(ENGINE_DIR, relpath)) as f:
        return f.read()


def engine_namespace():
    """The engine's module-level functions, constants and classes this file checks, read from its model.py (not
    imported: that needs ComfyUI's packages) and executed in a namespace holding only torch. Methods are called
    unbound on stand-in objects."""
    src = engine_source(os.path.join("comfy", "ldm", "minimax", "model.py"))
    ns = {"torch": torch, "math": math, "nn": torch.nn}
    constants = {"FRAME_PER_TOKEN", "FRAME_RESCALE", "VISUAL_COND_TIMESTEP", "AUDIO_COND_TIMESTEP"}
    functions = {"time_shift_sigma", "_axis_from_sqrt_area", "_frame_grid", "_video_t_spans", "_video_t_grid",
                 "_ref_t_span", "_audio_grid", "_video_grid", "_mod_row", "_mod_scale_shift", "_mod_gate"}
    classes = {"PackedLayout", "AdalnProj", "DiTBlock", "FinalLayer"}
    for node in ast.parse(src).body:
        keep = (isinstance(node, ast.Assign) and {getattr(t, "id", None) for t in node.targets} & constants) or \
            (isinstance(node, ast.FunctionDef) and node.name in functions) or \
            (isinstance(node, ast.ClassDef) and node.name in classes)
        if keep:
            exec(compile(ast.Module([node], []), "model.py", "exec"), ns)
    missing = (constants | functions | classes) - set(ns)
    if missing:
        raise AssertionError(f"the engine's model.py no longer defines {sorted(missing)}")
    return ns


# ----------------------------------------------------------------------------------------------------------------
# contracts
# ----------------------------------------------------------------------------------------------------------------

class TimestepPathContracts(unittest.TestCase):
    """What the recorded timestep becomes on both sides."""

    def test_labels_and_modulation_rows(self):
        g = torch.Generator().manual_seed(1)
        tags = (torch.rand(TEXT_LEN, generator=g) > 0.1).long()  # a few vision-tagged prompt rows
        n_audio, n_video = 2 * AUDIO_T, LATENT_T * (LATENT_H // 2) * (LATENT_W // 2)
        for sigma in SIGMAS:
            timestep = torch.tensor([1000.0 * sigma], dtype=torch.float32)  # what model_base hands the DiT
            sigma_v, sigma_a = traj.stream_sigmas(timestep)
            t_v, t_a = 1.0 - sigma_v, 1.0 - sigma_a
            e_tv, e_ta = engine_labels(timestep)
            self.assertEqual(float(t_v), e_tv, f"t_v at sigma {sigma}")
            self.assertEqual(float(t_a), e_ta, f"t_a at sigma {sigma}")
            rows, inverse, unique_t = t2v_rows(tags, n_audio, n_video, float(t_v), float(t_a))
            e_rows, e_final_v, e_final_a, e_unique = engine_mod_rows(e_tv, e_ta, tags, n_audio, n_video)
            self.assertTrue(torch.equal(rows, e_rows), f"block modulation rows at sigma {sigma}")
            self.assertEqual(unique_t.tolist(), torch.tensor(e_unique, dtype=torch.float32).tolist())
            self.assertTrue(bool((inverse[-n_video:] == e_final_v).all()), "final layer rows, video")
            self.assertTrue(bool((inverse[TEXT_LEN:TEXT_LEN + n_audio] == e_final_a).all()), "final layer rows, audio")
        self.assertEqual(len({engine_labels(torch.tensor([1000.0]))}), 1)  # sigma 1: one label for both streams

    def test_curve_sampling_is_the_engines(self):
        """The pruned adaLN coordinates (and so the turbo update's nearest-row lookup) bit for bit."""
        model = load_trainer(tiny_params(), pruned_state_dict(tiny_params()))
        ts = []
        for sigma in SIGMAS:
            t_v, t_a = engine_labels(torch.tensor([1000.0 * sigma], dtype=torch.float32))
            ts += [t_v, t_a]
        ts += [0.0, 1.0, 0.5, 7 / 1024, 1023 / 1024, 1e-7, 0.9999999] + torch.rand(64).tolist()
        t = torch.tensor(ts, dtype=torch.float32)
        ours = model._time_embedding(t)
        theirs = engine_curve(model.adaln_t_table.float(), t)
        self.assertTrue(torch.equal(ours, theirs))
        # the turbo injection's lookup lands on the same grid rows
        table = model.adaln_t_table.float()
        self.assertTrue(torch.equal(torch.cdist(ours, table).argmin(1), torch.cdist(theirs, table).argmin(1)))


class LayerContracts(unittest.TestCase):
    """The block and final-layer numerics in the engine's precision (bf16 stream, float32 modulation)."""

    def setUp(self):
        self.params = tiny_params()
        self.sd = pruned_state_dict(self.params)
        self.model = load_trainer(self.params, self.sd)
        self.model.match_engine_weight_precision()
        g = torch.Generator().manual_seed(5)
        self.s = 40
        self.x = (torch.randn(self.s, self.params.hidden_size, generator=g) * 3).to(BF16)
        self.t = torch.tensor([0.0118, 0.0455], dtype=torch.float32)
        self.temb = self.model._time_embedding(self.t)
        tags = torch.tensor([1] * 6 + [0] * 2 + [1] * 2)
        self.rows, self.inverse, _ = t2v_rows(tags, 10, 20, float(self.t[0]), float(self.t[1]))

    def test_final_layer_is_the_engines(self):
        final = self.model.final_layer
        ours = final(self.x.unsqueeze(0), self.temb, self.inverse.unsqueeze(0))
        theirs = engine_final(final, self.x, self.temb, self.inverse)
        for a, b in zip(ours, theirs):
            self.assertEqual(a.dtype, torch.float32)
            self.assertTrue(torch.equal(a.squeeze(0), b))
        # the old bf16 modulation is measurably off where 1 + scale is small
        shift, scale = final.adaln_proj(self.temb)
        idx = self.inverse
        old = final.norm(self.x) * (1.0 + scale[idx].to(BF16)) + shift[idx].to(BF16)
        new = final.norm(self.x) * (1.0 + scale[idx]) + shift[idx]
        self.assertGreater(rel(old, new), 1e-3)

    def test_block_is_the_engines(self):
        block = self.model.blocks[0]
        rope = self.model.rope(torch.zeros(1, self.s, 3))
        attn = block.attn
        block_view = types.SimpleNamespace(adaln_proj=block.adaln_proj, norm1=block.norm1, norm2=block.norm2,
                                           mlp=block.mlp, attn=lambda h, _: attn(h, rope))
        ours = block(self.x.unsqueeze(0), self.temb, self.rows.unsqueeze(0), rope).squeeze(0)
        theirs = engine_block(block_view, self.x, self.temb, self.rows)
        self.assertEqual(ours.dtype, BF16)
        self.assertTrue(torch.equal(ours, theirs))


class EnginePrecisionContracts(unittest.TestCase):
    """match_engine_weight_precision leaves every unquantized linear with the weights ComfyUI runs."""

    def engine_params(self, sd, assign):
        """Load ``sd`` into MixedPrecisionOps-style linears at the trainer's module paths."""
        model = MiniMaxH3Transformer(tiny_params())
        for name, module in list(model.named_modules()):
            for child, sub in list(module.named_children()):
                if isinstance(sub, torch.nn.Linear):
                    setattr(module, child, EngineMixedPrecisionLinear(sub.in_features, sub.out_features,
                                                                      sub.bias is not None))
        model.load_state_dict({k: v.clone() for k, v in sd.items()}, strict=True, assign=assign)
        return dict(model.named_parameters()), dict(model.named_buffers())

    def test_matches_both_comfy_loaders(self):
        params = tiny_params()
        sd = pruned_state_dict(params)
        for assign, round_biases in ((True, False), (False, True)):  # DynamicVRAM, legacy ModelPatcher
            model = load_trainer(params, sd)
            count = model.match_engine_weight_precision(BF16, round_biases=round_biases)
            eng_params, eng_buffers = self.engine_params(sd, assign)
            for name, p in model.named_parameters():
                self.assertTrue(torch.equal(p.float(), eng_params[name].float()), f"{name} (assign={assign})")
            for name, b in model.named_buffers():
                self.assertTrue(torch.equal(b, eng_buffers[name]), name)
            # where each call site computes: fp32 islands stay fp32 storage, the fp16 adaLN linears become bf16
            self.assertEqual(model.video_patch_proj.weight.dtype, torch.float32)
            self.assertEqual(model.final_layer.video_out.weight.dtype, torch.float32)
            self.assertEqual(model.blocks[0].adaln_proj.linear.weight.dtype, BF16)
            self.assertEqual(model.blocks[0].adaln_proj.linear.bias.dtype, BF16 if round_biases else torch.float16)
            self.assertGreater(count, 0)
            self.assertEqual(model.match_engine_weight_precision(BF16, round_biases=round_biases), 0, "idempotent")

    def test_quantized_linears_are_left_alone(self):
        params = tiny_params()
        model = load_trainer(params, pruned_state_dict(params))
        fc = model.blocks[0].mlp.fc1
        fc.weight.data = fc.weight.data.float()
        fc.is_ostris_quantized = True  # an OstrisLinear: its weight is a dequantizing view, never rounded here
        before = fc.weight.data.clone()
        model.match_engine_weight_precision()
        self.assertTrue(torch.equal(fc.weight.data, before))

    def test_model_switch(self):
        from extensions_built_in.diffusion_models.minimax_h3 import minimax_h3 as h3m
        params = tiny_params()
        sd = pruned_state_dict(params)
        m = object.__new__(h3m.MinimaxH3Model)
        for kwargs, quantized, expect in (({}, True, "dynamic"), ({"match_engine_precision": "legacy"}, True, "legacy"),
                                          ({"match_engine_precision": False}, True, None),
                                          ({"match_engine_precision": "off"}, True, None), ({}, False, None)):
            m.model_config = types.SimpleNamespace(model_kwargs=kwargs)
            model = load_trainer(params, sd)
            model.aitk_is_quantized = quantized
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                count = m.match_engine_precision(model)
            if expect is None:
                self.assertEqual(count, 0)
                self.assertEqual(out.getvalue(), "")
                self.assertEqual(model.video_patch_proj.weight.dtype, torch.float32)
                self.assertTrue(torch.equal(model.video_patch_proj.weight, sd["video_patch_proj.weight"]))
            else:
                self.assertGreater(count, 0)
                self.assertIn(f"engine precision ({expect})", out.getvalue())
                bias = model.blocks[0].adaln_proj.linear.bias
                self.assertEqual(bias.dtype, BF16 if expect == "legacy" else torch.float16)
        m.model_config = types.SimpleNamespace(model_kwargs={"match_engine_precision": "fp8"})
        self.assertRaises(ValueError, m.match_engine_precision, load_trainer(params, sd))


@unittest.skipUnless(ENGINE_DIR, "set H3_ENGINE_DIR to the ComfyUI checkout that runs MiniMax H3")
class EngineSourceContracts(unittest.TestCase):
    """The engine's own functions (read from its source) against the trainer, and the restated lines still there."""

    @classmethod
    def setUpClass(cls):
        cls.ns = engine_namespace()

    def test_restated_lines_are_in_the_engine(self):
        model_src = engine_source(os.path.join("comfy", "ldm", "minimax", "model.py"))
        for line in ('sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)',
                     't_v = float(1.0 - sigma_v)',
                     't_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))',
                     'seg_t = {"text": t_v, "video": t_v, "audio": t_a,',
                     'seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}',
                     'pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)',
                     'i0 = pos.floor().long().clamp(max=table.shape[0] - 2)',
                     't_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))',
                     'video_seg = (va, vb, t_row[seg_t["video"]])',
                     'audio_seg = (aa, ab, t_row[seg_t["audio"]])',
                     'return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]'):
            self.assertIn(line, model_src)
        ops_src = engine_source(os.path.join("comfy", "ops.py"))
        for line in ('self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}',
                     'module.weight = torch.nn.Parameter(weight.to(device=device, dtype=compute_dtype), requires_grad=False)',
                     'self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))'):
            self.assertIn(line, ops_src)
        sd_src = engine_source(os.path.join("comfy", "sd.py"))
        self.assertIn('model.load_model_weights(new_sd, "", assign=model_patcher.is_dynamic())', sd_src)

    def test_shift_positions_and_order(self):
        ns = self.ns
        for sigma in SIGMAS:
            s = torch.tensor(sigma, dtype=torch.float32)
            self.assertEqual(float(ns["time_shift_sigma"](s, SHIFT_V, SHIFT_A)), float(packing.remap_sigma(s)))
        for h, w in ((LATENT_H, LATENT_W), (LATENT_W, LATENT_H)):
            layout = ns["PackedLayout"](TEXT_LEN, LATENT_T, h, w, AUDIO_T)
            ours = packing.build_packed_sequence(text_token_tags=torch.ones(TEXT_LEN, dtype=torch.long),
                                                 num_latent_frames=LATENT_T, latent_height=h, latent_width=w,
                                                 num_audio_latents=AUDIO_T)
            self.assertEqual([k for _, _, k in layout.segments], ["text", "audio", "video"])
            _, (aa, ab, _), (va, vb, _) = layout.segments
            self.assertTrue(torch.equal(ours.audio_indices, torch.arange(aa, ab)))
            self.assertTrue(torch.equal(ours.video_indices, torch.arange(va, vb)))
            self.assertLess(float((ours.position_ids - layout.position_ids).abs().max()), 1e-12)
            # the rotary angles, as both compute them (float32 positions)
            self.assertTrue(torch.equal(ours.position_ids.float(), layout.position_ids.to(torch.float32)))

    def test_layers_are_the_engines(self):
        ns = self.ns
        params = tiny_params()
        model = load_trainer(params, pruned_state_dict(params))
        model.match_engine_weight_precision()
        g = torch.Generator().manual_seed(7)
        x = (torch.randn(40, params.hidden_size, generator=g) * 3).to(BF16)
        t = torch.tensor([0.0118, 0.0455], dtype=torch.float32)
        temb = model._time_embedding(t)
        tags = torch.tensor([1] * 6 + [0] * 2 + [1] * 2)
        rows, inverse, _ = t2v_rows(tags, 10, 20, float(t[0]), float(t[1]))
        segments = [(i, i + 1, int(r)) for i, r in enumerate(rows)]  # the engine's per-segment rows, one per token

        def engine_adaln(proj):
            # the engine's projection: its linear computes in the input's dtype (float32 t_emb) with the loaded weights
            obj = types.SimpleNamespace(expand=proj.expand, modalities=proj.modalities, hidden=proj.hidden,
                                        apply_silu=proj.apply_silu,
                                        linear=lambda e: torch.nn.functional.linear(e, proj.linear.weight.to(e.dtype),
                                                                                    proj.linear.bias.to(e.dtype)))
            return lambda e: ns["AdalnProj"].forward(obj, e)

        final = model.final_layer
        fobj = types.SimpleNamespace(adaln_proj=engine_adaln(final.adaln_proj), norm=final.norm,
                                     video_out=final.video_out, audio_out=final.audio_out)
        video_seg, audio_seg = (20, 40, int(inverse[-1])), (10, 20, int(inverse[10]))
        ev, ea = ns["FinalLayer"].forward(fobj, x, temb, video_seg, audio_seg)
        # the engine runs each head on its own stream's rows: same row slices here, so the GEMMs match too
        ov, _ = final(x[20:40].unsqueeze(0), temb, inverse[20:40].unsqueeze(0))
        _, oa = final(x[10:20].unsqueeze(0), temb, inverse[10:20].unsqueeze(0))
        self.assertTrue(torch.equal(ov[0], ev))
        self.assertTrue(torch.equal(oa[0], ea))

        block = model.blocks[0]
        rope = model.rope(torch.zeros(1, 40, 3))
        bobj = types.SimpleNamespace(adaln_proj=engine_adaln(block.adaln_proj), norm1=block.norm1, norm2=block.norm2,
                                     mlp=block.mlp,
                                     attn=lambda h, rope_freqs=None, transformer_options=None:
                                     block.attn(h.unsqueeze(0), rope).squeeze(0))
        theirs = ns["DiTBlock"].forward(bobj, x.clone(), temb, segments, None)
        ours = block(x.unsqueeze(0), temb, rows.unsqueeze(0), rope).squeeze(0)
        self.assertTrue(torch.equal(ours, theirs))


# The MiniMax-H3-Turbo node's curve-mode injection, verbatim (ComfyUI-MiniMax-H3-Turbo __init__.py, Apache License
# 2.0), for the whole-model comparison; test_h3_adaln.py checks the same function bit for bit and against the source.
NODE_REFERENCE = r'''
def _make_adaln_forward(base, a, b, shared, table=None, egrid=None):
    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)

        st = None
        if table is not None and egrid is not None and not base.apply_silu:
            try:
                tb = table.to(t_emb.device, torch.float32)
                idx = torch.cdist(t_emb.detach().float(), tb).argmin(dim=1)
                st = egrid.to(t_emb.device)[idx]          # [M, 2688], M from the model
            except Exception:
                st = None
        if st is None:
            st = shared.get("silu_temb")                  # legacy _unique_t path

        if st is not None and st.shape[0] == x.shape[0]:
            av = a.to(x.device, x.dtype)
            bv = b.to(x.device, x.dtype)
            sv = st.to(x.device, x.dtype)
            x = x + (bv @ (av @ sv.T)).T                              # [M, out]
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    return forward
'''
_node = {"torch": torch, "F": torch.nn.functional}
exec(NODE_REFERENCE, _node)


def import_engine():
    """ComfyUI's MiniMax H3 model on the CPU with SDPA attention (the no-sage engine's kernel)."""
    sys.path.insert(0, ENGINE_DIR)
    argv = sys.argv
    sys.argv = [argv[0], "--cpu", "--use-pytorch-cross-attention"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.model_management  # noqa: F401 (parses the args)
        import comfy.ops
        import comfy.ldm.minimax.model as engine_model
    finally:
        sys.argv = argv
    return comfy.ops, engine_model


@unittest.skipUnless(ENGINE_DIR, "set H3_ENGINE_DIR and run with the engine's Python for the whole-model comparison")
class WholeModelContracts(unittest.TestCase):
    """A tiny engine MiniMaxH3Model, loaded like a pre-quantized checkpoint (MixedPrecisionOps, DynamicVRAM's assign
    load) with the turbo node's adaLN injection, against the trainer's transformer (engine precision on, the same
    injection through src/adaln_injection.py) on the recorded T2V layout at the 8 turbo sigmas."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.ops, cls.em = import_engine()
        except Exception as exc:  # the engine's packages are not in this environment
            raise unittest.SkipTest(f"the engine does not import here ({exc!r}); run with its Python")

    def build(self, dtype, params):
        cfg = {k: getattr(params, k) for k in (
            "hidden_size", "num_layers", "token_refiner_num_layers", "num_attention_heads", "attention_head_dim",
            "ffn_hidden_size", "latents_dim", "audio_latents_dim", "text_dim", "time_embed_dim", "rope_inv_freq_len")}
        sd = pruned_state_dict(params)
        if dtype == torch.float32:
            sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
        engine = self.em.MiniMaxH3Model(adaln_curve_grid=params.adaln_t_table_size, dtype=dtype, device="cpu",
                                        operations=self.ops.mixed_precision_ops({}, dtype), **cfg)
        missing, unexpected = engine.load_state_dict({k: v.clone() for k, v in sd.items()}, strict=False, assign=True)
        self.assertFalse(missing or unexpected, (missing, unexpected))
        engine.eval().requires_grad_(False)
        trainer = load_trainer(params, sd)
        trainer.match_engine_weight_precision(dtype)
        # the turbo LoRA's adaLN pieces: the node's patched forwards on the engine, adaln_injection on the trainer
        g = torch.Generator().manual_seed(11)
        lora = {}
        for name, mod in engine.named_modules():
            if name.endswith("adaln_proj"):
                lora[name + ".linear.lora_A.weight"] = (torch.randn(4, 48, generator=g) * 0.3).to(BF16)
                lora[name + ".linear.lora_B.weight"] = (torch.randn(mod.linear.out_features, 4, generator=g)
                                                        * 0.3).to(BF16)
        grid = torch.randn(params.adaln_t_table_size, 48, generator=g).to(BF16)
        for name in sorted({k.rsplit(".linear.lora_", 1)[0] for k in lora}):
            mod = engine.get_submodule(name)
            mod.forward = _node["_make_adaln_forward"](mod, lora[name + ".linear.lora_A.weight"],
                                                       lora[name + ".linear.lora_B.weight"], {},
                                                       engine.adaln_t_table, grid)
        self.assertEqual(inj.attach(trainer, lora, grid), params.num_layers + 1)
        return engine, trainer

    def run_both(self, engine, trainer, dtype, sigma, t, h, w, ta, text, tags, video, audio):
        payload = {"text_token_tags": tags, "seed": 0, "audio_scale": 4.0,
                   "layout": self.em.PackedLayout(text.shape[1], t, h, w, ta)}
        ts = torch.tensor([1000.0 * sigma], dtype=torch.float32)
        with torch.no_grad():
            ctx = engine.preprocess_text_embeds(text.to(dtype))
            ev, ea = engine._forward([video.to(dtype), audio.to(dtype)], ts, ctx, {}, minimax_payload=payload)
            sigma_v, sigma_a = traj.stream_sigmas(ts)
            audio_rows = traj.audio_rows(audio.float())
            layout = packing.build_packed_sequence(text_token_tags=tags, num_latent_frames=t, latent_height=h,
                                                   latent_width=w, num_audio_latents=ta)
            pos, token_tags, vi, ai, ti, _ = packing.pad_layouts_to_batch([layout])
            row_t = (1.0 - sigma_v).view(-1, 1).expand(-1, token_tags.shape[1]).clone()
            row_t[:, ai] = (1.0 - sigma_a).view(1, 1)
            vp, ap = trainer(hidden_states=packing.patchify_video_latents(video.float()).to(dtype),
                             audio_hidden_states=audio_rows.to(dtype), encoder_hidden_states=text.to(dtype),
                             row_timesteps=row_t, token_tags=token_tags, position_ids=pos, video_indices=vi,
                             audio_indices=ai, text_indices=ti)
        return (-packing.unpatchify_video_tokens(vp.float(), t, h, w), -ap.float()), (ev.float(), traj.audio_rows(ea.float()))

    def test_forward_is_the_engines(self):
        params = tiny_params(hidden_size=64, num_layers=3, num_attention_heads=2, attention_head_dim=64,
                             ffn_hidden_size=128, rope_inv_freq_len=8)
        g = torch.Generator().manual_seed(3)
        t, h, w, ta, length = 10, 6, 8, 57, 40
        tags = torch.ones(length, dtype=torch.long)
        tags[3:6] = 0
        text = torch.randn(1, length, params.text_dim, generator=g)
        video = torch.randn(1, params.latents_dim, t, h, w, generator=g)
        audio = torch.randn(1, params.audio_latents_dim, 2, ta, generator=g)
        for dtype in (torch.float32, BF16):
            engine, trainer = self.build(dtype, params)
            for sigma in SIGMAS:
                (tv, tav), (ev, eav) = self.run_both(engine, trainer, dtype, sigma, t, h, w, ta, text, tags,
                                                     video.to(dtype), audio.to(dtype))
                if dtype == torch.float32:
                    self.assertLess(rel(tv, ev), 1e-5, f"video, sigma {sigma}")
                    self.assertLess(rel(tav, eav), 1e-5, f"audio, sigma {sigma}")
                else:
                    # the engine hands back bf16 (-out.to(x.dtype)): compare on that grid
                    for ours, theirs, what in ((tv, ev, "video"), (tav, eav, "audio")):
                        ours = ours.to(BF16).float()
                        differ = float((ours != theirs).float().mean())
                        self.assertLess(differ, 1e-3, f"{what}, sigma {sigma}: {differ:.2%} of the outputs differ")
                        self.assertLess(rel(ours, theirs), 1e-5, f"{what}, sigma {sigma}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
