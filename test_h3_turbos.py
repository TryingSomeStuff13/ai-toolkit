"""CPU contracts for turbo-agnostic MiniMax-H3 training (model_kwargs.turbo, src/turbos.py, src/runtime_lora.py).

- Registry names resolve to a kind (a LoRA applied live, or a checkpoint DiT), the file of the run's partition, the
  shifts and the sigma grid; the turbos the trainer cannot run are refused.
- Every tensor of a turbo LoRA maps onto a trainer module (nothing silently skipped); Larry's adaLN pieces go to the
  run-time adaLN update, a plain turbo may not carry any.
- The live application is the ComfyUI H3RuntimeLoRA node's math bit for bit. The node's functions are kept below
  verbatim as the reference (set H3_RUNTIME_LORA_DIR to also check the copy against the node's source).
- Several turbos share one base DiT and switch per sample; a trajectory sample's assistant is its recording's turbo,
  with that turbo's shifts; the per-step floor weighting scales each sample's loss.
- Real files, header only: set H3_TURBO_REGISTRY (the rental tooling's turbos.py) and H3_TURBO_LORA_DIR (a folder with
  the turbo files, or their safetensors headers saved as <file>.json).
"""
import ast, contextlib, io, json, math, os, struct, sys, tempfile, types, unittest
from unittest import mock
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
# load only the H3 module: the package __init__ imports every model, and one needs a GPU at import time
for name, path in (("extensions_built_in", "extensions_built_in"), ("extensions_built_in.diffusion_models", "extensions_built_in/diffusion_models")):
    if name not in sys.modules:
        stub = types.ModuleType(name); stub.__path__ = [os.path.join(ROOT, path)]; sys.modules[name] = stub
from extensions_built_in.diffusion_models.minimax_h3 import minimax_h3 as h3m
from extensions_built_in.diffusion_models.minimax_h3.src import runtime_lora as rt
from extensions_built_in.diffusion_models.minimax_h3.src import trajectory as traj
from extensions_built_in.diffusion_models.minimax_h3.src import turbos as tb
from extensions_built_in.diffusion_models.minimax_h3.src.packing import build_packed_sequence, remap_sigma
from extensions_built_in.diffusion_models.minimax_h3.src.pipeline import sampling_sigmas
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (MiniMaxH3Mlp, MiniMaxH3Transformer,
                                                                             MiniMaxH3TransformerParams)
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
import test_h3_trajectory as tt

# Copied verbatim from ComfyUI-H3-RuntimeLoRA (__init__.py): the reference for the live application.
NODE_REFERENCE = {
    "_pairs": r'''
def _pairs(sd, strength):
    """{module path without diffusion_model.: (A [r, in], B [out, r], alpha/rank * strength)}, tensors as stored."""
    parts = {}
    for key, tensor in sd.items():
        name = key[len("diffusion_model."):] if key.startswith("diffusion_model.") else key
        for suffix, slot in ((".lora_A.weight", "A"), (".lora_down.weight", "A"), (".lora_B.weight", "B"),
                             (".lora_up.weight", "B"), (".alpha", "alpha")):
            if name.endswith(suffix):
                parts.setdefault(name[:-len(suffix)], {})[slot] = tensor
                break
        else:
            raise ValueError(f"H3RuntimeLoRA: unexpected key {key}")
    pairs = {}
    for module, slots in parts.items():
        if "A" not in slots or "B" not in slots:
            raise ValueError(f"H3RuntimeLoRA: {module} lacks lora_A or lora_B")
        a, b = slots["A"], slots["B"]
        scale = (float(slots["alpha"]) / a.shape[0] if "alpha" in slots else 1.0) * strength
        pairs[module] = (a, b, scale)
    return pairs
''',
    "_Cast": r'''
class _Cast:
    """A LoRA pair moved once per device in its stored dtype (bf16: about 310 MB for the anti-burn LoRA), handed out in
    float32 per call (small: at most 28672 x rank), with its scale."""

    def __init__(self, a, b, scale):
        self.a, self.b, self.scale, self.on = a, b, scale, {}

    def __call__(self, device):
        hit = self.on.get(device)
        if hit is None:
            hit = self.on[device] = (self.a.to(device), self.b.to(device))
        return hit[0].float(), hit[1].float(), self.scale

    def clear(self):
        self.on.clear()
''',
    "_add_delta_": r'''
def _add_delta_(out, x, pair):
    """out += (((x @ A.T) @ B.T) * scale).to(out.dtype), float32 math, in row chunks, in place."""
    a, b, scale = pair(out.device)
    x2, o2 = x.reshape(-1, x.shape[-1]), out.view(-1, out.shape[-1])
    for r in range(0, x2.shape[0], ROWS):
        o2[r:r + ROWS] += (((x2[r:r + ROWS].float() @ a.T) @ b.T) * scale).to(o2.dtype)
''',
    "_add_swiglu_delta_": r'''
def _add_swiglu_delta_(out, h, pair):
    """out += fc2's delta on swiglu(h) = silu(gate) * up, h = [gate | up] (fc1's output), SwiGLU in float32."""
    a, b, scale = pair(out.device)
    h2, o2 = h.reshape(-1, h.shape[-1]), out.view(-1, out.shape[-1])
    half = h2.shape[-1] // 2
    for r in range(0, h2.shape[0], ROWS):
        act = F.silu(h2[r:r + ROWS, :half].float()) * h2[r:r + ROWS, half:].float()
        o2[r:r + ROWS] += (((act @ a.T) @ b.T) * scale).to(o2.dtype)
''',
}
NODE = {"torch": torch, "F": F, "ROWS": 2048}
for _src in NODE_REFERENCE.values():
    exec(_src, NODE)

# the registry's shape (the rental tooling's turbos.py), with its patch functions by name
FAKE_REGISTRY = r'''
def patch_lora_turbo(wf, turbo): return wf
def patch_hyperflow(wf, turbo): return wf
def patch_checkpoint_turbo(wf, turbo): return wf
GRID = (1.0, 0.9939097762107849, 0.9842875003814697, 0.9660636782646179, 0.9230769276618958, 0.8349423408508301,
        0.6968520283699036, 0.4687533378601074, 0.0)
TURBOS = {
    "larry": {"files": dict.fromkeys(("t2va", "fl2va", "ref2va"), "larry.safetensors"), "steps": 8,
              "shift": (12.0, 3.0), "sigmas": "simple", "sampler": "MiniMaxH3TurboSampler", "strength": 1.0,
              "patch": None},
    "lightx2v8": {"files": {"t2va": "lx_fl2v.safetensors", "fl2va": "lx_fl2v.safetensors", "ref2va": "lx_ref2v.safetensors"},
                  "steps": 8, "shift": (6.0, 3.0), "sigmas": "simple", "sampler": "euler", "strength": 1.0,
                  "patch": patch_lora_turbo},
    "hyperflow_lora": {"files": dict.fromkeys(("t2va", "fl2va", "ref2va"), "hf.safetensors"), "steps": 8,
                       "shift": (12.0, 3.0), "sigmas": GRID, "sampler": "euler", "strength": 1.0,
                       "patch": patch_lora_turbo},
    "hyperflow_refit": {"files": dict.fromkeys(("t2va", "fl2va", "ref2va"), "hfr.safetensors"), "steps": 8,
                        "shift": (12.0, 3.0), "sigmas": "file", "sampler": "euler", "strength": 1.0,
                        "patch": patch_hyperflow},
    "merged_turbo": {"files": dict.fromkeys(("t2va", "fl2va", "ref2va"), "merged.safetensors"),
              "dit": dict.fromkeys(("fl2va", "ref2va"), "merged.safetensors"), "steps": 8, "shift": (12.0, 7.0),
              "sigmas": "simple", "sampler": "euler", "strength": 1.0, "patch": patch_checkpoint_turbo},
    "pai_pdd": {"files": dict.fromkeys(("t2va", "fl2va", "ref2va"), "pdd.safetensors"), "steps": 8,
                "shift": (12.0, 3.0), "sigmas": "simple", "sampler": "euler", "strength": 1.0, "patch": None,
                "blocked": "no ComfyUI loader"},
}
'''
T_DIM, TABLE_ROWS, GRID_W = 8, 1025, 24


def tiny_params():
    return MiniMaxH3TransformerParams(hidden_size=16, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                                      attention_head_dim=16, ffn_hidden_size=32, text_dim=12, rope_inv_freq_len=2,
                                      time_embed_dim=T_DIM, adaln_t_table_size=TABLE_ROWS)


def tiny_transformer(seed=0):
    torch.manual_seed(seed)
    t = MiniMaxH3Transformer(tiny_params())
    with torch.no_grad():
        u = torch.linspace(0, 1, TABLE_ROWS).view(-1, 1)
        t.adaln_t_table.copy_(torch.sin(u * torch.arange(1, T_DIM + 1) * 2.0) * 0.5)
    return t


def tiny_inputs():
    tags = torch.ones(5, dtype=torch.long)
    layout = build_packed_sequence(text_token_tags=tags, num_latent_frames=2, latent_height=2, latent_width=2,
                                   num_audio_latents=3)
    s = layout.sequence_length
    row_t = torch.full((1, s), 0.3684)
    row_t[0, layout.audio_indices] = 0.5
    g = torch.Generator().manual_seed(9)
    return dict(hidden_states=torch.randn(1, 2, 96, generator=g), audio_hidden_states=torch.randn(1, 6, 32, generator=g),
                encoder_hidden_states=torch.randn(1, 5, 12, generator=g), row_timesteps=row_t,
                token_tags=layout.token_tags[None], position_ids=layout.position_ids[None],
                video_indices=layout.video_indices, audio_indices=layout.audio_indices, text_indices=layout.text_indices)


BACKBONE_SHAPES = {"attn.qkv_proj": (96, 16), "attn.out_proj": (16, 32), "mlp.fc1": (64, 16), "mlp.fc2": (16, 32)}
BLOCKS = ["blocks.0", "blocks.1", "token_refiner.blocks.0"]


def lora_file(rank=4, seed=1, prefix="diffusion_model.", alpha=True, adaln=False, down_up=False, scale=0.3):
    """A turbo LoRA over the tiny model in ComfyUI's generic layout (lightx2v / hyperflow: prefix + alpha), Larry's
    (no prefix, no alpha, adaLN pieces) or lora_down / lora_up."""
    g = torch.Generator().manual_seed(seed)
    a_name, b_name = (".lora_down.weight", ".lora_up.weight") if down_up else (".lora_A.weight", ".lora_B.weight")
    sd = {}
    for block in BLOCKS:
        for kind, (out, inp) in BACKBONE_SHAPES.items():
            key = prefix + f"{block}.{kind}"
            sd[key + a_name] = (torch.randn(rank, inp, generator=g) * scale).to(torch.bfloat16)
            sd[key + b_name] = (torch.randn(out, rank, generator=g) * scale).to(torch.bfloat16)
            if alpha:
                sd[key + ".alpha"] = torch.tensor(float(rank) * 0.5)
    if adaln:
        for path, out in (("blocks.0.adaln_proj", 288), ("blocks.1.adaln_proj", 288), ("final_layer.adaln_proj", 32)):
            sd[prefix + path + ".linear.lora_A.weight"] = (torch.randn(2, GRID_W, generator=g) * 0.2).to(torch.bfloat16)
            sd[prefix + path + ".linear.lora_B.weight"] = (torch.randn(out, 2, generator=g) * 0.2).to(torch.bfloat16)
    return sd


def grid(seed=3):
    return torch.randn(TABLE_ROWS, GRID_W, generator=torch.Generator().manual_seed(seed)) * 0.3


def turbo(name="lightx2v8", adaln=False, strength=1.0, shifts=(6.0, 3.0), sigmas=None):
    return tb.Turbo(name=name, kind="lora", file=name + ".safetensors", strength=strength, shift_video=shifts[0],
                    shift_audio=shifts[1], steps=8, sigmas=sigmas, adaln=adaln)


def bf16_block(t, path):
    """The block's linears in bf16, as the trainer runs them."""
    block = t.get_submodule(path)
    block.to(torch.bfloat16)
    return block


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.registry = os.path.join(self.dir, "turbos.py")
        with open(self.registry, "w") as f:
            f.write(FAKE_REGISTRY)

    def tearDown(self):
        self.tmp.cleanup()


class RegistryContracts(TempDir):
    def test_kinds_files_and_schedules_per_partition(self):
        reg = tb.load_registry(self.registry)
        lx = tb.resolve(reg, "lightx2v8", "ref2va_pruned")
        self.assertEqual((lx.kind, lx.file, lx.shifts, lx.sigmas, lx.adaln), ("lora", "lx_ref2v.safetensors", (6.0, 3.0), None, False))
        self.assertEqual(tb.resolve(reg, "lightx2v8", "fl2va_pruned").file, "lx_fl2v.safetensors", "per-partition file")
        larry = tb.resolve(reg, "larry", "ref2va")
        self.assertEqual((larry.kind, larry.adaln, larry.shifts), ("lora", True, (12.0, 3.0)))
        hf = tb.resolve(reg, "hyperflow_lora", "fl2va")
        self.assertEqual(len(hf.sigmas), 9)
        self.assertEqual((hf.sigmas[0], hf.sigmas[-1]), (1.0, 0.0))
        merged = tb.resolve(reg, "merged_turbo", "ref2va_pruned")
        self.assertEqual((merged.kind, merged.file, merged.shifts, merged.adaln), ("checkpoint", "merged.safetensors", (12.0, 7.0), False))
        for bad in ("hyperflow_refit", "pai_pdd", "nope"):
            self.assertRaises(ValueError, tb.resolve, reg, bad, "fl2va")
        self.assertRaises(ValueError, tb.resolve, reg, "larry", "fasth3")
        self.assertRaises(ValueError, tb.resolve_all, reg, ["merged_turbo", "larry"], "fl2va")  # a checkpoint runs alone
        self.assertRaises(ValueError, tb.names_of, ["larry", "larry"])

    def test_config_keys(self):
        kw = {"turbo": ["larry", "lightx2v8"], "turbo_registry": self.registry, "turbo_lora_dir": "/loras",
              "turbo_paths": {"larry": "/elsewhere/l.safetensors"}}
        turbos = tb.configured(kw, "ref2va_pruned")
        self.assertEqual(list(turbos), ["larry", "lightx2v8"], "config order")
        self.assertEqual(tb.file_path(turbos["lightx2v8"], kw), "/loras/lx_ref2v.safetensors")
        self.assertEqual(tb.file_path(turbos["larry"], kw), "/elsewhere/l.safetensors")
        self.assertIsNone(tb.configured({}, "fl2va"))
        self.assertRaises(ValueError, tb.configured, {"turbo_lora_dir": "/x"}, "fl2va")  # a turbo key without a turbo
        for stale in ({"assistant_adaln_lora_path": "/a"}, {"assistant_adaln_grid_path": "/g"}):
            self.assertRaises(ValueError, tb.configured, {**kw, **stale}, "fl2va")
        self.assertRaises(ValueError, tb.configured, kw, "fl2va", "/assistant.safetensors")
        self.assertRaises(ValueError, tb.configured, {**kw, "turbo_paths": {"merged_turbo": "/e"}}, "fl2va")
        self.assertRaises(ValueError, tb.configured, {**kw, "turbo_sample": "merged_turbo"}, "fl2va")
        self.assertRaises(ValueError, tb.file_path, turbos["lightx2v8"], {})
        self.assertRaises(FileNotFoundError, tb.configured, {"turbo": "larry"}, "fl2va")  # no registry

    def test_sampling_schedules(self):
        reg = tb.load_registry(self.registry)
        sv, sa = sampling_sigmas(None, 8)
        lx_v, lx_a = sampling_sigmas(tb.resolve(reg, "lightx2v8", "fl2va"), 8)
        self.assertTrue(torch.allclose(lx_v, 6 * torch.linspace(1, 0, 9) / (1 + 5 * torch.linspace(1, 0, 9)), atol=1e-6))
        self.assertTrue(torch.equal(lx_a, remap_sigma(lx_v, 6.0, 3.0)), "the audio clock of the turbo's shifts")
        self.assertFalse(torch.allclose(lx_v, sv))
        hf = tb.resolve(reg, "hyperflow_lora", "fl2va")
        with contextlib.redirect_stdout(io.StringIO()):
            hv, _ = sampling_sigmas(hf, 4)
        self.assertTrue(torch.equal(hv, torch.tensor(hf.sigmas, dtype=torch.float32)), "a fixed grid is used as is")


class KeyMappingContracts(TempDir):
    """Every tensor of a turbo file lands on a trainer module; anything that would not is an error."""

    def test_every_tensor_maps_in_each_layout(self):
        t = tiny_transformer()
        for kwargs, turbo_kw in ((dict(), {}), (dict(prefix="", alpha=False, adaln=True), dict(adaln=True)),
                                 (dict(down_up=True), {})):
            sd = lora_file(**kwargs)
            s = rt.TurboSet(t)
            p = s.add(turbo(**turbo_kw), sd, grid() if turbo_kw.get("adaln") else None)
            self.assertEqual(p.tensors, len(sd))
            self.assertEqual(len(p.backbone), 12)
            self.assertEqual(len(p.adaln), 3 if turbo_kw.get("adaln") else 0)
            self.assertEqual(set(s.pairs["lightx2v8"]), {f"{b}.{k}" for b in BLOCKS for k in BACKBONE_SHAPES})
            s.remove()

    def test_scale_is_the_nodes(self):
        sd = lora_file(rank=4)
        node = NODE["_pairs"](sd, 0.7)
        p = tb.plan({k: tuple(v.shape) for k, v in sd.items()}, rt.TurboSet(tiny_transformer()).expected)
        mine = rt.make_pairs(sd, p, 0.7)
        self.assertEqual(set(node), set(mine))
        for module, (a, b, scale) in node.items():
            self.assertEqual(mine[module].scale, scale)
            self.assertTrue(mine[module].a is a and mine[module].b is b, "the stored tensors, untouched")
        larry = lora_file(prefix="", alpha=False)
        p = tb.plan({k: tuple(v.shape) for k, v in larry.items()}, rt.TurboSet(tiny_transformer()).expected)
        self.assertTrue(all(pair.scale == 1.0 for pair in rt.make_pairs(larry, p, 1.0).values()), "no alpha: B @ A")

    def test_nothing_is_skipped(self):
        t = tiny_transformer()
        expected = rt.TurboSet(t).expected
        shapes = lambda sd: {k: tuple(v.shape) for k, v in sd.items()}
        good = lora_file()
        cases = {
            "an unknown tensor": {**good, "diffusion_model.blocks.0.attn.qkv_proj.dora_scale": torch.zeros(96)},
            "B without A": {k: v for k, v in good.items() if k != "diffusion_model.blocks.0.mlp.fc1.lora_A.weight"},
            "a module the model lacks": {**good, "diffusion_model.blocks.7.mlp.fc1.lora_A.weight": torch.zeros(4, 16),
                                         "diffusion_model.blocks.7.mlp.fc1.lora_B.weight": torch.zeros(64, 4)},
            "a wrong shape": {**good, "diffusion_model.blocks.0.mlp.fc2.lora_A.weight": torch.zeros(4, 16)},
            "a non-backbone linear": {**good, "diffusion_model.condition_proj.lora_A.weight": torch.zeros(4, 12),
                                      "diffusion_model.condition_proj.lora_B.weight": torch.zeros(16, 4)},
            "A and lora_down together": {**good, "diffusion_model.blocks.0.mlp.fc1.lora_down.weight": torch.zeros(4, 16)},
        }
        for why, sd in cases.items():
            with self.subTest(why):
                self.assertRaises(ValueError, tb.plan, shapes(sd), expected, "x")
        with_alpha = lora_file(adaln=True)
        self.assertRaises(ValueError, tb.plan, shapes({**with_alpha, "diffusion_model.blocks.0.adaln_proj.linear.alpha":
                                                       torch.tensor(2.0)}), expected, "x", GRID_W)
        s = rt.TurboSet(t)
        self.assertRaises(ValueError, s.add, turbo(), lora_file(adaln=True), grid())  # H3RuntimeLoRA refuses adaLN
        self.assertRaises(ValueError, s.add, turbo("larry", adaln=True), lora_file(prefix="", alpha=False), grid())
        self.assertRaises(ValueError, s.add, turbo("larry", adaln=True), lora_file(prefix="", alpha=False, adaln=True))

    def test_full_size_layout(self):
        """h3_expected (the header-only check's table) is the real pruned H3's module shapes, checked on a
        full-size model built on the meta device (nothing allocated)."""
        with torch.device("meta"):
            full = MiniMaxH3Transformer(MiniMaxH3TransformerParams(adaln_t_table_size=1025, time_embed_dim=8))
        expected = rt.TurboSet(full).expected
        paths = [f"{b}.{k}" for b in [f"blocks.{i}" for i in range(50)] + ["token_refiner.blocks.0", "token_refiner.blocks.1"]
                 for k in tb.BACKBONE]
        paths += [f"blocks.{i}.adaln_proj.linear" for i in range(50)] + ["final_layer.adaln_proj.linear"]
        for path in paths:
            self.assertEqual(tb.h3_expected(path), expected(path), path)
        for path in ("blocks.50.mlp.fc1", "token_refiner.blocks.2.attn.qkv_proj", "blocks.0.attn.to_gate_compress"):
            self.assertIsNone(tb.h3_expected(path))

    def test_damaged_files_are_named(self):
        """Random bytes or a truncated copy: a clear error naming the file, not an OverflowError from the header."""
        junk = os.path.join(self.dir, "junk.safetensors")
        with open(junk, "wb") as f:
            f.write(os.urandom(4096))
        good = os.path.join(self.dir, "good.safetensors")
        save_file({"a": torch.zeros(4, 4), "b": torch.ones(3)}, good, metadata={"k": "v"})
        header, meta = tb.read_header(good)
        self.assertEqual((sorted(header), meta), (["a", "b"], {"k": "v"}))
        cut = os.path.join(self.dir, "cut.safetensors")
        with open(good, "rb") as f, open(cut, "wb") as g:
            g.write(f.read()[:-10])
        tiny = os.path.join(self.dir, "tiny.safetensors")
        with open(tiny, "wb") as f:
            f.write(b"abc")
        for path, words in ((junk, "not a safetensors file"), (cut, "truncated"), (tiny, "not a safetensors file")):
            with self.assertRaises(ValueError) as caught:
                tb.read_header(path)
            self.assertIn(words, str(caught.exception))
            self.assertIn(path, str(caught.exception))
        self.assertRaises(ValueError, tb.check_lora_file, turbo(), junk)

    def test_real_files(self):
        """Header-only: every tensor of each registry turbo's file maps onto the pruned H3 (H3_TURBO_REGISTRY +
        H3_TURBO_LORA_DIR; a <file>.json header stands in for a file)."""
        registry, folder = os.environ.get("H3_TURBO_REGISTRY"), os.environ.get("H3_TURBO_LORA_DIR")
        if not (registry and folder):
            self.skipTest("set H3_TURBO_REGISTRY and H3_TURBO_LORA_DIR to check the real turbo files")
        reg = tb.load_registry(registry)
        checked = 0
        for name in ("larry", "lightx2v8", "hyperflow_lora"):
            for partition in ("fl2va", "ref2va"):
                t = tb.resolve(reg, name, partition)
                path = os.path.join(folder, t.file)
                if not os.path.exists(path) and os.path.exists(path + ".json"):
                    with open(path + ".json") as f:
                        header = {k: v for k, v in json.load(f).items() if k != "__metadata__"}
                    p = tb.plan({k: v["shape"] for k, v in header.items()}, tb.h3_expected, name)
                    tb.check_adaln(t, p)
                    self.assertEqual(p.tensors, len(header))
                elif os.path.exists(path):
                    grid_path = os.path.join(folder, "h3_silu_temb_grid.safetensors")
                    p = tb.check_lora_file(t, path, grid_path if os.path.exists(grid_path) else None) \
                        if not t.adaln or os.path.exists(grid_path) else None
                    if p is None:
                        header = tb.read_header(path)[0]
                        p = tb.plan({k: v["shape"] for k, v in header.items()}, tb.h3_expected, name)
                        tb.check_adaln(t, p)
                else:
                    continue
                with self.subTest(turbo=name, partition=partition):
                    self.assertEqual(len(p.backbone), 208, "50 blocks + 2 refiner blocks x 4 linears")
                    self.assertEqual(len(p.adaln), 51 if t.adaln else 0)
                checked += 1
        if not checked:
            self.skipTest(f"no registry turbo file in {folder}")


class NodeMathContracts(unittest.TestCase):
    """The live application against the node's functions, on real trainer modules in bf16."""

    def setUp(self):
        self.t = tiny_transformer()
        self.sd = lora_file(rank=4, seed=5, scale=0.5)
        self.node = {m: NODE["_Cast"](*v) for m, v in NODE["_pairs"](self.sd, 0.8).items()}
        g = torch.Generator().manual_seed(11)
        self.x = (torch.randn(1, 4500, 16, generator=g)).to(torch.bfloat16)  # three row chunks

    def test_reference_is_the_nodes_source(self):
        node_dir = os.environ.get("H3_RUNTIME_LORA_DIR")
        if not node_dir:
            self.skipTest("set H3_RUNTIME_LORA_DIR to compare the reference with the node's source")
        with open(os.path.join(node_dir, "__init__.py")) as f:
            src = f.read()
        tree = ast.parse(src)
        for name, ref in NODE_REFERENCE.items():
            fn = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
            self.assertEqual(ast.get_source_segment(src, fn), ref.strip("\n"), name)
        self.assertIn("ROWS = 2048 ", src)
        self.assertEqual(rt.ROWS, 2048)

    def test_attention_and_mlp_match_the_node_bit_for_bit(self):
        block = bf16_block(self.t, "blocks.1")
        plain = {}
        with torch.no_grad():
            for kind in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1"):
                linear = block.get_submodule(kind)
                x = self.x if linear.in_features == 16 else torch.randn(1, 4500, 32).to(torch.bfloat16)
                out = linear(x)
                NODE["_add_delta_"](out, x, self.node[f"blocks.1.{kind}"])
                plain[kind] = (x, out)
            # the node's MLP forward, with the trainer's own SwiGLU + fc2 for the base
            h = block.mlp.fc1(self.x)
            NODE["_add_delta_"](h, self.x, self.node["blocks.1.mlp.fc1"])
            gate, up = h.chunk(2, dim=-1)
            mlp_out = block.mlp.fc2(F.silu(gate) * up)
            NODE["_add_swiglu_delta_"](mlp_out, h, self.node["blocks.1.mlp.fc2"])
        s = rt.TurboSet(self.t)
        s.add(turbo(strength=0.8), self.sd)
        s.activate("lightx2v8")
        with torch.no_grad():
            for kind, (x, out) in plain.items():
                self.assertTrue(torch.equal(block.get_submodule(kind)(x), out), kind)
            self.assertTrue(torch.equal(block.mlp(self.x), mlp_out), "MLP: fc1 delta before the SwiGLU, fc2 on fp32 SwiGLU")
            s.activate(None)
            base = block.mlp(self.x)
        self.assertFalse(torch.equal(base, mlp_out))

    def test_gradients_are_the_float32_chain(self):
        a = torch.randn(4, 16, dtype=torch.bfloat16)
        b = torch.randn(24, 4, dtype=torch.bfloat16)
        x = torch.randn(3000, 16, dtype=torch.bfloat16, requires_grad=True)
        g = torch.randn(3000, 24, dtype=torch.bfloat16)
        rt.lora_delta(x, a, b, 0.37, torch.bfloat16).backward(g)
        xr = x.detach().float().requires_grad_(True)
        ((xr @ a.float().T) @ b.float().T * 0.37).backward(g.float())
        self.assertLess((x.grad.float() - xr.grad).abs().max().item(), 1e-2 * xr.grad.abs().max().item())
        h = torch.randn(3000, 64, dtype=torch.bfloat16, requires_grad=True)
        a2, b2 = torch.randn(4, 32, dtype=torch.bfloat16), torch.randn(16, 4, dtype=torch.bfloat16)
        g2 = torch.randn(3000, 16, dtype=torch.bfloat16)
        rt.swiglu_delta(h, a2, b2, 0.5, torch.bfloat16).backward(g2)
        hr = h.detach().float().requires_grad_(True)
        ((F.silu(hr[:, :32]) * hr[:, 32:]) @ a2.float().T @ b2.float().T * 0.5).backward(g2.float())
        diff = (h.grad.float() - hr.grad).abs().max().item()
        self.assertLess(diff, 1e-2 * hr.grad.abs().max().item())

    def test_training_graph(self):
        """With autograd on (the trained LoRA upstream), the turbo adds its delta and its gradient path."""
        block = bf16_block(self.t, "blocks.0").requires_grad_(False)
        grads = {}
        for live in (None, "lightx2v8"):
            s = rt.TurboSet(self.t)
            s.add(turbo(), self.sd)
            s.activate(live)
            x = self.x[:, :100].clone().requires_grad_(True)
            block.mlp(block.attn.out_proj(block.attn.qkv_proj(x)[..., :32])).float().pow(2).mean().backward()
            grads[live] = x.grad
            s.remove()
        self.assertTrue(all(g is not None for g in grads.values()))
        self.assertFalse(torch.equal(grads[None], grads["lightx2v8"]), "the turbo's delta is in the gradient")


class SwitchingContracts(TempDir):
    def test_one_live_turbo_at_a_time(self):
        t = tiny_transformer()
        inputs = tiny_inputs()
        with torch.no_grad():
            base = t(**inputs)
        singles = {}
        for name, sd, adaln in (("lightx2v8", lora_file(seed=2), False), ("larry", lora_file(seed=3, prefix="", alpha=False, adaln=True), True)):
            one = rt.TurboSet(t)
            one.add(turbo(name, adaln=adaln), sd, grid() if adaln else None)
            one.activate(name)
            with torch.no_grad():
                singles[name] = t(**inputs)
            one.remove()
        with torch.no_grad():
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(t(**inputs), base)), "remove() restores the base")
        both = rt.TurboSet(t)
        both.add(turbo("lightx2v8"), lora_file(seed=2))
        both.add(turbo("larry", adaln=True), lora_file(seed=3, prefix="", alpha=False, adaln=True), grid())
        self.assertEqual(both.adaln, {"larry": 3})
        for name in ("larry", "lightx2v8", None, "larry"):
            both.activate(name)
            with torch.no_grad():
                out = t(**inputs)
            want = base if name is None else singles[name]
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(out, want)), f"live: {name}")
            if name != "lightx2v8":
                self.assertTrue(all(not p.on for p in both.pairs["lightx2v8"].values()), "inactive: no device copy")
        both.keep_on_device = True
        both.activate("lightx2v8")
        with torch.no_grad():
            t(**inputs)
        both.activate("larry")
        self.assertTrue(all(p.on for p in both.pairs["lightx2v8"].values()), "keep_on_device keeps them")
        self.assertRaises(ValueError, both.activate, "hyperflow_lora")
        self.assertRaises(ValueError, both.add, turbo("larry", adaln=True), lora_file(prefix="", alpha=False, adaln=True), grid())


def tag_recording(path, block):
    """The recorder's header-only rewrite: the same tensors, info["turbo"] added."""
    with safe_open(path, "pt") as f:
        meta = dict(f.metadata())
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    info = json.loads(meta["info"])
    if block is not None:
        info["turbo"] = block
    if isinstance(block, dict) and "shift" in block:
        info["schedule"]["shift_video"], info["schedule"]["shift_audio"] = block["shift"]
    meta["info"] = json.dumps(info)
    save_file(tensors, path, metadata=meta)


def block_for(name, reg_path, partition="fl2va"):
    t = tb.resolve(tb.load_registry(reg_path), name, partition)
    return {"name": name, "apply": "runtime", "lora_file": t.file, "strength": t.strength, "shift": list(t.shifts),
            "steps": t.steps, "sigmas": list(t.sigmas) if t.sigmas else "simple", "sampler": "euler"}


class RecordingContracts(TempDir):
    def setUp(self):
        super().setUp()
        self.turbos = tb.resolve_all(tb.load_registry(self.registry), ["larry", "lightx2v8", "hyperflow_lora"], "fl2va")

    def info(self, **block):
        return {"schedule": {"shift_video": block.get("shift", [12.0, 3.0])[0], "shift_audio": block.get("shift", [12.0, 3.0])[1]},
                **({"turbo": block} if block else {})}

    def test_tags_and_checks(self):
        lx = block_for("lightx2v8", self.registry)
        self.assertEqual(tb.check_recording("/r/A_lightx2v8_001.safetensors", self.info(**lx), self.turbos), "lightx2v8")
        self.assertEqual(tb.check_recording("/r/x.safetensors", {**self.info(), "turbo": "larry"}, self.turbos), "larry")
        hf = block_for("hyperflow_lora", self.registry)
        self.assertEqual(tb.check_recording("/r/x.safetensors", self.info(**hf), self.turbos), "hyperflow_lora")
        # the stem when nothing else says (A_<turbo>_<idx>; the longest name wins)
        self.assertEqual(tb.recording_turbo("/r/A_hyperflow_lora_007.safetensors", {}, self.turbos), "hyperflow_lora")
        bad = {
            "untagged in a mix": ("/r/clip.safetensors", self.info()),
            "a turbo the run does not list": ("/r/x.safetensors", {**self.info(), "turbo": "merged_turbo"}),
            "shifts": ("/r/x.safetensors", {**self.info(**lx), "schedule": {"shift_video": 12.0, "shift_audio": 3.0}}),
            "the other partition's file": ("/r/x.safetensors", self.info(**{**lx, "lora_file": "lx_ref2v.safetensors"})),
            "strength": ("/r/x.safetensors", self.info(**{**lx, "strength": 0.8})),
            "a fixed grid for a simple turbo": ("/r/x.safetensors", self.info(**{**lx, "sigmas": hf["sigmas"]})),
            "simple for a fixed-grid turbo": ("/r/x.safetensors", self.info(**{**hf, "sigmas": "simple"})),
        }
        for why, (path, info) in bad.items():
            with self.subTest(why):
                self.assertRaises(ValueError, tb.check_recording, path, info, self.turbos)
        one = {"larry": self.turbos["larry"]}
        self.assertEqual(tb.check_recording("/r/clip.safetensors", self.info(), one), "larry", "untagged, one turbo")
        with open(os.path.join(self.dir, "B_7.json"), "w") as f:
            json.dump({"turbo": {"name": "lightx2v8"}}, f)
        self.assertEqual(tb.recording_turbo(os.path.join(self.dir, "B_7.safetensors"), {}, self.turbos), "lightx2v8")


def header_only(path, shapes, metadata=None):
    """A safetensors file with only its header (the header-only checks never read tensor data)."""
    header, offset = {}, 0
    for key, shape in shapes.items():
        n = 2 * math.prod(shape)
        header[key] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.truncate(8 + len(blob) + offset)  # the data as a sparse hole: the size the header describes, no disk


def h3_lora_shapes(rank, prefix="diffusion_model.", alpha=True, adaln=False):
    """A turbo LoRA's tensor shapes over the full pruned H3 (208 backbone linears; Larry: 51 adaLN pieces too)."""
    shapes = {}
    for block in [f"blocks.{i}" for i in range(50)] + ["token_refiner.blocks.0", "token_refiner.blocks.1"]:
        for kind, (out, inp) in tb.H3_LINEARS.items():
            key = f"{prefix}{block}.{kind}"
            shapes[key + ".lora_A.weight"], shapes[key + ".lora_B.weight"] = (rank, inp), (out, rank)
            if alpha:
                shapes[key + ".alpha"] = ()
    if adaln:
        for block, out in [(f"blocks.{i}", 96768) for i in range(50)] + [("final_layer", 10752)]:
            shapes[f"{prefix}{block}.adaln_proj.linear.lora_A.weight"] = (16, 2688)
            shapes[f"{prefix}{block}.adaln_proj.linear.lora_B.weight"] = (out, 16)
    return shapes


class TrajectoryTurboContracts(TempDir):
    """Each trajectory sample runs with its own recording's turbo live."""

    def setUp(self):
        super().setUp()
        self.rec = os.path.join(self.dir, "rec")
        os.makedirs(self.rec)
        self.paths = {}
        for i, name in enumerate(("larry", "lightx2v8")):
            stem = f"A_{name}_{i:03d}"
            path = tt.record(self.rec, stem, seed=i)[0]
            tag_recording(path, block_for(name, self.registry))
            self.paths[name] = stem

    def model(self, **kw):
        m = tt.trainer_model(self.rec, turbo=["larry", "lightx2v8"], turbo_registry=self.registry,
                             turbo_lora_dir=self.dir, continuation_trajectory_anchor_share=1.0,
                             continuation_trajectory_noised_share=0.0, **kw)
        m._turbo_config()
        log = []
        m.turbo_set = types.SimpleNamespace(activate=lambda name: log.append(name), active=None)
        return m, log

    def test_sample_assistant_is_its_recording_turbo(self):
        m, log = self.model()
        for name, stem in self.paths.items():
            tt.force_draws(m, "anchor", 2)
            batch = tt.Batch(stem)
            with contextlib.redirect_stdout(io.StringIO()):
                tt.predict(m, batch)
            self.assertEqual(batch.trajectory_samples[0].turbo, name)
            self.assertEqual(log[-1], name, "activated before the forward")
            self.assertEqual(m.turbo_active, name)
            self.assertEqual(m.stream_shifts(), (12.0, 3.0) if name == "larry" else (6.0, 3.0))
        report = m._trajectory_store.stats.report()
        self.assertIn("lightx2v8", report)
        self.assertIn(("anchor", 2, "larry"), m._trajectory_store.stats.rows)

    def test_mixed_batches_are_refused(self):
        m, _ = self.model()
        tt.force_draws(m, "anchor", 0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertRaises(ValueError, tt.predict, m, tt.Batch(*self.paths.values()), 2)

    def test_dataset_check_counts_turbos(self):
        m, _ = self.model()
        folder = os.path.join(self.dir, "data")
        os.makedirs(folder)
        for stem in self.paths.values():
            open(os.path.join(folder, stem + ".mp4"), "w").close()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            m.check_dataset_config(types.SimpleNamespace(folder_path=folder))
        self.assertIn("larry 1, lightx2v8 1", out.getvalue())
        untagged = tt.record(self.rec, "clip_9", seed=9)[0]
        open(os.path.join(folder, "clip_9.mp4"), "w").close()
        m2, _ = self.model()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertRaises(ValueError, m2.check_dataset_config, types.SimpleNamespace(folder_path=folder))
        os.remove(untagged)

    def test_check_model_kwargs_and_cli(self):
        """The pre-run check tooling calls (no trainer, header only): files, adaLN grid, every recording's turbo."""
        header_only(os.path.join(self.dir, "larry.safetensors"), h3_lora_shapes(64, prefix="", alpha=False, adaln=True))
        header_only(os.path.join(self.dir, "lx_fl2v.safetensors"), h3_lora_shapes(128))
        header_only(os.path.join(self.dir, "hf.safetensors"), h3_lora_shapes(256))
        header_only(os.path.join(self.dir, "grid.safetensors"), {tb.GRID_KEY: (1025, 2688)})
        folder = os.path.join(self.dir, "data")
        os.makedirs(folder)
        for stem in self.paths.values():
            open(os.path.join(folder, stem + ".mp4"), "w").close()
        kw = {"turbo": ["larry", "lightx2v8", "hyperflow_lora"], "turbo_registry": self.registry,
              "turbo_lora_dir": self.dir, "turbo_adaln_grid_path": os.path.join(self.dir, "grid.safetensors"),
              "continuation_trajectory_dir": self.rec}
        summary = tb.check_model_kwargs(kw, "fl2va_pruned", folder)
        self.assertIn("larry: larry.safetensors (208 live linears, 51 adaLN pieces)", summary)
        self.assertIn("2 recordings: larry 1, lightx2v8 1, hyperflow_lora 0", summary)
        self.assertRaises(FileNotFoundError, tb.check_model_kwargs, {**kw, "turbo_adaln_grid_path": None}, "fl2va")
        header_only(os.path.join(self.dir, "stripped.safetensors"), h3_lora_shapes(64, prefix="", alpha=False))
        self.assertRaises(ValueError, tb.check_model_kwargs,
                          {**kw, "turbo_paths": {"larry": os.path.join(self.dir, "stripped.safetensors")}}, "fl2va")
        self.assertRaises(ValueError, tb.check_model_kwargs, kw, "fl2va", folder, "/assistant.safetensors")
        config = os.path.join(self.dir, "config.yaml")
        with open(config, "w") as f:
            json.dump({"config": {"process": [{"model": {"model_kwargs": {**kw, "partition": "fl2va_pruned"}},
                                              "datasets": [{"folder_path": folder}]}]}}, f)  # JSON is YAML
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(tb.main([config]), 0)
        self.assertIn("hyperflow_lora: hf.safetensors (208 live linears)", out.getvalue())

    def test_floor_weighting_scales_each_loss(self):
        floors = {"larry": {"video": [0.04, 0.129, 0.166, 0.152, 0.116, 0.095, 0.062, 0.039],
                            "audio": [0.0167, 0.0311, 0.0377, 0.0517, 0.0453, 0.037, 0.0325, 0.0246]},
                  "lightx2v8": {"video": [0.05] * 8, "audio": [0.02] * 8}}
        weights = traj.inverse_variance_weights(floors["larry"]["video"])
        self.assertAlmostEqual(sum(weights) / 8, 1.0, places=9)
        self.assertGreater(weights[0], weights[2])
        results = {}
        for key, extra in (("plain", {}), ("floors", {"continuation_trajectory_step_floors": floors})):
            m, _ = self.model(**extra)
            m.model.lora.data.fill_(0.01)
            tt.force_draws(m, "anchor", 2)
            batch = tt.Batch(self.paths["larry"])
            with contextlib.redirect_stdout(io.StringIO()):
                pred = tt.predict(m, batch)
            target = m.get_loss_target(noise=torch.zeros((1,) + tt.VIDEO[1:]), batch=batch)
            results[key] = (pred.tensor - target, pred.audio - pred.audio_target)
        wv, wa = traj.inverse_variance_weights(floors["larry"]["video"])[2], traj.inverse_variance_weights(floors["larry"]["audio"])[2]
        self.assertTrue(torch.allclose(results["floors"][0], results["plain"][0] * math.sqrt(wv), rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.allclose(results["floors"][1], results["plain"][1] * math.sqrt(wa), rtol=1e-4, atol=1e-5))
        self.assertGreater(results["plain"][0].abs().max().item(), 1e-3)
        for bad in ({"larry": {"video": [0.1] * 8}}, {"larry": {"video": [0.1] * 8, "audio": [0.1] * 7}},
                    {"larry": {"video": [0.0] * 8, "audio": [0.1] * 8}}):
            self.assertRaises(ValueError, traj.parse_step_floors, bad)
        m, _ = self.model(continuation_trajectory_step_floors={"larry": floors["larry"]})
        tt.force_draws(m, "anchor", 2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertRaises(ValueError, tt.predict, m, tt.Batch(self.paths["lightx2v8"]))  # no entry for its turbo

    def test_measured_floors_paste_back(self):
        d = traj.Diagnostics()
        for step in range(8):
            for kind in ("anchor", "continuation"):
                d.add(kind, step, 1.0, {"video_mse": 0, "video_rel": 0, "audio_mse": 0, "audio_rel": 0,
                                        "video_floor": 0.1 + step / 100, "audio_floor": 0.02}, "larry")
        line = d.report().splitlines()[-1]
        self.assertTrue(line.startswith("measured floors (continuation_trajectory_step_floors): "))
        parsed = traj.parse_step_floors(json.loads(line.split(": ", 1)[1]))
        self.assertEqual(parsed[0][0], "larry")
        self.assertAlmostEqual(parsed[0][1][3], 0.13)


class FlowMatchTurboContracts(TempDir):
    def model(self, weights):
        m = object.__new__(h3m.MinimaxH3Model)
        m.model_config = types.SimpleNamespace(model_kwargs={"turbo": ["larry", "lightx2v8"], "turbo_registry": self.registry,
                                                             "turbo_lora_dir": self.dir, "turbo_weights": weights})
        m.noise_scheduler = m._train_scheduler()
        m.noise_scheduler.set_train_timesteps(1000, device="cpu", timestep_type="shift")
        return m

    def test_step_turbo_and_its_shift(self):
        m = self.model({"larry": 0.0, "lightx2v8": 1.0})
        self.assertEqual(float(m.noise_scheduler.shift), 12.0, "the first turbo's shift")
        g = torch.Generator().manual_seed(0)
        latents, noise = torch.randn(2, 24, 3, 4, 4, generator=g), torch.randn(2, 24, 3, 4, 4, generator=g)
        timesteps = torch.tensor([900.0, 400.0])
        s = (timesteps / 1000).view(-1, 1, 1, 1, 1)
        noisy = (1 - s) * latents + s * noise
        batch = types.SimpleNamespace(latents=latents)
        new_noisy, new_noise, new_t = m.prepare_training_step(batch, noisy, noise, timesteps)
        self.assertEqual(m.turbo_active, "lightx2v8")
        sigma = remap_sigma(timesteps / 1000, 12.0, 6.0)
        self.assertTrue(torch.allclose(new_t, sigma * 1000))
        s2 = sigma.view(-1, 1, 1, 1, 1)
        self.assertTrue(torch.allclose(new_noisy, (1 - s2) * latents + s2 * noise, atol=1e-6))
        self.assertTrue(new_noise is noise)
        self.assertEqual(m.stream_shifts(), (6.0, 3.0), "the audio clock follows the step's turbo")
        m2 = self.model({"larry": 1.0, "lightx2v8": 0.0})
        self.assertIsNone(m2.prepare_training_step(batch, noisy, noise, timesteps), "same shift: nothing moves")
        self.assertEqual(m2.turbo_active, "larry")


class CheckpointTurboContracts(TempDir):
    def test_quantization_metadata_becomes_markers(self):
        """A DiT that declares its int8 layers in the file metadata ((a merged-turbo checkpoint)) gets the per-layer markers ComfyUI
        builds (convert_old_quants); its unused adaLN basis is dropped like ComfyUI's unexpected keys."""
        layers = {"blocks.0.attn.qkv_proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}}
        path = os.path.join(self.dir, "merged.safetensors")
        save_file({"blocks.0.attn.qkv_proj.weight": torch.zeros(4, 4, dtype=torch.int8),
                   "blocks.0.attn.qkv_proj.weight_scale": torch.ones(4, 1), "adaln_basis": torch.zeros(8, 4),
                   "adaln_mean": torch.zeros(4)}, path,
                  metadata={"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})})
        seen = {}
        capture = classmethod(lambda cls, sd, dtype, **kw: seen.setdefault("sd", sd))
        with mock.patch.object(MiniMaxH3Transformer, "load_from_state_dict", capture):
            MiniMaxH3Transformer._load_single_file(path, torch.bfloat16)
        marker = seen["sd"]["blocks.0.attn.qkv_proj.comfy_quant"]
        self.assertEqual(marker.dtype, torch.uint8)
        self.assertEqual(bytes(marker.tolist()), json.dumps(layers["blocks.0.attn.qkv_proj"]).encode("utf-8"))
        converted = MiniMaxH3Transformer.convert_state_dict_on_load(dict(seen["sd"]))
        self.assertNotIn("adaln_basis", converted)
        self.assertNotIn("adaln_mean", converted)
        self.assertIn("blocks.0.attn.qkv_proj.weight", converted)

    def test_checkpoint_turbo_config(self):
        kw = {"turbo": "merged_turbo", "turbo_registry": self.registry, "turbo_dit_dir": self.dir}
        m = object.__new__(h3m.MinimaxH3Model)
        m.model_config = types.SimpleNamespace(model_kwargs=kw)
        self.assertEqual(m._checkpoint_turbo().name, "merged_turbo")
        self.assertEqual(m.stream_shifts(), (12.0, 7.0))
        self.assertEqual(tb.file_path(m._checkpoint_turbo(), kw), os.path.join(self.dir, "merged.safetensors"))
        self.assertIsNone(m._attach_turbos(None), "no live LoRA for a checkpoint turbo")


if __name__ == "__main__":
    unittest.main()
