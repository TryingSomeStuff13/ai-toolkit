"""CPU contracts for the run-time adaLN injection of a full-model LoRA on pruned MiniMax H3 checkpoints.

The ComfyUI MiniMax-H3-Turbo node applies the turbo LoRA's 51 adaLN pieces at run time on pruned checkpoints; the
trainer's MiniMaxH3AdalnProj must produce exactly what the node's patched forward produces. The node's function is
kept below verbatim as the reference (set H3_TURBO_NODE_DIR to also check the copy against the node's source).
"""
import ast, contextlib, io, os, sys, tempfile, types, unittest
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from safetensors.torch import save_file
# load only the H3 module: the package __init__ imports every model, and one needs a GPU at import time
for name, path in (("extensions_built_in", "extensions_built_in"), ("extensions_built_in.diffusion_models", "extensions_built_in/diffusion_models")):
    if name not in sys.modules:
        stub = types.ModuleType(name); stub.__path__ = [os.path.join(ROOT, path)]; sys.modules[name] = stub
from extensions_built_in.diffusion_models.minimax_h3 import minimax_h3 as h3m
from extensions_built_in.diffusion_models.minimax_h3.src import adaln_injection as inj
from extensions_built_in.diffusion_models.minimax_h3.src.packing import build_packed_sequence
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (MiniMaxH3AdalnProj, MiniMaxH3Transformer,
                                                                             MiniMaxH3TransformerParams)

# Copied verbatim from ComfyUI-MiniMax-H3-Turbo (__init__.py, commit 4274783), Apache License 2.0: the reference.
NODE_REFERENCE = r'''
def _make_adaln_forward(base, a, b, shared, table=None, egrid=None):
    """Curve-mode adaln injection as a *forward-attribute* patch: returns a
    replacement AdalnProj.forward that adds B @ A @ silu(t_emb) to the projection
    before the reference view/chunk. Installed via add_object_patch on the
    "<adaln_proj>.forward" attribute, so the module tree is left untouched.

    Why not a wrapper module: replacing the whole AdalnProj with an nn.Module that
    holds the original under .base injects a `.base` submodule (and its
    `.base.linear.weight`) into the model's parameter/buffer tree. ComfyUI's
    dynamic-VRAM streaming loader records every such path in its backup and, on
    unload, restores it by that path via set_attr_param/set_attr_buffer — but the
    object-patch has by then reverted AdalnProj to the plain module, so `.base`
    no longer resolves and it crashes with
    `AttributeError: 'AdalnProj' object has no attribute 'base'` (issue #4).
    Patching only the .forward attribute keeps adaln_proj.linear.weight at its
    natural path, so streaming backup/restore behaves exactly as unpatched.

    a/b are held as plain captured tensors (never registered), so they never enter
    the tree; they're cast to x's device/dtype per call, which also covers the
    VRAM-offload case where the projection runs on GPU while a/b sit on CPU."""

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
node_make_adaln_forward = _node["_make_adaln_forward"]

TABLE_ROWS, T_DIM, GRID_DIM, RANK = 1025, 8, 64, 4


def smooth_table(seed=0, rows=TABLE_ROWS, dim=T_DIM):
    """A pruned-checkpoint-like curve: one 8-wide embedding per t on linspace(0, 1, 1025)."""
    g = torch.Generator().manual_seed(seed)
    return torch.cumsum(torch.randn(rows, dim, generator=g) * 0.05, dim=0) + torch.randn(1, dim, generator=g)


def curve_rows(table, ts):
    """The trainer's _time_embedding: linear interpolation of table rows at t * (rows - 1)."""
    pos = torch.tensor(ts, dtype=torch.float32).clamp(0, 1) * (table.shape[0] - 1)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=table.shape[0] - 1)
    frac = (pos - lo.float()).unsqueeze(1)
    return table[lo] * (1.0 - frac) + table[hi] * frac


def trainer_proj(hidden, expand, modalities, seed):
    torch.manual_seed(seed)
    p = MiniMaxH3AdalnProj(T_DIM, hidden, expand, modalities, apply_silu=False, bias=True)
    with torch.no_grad():
        p.linear.weight.normal_(0, 0.3)
        p.linear.bias.normal_(0, 0.1)
    return p.to(torch.float16)  # pruned checkpoints store these projections fp16


def node_base(proj):
    """The ComfyUI AdalnProj the node patches: curve mode runs the projection in float32."""
    linear = torch.nn.Linear(proj.linear.in_features, proj.linear.out_features, bias=True)
    with torch.no_grad():
        linear.weight.copy_(proj.linear.weight.float())
        linear.bias.copy_(proj.linear.bias.float())
    return types.SimpleNamespace(linear=linear, apply_silu=False, modalities=proj.modalities, expand=proj.expand,
                                 hidden=proj.hidden)


class Holder(torch.nn.Module):
    def __init__(self, proj):
        super().__init__()
        self.adaln_proj = proj


class Container(torch.nn.Module):
    """The parts of the transformer the injection touches: the table and adaln projections at their paths."""

    def __init__(self, table, block_projs, final_proj):
        super().__init__()
        self.register_buffer("adaln_t_table", table)
        self.blocks = torch.nn.ModuleList([Holder(p) for p in block_projs])
        self.final_layer = Holder(final_proj)


def lora_for(module_paths_and_outs, seed=1, extra=True):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for path, out in module_paths_and_outs:
        sd[f"{path}.linear.lora_A.weight"] = (torch.randn(RANK, GRID_DIM, generator=g) * 0.2).to(torch.bfloat16)
        sd[f"{path}.linear.lora_B.weight"] = (torch.randn(out, RANK, generator=g) * 0.2).to(torch.bfloat16)
    if extra:  # backbone pieces ride in the same file and are ignored here
        sd["blocks.0.attn.qkv_proj.lora_A.weight"] = torch.zeros(RANK, 16, dtype=torch.bfloat16)
        sd["blocks.0.attn.qkv_proj.lora_B.weight"] = torch.zeros(48, RANK, dtype=torch.bfloat16)
    return sd


def grid_for(seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(TABLE_ROWS, GRID_DIM, generator=g).to(torch.bfloat16)


class AdalnInjectionContracts(unittest.TestCase):
    def setUp(self):
        self.table = smooth_table()
        self.grid = grid_for()
        g = torch.Generator().manual_seed(3)
        ts = [0.0, 0.1213, 0.3684, 0.5, 0.999, 1.0] + torch.rand(10, generator=g).tolist()
        self.temb = torch.cat([curve_rows(self.table, ts), self.table[[7, 512, 1024]],
                               torch.randn(4, T_DIM, generator=g)])

    def container(self):
        return Container(self.table.clone(), [trainer_proj(16, 6, 3, seed=5)], trainer_proj(16, 2, 1, seed=6))

    def test_patched_forward_equals_the_node(self):
        """(a) the trainer's projection with the update == the node's patched forward, bit for bit."""
        for strength in (1.0, 0.5):
            c = self.container()
            lora = lora_for([("blocks.0.adaln_proj", 6 * 16 * 3), ("final_layer.adaln_proj", 2 * 16)])
            self.assertEqual(inj.attach(c, lora, self.grid, strength), 2)
            for path in ("blocks.0.adaln_proj", "final_layer.adaln_proj"):
                proj = c.get_submodule(path)
                a = lora[path + ".linear.lora_A.weight"]
                b = lora[path + ".linear.lora_B.weight"] * strength
                node_forward = node_make_adaln_forward(node_base(proj), a, b, {}, self.table, self.grid)
                ours = proj(self.temb)
                theirs = node_forward(self.temb)
                self.assertEqual(len(ours), len(theirs))
                for x, y in zip(ours, theirs):
                    self.assertEqual(x.dtype, torch.float32)
                    self.assertTrue(torch.equal(x, y), f"{path} strength {strength}")
                plain = node_make_adaln_forward(node_base(proj), a, b, {}, None, None)(self.temb)
                self.assertFalse(all(torch.equal(x, y) for x, y in zip(ours, plain)), "the update is not a no-op")
                self.assertEqual(proj.adaln_delta.a.device.type, "cpu", "the pieces stay where they were loaded")
                self.assertEqual(proj.adaln_delta.b.dtype, torch.bfloat16)

    def test_node_reference_is_verbatim(self):
        node_dir = os.environ.get("H3_TURBO_NODE_DIR")
        if not node_dir:
            self.skipTest("set H3_TURBO_NODE_DIR to compare the reference with the node's source")
        with open(os.path.join(node_dir, "__init__.py")) as f:
            src = f.read()
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_make_adaln_forward")
        self.assertEqual(ast.get_source_segment(src, fn), NODE_REFERENCE.strip("\n"))

    def test_off_without_the_keys(self):
        """(b) nothing changes when the keys are absent."""
        c = self.container()
        for path in ("blocks.0.adaln_proj", "final_layer.adaln_proj"):
            proj = c.get_submodule(path)
            self.assertIsNone(proj.adaln_delta)
            plain = node_make_adaln_forward(node_base(proj), None, None, {}, None, None)(self.temb)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(proj(self.temb), plain)))
        m = object.__new__(h3m.MinimaxH3Model)
        m.model_config = types.SimpleNamespace(model_kwargs={})
        m.assistant_lora = None
        self.assertEqual(m.attach_assistant_adaln(c), 0)
        self.assertIsNone(c.blocks[0].adaln_proj.adaln_delta)
        m.model_config.model_kwargs = {"assistant_adaln_lora_path": "/x.safetensors"}
        self.assertRaises(ValueError, m.attach_assistant_adaln, c)
        m.model_config.model_kwargs = {"assistant_adaln_grid_path": "/g.safetensors"}
        self.assertRaises(ValueError, m.attach_assistant_adaln, c)

    def test_guards(self):
        lora = lora_for([("blocks.0.adaln_proj", 6 * 16 * 3), ("final_layer.adaln_proj", 2 * 16)])
        full = Container(self.table.clone(), [trainer_proj(16, 6, 3, 5)], trainer_proj(16, 2, 1, 6))
        del full.adaln_t_table
        self.assertRaises(ValueError, inj.attach, full, lora, self.grid)  # not a pruned checkpoint
        self.assertRaises(ValueError, inj.attach, self.container(), lora, self.grid[:1000])  # one grid row per table row
        empty = self.container()
        empty.adaln_t_table.zero_()
        self.assertRaises(ValueError, inj.attach, empty, lora, self.grid)  # attached before the checkpoint loaded
        bad = lora_for([("blocks.0.adaln_proj", 6 * 16 * 3 + 1)])
        self.assertRaises(ValueError, inj.attach, self.container(), bad, self.grid)  # B does not fit the projection
        stripped = {k: v for k, v in lora.items() if "adaln" not in k}
        self.assertRaises(ValueError, inj.attach, self.container(), stripped, self.grid)  # the stripped assistant file
        silu = self.container()
        silu.blocks[0].adaln_proj.apply_silu = True
        self.assertRaises(ValueError, inj.attach, silu, lora, self.grid)
        c = self.container()
        prefixed = {"diffusion_model." + k: v for k, v in lora.items()}
        self.assertEqual(inj.attach(c, prefixed, self.grid), 2)
        self.assertEqual(inj.detach(c), 2)
        self.assertIsNone(c.final_layer.adaln_proj.adaln_delta)


def tiny_params():
    return MiniMaxH3TransformerParams(hidden_size=16, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                                      attention_head_dim=16, ffn_hidden_size=32, text_dim=12, rope_inv_freq_len=2,
                                      time_embed_dim=T_DIM, adaln_t_table_size=TABLE_ROWS)


def tiny_inputs():
    tags = torch.ones(5, dtype=torch.long)
    layout = build_packed_sequence(text_token_tags=tags, num_latent_frames=2, latent_height=2, latent_width=2,
                                   num_audio_latents=3)
    s = layout.sequence_length
    row_t = torch.full((1, s), 0.3684)
    row_t[0, layout.audio_indices] = 0.5
    row_t[0, layout.video_indices[:1]] = 0.999
    g = torch.Generator().manual_seed(9)
    return dict(hidden_states=torch.randn(1, 2, 96, generator=g), audio_hidden_states=torch.randn(1, 6, 32, generator=g),
                encoder_hidden_states=torch.randn(1, 5, 12, generator=g), row_timesteps=row_t,
                token_tags=layout.token_tags[None], position_ids=layout.position_ids[None],
                video_indices=layout.video_indices, audio_indices=layout.audio_indices, text_indices=layout.text_indices)


class ModelAttachContracts(unittest.TestCase):
    """The loader path on a tiny real pruned transformer: files in, one console line, every forward updated while the
    assistant is on (training passes, the audio teacher's included) and not while the sampler has it off."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        torch.manual_seed(0)
        self.t = MiniMaxH3Transformer(tiny_params())
        with torch.no_grad():
            self.t.adaln_t_table.copy_(smooth_table(seed=4))
        paths = [(f"blocks.{i}.adaln_proj", 6 * 16 * 3) for i in range(2)] + [("final_layer.adaln_proj", 2 * 16)]
        self.lora_path = os.path.join(self.tmp.name, "turbo_full.safetensors")
        self.grid_path = os.path.join(self.tmp.name, "grid.safetensors")
        save_file(lora_for(paths, seed=7), self.lora_path)
        save_file({"silu_t_emb_grid": grid_for(seed=8)}, self.grid_path)
        self.m = object.__new__(h3m.MinimaxH3Model)
        self.m.model_config = types.SimpleNamespace(model_kwargs={"assistant_adaln_lora_path": self.lora_path,
                                                                  "assistant_adaln_grid_path": self.grid_path})
        self.m.assistant_lora = types.SimpleNamespace(is_active=True)

    def tearDown(self):
        self.tmp.cleanup()

    def forward(self):
        with torch.no_grad():
            return self.t(**tiny_inputs())

    def test_attach_prints_counts_and_follows_the_assistant(self):
        before = self.forward()
        keys = set(self.t.state_dict())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(self.m.attach_assistant_adaln(self.t), 3)
        self.assertIn("assistant adaLN injection active: 3 adaln projections", out.getvalue())
        self.assertEqual(set(self.t.state_dict()), keys, "nothing enters the state dict")
        deltas = [mod.adaln_delta for mod in self.t.modules() if isinstance(mod, MiniMaxH3AdalnProj)]
        self.assertEqual(len(deltas), 3)
        self.assertTrue(all(d is not None for d in deltas))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            on = self.forward()
            self.forward()
        self.assertEqual(out.getvalue().count("assistant adaLN update applied in a forward pass (first time)"), 1)
        self.assertTrue(all(d.calls == 2 for d in deltas), "every block and the final layer, every forward")
        self.assertFalse(torch.equal(on[0], before[0]) and torch.equal(on[1], before[1]))
        self.m.assistant_lora.is_active = False  # the sampler switches the assistant off for previews
        off = self.forward()
        self.assertTrue(torch.equal(off[0], before[0]) and torch.equal(off[1], before[1]))
        self.assertTrue(all(d.calls == 2 for d in deltas))
        self.m.assistant_lora.is_active = True
        again = self.forward()
        self.assertTrue(torch.equal(again[0], on[0]) and torch.equal(again[1], on[1]))
        # each projection against the node's patched forward, at the model's own time embeddings
        temb = self.t._time_embedding(torch.tensor([0.0, 0.3684, 0.5, 0.999, 1.0]))
        for i, proj in enumerate(mod for mod in self.t.modules() if isinstance(mod, MiniMaxH3AdalnProj)):
            d = proj.adaln_delta
            ref = node_make_adaln_forward(node_base(proj), d.a, d.b, {}, self.t.adaln_t_table, d.grid)(temb)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(proj(temb), ref)), f"projection {i}")

    def test_gradients_flow_as_before(self):
        self.m.attach_assistant_adaln(self.t)
        inputs = tiny_inputs()
        inputs["hidden_states"].requires_grad_(True)
        video, audio = self.t(**inputs)
        (video.square().mean() + audio.square().mean()).backward()
        self.assertTrue(torch.isfinite(inputs["hidden_states"].grad).all())
        self.assertIsNotNone(self.t.blocks[0].attn.qkv_proj.weight.grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
