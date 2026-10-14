"""Frozen turbo LoRAs applied live on the MiniMax-H3 trainer's linears, with the math of the ComfyUI H3RuntimeLoRA node.

The trainer's assistant is the turbo the recordings (or inference) ran with. The engine runs a plain LoRA turbo with the
H3RuntimeLoRA node (TURBO_APPLY=runtime): never merged into the int8 weights, each target linear's output gets

    out += (((x.float() @ A.T) @ B.T) * (alpha / rank * strength)).to(out.dtype)

where x is that linear's own input, in 2048-row chunks. This module does the same, op for op:

- attn.qkv_proj, attn.out_proj, mlp.fc1: forward hooks (output + delta of the input). fc1's delta lands before the
  SwiGLU, as in the node's MLP forward.
- mlp.fc2: the delta on swiglu(h) = silu(gate) * up computed in float32 from fc1's output h (the node takes it there
  because the engine's fused SwiGLU + int8 kernel never calls the fc2 module): MiniMaxH3Mlp.runtime_fc2.
- float32 math, scale after the two matmuls, cast to the output's dtype, added out of place (the same values as the
  node's in-place add); the same row chunks, so every GEMM has the node's shape.
- autograd: the deltas are custom Functions whose backward is the transposed chain in float32. A is [r, in] and B
  [out, r] frozen; nothing extra is saved (the delta is linear in x; the fc2 leg keeps a reference to h).

Several turbos share one set of hooks: ``activate(name)`` picks the turbo whose pairs they apply (None: all off).
Only the active turbo's pairs live on the compute device; switching drops the others' device copies.

A turbo with adaLN pieces (Larry: 51 ``*.adaln_proj.linear`` pairs) gets them through src/adaln_injection.py with the
turbo node's silu(t_emb) grid, applied only while it is the active turbo.
"""
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from . import adaln_injection
from . import turbos as h3_turbos

ROWS = 2048  # the node's row chunk
HOOKED = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1")
FC2 = "mlp.fc2"


class _LoraDelta(torch.autograd.Function):
    """((x.float() @ A.T) @ B.T * scale).to(dtype), in ROWS-row chunks: the node's _add_delta_ without the add."""

    @staticmethod
    def forward(ctx, x, a, b, scale, dtype):
        ctx.save_for_backward(a, b)  # as stored on the device (no float32 copies kept for the backward)
        ctx.scale, ctx.x_dtype, ctx.x_shape = scale, x.dtype, x.shape
        a, b = a.float(), b.float()
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.empty(x2.shape[0], b.shape[0], device=x.device, dtype=dtype)
        for r in range(0, x2.shape[0], ROWS):
            out[r:r + ROWS] = (((x2[r:r + ROWS].float() @ a.T) @ b.T) * scale).to(dtype)
        return out.view(*x.shape[:-1], b.shape[0])

    @staticmethod
    def backward(ctx, grad):
        a, b = (t.float() for t in ctx.saved_tensors)
        g2 = grad.reshape(-1, grad.shape[-1])
        gx = torch.empty(g2.shape[0], a.shape[1], device=grad.device, dtype=ctx.x_dtype)
        for r in range(0, g2.shape[0], ROWS):
            gx[r:r + ROWS] = (((g2[r:r + ROWS].float() * ctx.scale) @ b) @ a).to(ctx.x_dtype)
        return gx.view(ctx.x_shape), None, None, None, None


class _SwigluDelta(torch.autograd.Function):
    """fc2's delta on swiglu(h) = silu(gate) * up, float32, h = [gate | up] (fc1's output): the node's
    _add_swiglu_delta_ without the add."""

    @staticmethod
    def forward(ctx, h, a, b, scale, dtype):
        ctx.save_for_backward(h, a, b)
        ctx.scale = scale
        a, b = a.float(), b.float()
        h2 = h.reshape(-1, h.shape[-1])
        half = h2.shape[-1] // 2
        out = torch.empty(h2.shape[0], b.shape[0], device=h.device, dtype=dtype)
        for r in range(0, h2.shape[0], ROWS):
            act = F.silu(h2[r:r + ROWS, :half].float()) * h2[r:r + ROWS, half:].float()
            out[r:r + ROWS] = (((act @ a.T) @ b.T) * scale).to(dtype)
        return out.view(*h.shape[:-1], b.shape[0])

    @staticmethod
    def backward(ctx, grad):
        h, a, b = ctx.saved_tensors
        a, b = a.float(), b.float()
        h2 = h.reshape(-1, h.shape[-1])
        half = h2.shape[-1] // 2
        g2 = grad.reshape(-1, grad.shape[-1])
        gh = torch.empty_like(h2)
        for r in range(0, h2.shape[0], ROWS):
            g_act = ((g2[r:r + ROWS].float() * ctx.scale) @ b) @ a
            gate, up = h2[r:r + ROWS, :half].float(), h2[r:r + ROWS, half:].float()
            sig = torch.sigmoid(gate)
            gh[r:r + ROWS, :half] = (g_act * up * (sig * (1.0 + gate * (1.0 - sig)))).to(h.dtype)
            gh[r:r + ROWS, half:] = (g_act * gate * sig).to(h.dtype)
        return gh.view(h.shape), None, None, None, None


def lora_delta(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float, dtype: torch.dtype) -> torch.Tensor:
    return _LoraDelta.apply(x, a, b, float(scale), dtype)


def swiglu_delta(h: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float, dtype: torch.dtype) -> torch.Tensor:
    return _SwigluDelta.apply(h, a, b, float(scale), dtype)


class Pair:
    """One LoRA pair kept as stored (on the CPU), moved to a compute device once while its turbo is active, with its
    scale (the node's _Cast; the deltas upcast it to float32 per call, as the node does)."""

    __slots__ = ("a", "b", "scale", "on")

    def __init__(self, a: torch.Tensor, b: torch.Tensor, scale: float):
        self.a, self.b, self.scale, self.on = a, b, float(scale), {}

    def __call__(self, device) -> Tuple[torch.Tensor, torch.Tensor, float]:
        hit = self.on.get(device)
        if hit is None:
            hit = self.on[device] = (self.a.to(device), self.b.to(device))
        return hit[0], hit[1], self.scale

    def clear(self):
        self.on.clear()


def make_pairs(state: Mapping[str, torch.Tensor], p: "h3_turbos.Plan", strength: float) -> Dict[str, Pair]:
    """{module path: Pair} with the node's scale: alpha / rank (1 without alpha) times the strength."""
    out = {}
    for module, (a_key, b_key, alpha_key) in p.backbone.items():
        a, b = state[a_key], state[b_key]
        scale = (float(state[alpha_key]) / a.shape[0] if alpha_key else 1.0) * float(strength)
        out[module] = Pair(a, b, scale)
    return out


class _AdalnSwitch:
    """An adaln_proj's run-time update for whichever turbo is active (none: unchanged)."""

    def __init__(self, owner: "TurboSet", deltas: Dict[str, Callable]):
        self.owner, self.deltas = owner, deltas

    def __call__(self, temb: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        delta = self.deltas.get(self.owner.active)
        return x if delta is None else delta(temb, x)


class TurboSet:
    """The frozen turbos of a run on one transformer: ``add`` each, then ``activate`` the one a sample uses."""

    def __init__(self, transformer: torch.nn.Module, keep_on_device: bool = False):
        self.transformer = transformer
        self.keep_on_device = keep_on_device  # True: every turbo stays on the device once used (no transfers)
        self.pairs: Dict[str, Dict[str, Pair]] = {}
        self.adaln: Dict[str, int] = {}
        self.active: Optional[str] = None
        self._live: Dict[str, Pair] = {}
        self._handles = []
        self._hooked = set()
        self._mlps = set()
        self._switches: Dict[str, _AdalnSwitch] = {}

    def expected(self, module: str) -> Optional[Tuple[int, int]]:
        """(out, in) of the transformer's linear at ``module``; None when there is none."""
        try:
            linear = self.transformer.get_submodule(module)
        except AttributeError:
            return None
        if not hasattr(linear, "out_features") or not hasattr(linear, "in_features"):
            return None
        return int(linear.out_features), int(linear.in_features)

    def add(self, turbo: "h3_turbos.Turbo", state: Mapping[str, torch.Tensor],
            grid: Optional[torch.Tensor] = None) -> "h3_turbos.Plan":
        """Map every tensor of the turbo's file onto the transformer (h3_turbos.plan: nothing skipped), hook the
        linears it touches, attach its adaLN pieces. The turbo starts inactive."""
        if turbo.name in self.pairs:
            raise ValueError(f"turbo {turbo.name} is loaded twice")
        p = h3_turbos.plan({k: tuple(v.shape) for k, v in state.items()}, self.expected, turbo.name,
                           adaln_width=int(grid.shape[1]) if grid is not None else h3_turbos.ADALN_WIDTH)
        h3_turbos.check_adaln(turbo, p)
        if p.tensors != len(state):
            raise ValueError(f"turbo {turbo.name}: {len(state) - p.tensors} tensors would be unused")
        self.pairs[turbo.name] = make_pairs(state, p, turbo.strength)
        for module in p.backbone:
            kind = h3_turbos.kind_of(module)
            if kind in HOOKED and module not in self._hooked:
                self._handles.append(self.transformer.get_submodule(module).register_forward_hook(self._hook(module)))
                self._hooked.add(module)
            elif kind == FC2:
                mlp = module.rpartition(".")[0]
                if mlp not in self._mlps:
                    target = self.transformer.get_submodule(mlp)
                    if not hasattr(type(target), "runtime_fc2"):
                        raise RuntimeError(f"{mlp} ({type(target).__name__}) has no runtime_fc2 hook point")
                    target.runtime_fc2 = self._fc2(module)
                    self._mlps.add(mlp)
        if p.adaln:
            if grid is None:
                raise ValueError(f"turbo {turbo.name} carries {len(p.adaln)} adaLN pieces: set "
                                 "model_kwargs.turbo_adaln_grid_path (the turbo node's h3_silu_temb_grid)")
            lora = {key: state[key] for keys in p.adaln.values() for key in keys}
            self.adaln[turbo.name] = self._attach_adaln(turbo, lora, grid)
        return p

    def _attach_adaln(self, turbo, lora, grid) -> int:
        """adaln_injection.attach puts the turbo's own update on each projection it carries; each goes into that
        projection's switch, which applies whichever turbo is active."""
        count = adaln_injection.attach(self.transformer, lora, grid, turbo.strength,
                                       active=lambda name=turbo.name: self.active == name)
        for path in adaln_injection.adaln_pieces(lora):
            module = self.transformer.get_submodule(path)
            switch = self._switches.setdefault(path, _AdalnSwitch(self, {}))
            switch.deltas[turbo.name] = module.adaln_delta
            module.adaln_delta = switch
        return count

    def _hook(self, module: str):
        def hook(_module, args, output):
            pair = self._live.get(module)
            if pair is None:
                return None
            a, b, scale = pair(output.device)
            return output + lora_delta(args[0], a, b, scale, output.dtype)
        return hook

    def _fc2(self, module: str):
        def leg(out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            pair = self._live.get(module)
            if pair is None:
                return out
            a, b, scale = pair(out.device)
            return out + swiglu_delta(h, a, b, scale, out.dtype)
        return leg

    def activate(self, name: Optional[str]) -> None:
        """Make ``name`` the live turbo (None: none). The others' device copies are dropped unless keep_on_device."""
        if name is not None and name not in self.pairs:
            raise ValueError(f"turbo {name} is not loaded ({', '.join(self.pairs)})")
        if name == self.active:
            return
        if not self.keep_on_device and self.active is not None:
            for pair in self.pairs[self.active].values():
                pair.clear()
        self.active = name
        self._live = self.pairs[name] if name is not None else {}

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        for mlp in self._mlps:
            self.transformer.get_submodule(mlp).runtime_fc2 = None
        for path in self._switches:
            self.transformer.get_submodule(path).adaln_delta = None
        self._hooked, self._mlps, self._switches = set(), set(), {}
        self.activate(None)
