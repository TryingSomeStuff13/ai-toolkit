"""Run-time adaLN updates of a full-model LoRA on pruned MiniMax H3 checkpoints, as the ComfyUI turbo node applies them.

Pruned checkpoints replace the timestep MLP with a small curve table (``adaln_t_table``, 8 wide) that feeds the adaLN
projections directly, without the SiLU. A LoRA trained on the full model (the MiniMax-H3 turbo LoRA) carries adaLN
updates B @ A acting on the full model's 2688-wide silu(t_emb), so they cannot be applied as weights to the pruned
8-wide projections. The ComfyUI MiniMax-H3-Turbo node (``_inject_adaln_egrid`` / ``_make_adaln_forward``, curve-mode
path) re-injects them at run time. Inference therefore runs the turbo with them ("+ 51 adaln injected at run time" in
its log). For every adaln_proj it computes

    idx   = argmin over table rows of the Euclidean distance between the projection's input and the row
    st    = grid[idx]                         (a bundled 1025 x 2688 grid: silu(time_embedder(t)) per table row)
    x    += (B @ (A @ st.T)).T                (on the projection's output, before it is split into shift/scale/gate)

in the output's dtype. The adaLN update is plain B @ A: no alpha, strength applied to B once.
This module does the same inside MiniMaxH3AdalnProj.forward (its ``adaln_delta`` hook). Its tensors stay on the CPU
and are cast to the call's device and dtype per call, like the node's. They never enter the model's state dict,
quantization, offloading or LoRA networks.
"""
from typing import Callable, Dict, Optional

import torch

ADALN_LINEAR = ".adaln_proj.linear"
GRID_KEY = "silu_t_emb_grid"
KEY_PREFIXES = ("diffusion_model.", "transformer.", "model.diffusion_model.")


class AdalnLoraDelta:
    """The run-time adaLN update of one projection: ``delta(temb, x)`` returns x + (B @ (A @ st.T)).T.

    A plain object (not a module): the tensors stay where they were loaded and are cast per call."""

    def __init__(self, lora_a: torch.Tensor, lora_b: torch.Tensor, table: torch.Tensor, grid: torch.Tensor,
                 active: Optional[Callable[[], bool]] = None, on_first_use: Optional[Callable[[], None]] = None):
        self.a = lora_a
        self.b = lora_b
        self.table = table
        self.grid = grid
        self.active = active
        self.on_first_use = on_first_use
        self.calls = 0

    def __call__(self, temb: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.active is not None and not self.active():
            return x
        # the node's table/egrid path, op for op
        tb = self.table.to(temb.device, torch.float32)
        idx = torch.cdist(temb.detach().float(), tb).argmin(dim=1)
        st = self.grid.to(temb.device)[idx]
        if st.shape[0] != x.shape[0]:
            raise ValueError(f"adaLN injection: {st.shape[0]} grid rows for {x.shape[0]} projection rows")
        av = self.a.to(x.device, x.dtype)
        bv = self.b.to(x.device, x.dtype)
        sv = st.to(x.device, x.dtype)
        self.calls += 1
        if self.calls == 1 and self.on_first_use is not None:
            self.on_first_use()
        return x + (bv @ (av @ sv.T)).T


def _strip(name: str) -> str:
    for prefix in KEY_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def adaln_pieces(lora_state: Dict[str, torch.Tensor]) -> Dict[str, tuple]:
    """{module path of the adaln_proj ("blocks.0.adaln_proj"): (lora_A, lora_B)} from a LoRA state dict."""
    pieces = {}
    for key, value in lora_state.items():
        if not key.endswith(ADALN_LINEAR + ".lora_A.weight"):
            continue
        base = key[: -len(".lora_A.weight")]
        b_key = base + ".lora_B.weight"
        if b_key not in lora_state:
            raise ValueError(f"adaLN injection: {key} has no {b_key}")
        pieces[_strip(base)[: -len(".linear")]] = (value, lora_state[b_key])
    return pieces


def attach(transformer, lora_state: Dict[str, torch.Tensor], grid: torch.Tensor, strength: float = 1.0,
           active: Optional[Callable[[], bool]] = None, on_first_use: Optional[Callable[[], None]] = None) -> int:
    """Give every adaln_proj that the LoRA carries an update (its ``adaln_delta``). Returns how many projections got
    one. Only for pruned (curve) checkpoints; call it after the checkpoint is loaded (the table is snapshotted)."""
    table = getattr(transformer, "adaln_t_table", None)
    if table is None:
        raise ValueError("adaLN injection is for pruned H3 checkpoints (adaln_t_table); on the full model the LoRA's "
                         "adaLN pieces are ordinary LoRA weights")
    table = table.detach().to("cpu", torch.float32).clone()
    if not bool(torch.isfinite(table).all()) or not bool(table.abs().sum() > 0):
        raise ValueError("adaLN injection: adaln_t_table is empty; attach after the checkpoint is loaded")
    if grid.dim() != 2 or grid.shape[0] != table.shape[0]:
        raise ValueError(f"adaLN injection: the grid ({tuple(grid.shape)}) must have one row per adaln_t_table row "
                         f"({table.shape[0]})")
    pieces = adaln_pieces(lora_state)
    if not pieces:
        raise ValueError("adaLN injection: the LoRA has no *.adaln_proj.linear pieces (the stripped assistant file "
                         "has none; use the full LoRA file)")
    deltas = {}
    for path, (a, b) in sorted(pieces.items()):
        module = transformer.get_submodule(path)
        linear = module.linear
        if getattr(module, "apply_silu", True):
            raise ValueError(f"adaLN injection: {path} is not a curve-mode (pruned) projection")
        if a.dim() != 2 or b.dim() != 2 or a.shape[0] != b.shape[1] or a.shape[1] != grid.shape[1] \
                or b.shape[0] != linear.out_features:
            raise ValueError(f"adaLN injection: {path} A {tuple(a.shape)} / B {tuple(b.shape)} do not fit the grid "
                             f"width {grid.shape[1]} and the projection's {linear.out_features} outputs")
        deltas[path] = AdalnLoraDelta(a, b * strength, table, grid, active, on_first_use)
    for path, delta in deltas.items():
        transformer.get_submodule(path).adaln_delta = delta
    return len(deltas)


def detach(transformer) -> int:
    """Remove every run-time adaLN update; returns how many there were."""
    count = 0
    for module in transformer.modules():
        if getattr(module, "adaln_delta", None) is not None:
            module.adaln_delta = None
            count += 1
    return count


def load_and_attach(transformer, lora_path: str, grid_path: str, strength: float = 1.0,
                    active: Optional[Callable[[], bool]] = None,
                    on_first_use: Optional[Callable[[], None]] = None) -> int:
    from safetensors.torch import load_file

    grid = load_file(grid_path)
    if GRID_KEY not in grid:
        raise ValueError(f"{grid_path} has no {GRID_KEY!r} tensor (expected the turbo node's h3_silu_temb_grid)")
    lora = {k: v for k, v in load_file(lora_path).items() if ADALN_LINEAR in k}
    return attach(transformer, lora, grid[GRID_KEY], strength, active, on_first_use)
