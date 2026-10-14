"""MiniMax-H3 diffusion transformer (33B), weight-compatible with the original
``MiniMaxAI/MiniMax-H3`` checkpoint keys (``video_patch_proj``, ``blocks.N.*``,
``final_layer.*``, ...).

MiniMax-H3 runs one stack of blocks over a single packed 1-D sequence holding
``[text | keyframe-condition video rows | audio rows | target video rows]``.
Full self-attention, no cross-attention, no per-modality weights: modality-
specific behavior comes only from the two input patch projections, the per-row
AdaLN modality tag (0 = video, 1 = text, 2 = audio) and the two output heads.

Deviations from the reference implementations (both are pure generalizations —
with batch size 1 and no padding the math is identical):

  - every structural input (``position_ids``, ``token_tags``, ``row_timesteps``)
    carries a batch axis, so items with different prompt lengths can share a
    training batch (pad rows are tagged ``-1`` and masked out of attention)
  - timesteps are passed per row and deduplicated internally instead of the
    caller supplying ``timestep`` + ``timestep_indices``

Mixed precision follows the shipped checkpoint: ``video_patch_proj``,
``audio_patch_proj``, ``time_embedder``, ``final_layer.video_out/audio_out``
and ``rope.inv_freq`` are float32 islands, everything else runs bf16. Inputs
are aligned to each projection's own parameter dtype at the call site.

Numerics follow ComfyUI's ``comfy/ldm/minimax/model.py`` where it matters: the
final layer modulates in float32 (``1 + scale`` sits near 0.02-0.05 for most
channels there, so a bf16 round trip of ``scale`` would move it by up to ~10%),
gated residuals accumulate with one rounding (``addcmul``, like its
``_mod_gate``), and the adaLN curve is sampled with its exact formula.
``match_engine_weight_precision`` reproduces how ComfyUI loads pre-quantized
checkpoints (unquantized linear weights rounded to bf16).

Timesteps are consumed unscaled in [0, 1] with t = 1 - sigma (t=1 means clean),
and both heads predict the data-ward velocity ``clean - noise``.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from toolkit.models.v2._mixin import OstrisModelMixin

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

MODALITY_NUM = 3  # 0 = video, 1 = text, 2 = audio; -1 marks padding rows


@dataclass
class MiniMaxH3TransformerParams:
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128  # heads * head_dim = 7168 > hidden_size
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    # VSA-trained checkpoints (FastVideo FastH3) carry a per-token gate for
    # the coarse compression branch: blocks.N.attn.to_gate_compress
    gate_compress: bool = False
    # "pruned" checkpoints (e.g. Comfy-Org *_pruned_*) replace the timestep
    # MLP with a small lookup table: ``adaln_t_table`` of shape
    # (adaln_t_table_size, time_embed_dim) sampled by linear interpolation at
    # t * (size - 1), consumed by the AdaLN projections WITHOUT the SiLU.
    # These checkpoints also shrink time_embed_dim (8 in the released files)
    # and add a bias to the block AdaLN linears, which the original weights
    # lack (the final layer's AdaLN carries a bias in both variants).
    adaln_t_table_size: Optional[int] = None

    @property
    def adaln_apply_silu(self) -> bool:
        return self.adaln_t_table_size is None

    @property
    def adaln_bias(self) -> bool:
        return self.adaln_t_table_size is not None


class MiniMaxH3Rope(nn.Module):
    """3-axis rotary embedding over the packed (t, h, w) coordinates.

    One shared ``inv_freq`` of 16 frequencies per axis; the three 16-angle
    blocks concatenate to 48 and duplicate to 96, so the first 96 of the 128
    head channels rotate (rotate-half convention) and the last 32 pass through.
    """

    def __init__(self, inv_freq_len: int = 16, theta: float = 10000.0):
        super().__init__()
        dim = 2 * inv_freq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # present in the checkpoint, so persistent
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, position_ids: torch.Tensor):
        """position_ids (B, S, 3) -> cos, sin each (B, S, 96), float32."""
        # compute on the input's device: the frequency buffer may be left
        # CPU-resident by layer offloading / low_vram loads
        position_ids = position_ids.to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(position_ids.device)
        freqs = position_ids.unsqueeze(-1) * inv_freq.view(1, 1, 1, -1)
        # (B, S, 3, 16) -> (B, S, 48) in (t, h, w) axis order -> duplicate to 96
        freqs = freqs.flatten(2, 3)
        freqs = torch.cat([freqs, freqs], dim=-1)
        return freqs.cos(), freqs.sin()


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x (B, S, H, D); cos/sin (B, S, rot) rotate the leading ``rot`` channels."""
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    cos = cos.to(x.dtype).unsqueeze(2)
    sin = sin.to(x.dtype).unsqueeze(2)
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return torch.cat([x_rot * cos + rotated * sin, x_pass], dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    """Sinusoidal embedding (cos before sin, unscaled t in [0, 1]) -> MLP.

    A float32 island in the checkpoint; the output stays float32 so every
    block's AdaLN applies its SiLU at full precision before casting down.
    """

    def __init__(self, freq_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True)
        self.proj_out = nn.Linear(hidden, out_dim, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        emb = emb.to(self.proj_in.weight.dtype)
        return self.proj_out(F.silu(self.proj_in(emb)))


class MiniMaxH3Attention(nn.Module):
    """Fused-QKV self-attention with per-head RMSNorm on q/k and partial RoPE."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        head_dim: int,
        qk_norm_eps: float,
        gate_compress: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden, inner * 3, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(inner, hidden, bias=False)
        self.to_gate_compress = (
            nn.Linear(hidden, inner, bias=False) if gate_compress else None
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, S, hidden)
        rotary_emb=None,  # (cos, sin) each (B, S, rot) or None
        attn_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, S) bool, True = attend
        vsa=None,  # H3VSAContext or None (dense)
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(b, s, self.heads, self.head_dim)
        k = k.view(b, s, self.heads, self.head_dim)
        v = v.view(b, s, self.heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if rotary_emb is not None:
            q = apply_rotary_emb(q, *rotary_emb)
            k = apply_rotary_emb(k, *rotary_emb)

        if vsa is not None and self.to_gate_compress is not None:
            from .vsa import vsa_attention

            gate = self.to_gate_compress(x).view(b, s, self.heads, self.head_dim)
            out = vsa_attention(q, k, v, gate, vsa)
            return self.out_proj(out.reshape(b, s, -1))

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.out_proj(out)


class MiniMaxH3Mlp(nn.Module):
    """SwiGLU: fc1 packs [gate | up]; silu(gate) * up -> fc2. No biases.

    ``runtime_fc2`` (None by default) is a frozen turbo LoRA's fc2 leg ``leg(out, h)`` (src/runtime_lora.py): its
    delta on the float32 SwiGLU of fc1's output h, added to fc2's output, where the ComfyUI H3RuntimeLoRA node takes it
    (the engine's fused SwiGLU + int8 kernel never calls the fc2 module)."""

    runtime_fc2 = None

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        gate, up = h.chunk(2, dim=-1)
        out = self.fc2(F.silu(gate) * up)
        if self.runtime_fc2 is not None:
            out = self.runtime_fc2(out, h)
        return out


class MiniMaxH3AdalnProj(nn.Module):
    """Timestep embedding -> per-(timestep, modality) modulation parameters.

    (M, time_embed_dim) -> ``expand`` tensors of (M * modalities, hidden); row
    layout ``[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]``, addressed by
    ``timestep_index * MODALITY_NUM + tag``. The whole projection computes in
    float32 (pruned checkpoints store these linears fp16, whose 65504 ceiling
    overflows here); the weights stay at their stored dtype and are only
    upcast for the matmul, so whole-model casts can't undo this. Outputs are
    float32; the use sites cast them to the block dtype.

    ``adaln_delta`` (None by default) is a run-time update ``delta(temb, x)``
    applied to the projection's output before the split, e.g. a full-model
    LoRA's adaLN pieces on a pruned checkpoint (src/adaln_injection.py). The
    linear is called through F.linear, so LoRA modules on it never fire.
    """

    adaln_delta = None

    def __init__(
        self,
        t_dim: int,
        hidden: int,
        expand: int,
        modalities: int,
        apply_silu: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=bias)

    def forward(self, temb: torch.Tensor):
        if self.apply_silu:
            temb = F.silu(temb)
        # follow temb's device: the raw params may be CPU-resident under layer
        # offloading / low_vram, and F.linear bypasses the paging hooks
        weight = self.linear.weight.to(device=temb.device, dtype=torch.float32)
        bias = self.linear.bias
        if bias is not None:
            bias = bias.to(device=temb.device, dtype=torch.float32)
        x = F.linear(temb.float(), weight, bias)
        if self.adaln_delta is not None:
            x = self.adaln_delta(temb, x)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


class MiniMaxH3RefinerBlock(nn.Module):
    """Plain pre-norm block over the projected text stream. No AdaLN, no RoPE."""

    def __init__(self, p: MiniMaxH3TransformerParams):
        super().__init__()
        self.norm1 = nn.RMSNorm(p.hidden_size, eps=p.norm_eps)
        self.norm2 = nn.RMSNorm(p.hidden_size, eps=p.norm_eps)
        self.attn = MiniMaxH3Attention(
            p.hidden_size, p.num_attention_heads, p.attention_head_dim, p.qk_norm_eps
        )
        self.mlp = MiniMaxH3Mlp(p.hidden_size, p.ffn_hidden_size)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, p: MiniMaxH3TransformerParams):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MiniMaxH3RefinerBlock(p) for _ in range(p.token_refiner_num_layers)]
        )
        self.final_norm = nn.RMSNorm(p.hidden_size, eps=p.final_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask)
        return self.final_norm(x)


class MiniMaxH3Block(nn.Module):
    """Pre-norm attention + SwiGLU MLP, each modulated by AdaLN parameters
    gathered per row from the (timestep, modality) table."""

    def __init__(self, p: MiniMaxH3TransformerParams):
        super().__init__()
        self.norm1 = nn.RMSNorm(p.hidden_size, eps=p.norm_eps)
        self.norm2 = nn.RMSNorm(p.hidden_size, eps=p.norm_eps)
        self.attn = MiniMaxH3Attention(
            p.hidden_size,
            p.num_attention_heads,
            p.attention_head_dim,
            p.qk_norm_eps,
            gate_compress=p.gate_compress,
        )
        self.mlp = MiniMaxH3Mlp(p.hidden_size, p.ffn_hidden_size)
        self.adaln_proj = MiniMaxH3AdalnProj(
            p.time_embed_dim,
            p.hidden_size,
            expand=6,
            modalities=MODALITY_NUM,
            apply_silu=p.adaln_apply_silu,
            bias=p.adaln_bias,
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, S, hidden)
        temb: torch.Tensor,  # (M, time_embed_dim) float32
        adaln_indices: torch.Tensor,  # (B, S) long into the (M * 3) table
        rotary_emb,  # (cos, sin)
        attn_mask: Optional[torch.Tensor] = None,
        vsa=None,  # H3VSAContext or None (dense)
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_proj(temb)
        )
        dt = x.dtype  # pruned checkpoints store the adaln projections fp16

        h = self.norm1(x) * (1.0 + scale_msa[adaln_indices].to(dt)) + shift_msa[
            adaln_indices
        ].to(dt)
        # gated residual with a single rounding, as ComfyUI's _mod_gate
        # (x.addcmul_(out, gate)); x + gate * out rounds the product first
        x = torch.addcmul(
            x, self.attn(h, rotary_emb, attn_mask, vsa), gate_msa[adaln_indices].to(dt)
        )

        h = self.norm2(x) * (1.0 + scale_mlp[adaln_indices].to(dt)) + shift_mlp[
            adaln_indices
        ].to(dt)
        x = torch.addcmul(x, self.mlp(h), gate_mlp[adaln_indices].to(dt))
        return x


class MiniMaxH3FinalLayer(nn.Module):
    """Shared shift/scale-modulated RMSNorm + the two per-modality heads.

    Both heads run over every row (matching the reference); the caller selects
    each modality's rows from the results. Heads are float32 islands.
    """

    def __init__(self, p: MiniMaxH3TransformerParams):
        super().__init__()
        video_patch_dim = (
            p.latents_dim * p.patch_size[0] * p.patch_size[1] * p.patch_size[2]
        )
        self.norm = nn.RMSNorm(p.hidden_size, eps=p.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            p.time_embed_dim,
            p.hidden_size,
            expand=2,
            modalities=1,
            apply_silu=p.adaln_apply_silu,
            bias=True,
        )
        self.video_out = nn.Linear(p.hidden_size, video_patch_dim, bias=True)
        self.audio_out = nn.Linear(p.hidden_size, p.audio_latents_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,  # (B, S, hidden)
        temb: torch.Tensor,  # (M, time_embed_dim)
        timestep_indices: torch.Tensor,  # (B, S) long into the M rows
    ):
        shift, scale = self.adaln_proj(temb)  # float32
        # ComfyUI's FinalLayer modulates in float32: the (bf16) norm output
        # times the float32 (1 + scale), plus the float32 shift. Here most
        # channels have scale ~ -0.97, so rounding scale to bf16 first would
        # move 1 + scale by up to ~10% (the blocks' modulation, which ComfyUI
        # does round to the block dtype, stays in bf16 above)
        h = self.norm(x) * (1.0 + scale[timestep_indices]) + shift[timestep_indices]
        h = h.to(self.video_out.weight.dtype)
        return self.video_out(h), self.audio_out(h)


class MiniMaxH3Transformer(nn.Module, OstrisModelMixin):
    # comfy checkpoints carry a deliberate bf16/fp16/fp32 mix
    aitk_cast_on_load = False

    @classmethod
    def aitk_config_from_state_dict(cls, state_dict):
        params = MiniMaxH3TransformerParams()
        table = state_dict.get("adaln_t_table", None)
        if table is not None:
            # pruned checkpoint: factored timestep table instead of the MLP
            params.adaln_t_table_size = table.shape[0]
            params.time_embed_dim = table.shape[1]
        params.gate_compress = "blocks.0.attn.to_gate_compress.weight" in state_dict
        return params

    @classmethod
    def aitk_from_config(cls, config):
        from accelerate import init_empty_weights

        with init_empty_weights(include_buffers=False):
            return cls(config)

    # tensors a checkpoint may carry that the model does not use (ComfyUI loads them as unexpected keys and ignores
    # them): a merged-turbo checkpoint's adaLN basis of its hybrid turbo
    UNUSED_KEYS = ("adaln_basis", "adaln_mean")

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        for key in cls.UNUSED_KEYS:
            state_dict.pop(key, None)
        return state_dict

    @classmethod
    def _load_single_file(cls, file_path, dtype, config_path=None, config=None, subfolder=None, **kwargs):
        """Checkpoints that declare their int8 layers in the file metadata (``_quantization_metadata``, e.g. some
        merged-turbo DiTs) instead of per-layer ``.comfy_quant`` tensors get the markers ComfyUI builds from it
        (comfy.utils.convert_old_quants): one uint8 JSON tensor per layer."""
        import json

        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(file_path, framework="pt") as f:
            meta = f.metadata() or {}
        if "_quantization_metadata" not in meta:
            return super()._load_single_file(file_path, dtype, config_path=config_path, config=config,
                                             subfolder=subfolder, **kwargs)
        cls._readahead(file_path)
        state_dict = load_file(file_path)
        for layer, conf in json.loads(meta["_quantization_metadata"])["layers"].items():
            state_dict[f"{layer}.comfy_quant"] = torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)
        return cls.load_from_state_dict(state_dict, dtype, config_path=config_path, config=config,
                                        subfolder=subfolder, **kwargs)

    @classmethod
    def get_transformer_block_names(cls):
        return ["blocks"]

    def __init__(self, params: Optional[MiniMaxH3TransformerParams] = None):
        super().__init__()
        if params is None:
            params = MiniMaxH3TransformerParams()
        self.params = params
        p = params

        video_patch_dim = (
            p.latents_dim * p.patch_size[0] * p.patch_size[1] * p.patch_size[2]
        )
        self.video_patch_proj = nn.Linear(video_patch_dim, p.hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(p.audio_latents_dim, p.hidden_size, bias=True)
        self.condition_proj = nn.Linear(p.text_dim, p.hidden_size, bias=True)

        if p.adaln_t_table_size is not None:
            # pruned checkpoints: the timestep MLP is replaced by a lookup
            # table sampled with linear interpolation at t * (size - 1)
            self.time_embedder = None
            self.register_buffer(
                "adaln_t_table",
                torch.zeros(p.adaln_t_table_size, p.time_embed_dim),
                persistent=True,
            )
        else:
            self.time_embedder = MiniMaxH3TimeEmbedder(
                p.timestep_input_dim, p.time_embed_hidden_size, p.time_embed_dim
            )
        self.rope = MiniMaxH3Rope(p.rope_inv_freq_len, p.rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(p)
        self.blocks = nn.ModuleList([MiniMaxH3Block(p) for _ in range(p.num_layers)])
        self.final_layer = MiniMaxH3FinalLayer(p)

        self.gradient_checkpointing = False
        # None = dense attention. Set by the FastH3 model wrapper; only takes
        # effect on gate_compress checkpoints when the caller passes the grid.
        self.vsa_sparsity: Optional[float] = None

    # float32 islands of the shipped checkpoint; used by the loader to keep
    # these keys at full precision when the rest is cast to bf16
    FP32_KEY_PREFIXES = (
        "video_patch_proj",
        "audio_patch_proj",
        "time_embedder",
        "final_layer.video_out",
        "final_layer.audio_out",
        "rope",
    )

    def enable_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = enable
        self.token_refiner.gradient_checkpointing = enable

    def disable_gradient_checkpointing(self):
        self.enable_gradient_checkpointing(False)

    @property
    def device(self):
        return self.condition_proj.weight.device

    @property
    def dtype(self):
        # working dtype of the block stack (norm weights are never quantized)
        return self.token_refiner.final_norm.weight.dtype

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """(M,) float32 in [0, 1] -> (M, time_embed_dim) float32."""
        if self.time_embedder is not None:
            return self.time_embedder(t)
        table = self.adaln_t_table.float()
        # ComfyUI's curve sampling op for op (lower row clamped to the last
        # interval, torch.lerp), so the curve coordinates, and the turbo
        # injection's nearest-row lookup on them, match the engine bit for bit.
        # The table may stay CPU-resident under offloading while timesteps
        # arrive on cuda: interpolate on the table's device, return on t's
        pos = t.to(device=table.device, dtype=torch.float32).clamp(0.0, 1.0) * (
            table.shape[0] - 1
        )
        i0 = pos.floor().long().clamp(min=0, max=max(table.shape[0] - 2, 0))
        i1 = (i0 + 1).clamp(max=table.shape[0] - 1)
        return torch.lerp(table[i0], table[i1], (pos - i0).unsqueeze(1)).to(t.device)

    @torch.no_grad()
    def match_engine_weight_precision(
        self, compute_dtype: torch.dtype = torch.bfloat16, round_biases: bool = False
    ) -> int:
        """Round the unquantized linears' weights the way ComfyUI loads a
        pre-quantized (``comfy_quant``) checkpoint. Returns how many tensors
        changed.

        ComfyUI builds every Linear of such a checkpoint with MixedPrecisionOps,
        whose loader converts an unquantized layer's weight to the compute dtype
        (bf16), whatever dtype the model code declares. So the engine runs the
        fp16 adaLN projections and the fp32 islands (patch projections, output
        heads, a full checkpoint's time embedder) with bf16-rounded weights,
        each still computed in its input's dtype (float32 for all of them).
        Biases keep their stored dtype under DynamicVRAM (its load assigns the
        checkpoint tensors); the legacy ModelPatcher copies them into
        compute-dtype parameters (``round_biases=True``).

        Rounded values are kept at float32 storage where the weight was float32
        (so its call site still computes in float32, like the engine) and moved
        to the compute dtype otherwise (the adaLN projections upcast per call).
        Quantized linears are left alone; idempotent."""
        changed = 0
        for module in self.modules():
            if not isinstance(module, nn.Linear) or getattr(
                module, "is_ostris_quantized", False
            ):
                continue
            for name in ("weight", "bias") if round_biases else ("weight",):
                param = module._parameters.get(name)
                if (
                    param is None
                    or not param.is_floating_point()
                    or param.dtype == compute_dtype
                ):
                    continue
                rounded = param.data.to(compute_dtype)
                new = rounded.to(param.dtype) if param.dtype == torch.float32 else rounded
                if new.dtype == param.dtype and torch.equal(new, param.data):
                    continue
                param.data = new
                changed += 1
        return changed

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, Nv, 96) patchified video rows (cond + target)
        audio_hidden_states: torch.Tensor,  # (B, Na, 32) audio rows
        encoder_hidden_states: torch.Tensor,  # (B, L, 5120) text conditioning
        row_timesteps: torch.Tensor,  # (B, S) float in [0, 1], t = 1 - sigma
        token_tags: torch.Tensor,  # (B, S) long: 0 video, 1 text, 2 audio, -1 pad
        position_ids: torch.Tensor,  # (B, S, 3) float (t, h, w) rotary coords
        video_indices: torch.Tensor,  # (Nv,) long positions of video rows in the pack
        audio_indices: torch.Tensor,  # (Na,) long
        text_indices: torch.Tensor,  # (L,) long
        vsa_video_grid: Optional[
            Tuple[int, int, int]
        ] = None,  # target-video token grid (t, h, w)
    ):
        """Returns (video_out (B, Nv, 96), audio_out (B, Na, 32)) — the
        data-ward velocity ``clean - noise`` for every row, in input order.
        Conditioning rows come back unmasked; discarding them is the caller's
        job."""
        batch_size, seq_len = token_tags.shape

        rotary_emb = self.rope(position_ids)

        video_embeds = self.video_patch_proj(
            hidden_states.to(self.video_patch_proj.weight.dtype)
        )
        audio_embeds = self.audio_patch_proj(
            audio_hidden_states.to(self.audio_patch_proj.weight.dtype)
        )
        text_embeds = self.condition_proj(encoder_hidden_states.to(self.dtype))

        # pad rows never act as attention keys; as queries they see everything
        # (their outputs are discarded), which keeps SDPA rows finite
        attn_mask = None
        text_attn_mask = None
        is_pad = token_tags < 0
        if bool(is_pad.any()):
            live = ~is_pad
            attn_mask = live[:, None, None, :]
            text_attn_mask = live[:, text_indices][:, None, None, :]

        text_embeds = self.token_refiner(text_embeds, text_attn_mask)

        x = text_embeds.new_zeros((batch_size, seq_len, text_embeds.shape[-1]))
        x = x.index_copy(1, text_indices, text_embeds)
        x = x.index_copy(1, video_indices, video_embeds.to(x.dtype))
        x = x.index_copy(1, audio_indices, audio_embeds.to(x.dtype))

        # one timestep embedding per distinct noise level across the batch
        unique_t, inverse = torch.unique(
            row_timesteps.to(torch.float32), sorted=True, return_inverse=True
        )
        temb = self._time_embedding(unique_t)
        adaln_indices = inverse * MODALITY_NUM + token_tags.clamp(min=0)

        vsa_ctx = None
        if (
            self.vsa_sparsity is not None
            and self.params.gate_compress
            and vsa_video_grid is not None
        ):
            from .vsa import build_vsa_context, vsa_is_available

            # no triton -> warn once and run every block dense instead
            vsa_ctx = (
                None
                if not vsa_is_available()
                else build_vsa_context(
                    seq_len=seq_len,
                    num_text_rows=int(text_indices.shape[0]),
                    video_grid=tuple(int(g) for g in vsa_video_grid),
                    token_tags=token_tags,
                    sparsity=self.vsa_sparsity,
                    device=x.device,
                )
            )

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = checkpoint(
                    block,
                    x,
                    temb,
                    adaln_indices,
                    rotary_emb,
                    attn_mask,
                    vsa_ctx,
                    use_reentrant=False,
                )
            else:
                x = block(x, temb, adaln_indices, rotary_emb, attn_mask, vsa_ctx)

        video_all, audio_all = self.final_layer(x, temb, inverse)
        video_out = video_all.index_select(1, video_indices)
        audio_out = audio_all.index_select(1, audio_indices)
        return video_out, audio_out
