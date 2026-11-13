"""Decoder-only transformer: pre-norm, RoPE, grouped-query attention, SwiGLU.

The language-model head is tied to the input embedding when the config says so.
Labels are next-token ids aligned with ``input_ids``; index ``-100`` is ignored.
The module does not shift labels itself, because packed sequences already place
the target token at the same index and cut the target across document boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from codeslm.config.schema import ModelArch, ModelConfig
from codeslm.models.mask import build_attention_mask
from codeslm.models.norm import RMSNorm
from codeslm.models.rope import apply_rope, rope_angles

IGNORE_INDEX = -100


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class CausalSelfAttention(nn.Module):
    """Grouped-query causal attention through PyTorch SDPA."""

    def __init__(self, arch: ModelArch) -> None:
        super().__init__()
        if arch.d_model % arch.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if arch.n_heads % arch.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        head_dim = arch.d_model // arch.n_heads
        if head_dim % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")
        self.n_heads = arch.n_heads
        self.n_kv_heads = arch.n_kv_heads
        self.head_dim = head_dim
        self.dropout = arch.dropout
        self.q_proj = nn.Linear(arch.d_model, arch.d_model, bias=arch.bias)
        kv_width = arch.n_kv_heads * head_dim
        self.k_proj = nn.Linear(arch.d_model, kv_width, bias=arch.bias)
        self.v_proj = nn.Linear(arch.d_model, kv_width, bias=arch.bias)
        self.out_proj = nn.Linear(arch.d_model, arch.d_model, bias=arch.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        query = self._shape(self.q_proj(hidden), batch, sequence, self.n_heads)
        key = self._shape(self.k_proj(hidden), batch, sequence, self.n_kv_heads)
        value = self._shape(self.v_proj(hidden), batch, sequence, self.n_kv_heads)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        dropout = self.dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            is_causal=is_causal,
            enable_gqa=self.n_heads != self.n_kv_heads,
        )
        merged = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.out_proj(merged)

    def _shape(self, projected: torch.Tensor, batch: int, sequence: int, heads: int) -> torch.Tensor:
        return projected.view(batch, sequence, heads, self.head_dim).transpose(1, 2)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network with no biases unless the config asks for them."""

    def __init__(self, arch: ModelArch) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(arch.d_model, arch.intermediate_size, bias=arch.bias)
        self.up_proj = nn.Linear(arch.d_model, arch.intermediate_size, bias=arch.bias)
        self.down_proj = nn.Linear(arch.intermediate_size, arch.d_model, bias=arch.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TransformerBlock(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, arch: ModelArch) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(arch.d_model, arch.rms_norm_eps)
        self.attention = CausalSelfAttention(arch)
        self.mlp_norm = RMSNorm(arch.d_model, arch.rms_norm_eps)
        self.mlp = SwiGLU(arch)
        self.dropout = arch.dropout

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        attended = self.attention(self.attention_norm(hidden), cos, sin, attention_mask, is_causal)
        hidden = hidden + self._drop(attended)
        hidden = hidden + self._drop(self.mlp(self.mlp_norm(hidden)))
        return hidden

    def _drop(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.dropout == 0.0:
            return hidden
        return F.dropout(hidden, p=self.dropout, training=self.training)


class CodeModel(nn.Module):
    """Causal language model sized by a :class:`ModelArch` config."""

    def __init__(self, arch: ModelArch) -> None:
        super().__init__()
        self.arch = arch
        self.embed = nn.Embedding(arch.vocab_size, arch.d_model)
        self.layers = nn.ModuleList(TransformerBlock(arch) for _ in range(arch.n_layers))
        self.norm = RMSNorm(arch.d_model, arch.rms_norm_eps)
        self.lm_head = nn.Linear(arch.d_model, arch.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.reset_parameters()
        if arch.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    @classmethod
    def from_config(cls, config: ModelConfig) -> CodeModel:
        return cls(config.model)

    def reset_parameters(self) -> None:
        std = self.arch.init_std
        residual_std = std / math.sqrt(2 * self.arch.n_layers)
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        for block in self.layers:
            _init_linear(block.attention.q_proj, std)
            _init_linear(block.attention.k_proj, std)
            _init_linear(block.attention.v_proj, std)
            _init_linear(block.attention.out_proj, residual_std)
            _init_linear(block.mlp.gate_proj, std)
            _init_linear(block.mlp.up_proj, std)
            _init_linear(block.mlp.down_proj, residual_std)
            nn.init.ones_(block.attention_norm.weight)
            nn.init.ones_(block.mlp_norm.weight)
        nn.init.ones_(self.norm.weight)
        if not self.arch.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        block_ids: torch.Tensor | None = None,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        batch, sequence = input_ids.shape
        if sequence > self.arch.max_seq_len:
            raise ValueError(f"sequence length {sequence} exceeds max_seq_len {self.arch.max_seq_len}")
        if position_ids is None:
            position_ids = torch.arange(sequence, device=input_ids.device).unsqueeze(0).expand(batch, sequence)
        cos, sin = rope_angles(position_ids, self.layers[0].attention.head_dim, self.arch.rope_theta)
        attn_mask = build_attention_mask(
            sequence_length=sequence,
            attention_mask=attention_mask,
            block_ids=block_ids,
            device=input_ids.device,
        )
        is_causal = attn_mask is None
        hidden = self.embed(input_ids)
        for block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    block,
                    hidden,
                    cos,
                    sin,
                    attn_mask,
                    is_causal,
                    use_reentrant=False,
                )
            else:
                hidden = block(hidden, cos, sin, attn_mask, is_causal)
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.arch.vocab_size),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return ModelOutput(logits=logits, loss=loss)


def _init_linear(layer: nn.Linear, std: float) -> None:
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
