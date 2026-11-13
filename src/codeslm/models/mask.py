"""Attention masks for causal and document-blocked sequences.

A True entry means that query position may attend to that key. Pad positions
and tokens from earlier documents are blocked. Every query keeps its own
diagonal so a fully masked row cannot produce a NaN softmax.
"""

from __future__ import annotations

import torch


def build_attention_mask(
    *,
    sequence_length: int,
    attention_mask: torch.Tensor | None = None,
    block_ids: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Return a boolean mask of shape ``(batch, 1, sequence, sequence)``.

    ``None`` means the sequence is one causal span with no padding, so the
    caller can use SDPA's built-in causal kernel.
    """
    if block_ids is None and attention_mask is None:
        return None
    if attention_mask is not None and torch.all(attention_mask == 1) and block_ids is None:
        return None
    if device is None:
        source = block_ids if block_ids is not None else attention_mask
        assert source is not None
        device = source.device
    blocks = _blocks(sequence_length, attention_mask, block_ids, device)
    valid = blocks >= 0
    same_document = valid[:, :, None] & valid[:, None, :] & (blocks[:, :, None] == blocks[:, None, :])
    positions = torch.arange(sequence_length, device=device)
    causal = positions.view(1, 1, sequence_length) <= positions.view(1, sequence_length, 1)
    allowed = same_document & causal
    empty_rows = ~allowed.any(dim=-1, keepdim=True)
    diagonal = torch.eye(sequence_length, device=device, dtype=torch.bool).view(1, sequence_length, sequence_length)
    allowed = allowed | (empty_rows & diagonal)
    return allowed.unsqueeze(1)


def _blocks(
    sequence_length: int,
    attention_mask: torch.Tensor | None,
    block_ids: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if block_ids is None:
        assert attention_mask is not None
        blocks = torch.where(
            attention_mask.to(device=device) > 0,
            torch.zeros((), device=device, dtype=torch.long),
            torch.full((), -1, device=device, dtype=torch.long),
        )
        return blocks.expand(attention_mask.shape[0], sequence_length).clone()
    blocks = block_ids.to(device=device, dtype=torch.long)
    if attention_mask is not None:
        blocks = blocks.masked_fill(attention_mask.to(device=device) == 0, -1)
    return blocks
