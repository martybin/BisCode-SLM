"""Held-out perplexity from next-token loss.

The score uses the labels on each batch. Index ``-100`` is ignored, matching
the training loss. This module does not load a dataset.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from codeslm.evaluation.errors import EvaluationError
from codeslm.training.trainer import IGNORE_INDEX, Batch


@dataclass(frozen=True)
class PerplexityResult:
    loss: float
    perplexity: float
    tokens: int


@torch.no_grad()
def perplexity(model: nn.Module, batches: Sequence[Batch], device: torch.device | None = None) -> PerplexityResult:
    """Return the token-weighted cross-entropy and its exponential."""
    if not batches:
        raise EvaluationError("perplexity received no batches")
    device = device or _device(model)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        for batch in batches:
            moved = batch.to(device)
            tokens = int(moved.labels.ne(IGNORE_INDEX).sum().item())
            if tokens < 1:
                raise EvaluationError("batch has no scored tokens")
            output = model(
                moved.input_ids,
                labels=moved.labels,
                attention_mask=moved.attention_mask,
                position_ids=moved.position_ids,
                block_ids=moved.block_ids,
            )
            loss = output.loss
            if loss is None or not torch.isfinite(loss):
                raise EvaluationError("non-finite evaluation loss")
            total_loss += float(loss.detach().cpu()) * tokens
            total_tokens += tokens
    finally:
        model.train(was_training)
    mean_loss = total_loss / total_tokens
    return PerplexityResult(loss=mean_loss, perplexity=math.exp(mean_loss), tokens=total_tokens)


def _device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
