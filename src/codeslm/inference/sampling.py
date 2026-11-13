"""Token sampling for greedy and random generation."""

from __future__ import annotations

import torch

from codeslm.inference.errors import InferenceError


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    previous_ids: list[int],
    generator: torch.Generator | None,
) -> int:
    """Draw one token from a 1-D logit vector. Temperature 0 is greedy."""
    if logits.ndim != 1:
        raise InferenceError("logits must be a vector")
    if temperature < 0 or top_k < 0 or not 0.0 < top_p <= 1.0 or repetition_penalty <= 0:
        raise InferenceError("sampling settings are out of range")
    scores = apply_repetition_penalty(logits.detach().float().cpu(), previous_ids, repetition_penalty)
    if temperature == 0.0:
        return int(torch.argmax(scores).item())
    scores = apply_top_k(scores / temperature, top_k)
    scores = apply_top_p(scores, top_p)
    probabilities = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
        raise InferenceError("sampling distribution is empty")
    drawn = torch.multinomial(probabilities, num_samples=1, generator=generator)
    return int(drawn.item())


def apply_repetition_penalty(logits: torch.Tensor, previous_ids: list[int], penalty: float) -> torch.Tensor:
    """Down-weight tokens that already appeared. Penalty 1 leaves logits unchanged."""
    if penalty == 1.0 or not previous_ids:
        return logits
    scores = logits.clone()
    for token_id in set(previous_ids):
        value = scores[token_id]
        scores[token_id] = value / penalty if value > 0 else value * penalty
    return scores


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    cutoff = torch.topk(logits, top_k).values[-1]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    ordered, indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(torch.softmax(ordered, dim=-1), dim=-1)
    remove = cumulative - torch.softmax(ordered, dim=-1) > top_p
    filtered = ordered.masked_fill(remove, float("-inf"))
    return filtered.scatter(0, indices, filtered)
