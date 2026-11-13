"""Learning-rate schedule and training-FLOP estimate."""

from __future__ import annotations

import math


def learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    min_ratio: float,
) -> float:
    """Linear warmup, then cosine decay from the peak to ``peak * min_ratio``.

    ``step`` is the zero-based index of the optimizer update about to run.
    """
    if step < 0 or total_steps < 1 or warmup_steps < 0:
        raise ValueError("step, total_steps, and warmup_steps are out of range")
    if peak <= 0 or not 0.0 <= min_ratio <= 1.0:
        raise ValueError("peak learning rate and min_ratio are out of range")
    if warmup_steps > 0 and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return peak
    span = total_steps - warmup_steps - 1
    if span <= 0:
        progress = 1.0
    else:
        progress = (step - warmup_steps) / span
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = peak * min_ratio
    return floor + (peak - floor) * cosine


def estimate_flops(num_parameters: int, n_layers: int, d_model: int, seq_len: int, tokens: int) -> int:
    """Approximate training FLOPs for a decoder-only step.

    Linear layers contribute ``6 * parameters`` per token. Attention score and
    context matmuls contribute ``12 * layers * d_model * seq_len`` per token.
    A multiply-add counts as two FLOPs.
    """
    if min(num_parameters, n_layers, d_model, seq_len, tokens) < 0:
        raise ValueError("FLOP inputs must be non-negative")
    per_token = (6 * num_parameters) + (12 * n_layers * d_model * seq_len)
    return per_token * tokens
