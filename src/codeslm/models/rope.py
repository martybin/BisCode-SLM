"""Rotary position embeddings computed from explicit position ids.

Position 0 is the identity rotation. There is no learned position table and no
cached maximum length, so a later context size does not need a new embedding
matrix. The rotation pairs dimension ``i`` with ``i + head_dim / 2``.
"""

from __future__ import annotations

import torch


def rotate_half(states: torch.Tensor) -> torch.Tensor:
    first, second = states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rope_angles(position_ids: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cosine and sine with shape ``(batch, 1, sequence, head_dim)``."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE head dimension must be even")
    if theta <= 0:
        raise ValueError("RoPE theta must be positive")
    half = head_dim // 2
    exponents = torch.arange(half, device=position_ids.device, dtype=torch.float32) / half
    frequencies = 1.0 / (theta ** exponents)
    angles = position_ids.to(dtype=torch.float32).unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)


def apply_rope(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``states`` of shape ``(batch, heads, sequence, head_dim)``."""
    rotated = (states.float() * cos) + (rotate_half(states.float()) * sin)
    return rotated.to(dtype=states.dtype)
