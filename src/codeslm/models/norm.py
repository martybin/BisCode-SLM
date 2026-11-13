"""Root-mean-square normalization."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm with a learned per-channel scale and no bias."""

    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        if size < 1:
            raise ValueError("RMSNorm size must be positive")
        if eps <= 0:
            raise ValueError("RMSNorm eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(dtype=hidden.dtype)
