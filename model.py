"""Score network for TNP-Diffusion."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


def scalar_sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    exponent = -math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=values.device, dtype=torch.float32) * exponent)
    angles = values.float().unsqueeze(1) * frequencies.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        logsnr_like = torch.logit(t.clamp(1e-5, 1.0 - 1e-5))
        return self.net(scalar_sinusoidal_embedding(logsnr_like, self.dim))


class AdaLN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mod(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class StandardAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"model_dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, dim))


class TNPBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = AdaLN(cfg.model_dim)
        self.attn = StandardAttention(cfg.model_dim, cfg.heads, cfg.dropout)
        self.ff_norm = AdaLN(cfg.model_dim)
        hidden = cfg.model_dim * cfg.ff_mult
        self.ff = nn.Sequential(
            nn.Linear(cfg.model_dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.model_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x, cond))
        x = x + self.ff(self.ff_norm(x, cond))
        return x


class ScoreTransformer(nn.Module):
    """Transformer epsilon-prediction score network over time tokens."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_norm = nn.LayerNorm(cfg.input_channels)
        self.input_proj = nn.Linear(cfg.input_channels, cfg.model_dim)
        self.local_conv = nn.Sequential(
            nn.Conv1d(cfg.model_dim, cfg.model_dim, kernel_size=cfg.conv_kernel, padding=cfg.conv_kernel // 2, groups=cfg.model_dim),
            nn.GELU(),
            nn.Conv1d(cfg.model_dim, cfg.model_dim, kernel_size=1),
        )
        self.position = nn.Parameter(torch.zeros(1, cfg.target_length, cfg.model_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.time_embedding = TimeEmbedding(cfg.model_dim)
        self.blocks = nn.ModuleList([TNPBlock(cfg) for _ in range(cfg.depth)])
        self.final_norm = nn.LayerNorm(cfg.model_dim)
        self.out = nn.Linear(cfg.model_dim, cfg.input_channels)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.shape[1] > self.position.shape[1]:
            raise ValueError(f"Input length {x_t.shape[1]} exceeds configured target_length {self.position.shape[1]}.")
        cond = self.time_embedding(t)
        h = self.input_proj(self.input_norm(x_t))
        h = h + self.local_conv(h.transpose(1, 2)).transpose(1, 2)
        h = h + self.position[:, : h.shape[1]]
        for block in self.blocks:
            h = block(h, cond)
        return self.out(self.final_norm(h))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
