"""CCSD-QKV denoising Transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


def sinusoidal_scalar_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Builds a sinusoidal embedding for a scalar tensor of shape [B]."""
    half = dim // 2
    exponent = -math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=values.device, dtype=torch.float32) * exponent)
    angles = values.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class SigmaEmbedding(nn.Module):
    """MLP embedding of log sigma."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        log_sigma = torch.log(torch.clamp(sigma.float(), min=1e-8))
        return self.mlp(sinusoidal_scalar_embedding(log_sigma, self.embed_dim))


class SigmaAdaLN(nn.Module):
    """LayerNorm modulated by a per-sample sigma embedding."""

    def __init__(self, dim: int, sigma_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(sigma_dim, dim * 2),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, sigma_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(sigma_emb).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SigmaConditionedQKVAttention(nn.Module):
    """Full self-attention with log-sigma-conditioned Q/K/V representations."""

    def __init__(self, dim: int, heads: int, sigma_dim: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"model_dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.adaln = SigmaAdaLN(dim, sigma_dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, sigma_emb: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        h = self.adaln(x, sigma_emb)
        qkv = self.qkv(h).view(batch, length, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.out(attn)


class SigmaConditionedFFN(nn.Module):
    """Feed-forward block with sigma-conditioned normalization."""

    def __init__(self, dim: int, sigma_dim: int, ff_mult: int, dropout: float):
        super().__init__()
        self.adaln = SigmaAdaLN(dim, sigma_dim)
        hidden = dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, sigma_emb: torch.Tensor) -> torch.Tensor:
        return self.net(self.adaln(x, sigma_emb))


class CCSDQKVBlock(nn.Module):
    """Residual Transformer block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = SigmaConditionedQKVAttention(
            dim=cfg.model_dim,
            heads=cfg.heads,
            sigma_dim=cfg.sigma_embed_dim,
            dropout=cfg.dropout,
        )
        self.ffn = SigmaConditionedFFN(
            dim=cfg.model_dim,
            sigma_dim=cfg.sigma_embed_dim,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor, sigma_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x, sigma_emb)
        x = x + self.ffn(x, sigma_emb)
        return x


class CCSDQKVDenoiser(nn.Module):
    """Cycle-conditional score diffusion denoiser with Q/K/V sigma conditioning."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        token_dim = cfg.input_channels * 2
        self.input_proj = nn.Linear(token_dim, cfg.model_dim)
        self.position = nn.Parameter(torch.zeros(1, cfg.target_length, cfg.model_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.sigma_embedding = SigmaEmbedding(cfg.sigma_embed_dim)
        self.blocks = nn.ModuleList([CCSDQKVBlock(cfg) for _ in range(cfg.depth)])
        self.final_norm = nn.LayerNorm(cfg.model_dim)
        self.out = nn.Linear(cfg.model_dim, cfg.input_channels)

    def forward(self, noised_context: torch.Tensor, target_mask: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Predicts epsilon for [B, T, C] noised/context inputs."""
        if noised_context.shape != target_mask.shape:
            raise ValueError("noised_context and target_mask must have the same shape.")
        if noised_context.shape[1] > self.position.shape[1]:
            raise ValueError("Input length exceeds configured target_length.")
        sigma_emb = self.sigma_embedding(sigma)
        tokens = torch.cat([noised_context, target_mask.float()], dim=-1)
        h = self.input_proj(tokens)
        h = h + self.position[:, : h.shape[1]]
        for block in self.blocks:
            h = block(h, sigma_emb)
        return self.out(self.final_norm(h))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
