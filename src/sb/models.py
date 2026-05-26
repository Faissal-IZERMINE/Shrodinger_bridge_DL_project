"""Drift networks v_theta(t, x, s) used by alpha-DSBM.

The same network is queried with s=1 for the forward drift and s=0 for the
backward drift (paper §3). Three architectures are provided:

- ``BidirectionalMLP``: 2D toy datasets.
- ``SimpleUNet``: small image U-Net (no residual blocks).
- ``EnhancedUNet`` (~5.1M params): the architecture used for the constrained
  MNIST -> EMNIST experiments. ResBlocks with double AdaIN conditioning plus
  self-attention at the 8x8 bottleneck.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbeddings(nn.Module):
    """Standard sinusoidal positional embedding for the time conditioning."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        emb = time[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class EmbeddingBlock(nn.Module):
    """Sinusoidal -> Linear -> SiLU -> Linear MLP head for the time embed."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            SinusoidalPositionalEmbeddings(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class AdaIN(nn.Module):
    """Adaptive (group) instance norm with FiLM-style condition injection."""

    def __init__(self, feature_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, feature_channels), feature_channels, affine=False)
        self.fc = nn.Linear(condition_dim, feature_channels * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.fc(condition).view(condition.size(0), -1, 1, 1)
        gamma, beta = torch.chunk(h, 2, dim=1)
        return gamma * self.norm(x) + beta


class ResBlock(nn.Module):
    """Two-conv residual block with double AdaIN conditioning + 1x1 shortcut."""

    def __init__(self, in_channels: int, out_channels: int,
                 condition_dim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.adain1 = AdaIN(out_channels, condition_dim)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.adain2 = AdaIN(out_channels, condition_dim)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.adain1(self.conv1(x), cond))
        h = self.dropout(h)
        h = self.adain2(self.conv2(h), cond)
        return F.silu(h + self.shortcut(x))


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over spatial positions, placed at the 8x8
    bottleneck so the model can reason about global relationships."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return x + h


class EnhancedUNet(nn.Module):
    """~5.1M-parameter drift network for the constrained MNIST experiments.

    Treats the input as a flattened 32x32 image. Conditioning: sinusoidal
    time embedding concatenated with a learned direction embedding s in {0, 1}.
    """

    def __init__(self, time_embed_dim: int = 32, dir_embed_dim: int = 32,
                 base_channels: int = 128, dropout_rate: float = 0.1) -> None:
        super().__init__()
        c1, c2 = base_channels, base_channels * 2
        condition_dim = time_embed_dim + dir_embed_dim

        self.time_embed = EmbeddingBlock(time_embed_dim, time_embed_dim)
        self.dir_embed = nn.Sequential(
            nn.Linear(1, dir_embed_dim),
            nn.SiLU(),
            nn.Linear(dir_embed_dim, dir_embed_dim),
        )

        self.enc1 = ResBlock(1, c1, condition_dim, dropout_rate)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ResBlock(c1, c2, condition_dim, dropout_rate)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = ResBlock(c2, c2, condition_dim, dropout_rate)
        self.attn = SelfAttention2d(c2, num_heads=4)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResBlock(c1 + c2, c2, condition_dim, dropout_rate)

        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = ResBlock(c1 + c1, c1, condition_dim, dropout_rate)

        self.final_conv = nn.Conv2d(c1, 1, kernel_size=3, padding=1)

    def _build_cond(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        t_emb = self.time_embed(t.squeeze(-1))
        s_emb = self.dir_embed(s)
        return torch.cat([t_emb, s_emb], dim=1)

    def forward(self, x_vec: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        cond = self._build_cond(t, s)
        x = x_vec.reshape(x_vec.shape[0], 1, 32, 32)

        h1 = self.enc1(x, cond)
        h2 = self.enc2(self.pool1(h1), cond)
        hb = self.bottleneck(self.pool2(h2), cond)
        hb = self.attn(hb)

        d1 = self.up1(hb)
        d1 = self.dec1(torch.cat([d1, h2], 1), cond)
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, h1], 1), cond)
        return self.final_conv(d2).reshape(x_vec.shape[0], -1)


class SimpleUNet(nn.Module):
    """Smaller U-Net variant without residual blocks. Kept for ablations
    showing that the ResBlock + attention upgrades in ``EnhancedUNet`` are
    necessary for stability under constrained compute."""

    def __init__(self, time_embed_dim: int = 32, dir_embed_dim: int = 32,
                 base_channels: int = 64, dropout_rate: float = 0.1) -> None:
        super().__init__()
        c1, c2 = base_channels, base_channels * 2
        condition_dim = time_embed_dim + dir_embed_dim

        self.time_embed = EmbeddingBlock(time_embed_dim, time_embed_dim)
        self.dir_embed = nn.Sequential(
            nn.Linear(1, dir_embed_dim),
            nn.SiLU(),
            nn.Linear(dir_embed_dim, dir_embed_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.enc1_conv = nn.Conv2d(1, c1, 3, padding=1)
        self.enc1_adain = AdaIN(c1, condition_dim)
        self.enc2_conv = nn.Conv2d(c1, c2, 3, padding=1)
        self.enc2_adain = AdaIN(c2, condition_dim)
        self.pool = nn.MaxPool2d(2, 2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1_conv = nn.Conv2d(c1 + c2, c2, 3, padding=1)
        self.dec1_adain = AdaIN(c2, condition_dim)

        self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec2_conv = nn.Conv2d(c1 + c1, c1, 3, padding=1)
        self.dec2_adain = AdaIN(c1, condition_dim)

        self.final_conv = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, x_vec: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        t_emb = self.time_embed(t.squeeze(-1))
        s_emb = self.dir_embed(s)
        cond = torch.cat([t_emb, s_emb], dim=1)

        x = x_vec.reshape(x_vec.shape[0], 1, 32, 32)

        h1 = F.leaky_relu(self.enc1_adain(self.enc1_conv(x), cond))
        h1 = self.dropout(h1)
        h2 = F.leaky_relu(self.enc2_adain(self.enc2_conv(self.pool(h1)), cond))
        h2 = self.dropout(h2)
        h_bot = self.pool(h2)

        d1 = self.up1(h_bot)
        d1 = torch.cat([d1, h2], dim=1)
        d1 = F.leaky_relu(self.dec1_adain(self.dec1_conv(d1), cond))
        d1 = self.dropout(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, h1], dim=1)
        d2 = F.leaky_relu(self.dec2_adain(self.dec2_conv(d2), cond))

        return self.final_conv(d2).reshape(x_vec.shape[0], -1)


class BidirectionalMLP(nn.Module):
    """Tiny MLP used for 2D toy experiments (Blobs->Moons etc.).

    Direction is an Embedding(2, dir_embed_dim) lookup (s expected to be a
    {0, 1} long tensor), unlike the image networks which use a linear projection
    of a float s.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128,
                 time_embed_dim: int = 32, dir_embed_dim: int = 16) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.dir_embed = nn.Embedding(2, dir_embed_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim + dir_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if s.dim() == 2:
            s = s.squeeze(-1)
        t_emb = self.time_embed(t)
        s_emb = self.dir_embed(s.long())
        return self.net(torch.cat([x, t_emb, s_emb], dim=1))
