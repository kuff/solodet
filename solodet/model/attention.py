"""Attention modules: CBAM and ECA for small-object detection."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention from CBAM: global avg+max pool -> shared MLP -> sigmoid."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_pool = x.mean(dim=(2, 3))  # (B, C)
        max_pool = x.amax(dim=(2, 3))  # (B, C)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention from CBAM: channel avg+max -> 7x7 conv -> sigmoid."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(combined))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Sequentially applies channel attention then spatial attention.

    Args:
        channels: Number of input channels (c1 in Ultralytics convention).
        reduction: Channel reduction ratio for channel attention MLP.
        kernel_size: Kernel size for spatial attention convolution.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class ECA(nn.Module):
    """Efficient Channel Attention module.

    Uses adaptive kernel size based on channel count. Lighter than SE/CBAM.

    Args:
        channels: Number of input channels.
        gamma: Hyperparameter for adaptive kernel size.
        beta: Hyperparameter for adaptive kernel size.
    """

    def __init__(self, channels: int, gamma: int = 2, beta: int = 1):
        super().__init__()
        import math
        t = int(abs(math.log2(channels) + beta) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = torch.sigmoid(y)
        return x * y
