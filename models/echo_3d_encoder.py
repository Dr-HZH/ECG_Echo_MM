from __future__ import annotations

import torch
from torch import nn


class SeparableConv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int, int]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm3d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EchoLight3DEncoder(nn.Module):
    """Small X3D-style encoder trained from scratch on A4C echo clips."""

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 256,
        base_channels: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(base_channels),
            nn.SiLU(inplace=True),
        )
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 6]
        self.blocks = nn.Sequential(
            SeparableConv3DBlock(channels[0], channels[0], stride=(1, 1, 1)),
            SeparableConv3DBlock(channels[0], channels[1], stride=(1, 2, 2)),
            SeparableConv3DBlock(channels[1], channels[2], stride=(2, 2, 2)),
            SeparableConv3DBlock(channels[2], channels[3], stride=(2, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, echo: torch.Tensor) -> torch.Tensor:
        if echo.ndim != 5:
            raise ValueError(f"Expected Echo tensor [B, C, T, H, W], got {tuple(echo.shape)}")
        x = self.stem(echo)
        x = self.blocks(x)
        x = self.pool(x)
        return self.proj(x)
