from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.shortcut(x))


class ECGResNet1DEncoder(nn.Module):
    """Compact from-scratch 12-lead ECG encoder for 10-second records."""

    def __init__(
        self,
        in_channels: int = 12,
        embedding_dim: int = 256,
        base_channels: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        self.blocks = nn.Sequential(
            ResidualBlock1D(channels[0], channels[0], stride=1, dropout=dropout),
            ResidualBlock1D(channels[0], channels[1], stride=2, dropout=dropout),
            ResidualBlock1D(channels[1], channels[2], stride=2, dropout=dropout),
            ResidualBlock1D(channels[2], channels[3], stride=2, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        if ecg.ndim != 3:
            raise ValueError(f"Expected ECG tensor [B, C, L], got {tuple(ecg.shape)}")
        x = self.stem(ecg)
        x = self.blocks(x)
        x = self.pool(x)
        return self.proj(x)
