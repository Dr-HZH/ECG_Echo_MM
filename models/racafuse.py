from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.ecg_encoder import ECGResNet1DEncoder
from models.echo_3d_encoder import EchoLight3DEncoder


class ReliabilityAwareFusion(nn.Module):
    def __init__(self, embedding_dim: int = 256, meta_dim: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2 + meta_dim, embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 2),
        )
        self.concat_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, z_ecg: torch.Tensor, z_echo: torch.Tensor, meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(torch.cat([z_ecg, z_echo, meta], dim=-1))
        alpha = torch.softmax(gate_logits, dim=-1)
        weighted = alpha[:, :1] * z_ecg + alpha[:, 1:] * z_echo
        fused = weighted + self.concat_mlp(torch.cat([z_ecg, z_echo], dim=-1))
        return fused, alpha


class ModalityConditionedTemperature(nn.Module):
    def __init__(self, embedding_dim: int = 256, meta_dim: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + 2 + meta_dim, embedding_dim // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, fused: torch.Tensor, alpha: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(torch.cat([fused, alpha, meta], dim=-1))) + 1.0


class RACaFuse(nn.Module):
    """Reliability-aware calibrated ECG-Echo fusion network trained from scratch."""

    def __init__(
        self,
        ecg_in_channels: int = 12,
        echo_in_channels: int = 3,
        embedding_dim: int = 256,
        ecg_base_channels: int = 48,
        echo_base_channels: int = 24,
        dropout: float = 0.1,
        meta_dim: int = 3,
    ) -> None:
        super().__init__()
        self.ecg_encoder = ECGResNet1DEncoder(
            in_channels=ecg_in_channels,
            embedding_dim=embedding_dim,
            base_channels=ecg_base_channels,
            dropout=dropout,
        )
        self.echo_encoder = EchoLight3DEncoder(
            in_channels=echo_in_channels,
            embedding_dim=embedding_dim,
            base_channels=echo_base_channels,
            dropout=dropout,
        )
        self.fusion = ReliabilityAwareFusion(embedding_dim=embedding_dim, meta_dim=meta_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )
        self.temperature = ModalityConditionedTemperature(embedding_dim=embedding_dim, meta_dim=meta_dim, dropout=dropout)

    def forward(
        self,
        ecg: torch.Tensor,
        echo: torch.Tensor,
        meta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if meta is None:
            meta = torch.zeros(ecg.shape[0], 3, dtype=ecg.dtype, device=ecg.device)
        z_ecg = self.ecg_encoder(ecg)
        z_echo = self.echo_encoder(echo)
        fused, alpha = self.fusion(z_ecg, z_echo, meta)
        raw_logit = self.classifier(fused).squeeze(-1)
        temperature = self.temperature(fused, alpha, meta).squeeze(-1)
        calibrated_logit = raw_logit / temperature.clamp_min(1.0)
        return {
            "raw_logit": raw_logit,
            "calibrated_logit": calibrated_logit,
            "temperature": temperature,
            "alpha_ecg": alpha[:, 0],
            "alpha_echo": alpha[:, 1],
            "z_ecg": z_ecg,
            "z_echo": z_echo,
            "fused": fused,
        }
