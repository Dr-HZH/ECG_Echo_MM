from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class FeatureGate(nn.Module):
    def __init__(self, feature_dim: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(feature_dim // reduction, 32)
        self.gate = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(features)
        return features * gate, gate


class TemporalConvEnhancer(nn.Module):
    def __init__(self, feature_dim: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding, groups=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        residual = sequence
        x = self.block[0](sequence).transpose(1, 2)
        for layer in self.block[1:]:
            x = layer(x)
        return residual + x.transpose(1, 2)


class StableTemporalPool(nn.Module):
    """Mean + attention-gated temporal fusion for stable video pooling."""

    def __init__(self, feature_dim: int, last_frame_weight: float = 0.0) -> None:
        super().__init__()
        self.last_frame_weight = float(last_frame_weight)
        hidden = max(feature_dim // 2, 32)
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_feat = sequence.mean(dim=1)
        gate = torch.sigmoid(self.attn(mean_feat))
        weighted = (sequence * gate.unsqueeze(1)).mean(dim=1)
        pooled = mean_feat + weighted
        if self.last_frame_weight > 0:
            pooled = (1.0 - self.last_frame_weight) * pooled + self.last_frame_weight * sequence[:, -1, :]
        return pooled, mean_feat, gate


class PanEchoIndustrialClassifier(nn.Module):
    """Deployment-oriented PanEcho head: stable temporal pooling + MLP head."""

    def __init__(
        self,
        panecho_backbone: nn.Module,
        feature_dim: int,
        dropout: float = 0.2,
        last_frame_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = panecho_backbone
        self.feature_dim = feature_dim
        self.pool = StableTemporalPool(feature_dim, last_frame_weight=last_frame_weight)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )
        self.pretrained_info: dict[str, Any] = {}

    def _frame_sequence(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.reshape(b * t, c, h, w)
        frame_features = self.backbone.encoder(frames).reshape(b, t, self.feature_dim)
        frame_features = self.backbone.time_encoder(frame_features)
        return self.backbone.transformer(frame_features)

    def forward_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        sequence = self._frame_sequence(x)
        pooled, mean_features, gate = self.pool(sequence)
        return {
            "main_logit": self.head(pooled).squeeze(1),
            "features": pooled,
            "mean_features": mean_features,
            "gate": gate,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_dict(x)["main_logit"]

    def configure_freezing(
        self,
        freeze_mode: str = "frozen",
        unfreeze_patterns: list[str] | None = None,
    ) -> None:
        unfreeze_patterns = unfreeze_patterns or []
        if freeze_mode == "full":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif freeze_mode == "frozen":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_mode == "partial":
            for name, param in self.backbone.named_parameters():
                param.requires_grad = any(re.search(pattern, name) for pattern in unfreeze_patterns)
        else:
            raise ValueError(f"Unsupported freeze_mode: {freeze_mode}")

        for module in (self.pool, self.head):
            for param in module.parameters():
                param.requires_grad = True


class PanEchoRefinementClassifier(nn.Module):
    """PanEcho frame-sequence refinement head for ischemia alignment.

    The official PanEcho FrameTransformer normally returns a pooled 768-d vector.
    This module reuses the same loaded PanEcho submodules but keeps the temporal
    sequence before pooling, then applies a lightweight temporal conv, attention
    pooling, feature gate, and task-specific heads.
    """

    def __init__(
        self,
        panecho_backbone: nn.Module,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        temporal_kernel_size: int = 3,
        aux_tasks: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = panecho_backbone
        self.feature_dim = feature_dim
        self.temporal = TemporalConvEnhancer(feature_dim, kernel_size=temporal_kernel_size, dropout=dropout)
        self.attention_pool = AttentionPooling(feature_dim)
        self.feature_gate = FeatureGate(feature_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_tasks = aux_tasks or []
        self.aux_heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for task in self.aux_tasks
            }
        )
        self.pretrained_info: dict[str, Any] = {}

    def _frame_sequence(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.reshape(b * t, c, h, w)
        frame_features = self.backbone.encoder(frames).reshape(b, t, self.feature_dim)
        frame_features = self.backbone.time_encoder(frame_features)
        return self.backbone.transformer(frame_features)

    def forward_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        sequence = self._frame_sequence(x)
        refined_sequence = self.temporal(sequence)
        pooled, attention = self.attention_pool(refined_sequence)
        gated, gate = self.feature_gate(pooled)
        return {
            "main_logit": self.head(gated).squeeze(1),
            "aux_logits": {name: head(gated).squeeze(1) for name, head in self.aux_heads.items()},
            "features": gated,
            "attention": attention,
            "gate": gate,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_dict(x)["main_logit"]

    def configure_freezing(
        self,
        freeze_mode: str = "frozen",
        unfreeze_patterns: list[str] | None = None,
    ) -> None:
        unfreeze_patterns = unfreeze_patterns or []
        if freeze_mode == "full":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif freeze_mode == "frozen":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_mode == "partial":
            for name, param in self.backbone.named_parameters():
                param.requires_grad = any(re.search(pattern, name) for pattern in unfreeze_patterns)
        else:
            raise ValueError(f"Unsupported freeze_mode: {freeze_mode}")

        for module in (self.temporal, self.attention_pool, self.feature_gate, self.head, self.aux_heads):
            for param in module.parameters():
                param.requires_grad = True
