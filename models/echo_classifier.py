from __future__ import annotations

import re

from torch import nn


class EchoBinaryClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).squeeze(1)

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
                param.requires_grad = False
                if any(re.search(pattern, name) for pattern in unfreeze_patterns):
                    param.requires_grad = True
        else:
            raise ValueError(f"Unsupported freeze_mode: {freeze_mode}")

        for param in self.head.parameters():
            param.requires_grad = True
