from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class WeightedFocalBCE(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, pos_weight: float | None = None) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight), dtype=torch.float32) if pos_weight is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


def pairwise_auc_margin_loss(logits: torch.Tensor, targets: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    targets = targets.float()
    pos = logits[targets > 0.5]
    neg = logits[targets <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)
    diffs = pos[:, None] - neg[None, :]
    return F.relu(margin - diffs).mean()


def brier_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    return torch.mean((prob - targets.float()).pow(2))


class RACaFuseLoss(nn.Module):
    def __init__(
        self,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        pos_weight: float | None = None,
        lambda_auc: float = 0.1,
        lambda_brier: float = 0.05,
        auc_margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.focal = WeightedFocalBCE(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
        self.lambda_auc = float(lambda_auc)
        self.lambda_brier = float(lambda_brier)
        self.auc_margin = float(auc_margin)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        focal = self.focal(logits, targets)
        auc = pairwise_auc_margin_loss(logits, targets, margin=self.auc_margin)
        brier = brier_loss(logits, targets)
        total = focal + self.lambda_auc * auc + self.lambda_brier * brier
        return {"loss": total, "focal": focal, "auc": auc, "brier": brier}
