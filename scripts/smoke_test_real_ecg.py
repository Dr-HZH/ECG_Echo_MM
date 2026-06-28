#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ecg_wfdb_dataset import ECGWfdbDataset


class TinyECGModel(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x).flatten(1)
        return self.head(feats).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test real ECG WFDB loading.")
    parser.add_argument("--csv_path", default="manifests/301/ecg_available_manifest.csv")
    parser.add_argument("--ecg_root", default="/media/data1/huangzihao/ECG_Echo_MM/301/ecg-diagnostic-electrocardiogram-matched-subset")
    parser.add_argument("--record_col", default="ecg_record_path")
    parser.add_argument("--label_col", default="is_ischemic")
    parser.add_argument("--target_length", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_path, low_memory=False)
    valid_idx = df.index[df[args.label_col].notna()].tolist()
    if not valid_idx:
        raise SystemExit(f"No non-null labels found in {args.csv_path} for {args.label_col}")

    dataset = ECGWfdbDataset(
        csv_path=args.csv_path,
        record_col=args.record_col,
        label_col=args.label_col,
        patient_id_col="subject_id" if "subject_id" in df.columns else None,
        split="val",
        target_length=args.target_length,
        data_root=args.ecg_root,
    )
    subset = Subset(dataset, valid_idx[:2])
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    batch = next(iter(loader))
    model = TinyECGModel(in_channels=batch["ecg"].shape[1]).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    x = batch["ecg"].to(args.device)
    y = batch["label"].to(args.device).float()
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    print(f"ecg_smoke_ok batch_shape={tuple(x.shape)} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
