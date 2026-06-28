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

from datasets.multimodal_dataset import MultimodalDataset


class TinyMultimodalModel(nn.Module):
    def __init__(self, echo_channels: int, ecg_channels: int) -> None:
        super().__init__()
        self.echo_net = nn.Sequential(
            nn.Conv3d(echo_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.ecg_net = nn.Sequential(
            nn.Conv1d(ecg_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(24, 1)

    def forward(self, echo: torch.Tensor, ecg: torch.Tensor) -> torch.Tensor:
        echo_feat = self.echo_net(echo).flatten(1)
        ecg_feat = self.ecg_net(ecg).flatten(1)
        return self.head(torch.cat([echo_feat, ecg_feat], dim=1)).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test real multimodal loading.")
    parser.add_argument("--csv_path", default="manifests/301/multimodal_pairs_manifest.csv")
    parser.add_argument("--echo_root", default="/media/data1/huangzihao/ECG_Echo_MM/301/echo_only_dcm")
    parser.add_argument("--ecg_root", default="/media/data1/huangzihao/ECG_Echo_MM/301/ecg-diagnostic-electrocardiogram-matched-subset")
    parser.add_argument("--echo_path_col", default="echo_dcm_path")
    parser.add_argument("--ecg_record_col", default="ecg_record_path")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--echo_num_frames", type=int, default=8)
    parser.add_argument("--echo_image_size", type=int, default=64)
    parser.add_argument("--ecg_target_length", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_path, low_memory=False)
    valid_idx = df.index[df[args.label_col].notna()].tolist()
    if not valid_idx:
        raise SystemExit(f"No non-null labels found in {args.csv_path} for {args.label_col}")

    dataset = MultimodalDataset(
        csv_path=args.csv_path,
        echo_path_col=args.echo_path_col,
        ecg_record_col=args.ecg_record_col,
        label_col=args.label_col,
        patient_id_col="subject_id" if "subject_id" in df.columns else None,
        split="val",
        echo_format="dicom",
        ecg_format="wfdb",
        echo_num_frames=args.echo_num_frames,
        echo_image_size=args.echo_image_size,
        echo_target_channels=1,
        ecg_target_length=args.ecg_target_length,
        echo_data_root=args.echo_root,
        ecg_data_root=args.ecg_root,
    )
    subset = Subset(dataset, valid_idx[:2])
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    batch = next(iter(loader))
    model = TinyMultimodalModel(
        echo_channels=batch["echo"].shape[1],
        ecg_channels=batch["ecg"].shape[1],
    ).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    echo = batch["echo"].to(args.device)
    ecg = batch["ecg"].to(args.device)
    y = batch["label"].to(args.device).float()
    optimizer.zero_grad()
    logits = model(echo, ecg)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    print(
        f"multimodal_smoke_ok echo_shape={tuple(echo.shape)} ecg_shape={tuple(ecg.shape)} "
        f"loss={loss.item():.6f}"
    )


if __name__ == "__main__":
    main()
