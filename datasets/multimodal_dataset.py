from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.echo_dicom_dataset import EchoDicomDataset
from datasets.ecg_wfdb_dataset import ECGWfdbDataset


class MultimodalDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        echo_path_col: str = "echo_dcm_path",
        ecg_record_col: str = "ecg_record_path",
        label_col: str | None = None,
        patient_id_col: str | None = "subject_id",
        split: str = "train",
        echo_format: str = "dicom",
        ecg_format: str = "wfdb",
        echo_num_frames: int | None = None,
        echo_image_size: int | tuple[int, int] | None = None,
        echo_target_channels: int | None = None,
        ecg_target_length: int | None = None,
        ecg_target_fs: float | None = None,
        ecg_resample_if_needed: bool = False,
        ecg_lead_order: list[str] | None = None,
        echo_data_root: str | Path | None = None,
        ecg_data_root: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        self.echo_path_col = echo_path_col
        self.ecg_record_col = ecg_record_col
        self.label_col = label_col
        self.patient_id_col = patient_id_col

        self.echo_dataset = EchoDicomDataset(
            csv_path=csv_path,
            echo_path_col=echo_path_col,
            label_col=label_col,
            patient_id_col=patient_id_col,
            split=split,
            num_frames=echo_num_frames,
            image_size=echo_image_size,
            target_channels=echo_target_channels,
            input_format=echo_format,
            data_root=echo_data_root,
        )
        self.ecg_dataset = ECGWfdbDataset(
            csv_path=csv_path,
            record_col=ecg_record_col,
            label_col=label_col,
            patient_id_col=patient_id_col,
            split=split,
            target_length=ecg_target_length,
            target_fs=ecg_target_fs,
            resample_if_needed=ecg_resample_if_needed,
            lead_order=ecg_lead_order,
            input_format=ecg_format,
            data_root=ecg_data_root,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        echo_sample = self.echo_dataset[index]
        ecg_sample = self.ecg_dataset[index]
        row = self.df.iloc[index]
        sample: dict[str, Any] = {
            "echo": echo_sample["echo"],
            "ecg": ecg_sample["ecg"],
            "echo_path": echo_sample["echo_path"],
            "ecg_record_path": ecg_sample["ecg_record_path"],
            "row_index": int(index),
            "meta": {
                "echo": echo_sample.get("meta", {}),
                "ecg": ecg_sample.get("meta", {}),
            },
        }
        if self.label_col is not None and self.label_col in row.index:
            sample["label"] = echo_sample.get("label", ecg_sample.get("label"))
        if self.patient_id_col is not None and self.patient_id_col in row.index:
            sample["patient_id"] = row[self.patient_id_col]
        return sample
