from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.dicom_io import adapt_channel_count, sample_video_frames
from utils.hdf5_io import HDF5ShardReader, decode_wfdb_digital
from utils.video_transforms import (
    crop_video,
    jitter_brightness_contrast,
    minmax_scale_video,
    normalize_video,
    resize_video,
    translate_video,
)


class RACaFuseDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        dataframe: pd.DataFrame | None = None,
        split: str = "train",
        data_root: str | Path = ".",
        echo_num_frames: int = 16,
        echo_image_size: int = 112,
        echo_channels: int = 3,
        echo_intensity_mode: str = "uint8_scale",
        echo_crop_scale_range: tuple[float, float] | None = None,
        echo_brightness_jitter: float = 0.0,
        echo_contrast_jitter: float = 0.0,
        echo_translate_fraction: float = 0.0,
        echo_normalize_mean: Sequence[float] | None = None,
        echo_normalize_std: Sequence[float] | None = None,
        ecg_target_length: int = 5000,
        label_col: str = "label",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = dataframe.copy().reset_index(drop=True) if dataframe is not None else pd.read_csv(csv_path, low_memory=False)
        self.split = split
        self.label_col = label_col
        self.echo_num_frames = echo_num_frames
        self.echo_image_size = echo_image_size
        self.echo_channels = echo_channels
        self.echo_intensity_mode = echo_intensity_mode
        self.echo_crop_scale_range = echo_crop_scale_range
        self.echo_brightness_jitter = echo_brightness_jitter
        self.echo_contrast_jitter = echo_contrast_jitter
        self.echo_translate_fraction = echo_translate_fraction
        self.echo_normalize_mean = list(echo_normalize_mean) if echo_normalize_mean is not None else None
        self.echo_normalize_std = list(echo_normalize_std) if echo_normalize_std is not None else None
        self.ecg_target_length = int(ecg_target_length)
        self.h5_reader = HDF5ShardReader(data_root)
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.df)

    def _load_echo(self, row: pd.Series) -> torch.Tensor:
        stored = self.h5_reader.read(str(row["echo_h5_path"]), str(row["echo_h5_key"]))
        if stored.ndim != 4 or stored.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Expected echo HDF5 [T,H,W,C], got {stored.shape}")
        array = np.ascontiguousarray(np.transpose(stored, (3, 0, 1, 2)))
        array = sample_video_frames(array, num_frames=self.echo_num_frames, split=self.split, rng=self.rng)
        array = adapt_channel_count(array, target_channels=self.echo_channels)
        tensor = torch.from_numpy(array.astype(np.float32, copy=False))
        if self.echo_intensity_mode == "uint8_scale":
            tensor = tensor / 255.0
        elif self.echo_intensity_mode == "minmax":
            tensor = minmax_scale_video(tensor)
        tensor = crop_video(tensor, split=self.split, crop_scale_range=self.echo_crop_scale_range, rng=self.rng)
        tensor = translate_video(tensor, split=self.split, max_translate_fraction=self.echo_translate_fraction, rng=self.rng)
        tensor = resize_video(tensor, image_size=self.echo_image_size)
        tensor = jitter_brightness_contrast(
            tensor,
            split=self.split,
            brightness=self.echo_brightness_jitter,
            contrast=self.echo_contrast_jitter,
            rng=self.rng,
        )
        return normalize_video(tensor, mean=self.echo_normalize_mean, std=self.echo_normalize_std)

    def _load_ecg(self, row: pd.Series) -> torch.Tensor:
        signal, attrs = self.h5_reader.read_with_attrs(str(row["ecg_h5_path"]), str(row["ecg_h5_key"]))
        if signal.ndim != 2:
            raise ValueError(f"Expected ECG HDF5 [C,L], got {signal.shape}")
        if attrs.get("encoding") == "wfdb_digital":
            signal = decode_wfdb_digital(signal, attrs)
        arr = signal.astype(np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[1] > self.ecg_target_length:
            start = 0 if self.split != "train" else int(self.rng.integers(0, arr.shape[1] - self.ecg_target_length + 1))
            arr = arr[:, start : start + self.ecg_target_length]
        elif arr.shape[1] < self.ecg_target_length:
            pad = self.ecg_target_length - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant")
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True)
        arr = (arr - mean) / np.clip(std, 1e-3, None)
        arr = np.clip(arr, -8.0, 8.0)
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        time_delta = float(row.get("time_delta_days", 0.0))
        meta = torch.tensor([min(time_delta / 30.0, 1.0), 0.0, 0.0], dtype=torch.float32)
        return {
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
            "echo": self._load_echo(row),
            "ecg": self._load_ecg(row),
            "meta": meta,
            "label": torch.tensor(float(row[self.label_col]), dtype=torch.float32),
            "time_delta_days": torch.tensor(time_delta, dtype=torch.float32),
        }
