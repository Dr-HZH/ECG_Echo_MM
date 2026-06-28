from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.wfdb_io import read_wfdb_record
from utils.hdf5_io import HDF5ShardReader, decode_wfdb_digital


def _parse_label(value: Any) -> torch.Tensor:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return torch.tensor(float("nan"), dtype=torch.float32)
    try:
        return torch.tensor(float(value), dtype=torch.float32)
    except Exception:
        return torch.tensor(float("nan"), dtype=torch.float32)


def _normalize_ecg_npy(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        if arr.shape[0] <= 32:
            return arr
        if arr.shape[1] <= 32:
            return arr.T
    raise ValueError(f"Unsupported ECG npy shape: {arr.shape}")


class ECGWfdbDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        record_col: str = "ecg_record_path",
        label_col: str | None = None,
        patient_id_col: str | None = None,
        split: str = "train",
        target_length: int | None = None,
        target_fs: float | None = None,
        resample_if_needed: bool = False,
        lead_order: list[str] | None = None,
        input_format: str = "wfdb",
        data_root: str | Path | None = None,
        h5_path_col: str = "h5_path",
        h5_key_col: str = "h5_key",
        h5_root: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        self.record_col = record_col
        self.label_col = label_col
        self.patient_id_col = patient_id_col
        self.split = split
        self.target_length = target_length
        self.target_fs = target_fs
        self.resample_if_needed = resample_if_needed
        self.lead_order = lead_order
        self.input_format = input_format
        self.data_root = Path(data_root) if data_root is not None else self.csv_path.parent
        self.h5_path_col = h5_path_col
        self.h5_key_col = h5_key_col
        self.h5_reader = HDF5ShardReader(h5_root) if input_format == "hdf5" else None
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.data_root / value

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        path_col = self.h5_path_col if self.input_format == "hdf5" else self.record_col
        path_value = row[path_col]
        if pd.isna(path_value):
            raise FileNotFoundError(f"Missing ECG record path at row {index}")

        abs_path = Path(str(path_value)) if self.input_format == "hdf5" else self._resolve_path(str(path_value))
        if self.input_format == "wfdb":
            result = read_wfdb_record(
                abs_path,
                target_fs=self.target_fs,
                target_length=self.target_length,
                split=self.split,
                lead_order=self.lead_order,
                resample_if_needed=self.resample_if_needed,
                rng=self.rng,
            )
            signal = result.signal
            meta = {
                "fs": result.fs,
                "sig_len": result.sig_len,
                "n_sig": result.n_sig,
                "sig_name": result.sig_name,
                "units": result.units,
                "nan_count": result.nan_count,
                "inf_count": result.inf_count,
                "zero_leads": result.zero_leads,
                "missing_leads": result.missing_leads,
            }
        elif self.input_format == "npy":
            signal = _normalize_ecg_npy(np.load(abs_path))
            meta = {
                "fs": None,
                "sig_len": int(signal.shape[1]),
                "n_sig": int(signal.shape[0]),
                "sig_name": [],
                "units": [],
                "nan_count": int(np.isnan(signal).sum()),
                "inf_count": int(np.isinf(signal).sum()),
                "zero_leads": [],
                "missing_leads": [],
            }
        elif self.input_format == "hdf5":
            if self.h5_reader is None:
                raise ValueError("HDF5 reader is not initialized.")
            h5_key = str(row[self.h5_key_col])
            signal, h5_attrs = self.h5_reader.read_with_attrs(abs_path, h5_key)
            if signal.ndim != 2:
                raise ValueError(f"Expected HDF5 ECG [C,L], got {signal.shape}")
            stored_dtype = str(signal.dtype)
            if h5_attrs.get("encoding") == "wfdb_digital":
                signal = decode_wfdb_digital(signal, h5_attrs)
            meta = {
                "fs": float(h5_attrs.get("fs", self.target_fs or 0.0)),
                "sig_len": int(signal.shape[1]),
                "n_sig": int(signal.shape[0]),
                "sig_name": self.lead_order or [],
                "units": [],
                "nan_count": int(np.isnan(signal).sum()),
                "inf_count": int(np.isinf(signal).sum()),
                "zero_leads": [],
                "missing_leads": [],
                "h5_key": h5_key,
                "stored_dtype": stored_dtype,
                "encoding": str(h5_attrs.get("encoding", stored_dtype)),
            }
        else:
            raise ValueError(f"Unsupported ECG input_format={self.input_format}")

        tensor = torch.from_numpy(signal.astype(np.float32, copy=False))
        sample: dict[str, Any] = {
            "ecg": tensor,
            "ecg_record_path": str(abs_path),
            "row_index": int(index),
            "meta": meta,
        }
        if self.label_col is not None and self.label_col in row.index:
            sample["label"] = _parse_label(row[self.label_col])
        if self.patient_id_col is not None and self.patient_id_col in row.index:
            sample["patient_id"] = row[self.patient_id_col]
        return sample
