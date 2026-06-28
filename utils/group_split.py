from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitArtifacts:
    patient_splits: pd.DataFrame
    patient_table: pd.DataFrame
    summary: dict


def _import_sklearn():
    try:
        from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for patient-level split generation."
        ) from exc
    return StratifiedShuffleSplit


def build_patient_table(df: pd.DataFrame, patient_col: str, label_col: str) -> pd.DataFrame:
    labeled = df[[patient_col, label_col]].copy()
    grouped = labeled.groupby(patient_col, dropna=True)
    patient_table = grouped[label_col].agg(
        n_samples="size",
        n_positive=lambda s: int((s == 1).sum()),
        n_negative=lambda s: int((s == 0).sum()),
        n_missing=lambda s: int(s.isna().sum()),
    ).reset_index()
    patient_table["has_conflict"] = (
        (patient_table["n_positive"] > 0) & (patient_table["n_negative"] > 0)
    )
    patient_table["patient_label"] = np.where(
        patient_table["n_positive"] > 0,
        1,
        np.where(patient_table["n_negative"] > 0, 0, np.nan),
    )
    return patient_table


def _split_indices_stratified(
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    StratifiedShuffleSplit = _import_sklearn()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    indices = np.arange(len(labels))
    train_idx, test_idx = next(splitter.split(indices, labels))
    return train_idx, test_idx


def create_patient_splits(
    df: pd.DataFrame,
    patient_col: str,
    label_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitArtifacts:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    patient_table = build_patient_table(df, patient_col=patient_col, label_col=label_col)
    usable = patient_table[patient_table["patient_label"].notna()].copy()
    if usable.empty:
        raise ValueError("No patients with non-missing labels are available for splitting.")

    labels = usable["patient_label"].astype(int).to_numpy()
    stratify_possible = len(np.unique(labels)) > 1 and np.min(np.bincount(labels)) >= 2
    warnings: list[str] = []

    if stratify_possible:
        train_val_idx, test_idx = _split_indices_stratified(labels, test_size=test_ratio, seed=seed)
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(usable))
        test_n = max(1, int(round(len(usable) * test_ratio)))
        test_idx = perm[:test_n]
        train_val_idx = perm[test_n:]
        warnings.append("Stratified split was not possible; used random patient split.")

    train_val = usable.iloc[train_val_idx].reset_index(drop=True)
    test = usable.iloc[test_idx].reset_index(drop=True)

    remaining_ratio = train_ratio + val_ratio
    val_within_remaining = val_ratio / remaining_ratio
    rem_labels = train_val["patient_label"].astype(int).to_numpy()
    rem_stratify_possible = len(np.unique(rem_labels)) > 1 and np.min(np.bincount(rem_labels)) >= 2

    if rem_stratify_possible:
        train_idx, val_idx = _split_indices_stratified(
            rem_labels,
            test_size=val_within_remaining,
            seed=seed + 1,
        )
    else:
        rng = np.random.default_rng(seed + 1)
        perm = rng.permutation(len(train_val))
        val_n = max(1, int(round(len(train_val) * val_within_remaining)))
        val_idx = perm[:val_n]
        train_idx = perm[val_n:]
        warnings.append("Second-stage stratified split was not possible; used random split.")

    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    assignments = pd.concat([train, val, test], ignore_index=True)

    if assignments[patient_col].duplicated().any():
        raise RuntimeError("Patient leakage detected in patient-level split assignments.")

    sample_df = df.merge(assignments[[patient_col, "split"]], on=patient_col, how="inner")
    summary = {
        "n_patients_total": int(len(patient_table)),
        "n_patients_usable": int(len(usable)),
        "n_patients_label_conflict": int(patient_table["has_conflict"].sum()),
        "patient_label_aggregation_rule": "patient_label=1 if any positive sample exists, else 0 if any negative sample exists",
        "warnings": warnings,
        "splits": {},
    }

    for split_name in ("train", "val", "test"):
        split_patients = assignments[assignments["split"] == split_name]
        split_samples = sample_df[sample_df["split"] == split_name]
        summary["splits"][split_name] = {
            "n_patients": int(len(split_patients)),
            "n_samples": int(len(split_samples)),
            "positive_rate_patients": float(split_patients["patient_label"].mean()) if len(split_patients) else None,
            "positive_rate_samples": float(split_samples[label_col].mean()) if len(split_samples) else None,
            "unique_echo": int(split_samples["echo_dcm_path"].nunique()) if "echo_dcm_path" in split_samples.columns else None,
            "unique_ecg": int(split_samples["ecg_record_path"].nunique()) if "ecg_record_path" in split_samples.columns else None,
        }

    return SplitArtifacts(patient_splits=assignments, patient_table=patient_table, summary=summary)
