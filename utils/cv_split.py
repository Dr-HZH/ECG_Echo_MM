from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NestedCVArtifacts:
    assignments: pd.DataFrame
    summary: dict


def _import_sklearn():
    try:
        from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
    except Exception as exc:
        raise RuntimeError("scikit-learn with StratifiedGroupKFold is required.") from exc
    return StratifiedGroupKFold


def _validate_binary_present(y: pd.Series, context: str) -> None:
    uniq = np.sort(y.dropna().astype(int).unique())
    if uniq.shape[0] < 2:
        raise ValueError(f"{context} contains fewer than 2 classes: {uniq.tolist()}")


def build_nested_stratified_group_splits(
    df: pd.DataFrame,
    id_col: str,
    patient_col: str,
    label_col: str,
    n_splits: int = 5,
    inner_val_ratio: float = 0.10,
    seed: int = 42,
) -> NestedCVArtifacts:
    if id_col not in df.columns:
        raise KeyError(id_col)
    if patient_col not in df.columns:
        raise KeyError(patient_col)
    if label_col not in df.columns:
        raise KeyError(label_col)

    working = df.copy().reset_index(drop=True)
    working[label_col] = pd.to_numeric(working[label_col], errors="coerce")
    working = working[working[id_col].notna() & working[patient_col].notna() & working[label_col].notna()].copy()
    working[label_col] = working[label_col].astype(int)
    _validate_binary_present(working[label_col], context="full dataset")

    StratifiedGroupKFold = _import_sklearn()
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    summaries = {}
    all_test_patients: set[str] = set()
    inner_splits = max(2, int(round(1.0 / inner_val_ratio)))

    for fold_idx, (train_val_idx, test_idx) in enumerate(
        outer.split(working, working[label_col], groups=working[patient_col]),
        start=1,
    ):
        train_val = working.iloc[train_val_idx].reset_index(drop=True)
        test = working.iloc[test_idx].reset_index(drop=True)
        _validate_binary_present(test[label_col], context=f"outer test fold {fold_idx}")

        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold_idx)
        inner_train_idx, inner_val_idx = next(
            inner.split(train_val, train_val[label_col], groups=train_val[patient_col])
        )
        train = train_val.iloc[inner_train_idx].reset_index(drop=True)
        val = train_val.iloc[inner_val_idx].reset_index(drop=True)
        _validate_binary_present(val[label_col], context=f"inner val fold {fold_idx}")
        _validate_binary_present(train[label_col], context=f"train fold {fold_idx}")

        train_patients = set(train[patient_col].astype(str).unique())
        val_patients = set(val[patient_col].astype(str).unique())
        test_patients = set(test[patient_col].astype(str).unique())
        if train_patients & val_patients or train_patients & test_patients or val_patients & test_patients:
            raise RuntimeError(f"Patient overlap detected in fold {fold_idx}")
        if all_test_patients & test_patients:
            raise RuntimeError(f"Outer test patient reused across folds: fold {fold_idx}")
        all_test_patients |= test_patients

        for split_name, split_df in [("train", train), ("inner_val", val), ("outer_test", test)]:
            fold_rows = split_df[[id_col, patient_col, label_col]].copy()
            fold_rows["fold"] = fold_idx
            fold_rows["split"] = split_name
            rows.append(fold_rows)

        patient_label_conflicts = (
            train_val.groupby(patient_col)[label_col].nunique(dropna=True).gt(1).sum()
        )
        summaries[f"fold_{fold_idx}"] = {
            "n_train_samples": int(len(train)),
            "n_inner_val_samples": int(len(val)),
            "n_outer_test_samples": int(len(test)),
            "n_train_patients": int(train[patient_col].nunique()),
            "n_inner_val_patients": int(val[patient_col].nunique()),
            "n_outer_test_patients": int(test[patient_col].nunique()),
            "train_positive_rate": float(train[label_col].mean()),
            "inner_val_positive_rate": float(val[label_col].mean()),
            "outer_test_positive_rate": float(test[label_col].mean()),
            "patients_with_label_conflict_in_trainval": int(patient_label_conflicts),
        }

    assignments = pd.concat(rows, ignore_index=True)
    summary = {
        "n_splits": n_splits,
        "inner_val_ratio": inner_val_ratio,
        "seed": seed,
        "n_unique_samples": int(working[id_col].nunique()),
        "n_unique_patients": int(working[patient_col].nunique()),
        "folds": summaries,
    }
    return NestedCVArtifacts(assignments=assignments, summary=summary)
