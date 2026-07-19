"""
preprocess.py
-------------
Stage 2 of the pipeline: clean the raw ingested data.

Input:  data/raw/crossfit.csv
Output: data/processed/crossfit.csv

This is the ONLY cleaning path (no parallel v1/v2 branching in code).
Dataset versioning (v1 = raw, v2 = this cleaned output) is handled
externally via DVC + git tags, not by producing multiple files here.
See DVC_SETUP.md for the exact `dvc add` / `git tag` workflow.

Cleaning steps:
1. Baseline logic exactly as given in the assignment spec (drop rows
   missing core columns, drop irrelevant columns, remove physically
   implausible outliers via fixed thresholds, clean decline-to-answer
   survey responses).
2. EXTENSION (documented): an additional z-score-based statistical
   outlier filter on top of the baseline, since fixed thresholds catch
   physically-impossible values but not statistical outliers within
   the plausible range. Controlled by params.yaml -> preprocess.outlier_zscore.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "crossfit.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "crossfit.csv"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict:
    """Load preprocessing parameters from params.yaml (with safe defaults)."""
    defaults = {"outlier_zscore": 3.0}
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            all_params = yaml.safe_load(f) or {}
        return {**defaults, **all_params.get("preprocess", {})}
    return defaults


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline cleaning pipeline, exactly as specified in the assignment.
    """
    data = data.dropna(
        subset=[
            'region', 'age', 'weight', 'height', 'howlong',
            'gender', 'eat', 'background', 'experience',
            'schedule', 'deadlift', 'candj', 'snatch', 'backsq'
        ]
    )

    data = data.drop(
        columns=[
            'affiliate', 'team', 'name',
            'fran', 'helen', 'grace', 'filthy50',
            'fgonebad', 'run400', 'run5k', 'pullups', 'train'
        ]
    )

    data = data[data['weight'] < 1500]
    data = data[data['gender'] != '--']
    data = data[data['age'] >= 18]
    data = data[(data['height'] < 96) & (data['height'] > 48)]
    data = data[
        ((data['gender'] == 'Male') & (data['deadlift'] <= 1105)) |
        ((data['gender'] == 'Female') & (data['deadlift'] <= 636))
    ]
    data = data[(data['candj'] > 0) & (data['candj'] <= 395)]
    data = data[(data['snatch'] > 0) & (data['snatch'] <= 496)]
    data = data[(data['backsq'] > 0) & (data['backsq'] <= 1069)]

    decline_dict = {
        'Decline to answer|': np.nan,
        'Decline to answer': np.nan,
    }
    data = data.replace(decline_dict)
    data = data.dropna(
        subset=[
            'background', 'experience',
            'schedule', 'howlong', 'eat'
        ]
    )

    return data


def remove_statistical_outliers(data: pd.DataFrame, zscore_threshold: float,
                                 numeric_cols=None) -> pd.DataFrame:
    """
    EXTENSION: additional z-score outlier filter on top of the baseline's
    fixed-threshold outlier removal. See module docstring for justification.
    """
    if numeric_cols is None:
        numeric_cols = ['age', 'weight', 'height', 'deadlift', 'candj', 'snatch', 'backsq']

    before_shape = data.shape
    mask = pd.Series(True, index=data.index)
    for col in numeric_cols:
        if col not in data.columns:
            continue
        col_mean = data[col].mean()
        col_std = data[col].std()
        if col_std == 0 or np.isnan(col_std):
            continue
        z_scores = (data[col] - col_mean) / col_std
        mask &= z_scores.abs() <= zscore_threshold

    data = data[mask]
    print(f"[preprocess] Z-score outlier filter (threshold={zscore_threshold}): "
          f"{before_shape} -> {data.shape}")
    return data


def main():
    """Run the preprocessing stage: load raw data, clean it, write one output."""
    params = load_params()
    print(f"[preprocess] Loaded params: {params}")

    if not RAW_PATH.exists():
        print(f"[preprocess] ERROR: raw file not found at {RAW_PATH}. "
              f"Run src/ingest.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[preprocess] Reading raw data from {RAW_PATH}")
    data = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"[preprocess] Raw shape: {data.shape}")

    data = clean_data(data)
    print(f"[preprocess] Cleaned shape (baseline): {data.shape}")

    data = remove_statistical_outliers(data, zscore_threshold=params["outlier_zscore"])
    print(f"[preprocess] Cleaned shape (after z-score extension): {data.shape}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"[preprocess] Wrote cleaned data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()