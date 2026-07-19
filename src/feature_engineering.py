"""
feature_engineering.py
-----------------------
Stage 3 of the pipeline: turn cleaned data into model-ready features.

Input:  data/processed/crossfit.csv
Output: data/processed/crossfit_features.csv

Design decisions:
- `region` dropped: high-cardinality geographic category, weak expected
  predictive signal, would bloat one-hot encoding.
- `gender` binary-encoded.
- `howlong` ordinal-encoded (raw values carry a trailing '|' artifact,
  stripped before mapping).
- `background`, `experience`, `schedule`, `eat` are multi-select survey
  fields (pipe-delimited); MULTI-HOT encoded rather than naively
  one-hot encoded, since one-hot on the raw string would treat every
  unique combination of answers as an unrelated category.
  "Decline to answer" is treated as a non-answer and excluded; rows
  with no other answer for a field are dropped.
- total_lift = deadlift + candj + snatch + backsq, added per professor's
  requirement.
- athlete_id is RETAINED as the Feast feature store entity key (not
  used as a model feature — excluded in train.py the same way the lift
  components are).
- A synthetic event_timestamp column is added for all rows, since this
  dataset has no real temporal dimension but Feast requires one for
  point-in-time feature retrieval. This is a documented simplification:
  every row is treated as "current" (same timestamp), not a true
  historical feature log.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "crossfit.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "crossfit_features.csv"
FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"
FEATURE_STORE_PARQUET_PATH = FEATURE_STORE_DIR / "crossfit_features.parquet"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

HOWLONG_ORDER = {
    'Less than 6 months': 0,
    '6-12 months': 1,
    '1-2 years': 2,
    '2-4 years': 3,
    '4+ years': 4,
}

DROP_COLS = ['region']
MULTI_SELECT_COLS = ['background', 'experience', 'schedule', 'eat']


def load_params() -> dict:
    """Load feature-engineering parameters from params.yaml."""
    defaults = {"selected": None}
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            all_params = yaml.safe_load(f) or {}
        return {**defaults, **all_params.get("features", {})}
    return defaults


def encode_gender(data: pd.DataFrame) -> pd.DataFrame:
    """Binary-encode gender: Male=1, Female=0."""
    data = data.copy()
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    return data


def strip_trailing_pipe(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Strip a single trailing '|' delimiter artifact from single-value columns."""
    data = data.copy()
    for col in cols:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip().str.rstrip('|').str.strip()
            data.loc[data[col].isin(['nan', '']), col] = pd.NA
    return data


def encode_howlong(data: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode howlong; drop rows with an unmapped value after cleanup."""
    data = data.copy()
    data = strip_trailing_pipe(data, ['howlong'])
    data['howlong'] = data['howlong'].map(HOWLONG_ORDER)
    before = len(data)
    data = data.dropna(subset=['howlong'])
    dropped = before - len(data)
    if dropped:
        print(f"[feature_engineering] Dropped {dropped} rows with unmapped 'howlong' values")
    data['howlong'] = data['howlong'].astype(int)
    return data

# Expected survey responses used by Feast feature definitions.
# The values should match the cleaned survey responses after stripping.
EXPECTED_MULTI_SELECT = {
    "background": [
        "football",
        "weightlifting",
    ],
    "experience": [
        "beginner",
        "advanced",
    ],
    "schedule": [
        "5 days",
    ],
    "eat": [
        "paleo",
    ],
}


def normalize_answer(answer: str) -> str:
    """
    Normalize survey responses so feature names are consistent.
    Example:
        '5 Days' -> '5_days'
        'Weightlifting' -> 'weightlifting'
    """
    return (
        answer.lower()
        .strip()
        .replace("/", "_")
        .replace(" ", "_")
    )


def multi_hot_encode(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Multi-hot encode pipe-delimited survey responses.

    Unlike the previous implementation, this version:
      • keeps all rows (missing answers become all zeros)
      • always creates the same feature columns required by Feast
      • normalizes feature names to lowercase
    """
    data = data.copy()
    decline_phrase = "decline to answer"

    for col in cols:

        if col not in data.columns:
            print(f"[feature_engineering] WARNING: missing column '{col}', skipping.")
            continue

        split_series = data[col].fillna("").astype(str).apply(
            lambda s: [
                normalize_answer(ans)
                for ans in s.split("|")
                if ans.strip()
                and ans.strip().lower() != decline_phrase
            ]
        )

        expected_answers = [
            normalize_answer(ans)
            for ans in EXPECTED_MULTI_SELECT.get(col, [])
        ]

        for answer in expected_answers:
            feature_name = f"{col}__{answer}"
            data[feature_name] = split_series.apply(
                lambda answers, a=answer: int(a in answers)
            )

        data = data.drop(columns=[col])

    return data



def add_total_lift(data: pd.DataFrame) -> pd.DataFrame:
    """Add total_lift = deadlift + candj + snatch + backsq (professor's requirement)."""
    data = data.copy()
    data['total_lift'] = (
        data['deadlift'] + data['candj'] + data['snatch'] + data['backsq']
    )
    return data


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline on cleaned data."""
    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns])
    data = encode_gender(data)
    data = encode_howlong(data)
    data = multi_hot_encode(data, MULTI_SELECT_COLS)
    data = add_total_lift(data)
    return data


def add_event_timestamp(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a synthetic event_timestamp column, required by Feast for
    point-in-time feature retrieval. LIMITATION: this dataset has no
    real temporal dimension, so every row is stamped with the same
    "now" timestamp at feature-engineering time rather than a true
    historical event time. This is sufficient to demonstrate the
    feature store integration but should not be read as real
    time-travel/point-in-time correctness over athlete history.
    """
    data = data.copy()
    data['event_timestamp'] = pd.Timestamp.now()
    return data


def write_feature_store_source(data: pd.DataFrame):
    """
    Write the Feast-compatible parquet source file: entity key
    (athlete_id), event_timestamp, and all engineered feature columns.
    This parquet file is the FileSource that feature_repo/definitions.py
    points to.

    athlete_id is explicitly cast to int64: Feast's entity key
    serialization only supports int/string entity keys, and pandas
    reads athlete_id as float64 by default (e.g. 2554.0) since the raw
    CSV formats IDs with a trailing .0. Any rows with a missing
    athlete_id are dropped first, since they can't serve as a valid
    entity key.
    """
    data = data.copy()
    before = len(data)
    data = data.dropna(subset=['athlete_id'])
    dropped = before - len(data)
    if dropped:
        print(f"[feature_engineering] Dropped {dropped} rows with missing athlete_id "
              f"(required as Feast entity key)")
    data['athlete_id'] = data['athlete_id'].astype('int64')

    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(FEATURE_STORE_PARQUET_PATH, index=False)
    print(f"[feature_engineering] Wrote Feast feature source to {FEATURE_STORE_PARQUET_PATH}")


def main():
    """Load cleaned data, build features, and write the model-ready dataset."""
    params = load_params()
    print(f"[feature_engineering] Loaded params: {params}")

    if not INPUT_PATH.exists():
        print(f"[feature_engineering] ERROR: input not found at {INPUT_PATH}. "
              f"Run src/preprocess.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[feature_engineering] Reading {INPUT_PATH}")
    data = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"[feature_engineering] Input shape: {data.shape}")

    data = build_features(data)
    print(f"[feature_engineering] Output shape: {data.shape}")

    data = add_event_timestamp(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"[feature_engineering] Wrote features to {OUTPUT_PATH}")

    write_feature_store_source(data)


if __name__ == "__main__":
    main()