"""
split_data.py
--------------
Stage 4 of the pipeline: create a reproducible train/test split.

Feature store integration: core registered features are retrieved via
Feast's `get_historical_features()` API against a versioned
FeatureService (crossfit_fs_v1 or crossfit_fs_v2, selected via
params.yaml -> features.version) -- demonstrating real point-in-time
feature retrieval, not just a file read. Any remaining engineered
columns that are not individually registered with Feast (e.g. wide,
dynamic multi-hot survey encodings whose names vary depending on which
answers appear in the raw data) are merged in directly from
data/processed/crossfit_features.csv. This hybrid approach reflects a
common real-world pattern: a feature store serves a core, versioned
feature set, while wide/dynamic one-off features are still handled by
the regular data pipeline.

IMPORTANT: "non-Feast columns" are determined against the UNION of all
known feature versions' fields (FEATURES_V1 | FEATURES_V2), not just the
fields belonging to whichever version is currently selected. Otherwise a
v1 request would incorrectly re-merge v2's fields back in from the CSV
(since they wouldn't be in v1's feast_cols list), erasing the difference
between versions. Only columns that are Feast-unmanaged in EVERY version
(e.g. the raw leakage columns deadlift/candj/snatch/backsq) should ever
land in non_feast_cols.

Input:  data/processed/crossfit_features.csv (for entity ids/timestamps
        and non-Feast-managed columns)
        Feast offline store (for the registered core features)
Outputs:
    data/train/train.csv
    data/test/test.csv
    reports/split_documentation.txt  (split ratio, seed, feature version, features used)
"""

import sys
from pathlib import Path

import pandas as pd
import yaml
from feast import FeatureStore
from sklearn.model_selection import train_test_split
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from random_seed import set_global_seed  # noqa: E402  pylint: disable=wrong-import-position,import-error
from feature_definitions import FEATURES_V1, FEATURES_V2  # noqa: E402  pylint: disable=wrong-import-position,import-error

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "crossfit_features.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "train" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test" / "test.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
SPLIT_DOC_PATH = REPORTS_DIR / "split_documentation.txt"

TARGET_COLUMN = "total_lift"

# Union of every field defined across all known feature versions. Used to
# determine which raw CSV columns are genuinely Feast-unmanaged (as opposed
# to just "not part of the currently selected version").
ALL_FEAST_FIELDS = {f.name for f in FEATURES_V1} | {f.name for f in FEATURES_V2}

# TARGET_COLUMN (total_lift) is defined in FEATURES_V2 but not FEATURES_V1.
# It must always reach full_df regardless of feature version, since it's
# the training target, not just another feature. Excluding it here means:
# if the active version's Feast service doesn't return it (v1), it still
# falls through to the CSV-merge path below rather than disappearing.
FEAST_GATED_FIELDS = ALL_FEAST_FIELDS - {TARGET_COLUMN}


def load_params() -> dict:
    """Load split parameters and feature version from params.yaml."""
    defaults = {
        "test_size": 0.2,
        "random_seed": 42,
        "feature_version": "v2",
    }

    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            all_params = yaml.safe_load(f) or {}

        split_params = all_params.get("split", {})
        feature_params = all_params.get("features", {})

        return {
            **defaults,
            **split_params,
            "feature_version": feature_params.get("version", defaults["feature_version"]),
        }

    return defaults


def fetch_features_via_feast(raw_features: pd.DataFrame, feature_version: str) -> pd.DataFrame:
    """
    Retrieve the registered features for the given version via Feast's
    get_historical_features() against the matching FeatureService, then
    merge in any remaining columns (e.g. dynamic multi-hot survey
    encodings) that aren't individually registered under ANY feature
    version.
    """
    if "athlete_id" not in raw_features.columns or "event_timestamp" not in raw_features.columns:
        raise ValueError(
            "raw_features must contain 'athlete_id' and 'event_timestamp' columns "
            "for point-in-time Feast retrieval."
        )

    entity_df = raw_features[["athlete_id", "event_timestamp"]].copy()
    entity_df = entity_df.dropna(subset=["athlete_id"])
    entity_df["athlete_id"] = entity_df["athlete_id"].astype("int64")
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    store = FeatureStore(repo_path=str(PROJECT_ROOT))

    service_name = f"crossfit_fs_{feature_version}"
    try:
        feature_service = store.get_feature_service(service_name)
    except Exception as exc:  # feast raises its own lookup error type
        raise ValueError(
            f"No registered FeatureService named '{service_name}'. "
            f"Check params.yaml 'features.version' and feature_definitions.py."
        ) from exc

    feast_result = store.get_historical_features(
        entity_df=entity_df,
        features=feature_service,
    ).to_df()

    # Fields actually returned for THIS version's service call.
    feast_cols = [
        f.name
        for projection in feature_service.feature_view_projections
        for f in projection.features
    ]

    # A column only counts as "non-Feast" (i.e. eligible for CSV merge) if it
    # isn't managed by Feast under ANY version -- not just the current one.
    # This is what keeps v1 and v2 requests from converging on the same
    # final column set.
    non_feast_cols = [
        c for c in raw_features.columns
        if c not in FEAST_GATED_FIELDS
        and c not in feast_cols
        and c not in ["athlete_id", "event_timestamp"]
    ]

    non_feast_df = raw_features[["athlete_id"] + non_feast_cols].copy()
    non_feast_df = non_feast_df.dropna(subset=["athlete_id"])
    non_feast_df["athlete_id"] = non_feast_df["athlete_id"].astype("int64")

    merged = feast_result.merge(non_feast_df, on="athlete_id", how="left")

    # total_lift is only defined in FEATURES_V2's schema. For a v1 request,
    # Feast won't return it and it isn't a literal CSV column either -- but
    # its raw components (deadlift/candj/snatch/backsq) are always carried
    # through as non-Feast-managed columns, so compute it directly rather
    # than treating its absence as an error. This mirrors train.py's own
    # prepare_xy() fallback for the same target.
    if TARGET_COLUMN not in merged.columns:
        components = ["deadlift", "candj", "snatch", "backsq"]
        if all(c in merged.columns for c in components):
            merged[TARGET_COLUMN] = merged[components].sum(axis=1)
            print(
                f"[split_data] Computed '{TARGET_COLUMN}' from components "
                f"{components} (not directly supplied by Feast/CSV for "
                f"feature version {feature_version})."
            )

    print(
        f"[split_data] Retrieved {len(feast_cols)} Feast features "
        f"using Feature Version {feature_version.upper()} "
        f"({service_name}); {len(non_feast_cols)} additional columns merged from CSV."
    )

    return merged.drop(columns=["event_timestamp"], errors="ignore")


def write_split_documentation(
    feature_version: str,
    test_size: float,
    random_seed: int,
    feature_columns: list,
    train_rows: int,
    test_rows: int,
) -> None:
    """Write a plain-text record of how the split was produced."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "Train/Test Split Documentation",
        "=" * 40,
        f"Feature version : {feature_version}",
        f"Test size       : {test_size}",
        f"Random seed     : {random_seed}",
        f"Train rows      : {train_rows}",
        f"Test rows       : {test_rows}",
        "",
        "Feature columns used:",
    ]
    lines.extend(f"  - {col}" for col in feature_columns)

    with open(SPLIT_DOC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[split_data] Wrote split documentation to {SPLIT_DOC_PATH}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-version",
        type=str,
        choices=["v1", "v2"],
        default=None,
        help="Override feature version from params.yaml.",
    )

    args = parser.parse_args()

    params = load_params()

    test_size = params["test_size"]
    random_seed = set_global_seed()

    feature_version = (
        args.feature_version
        if args.feature_version is not None
        else params["feature_version"]
    )

    if not INPUT_PATH.exists():
        print(f"[split_data] ERROR: {INPUT_PATH} not found. "
              f"Run src/feature_engineering.py first.", file=sys.stderr)
        sys.exit(1)

    raw_features = pd.read_csv(INPUT_PATH)

    full_df = fetch_features_via_feast(raw_features, feature_version)

    if TARGET_COLUMN not in full_df.columns:
        print(f"[split_data] ERROR: target column '{TARGET_COLUMN}' not found "
              f"after Feast retrieval/merge. Available columns: "
              f"{list(full_df.columns)}", file=sys.stderr)
        sys.exit(1)

    train_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        random_state=random_seed,
    )

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    feature_columns = [c for c in full_df.columns if c != "athlete_id"]
    write_split_documentation(
        feature_version=feature_version,
        test_size=test_size,
        random_seed=random_seed,
        feature_columns=feature_columns,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )

    print(
        f"[split_data] Wrote {len(train_df)} train rows to {TRAIN_PATH} "
        f"and {len(test_df)} test rows to {TEST_PATH}."
    )


if __name__ == "__main__":
    main()