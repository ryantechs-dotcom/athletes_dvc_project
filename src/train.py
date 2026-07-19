"""
train.py
--------
Stage 5 of the pipeline: train a model on the train split.

Target variable: total_lift (deadlift + candj + snatch + backsq) — a
regression problem predicting overall competition strength from
demographic, lifestyle, and experience features.

Leakage prevention: deadlift, candj, snatch, and backsq are EXCLUDED
from the feature set, since total_lift is their direct sum. athlete_id
is also EXCLUDED — it is retained through the pipeline only as the
Feast entity key, not as a model feature (see feature_engineering.py).

Input:  data/train/train.csv
Output: models/model_{feature_version}.pkl
"""

import argparse
import sys
from pathlib import Path

import joblib
import mlflow
from mlflow.models import infer_signature
import pandas as pd
import yaml
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from random_seed import set_global_seed  # noqa: E402  pylint: disable=wrong-import-position,import-error

TRAIN_PATH = PROJECT_ROOT / "data" / "train" / "train.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

LIFT_COMPONENT_COLS = ['deadlift', 'candj', 'snatch', 'backsq']
TARGET_COL = 'total_lift'
NON_FEATURE_COLS = ['athlete_id']


def load_params() -> dict:
    """Load training parameters from params.yaml with defaults.

    random_seed is sourced from the same 'split' section split_data.py
    reads, so both stages use one seed value from a single place in
    params.yaml rather than each defaulting independently.
    """

    defaults = {
        "model_type": "xgboost",
        "random_seed": 42,
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
        },
    }

    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        all_params = yaml.safe_load(f) or {}

    train_params = all_params.get("train", {})
    split_params = all_params.get("split", {})
    feature_version = all_params.get("features", {}).get("version", "v1")

    return {
        **defaults,
        **train_params,
        "random_seed": split_params.get("random_seed", defaults["random_seed"]),
        "feature_version": feature_version,
    }


def prepare_xy(data: pd.DataFrame):
    """Build (x, y) from a dataframe, excluding lift components, the
    target, and non-feature identifier columns to prevent leakage."""
    data = data.copy()

    if TARGET_COL not in data.columns:
        missing = [c for c in LIFT_COMPONENT_COLS if c not in data.columns]
        if missing:
            raise ValueError(
                f"Cannot compute '{TARGET_COL}': missing component columns {missing}"
            )
        data[TARGET_COL] = (
            data['deadlift'] + data['candj'] + data['snatch'] + data['backsq']
        )

    y = data[TARGET_COL]
    excluded = LIFT_COMPONENT_COLS + [TARGET_COL] + NON_FEATURE_COLS
    x = data.drop(columns=[c for c in excluded if c in data.columns])

    return x, y


def build_model(params: dict, random_seed: int):
    """Construct the model specified by params['model_type'] with fixed hyperparameters."""
    model_type = params["model_type"].lower()
    hp = params["hyperparameters"]

    if model_type == "xgboost":
        return XGBRegressor(
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth", 6),
            learning_rate=hp.get("learning_rate", 0.05),
            random_state=random_seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model_type: {model_type}. Add support in build_model().")


def main():
    """Train the model on the train split and save it with its feature list."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-version", type=str, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    params = load_params()
    random_seed = set_global_seed(params["random_seed"])

    if args.feature_version is not None:
        params["feature_version"] = args.feature_version

    if args.n_estimators is not None:
        params["hyperparameters"]["n_estimators"] = args.n_estimators

    if args.max_depth is not None:
        params["hyperparameters"]["max_depth"] = args.max_depth

    if args.learning_rate is not None:
        params["hyperparameters"]["learning_rate"] = args.learning_rate

    if not TRAIN_PATH.exists():
        print(f"[train] ERROR: {TRAIN_PATH} not found. Run src/split_data.py first.",
              file=sys.stderr)
        sys.exit(1)

    data = pd.read_csv(TRAIN_PATH)
    print(f"[train] Train shape: {data.shape}")

    x, y = prepare_xy(data)
    print(
        f"[train] Feature count: {x.shape[1]} "
        f"(excluded {LIFT_COMPONENT_COLS + NON_FEATURE_COLS} to avoid leakage)"
    )

    mlflow.set_experiment("crossfit-total-lift")
    feature_version = params["feature_version"]

    run_name = (
        args.run_name
        if args.run_name
        else f"{feature_version}_xgboost_{params['hyperparameters']['n_estimators']}"
    )

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "feature_version": feature_version,
            "model_type": params["model_type"],
            "random_seed": random_seed,
            "n_estimators": params["hyperparameters"].get("n_estimators"),
            "max_depth": params["hyperparameters"].get("max_depth"),
            "learning_rate": params["hyperparameters"].get("learning_rate"),
            "feature_count": x.shape[1],
            "train_rows": len(data),
        })

        model = build_model(params, random_seed)
        model.fit(x, y)

        print(
            f"[train] Model fit complete ({params['model_type']}, "
            f"seed={random_seed})"
        )

        # --- signature + input example ---
        signature = infer_signature(x, model.predict(x))
        input_example = x.iloc[:5]

        # --- log + register the model ---
        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="crossfit_total_lift_model",
            signature=signature,
            input_example=input_example,
        )

        # --- tag the model VERSION (not just the run) with feature version ---
        client = mlflow.MlflowClient()
        client.set_model_version_tag(
            name="crossfit_total_lift_model",
            version=model_info.registered_model_version,
            key="feature_version",
            value=feature_version,
        )
        client.set_model_version_tag(
            name="crossfit_total_lift_model",
            version=model_info.registered_model_version,
            key="n_estimators",
            value=str(params["hyperparameters"].get("n_estimators")),
        )
        client.set_model_version_tag(
            name="crossfit_total_lift_model",
            version=model_info.registered_model_version,
            key="feature_count",
            value=str(x.shape[1]),
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"model_{feature_version}.pkl"
    joblib.dump({"model": model, "features": list(x.columns)}, model_path)
    print(f"[train] Saved model to {model_path}")


if __name__ == "__main__":
    main()