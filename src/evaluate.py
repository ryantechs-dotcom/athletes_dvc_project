"""
evaluate.py
-----------
Stage 6 of the pipeline: evaluate the trained model on the held-out
test set.

Input:  models/model_{feature_version}.pkl, data/test/test.csv
Output: reports/evaluation.txt

Metrics are logged onto the SAME MLflow run that trained the model
(looked up via the model version's run_id), so RMSE/MAE/R2 stay
attached to the run carrying feature_version/n_estimators tags rather
than living in a disconnected, generically-named run.
"""

import argparse
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import prepare_xy  # noqa: E402  pylint: disable=wrong-import-position,no-name-in-module

TEST_PATH = PROJECT_ROOT / "data" / "test" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_feature_version(cli_value: str) -> str:
    """Resolve feature_version from CLI arg, falling back to params.yaml."""
    if cli_value is not None:
        return cli_value
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        all_params = yaml.safe_load(f) or {}
    return all_params.get("features", {}).get("version", "v1")


def find_latest_run_id(feature_version: str):
    """Find the most recent MLflow run tagged with this feature_version,
    so evaluation metrics can be logged onto the training run itself."""
    mlflow.set_experiment("crossfit-total-lift")
    runs = mlflow.search_runs(
        filter_string=f"params.feature_version = '{feature_version}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return None
    return runs.iloc[0]["run_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-version", type=str, default=None)
    args = parser.parse_args()

    feature_version = load_feature_version(args.feature_version)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"model_{feature_version}.pkl"
    report_path = REPORTS_DIR / f"evaluation_{feature_version}.txt"

    if not model_path.exists():
        print(f"[evaluate] ERROR: {model_path} not found. "
              f"Run src/train.py --feature-version {feature_version} first.",
              file=sys.stderr)
        sys.exit(1)
    if not TEST_PATH.exists():
        print(f"[evaluate] ERROR: {TEST_PATH} not found. Run src/split_data.py first.",
              file=sys.stderr)
        sys.exit(1)

    bundle = joblib.load(model_path)
    model = bundle["model"]
    trained_features = bundle["features"]

    test_data = pd.read_csv(TEST_PATH)
    x_test, y_test = prepare_xy(test_data)
    x_test = x_test.reindex(columns=trained_features, fill_value=0)

    preds = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    print(f"[evaluate] feature_version={feature_version}  RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")

    run_id = find_latest_run_id(feature_version)
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
            mlflow.log_param("test_rows", len(test_data))
        print(f"[evaluate] Logged metrics onto training run {run_id}")
    else:
        print(f"[evaluate] WARNING: no MLflow run found for feature_version="
              f"'{feature_version}'; metrics not logged to MLflow.", file=sys.stderr)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Feature version:   {feature_version}\n")
        f.write(f"Test set size:     {len(test_data)}\n")
        f.write(f"Feature count:     {len(trained_features)}\n")
        f.write("Target:            total_lift (deadlift + candj + snatch + backsq)\n\n")
        f.write(f"RMSE:              {rmse:.3f}\n")
        f.write(f"MAE:               {mae:.3f}\n")
        f.write(f"R^2:               {r2:.4f}\n")
    print(f"[evaluate] Wrote {report_path}")


if __name__ == "__main__":
    main()