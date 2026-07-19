"""
compare_versions.py
--------------------
Reads the latest MLflow run for each feature_version and writes a
comparison report, so the comparison is generated from the same
tracked metrics/params MLflow already holds, rather than re-parsing
text report files.

Output:
    reports/model_comparison.txt
"""

import sys
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.txt"


def get_latest_run(feature_version: str):
    mlflow.set_experiment("crossfit-total-lift")
    runs = mlflow.search_runs(
        filter_string=f"params.feature_version = '{feature_version}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return None if runs.empty else runs.iloc[0]


def main():
    v1 = get_latest_run("v1")
    v2 = get_latest_run("v2")

    if v1 is None or v2 is None:
        print("[compare_versions] ERROR: missing MLflow run for v1 or v2. "
              "Run train.py --feature-version v1 and --feature-version v2, "
              "then evaluate.py for each, first.", file=sys.stderr)
        sys.exit(1)

    rmse_diff = v2["metrics.rmse"] - v1["metrics.rmse"]
    r2_diff = v2["metrics.r2"] - v1["metrics.r2"]
    rmse_pct = (rmse_diff / v1["metrics.rmse"] * 100) if v1["metrics.rmse"] else float("nan")
    better_worse = "improved" if rmse_diff < 0 else "worsened" if rmse_diff > 0 else "was unchanged"

    lines = [
        "Feature Version Comparison (v1 vs v2, via MLflow)",
        "=" * 60,
        "",
        f"{'Metric':<18}{'v1':>14}{'v2':>16}{'Diff (v2-v1)':>16}",
        f"{'RMSE':<18}{v1['metrics.rmse']:>14.3f}{v2['metrics.rmse']:>16.3f}{rmse_diff:>16.3f}",
        f"{'MAE':<18}{v1['metrics.mae']:>14.3f}{v2['metrics.mae']:>16.3f}{v2['metrics.mae']-v1['metrics.mae']:>16.3f}",
        f"{'R^2':<18}{v1['metrics.r2']:>14.4f}{v2['metrics.r2']:>16.4f}{r2_diff:>16.4f}",
        "",
        f"Feature count:  v1={int(v1['params.feature_count'])}   v2={int(v2['params.feature_count'])}",
        f"MLflow run IDs: v1={v1['run_id']}   v2={v2['run_id']}",
        "",
        f"RMSE {better_worse} by {abs(rmse_diff):.3f} ({rmse_pct:+.1f}%) going from v1 to v2.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[compare_versions] Wrote {COMPARISON_PATH}")


if __name__ == "__main__":
    main()