"""
pipeline.py
-----------
Task 7: single-command, end-to-end reproducible pipeline.

Runs every stage in order:
    1. Raw data ingestion       (ingest.py)
    2. Data cleaning            (preprocess.py)
    3. Feature engineering      (feature_engineering.py)
    4. Train/test split         (split_data.py)
    5. Model training           (train.py)
    6. Model evaluation         (evaluate.py)

Requires no manual intervention (assignment instruction #5):

    python src/pipeline.py

Dataset versioning (v1 = raw, v2 = cleaned) is handled externally via
DVC + git tags — see DVC_SETUP.md — not by this script.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

STAGES = [
    ("Ingestion", SRC_DIR / "ingest.py"),
    ("Data cleaning", SRC_DIR / "preprocess.py"),
    ("Feature engineering", SRC_DIR / "feature_engineering.py"),
    ("Train/test split", SRC_DIR / "split_data.py"),
    ("Model training", SRC_DIR / "train.py"),
    ("Model evaluation", SRC_DIR / "evaluate.py"),
]


def run_stage(name: str, script_path: Path):
    """Run a single pipeline stage as a subprocess and exit on failure."""
    print("\n" + "=" * 60)
    print(f"STAGE: {name}  ({script_path.name})")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    if result.returncode != 0:
        print(f"\n[pipeline] FAILED at stage '{name}' "
              f"(exit code {result.returncode}). Stopping pipeline.",
              file=sys.stderr)
        sys.exit(result.returncode)


def main():
    """Run every pipeline stage in order, end-to-end, with no manual intervention."""
    print("[pipeline] Starting end-to-end pipeline run.")
    print(f"[pipeline] Project root: {PROJECT_ROOT}")
    print(f"[pipeline] Python interpreter: {sys.executable}")

    for name, script_path in STAGES:
        if not script_path.exists():
            print(f"[pipeline] ERROR: expected script not found: {script_path}",
                  file=sys.stderr)
            sys.exit(1)
        run_stage(name, script_path)

    print("\n" + "=" * 60)
    print("[pipeline] All stages completed successfully.")
    print("=" * 60)
    print("\nOutputs:")
    print("  data/processed/crossfit.csv")
    print("  data/processed/crossfit_features.csv")
    print("  data/train/train.csv, data/test/test.csv")
    print("  models/model.pkl")
    print("  reports/evaluation.txt, split_documentation.txt")


if __name__ == "__main__":
    main()