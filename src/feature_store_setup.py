"""
feature_store_setup.py
------------------------
Registers the Feast feature definitions (entity, source, feature view)
and materializes them into the online store. Run after
feature_engineering.py has written the parquet source, and before
split_data.py retrieves features for training.

Equivalent to running, from the project root:
    feast apply
    feast materialize-incremental <now>

but wrapped as a Python script so it's part of the automated pipeline
(dvc.yaml / run_all.py) rather than a manual CLI step.
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list):
    """Run a command from the project root (where feature_store.yaml lives)."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if result.returncode != 0:
        print(f"[feature_store_setup] Command failed (exit {result.returncode}): "
              f"{' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    """Apply Feast feature definitions and materialize the online store."""
    parquet_path = PROJECT_ROOT / "data" / "feature_store" / "crossfit_features.parquet"
    if not parquet_path.exists():
        print(f"[feature_store_setup] ERROR: {parquet_path} not found. "
              f"Run src/feature_engineering.py first.", file=sys.stderr)
        sys.exit(1)

    run(["feast", "apply"])

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run(["feast", "materialize-incremental", now])

    print("[feature_store_setup] Feast feature definitions applied and materialized.")


if __name__ == "__main__":
    main()