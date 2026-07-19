"""
run_experiments.py
------------------

Runs all experiment combinations defined in params.yaml.

For each experiment:
    1. Updates the active feature version.
    2. Updates the model hyperparameters.
    3. Runs:
            split_data.py
            train.py
            evaluate.py

Each train.py execution creates its own MLflow run.

This provides:
    - 2 Feast feature versions
    - 2 hyperparameter configurations
    - 4 MLflow experiments
"""

import copy
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_config():
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config):
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run(command):
    print(f"\n$ {' '.join(command)}")

    result = subprocess.run(command, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        sys.exit(result.returncode)


def main():

    original_config = load_config()

    try:

        experiments = original_config.get("experiments", [])

        if not experiments:
            raise ValueError("No experiments defined in params.yaml")

        print(f"\nRunning {len(experiments)} experiments...")

        for experiment in experiments:

            print("\n" + "=" * 70)
            print(f"Experiment: {experiment['name']}")
            print("=" * 70)

            config = copy.deepcopy(original_config)

            #
            # Select Feast Feature Version
            #
            config["features"]["version"] = experiment["feature_version"]

            #
            # Select Hyperparameters
            #
            config["train"]["hyperparameters"] = experiment["hyperparameters"]

            #
            # Save temporary params.yaml
            #
            save_config(config)

            #
            # Execute pipeline stages
            #
            run(["python", "src/split_data.py"])
            run(["python", "src/train.py"])
            run(["python", "src/evaluate.py"])

        print("\nAll experiments completed successfully.")

    finally:
        #
        # Restore original params.yaml
        #
        save_config(original_config)
        print("\nparams.yaml restored.")


if __name__ == "__main__":
    main()