"""
ingest.py
---------
Stage 1 of the pipeline: ingest raw data.

Locates a zip archive one parent directory above the project root
(i.e. sibling to `crossfit-ml-project/`), extracts the target CSV
(default: `athletes.csv`), and copies it into `data/raw/` as
`crossfit.csv` so downstream stages always read from a stable,
version-controlled path.

Usage:
    python src/ingest.py
    python src/ingest.py --zip-name athletes.zip --csv-name athletes.csv

This script is idempotent: re-running it simply re-extracts and
overwrites data/raw/crossfit.csv, which keeps it safe for `dvc repro`.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

# Project root = parent of this file's parent (src/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_CSV_NAME = "crossfit.csv"  # standardized name expected by preprocess.py


def find_zip(zip_name: str) -> Path:
    """
    Look for `zip_name` one directory above the project root.

    Example layout expected:
        parent_folder/
        ├── athletes.zip          <-- we look here
        └── crossfit-ml-project/  <-- PROJECT_ROOT
    """
    candidate = PROJECT_ROOT.parent / zip_name
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find '{zip_name}' at expected location: {candidate}\n"
            f"Expected the zip to sit one folder above the project root "
            f"({PROJECT_ROOT})."
        )
    return candidate


def extract_csv(zip_path: Path, csv_name: str, extract_to: Path) -> Path:
    """Extract a single CSV member from the zip into extract_to."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Allow the csv to be nested inside a subfolder in the zip
        matches = [n for n in names if n.endswith(csv_name)]
        if not matches:
            raise FileNotFoundError(
                f"'{csv_name}' not found inside {zip_path.name}. "
                f"Files in archive: {names}"
            )
        member = matches[0]
        zf.extract(member, path=extract_to)
        return extract_to / member


def main():
    """Run the ingestion stage end-to-end: locate zip, extract CSV, save standardized copy."""
    parser = argparse.ArgumentParser(description="Ingest raw athletes data from zip.")
    parser.add_argument(
        "--zip-name", default="athletes.zip",
        help="Name of the zip file located one parent directory above the project."
    )
    parser.add_argument(
        "--csv-name", default="athletes.csv",
        help="Name of the CSV file inside the zip to extract."
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] Project root: {PROJECT_ROOT}")
    zip_path = find_zip(args.zip_name)
    print(f"[ingest] Found zip: {zip_path}")

    tmp_extract_dir = RAW_DIR / "_tmp_extract"
    tmp_extract_dir.mkdir(exist_ok=True)

    try:
        extracted_path = extract_csv(zip_path, args.csv_name, tmp_extract_dir)
        print(f"[ingest] Extracted: {extracted_path}")

        final_path = RAW_DIR / OUTPUT_CSV_NAME
        shutil.copy(extracted_path, final_path)
        print(f"[ingest] Saved standardized copy to: {final_path}")
    finally:
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)

    print("[ingest] Done.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"[ingest] ERROR: {e}", file=sys.stderr)
        sys.exit(1)