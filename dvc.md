# Dataset Versioning with DVC + Git

## Version Definitions

This project uses **DVC for dataset versioning** and **Git tags for identifying dataset snapshots**.

The dataset versions are:

- **v1** = `data/raw/crossfit.csv`
  - Raw dataset immediately after ingestion
  - No cleaning or preprocessing applied
  - Used for the raw baseline comparison model

- **v2** = `data/processed/crossfit.csv`
  - Cleaned and transformed dataset after preprocessing
  - Used for feature engineering and final model training


The dataset versions are created automatically through the DVC pipeline and
tracked through Git commits and tags.

---

# 1. One-Time Setup

Install required tools:

```powershell
pip install dvc
```

Initialize Git and DVC:

```powershell
git init
dvc init
```

Configure a local DVC remote:

```powershell
mkdir ..\dvc_storage

dvc remote add -d localstorage ..\dvc_storage
```

Commit the initial configuration:

```powershell
git add .dvc/config .gitignore

git commit -m "Configure DVC"
```

---

# 2. Creating Dataset Version v1

Dataset version 1 is created during the ingestion stage.

Run:

```powershell
python src/ingest.py --zip-name athletes.zip --csv-name athletes.csv
```

This creates:

```
data/raw/crossfit.csv
```

The pipeline then commits this state and creates the Git tag:

```
v1
```

The v1 dataset represents the original raw data before cleaning.

---

# 3. Creating Dataset Version v2

Dataset version 2 is created during preprocessing.

Run:

```powershell
python src/preprocess.py
```

This creates:

```
data/processed/crossfit.csv
```

The cleaned dataset is then committed and tagged:

```
v2
```

The v2 dataset includes:

- Data cleaning
- Missing value handling
- Invalid record removal
- Transformations required for modeling

---

# 4. Automated Versioning Workflow

The entire pipeline, including dataset versioning, can be executed using:

```powershell
python run_all.py
```

This performs:

1. Ingestion
2. Creation of dataset v1
3. Git commit and tag (`v1`)
4. Preprocessing
5. Creation of dataset v2
6. Git commit and tag (`v2`)
7. Feature engineering
8. Model training
9. Evaluation
10. v1 vs v2 comparison


DVC automatically creates and updates:

```
dvc.lock
```

which records:

- pipeline stages
- dependencies
- parameters
- dataset hashes
- output hashes

---

# 5. Viewing Dataset Versions

View available dataset versions:

```powershell
git tag
```

Example:

```
v1
v2
```

View commit history:

```powershell
git log --oneline --decorate
```

Example:

```
abc123 (tag: v2) v2: cleaned and transformed dataset
def456 (tag: v1) v1: raw ingested dataset
```

---

# 6. Switching Between Versions

To view the raw dataset version:

```powershell
git checkout v1
dvc checkout
```

The repository will restore the v1 pipeline state.

The raw dataset will be available at:

```
data/raw/crossfit.csv
```

---

To view the cleaned dataset version:

```powershell
git checkout v2
dvc checkout
```

The cleaned dataset will be available at:

```
data/processed/crossfit.csv
```

---

Return to the latest branch:

```powershell
git checkout main
dvc checkout
```

---

# 7. Reproducing the Pipeline

The complete pipeline can be reproduced using:

```powershell
dvc repro compare
```

DVC automatically executes all required upstream stages:

```
ingest
   |
preprocess
   |
feature_engineering
   |
split_data
   |
train
   |
evaluate
   |
compare
```

The EDA stage can be executed separately:

```powershell
dvc repro eda
```

---

# 8. Updating Dataset Versions

If preprocessing logic changes:

Example:

- modifying cleaning rules
- changing outlier thresholds
- updating feature transformations

Run:

```powershell
dvc repro
```

DVC will rerun affected stages and update:

```
dvc.lock
```

After verifying results:

```powershell
git add dvc.lock params.yaml

git commit -m "Updated preprocessing pipeline"

git tag v2.1
```

---

# Notes

- Git tags identify important pipeline states.
- DVC tracks the actual dataset files and their hashes.
- `dvc.lock` guarantees reproducibility by recording the exact pipeline configuration.
- Dataset files are included in this submission so the pipeline can be reproduced without requiring access to a remote DVC storage location.
- The raw v1 dataset is intentionally less accurate because it does not include preprocessing or feature engineering.
- The v2 dataset is the final modeling dataset used by:

```
data/processed/crossfit.csv
        |
        v
feature_engineering.py
        |
        v
train.py
```

This workflow demonstrates the impact of data quality improvements through reproducible dataset versioning.