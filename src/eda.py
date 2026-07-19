"""
eda.py
------
Task 4: Exploratory Data Analysis comparing the raw dataset
(data/raw/crossfit.csv) and the cleaned/transformed dataset
(data/processed/crossfit.csv).

Produces, for each dataset stage:
    - Summary statistics
    - Missing value analysis
    - Distribution analysis (selected columns)
    - Outlier visualization (selected columns)
    - Correlation analysis (selected columns)
    - Key observations

All plots are saved to outputs/eda/, prefixed "raw_" or "processed_".

Naming note: this script labels the two dataset stages "raw" and
"processed" (NOT "v1"/"v2"). "v1"/"v2" is reserved exclusively for
Feast feature versions defined in feature_definitions.py; using it
here as well would create a naming collision between two unrelated
versioning axes (dataset snapshot vs. feature definition).

total_lift is intentionally NOT analyzed here: it doesn't exist in
either data/raw/crossfit.csv or data/processed/crossfit.csv -- it's
only computed downstream in feature_engineering.py. EDA on total_lift
would need to run against data/processed/crossfit_features.csv
instead; out of scope for this stage.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAW_PATH = "data/raw/crossfit.csv"
PROCESSED_PATH = "data/processed/crossfit.csv"

OUTPUT_DIR = "outputs/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# Columns to focus on for EDA. Names match the actual raw dataset
# columns (see ingest.py / preprocess.py), NOT the engineered
# total_lift/target naming used later in the pipeline.
EDA_COLUMNS = [
    "age",
    "height",
    "weight",
    "backsq",
    "candj",
    "snatch",
    "deadlift",
]


def get_numeric_columns(df):
    """Return only the selected numeric columns that exist in the dataset."""
    return [
        col
        for col in EDA_COLUMNS
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]


def print_summary_statistics(df: pd.DataFrame):
    """Print dataset overview and summary statistics."""
    print("\nDataset Shape")
    print(df.shape)

    print("\nColumn Types")
    print(df.dtypes)

    print("\nSummary Statistics")
    print(df.describe(include="all"))


def plot_missing_values(df: pd.DataFrame, stage: str):
    """Plot missing values."""
    print("\nMissing Values")
    missing = df.isnull().sum()
    print(missing)

    missing = missing[missing > 0]

    if missing.empty:
        return

    plt.figure(figsize=(10, 5))
    missing.sort_values().plot(kind="bar")
    plt.title(f"{stage} Missing Values")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{stage}_missing_values.png")
    plt.close()


def plot_distributions(df: pd.DataFrame, stage: str, numeric_cols):
    """Create histograms for selected numeric columns."""
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), bins=30)
        plt.title(f"{stage} Distribution: {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{stage}_{col}_distribution.png")
        plt.close()


def plot_boxplots(df: pd.DataFrame, stage: str, numeric_cols):
    """Create boxplots for selected numeric columns."""
    for col in numeric_cols:
        plt.figure(figsize=(6, 2.5))
        sns.boxplot(x=df[col])
        plt.title(f"{stage} Boxplot: {col}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{stage}_{col}_boxplot.png")
        plt.close()


def plot_correlation_matrix(df: pd.DataFrame, stage: str, numeric_cols):
    """Create a correlation heatmap."""
    if len(numeric_cols) < 2:
        return

    corr = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
    )
    plt.title(f"{stage} Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{stage}_correlation_matrix.png")
    plt.close()


def print_key_observations(df: pd.DataFrame):
    """Print simple dataset observations."""
    print("\nKey Observations")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Total Missing Values: {df.isnull().sum().sum()}")
    print("\nEDA Complete.\n")


def perform_eda(df: pd.DataFrame, stage: str):
    """Run the full EDA pipeline."""
    print("=" * 80)
    print(f"EDA FOR {stage.upper()}")
    print("=" * 80)

    numeric_cols = get_numeric_columns(df)

    print_summary_statistics(df)
    plot_missing_values(df, stage)
    plot_distributions(df, stage, numeric_cols)
    plot_boxplots(df, stage, numeric_cols)
    plot_correlation_matrix(df, stage, numeric_cols)
    print_key_observations(df)


def main():
    """Run EDA on both the raw and processed dataset stages."""
    df_raw = pd.read_csv(RAW_PATH)
    df_processed = pd.read_csv(PROCESSED_PATH)

    perform_eda(df_raw, "raw")
    perform_eda(df_processed, "processed")

    print("=" * 80)
    print("EDA COMPLETE")
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()