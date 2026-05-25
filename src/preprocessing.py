"""
Step 1 of the pipeline: load raw CSV files from ``data/raw/`` and
aggregate them to monthly level by ``insurance_type`` and ``financial_class``.

Run from the project root with:
    python -m src.preprocessing
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import (
    MONTHLY_DENIALS_FILE,
    MONTHLY_FEATURES_FILE,
    MONTHLY_INSURANCE_PAID_FILE,
    RAW_DIR,
    RAW_REQUIRED_COLUMNS,
)


def _read_one_csv(path: Path) -> pd.DataFrame:
    """Read one CSV and validate that the required columns exist."""
    df = pd.read_csv(path)
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return df[RAW_REQUIRED_COLUMNS].copy()


def load_raw_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Concatenate every CSV in ``raw_dir`` into one DataFrame."""
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            "Add files like claims_2023.csv, claims_2024.csv, claims_2025.csv."
        )

    frames = [_read_one_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # Parse dates and clean numeric columns
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    df = df.dropna(subset=["service_date"])
    numeric_cols = ["charge", "insurance_paid", "denial_amount", "patient_responsibility"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the claim-level DataFrame to a single monthly row per
    (month, insurance_type, financial_class) combination.
    """
    df = df.copy()
    # First day of the month -> nice for time-series indexes
    df["month"] = df["service_date"].dt.to_period("M").dt.to_timestamp()

    grouped = (
        df.groupby(["month", "insurance_type", "financial_class"], as_index=False)
        .agg(
            total_insurance_paid=("insurance_paid", "sum"),
            total_denials=("denial_amount", "sum"),
            total_charges=("charge", "sum"),
            sample_count=("patient_id", "count"),
            total_patient_responsibility=("patient_responsibility", "sum"),
            denial_count=("denial_amount", lambda s: int((s > 0).sum())),
        )
        .sort_values(["insurance_type", "financial_class", "month"])
        .reset_index(drop=True)
    )
    return grouped


def save_processed(monthly: pd.DataFrame) -> None:
    """Save monthly aggregates as parquet files."""
    # Insurance-paid view
    insurance_paid_df = monthly[
        [
            "month",
            "insurance_type",
            "financial_class",
            "total_insurance_paid",
            "sample_count",
            "total_charges",
        ]
    ].copy()
    insurance_paid_df.to_parquet(MONTHLY_INSURANCE_PAID_FILE, index=False)

    # Denials view
    denials_df = monthly[
        [
            "month",
            "insurance_type",
            "financial_class",
            "total_denials",
            "denial_count",
            "sample_count",
        ]
    ].copy()
    denials_df.to_parquet(MONTHLY_DENIALS_FILE, index=False)

    # Combined feature table (used by XGBoost)
    monthly.to_parquet(MONTHLY_FEATURES_FILE, index=False)


def run() -> pd.DataFrame:
    """End-to-end preprocessing entry point."""
    raw = load_raw_data()
    monthly = aggregate_monthly(raw)
    save_processed(monthly)
    print(f"Saved {len(monthly):,} monthly rows to data/processed/")
    return monthly


if __name__ == "__main__":
    run()
