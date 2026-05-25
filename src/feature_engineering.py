"""
Step 2 of the pipeline: enrich the monthly DataFrame with calendar /
seasonality / holiday features used by XGBoost.

Run from the project root with:
    python -m src.feature_engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.holidays import get_holiday_dates
from config.settings import MONTHLY_FEATURES_FILE


def add_time_features(df: pd.DataFrame, date_col: str = "month") -> pd.DataFrame:
    """Add calendar features derived from the ``date_col`` column."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])

    df["year"] = dt.dt.year
    df["month_num"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12.0)
    return df


def add_holiday_flag(df: pd.DataFrame, date_col: str = "month") -> pd.DataFrame:
    """Flag months that contain at least one holiday."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])

    holiday_dates = get_holiday_dates(
        start=str(dt.min().date()),
        end=str((dt.max() + pd.offsets.MonthEnd(12)).date()),
    )
    holiday_months = {(d.year, d.month) for d in holiday_dates}

    df["is_holiday"] = [int((d.year, d.month) in holiday_months) for d in dt]
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lag features within each (insurance_type, financial_class) group."""
    group_cols = group_cols or ["insurance_type", "financial_class"]
    target_cols = target_cols or ["total_insurance_paid", "total_denials", "sample_count"]
    lags = lags or [1, 2, 3, 6, 12]

    df = df.sort_values(group_cols + ["month"]).copy()

    for col in target_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby(group_cols)[col].shift(lag)

        df[f"{col}_roll3"] = df.groupby(group_cols)[col].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )

    return df


def build_feature_table(monthly: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature step to the monthly DataFrame."""
    df = add_time_features(monthly)
    df = add_holiday_flag(df)
    df = add_lag_features(df)
    return df


def run() -> pd.DataFrame:
    """Load the monthly features parquet, enrich it and overwrite it."""
    monthly = pd.read_parquet(MONTHLY_FEATURES_FILE)
    enriched = build_feature_table(monthly)
    enriched.to_parquet(MONTHLY_FEATURES_FILE, index=False)
    print(f"Feature table now has {enriched.shape[1]} columns and {len(enriched):,} rows")
    return enriched


if __name__ == "__main__":
    run()