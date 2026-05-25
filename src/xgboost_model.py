"""
Train and use XGBoost regressors for ``total_insurance_paid`` and
``total_denials``.

Run from the project root with:
    python -m src.xgboost_model
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from config.settings import (
    MONTHLY_FEATURES_FILE,
    XGB_DENIALS_FILE,
    XGB_INSURANCE_PAID_FILE,
)
from src.feature_engineering import add_holiday_flag, add_time_features


# Columns NEVER used as features (targets or identifiers)
NON_FEATURE_COLS = {
    "month",
    "total_insurance_paid",
    "total_denials",
    "denial_count",
    "total_charges",
    "total_patient_responsibility",
    "sample_count",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _split_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y), with categorical columns one-hot encoded."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()

    # One-hot encode the two categorical columns
    X = pd.get_dummies(X, columns=["insurance_type", "financial_class"], drop_first=False)
    # XGBoost expects numeric only
    X = X.fillna(0)

    y = df[target].astype(float)
    return X, y


def _train_one_target(df: pd.DataFrame, target: str, save_path: Path) -> XGBRegressor:
    """Train one XGBoost regressor for ``target`` and save it with joblib."""
    X, y = _split_xy(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"{target}: MAE on holdout = {mae:,.2f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist model AND the column order so we can re-align at inference time
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, save_path)
    print(f"Saved XGBoost model -> {save_path.name}")
    return model


def train_all() -> None:
    df = pd.read_parquet(MONTHLY_FEATURES_FILE)
    # Drop rows where lag features are missing (start of the history)
    df = df.dropna()

    _train_one_target(df, "total_insurance_paid", XGB_INSURANCE_PAID_FILE)
    _train_one_target(df, "total_denials", XGB_DENIALS_FILE)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
def _build_future_frame(
    history: pd.DataFrame,
    insurance_type: str,
    financial_class: str,
    horizon_months: int,
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per future month, populated with
    time-based features and lag features that walk forward from the last
    known month of history.
    """
    sub = (
        history[
            (history["insurance_type"] == insurance_type)
            & (history["financial_class"] == financial_class)
        ]
        .sort_values("month")
        .copy()
    )
    if sub.empty:
        return sub

    last_month = pd.to_datetime(sub["month"].max())
    future_months = [last_month + pd.DateOffset(months=i + 1) for i in range(horizon_months)]

    rows: List[dict] = []
    history_window = sub.tail(12).copy()

    for fm in future_months:
        # Use the last 12 actuals (extended by previously predicted rows) for lags
        target_cols = ["total_insurance_paid", "total_denials", "sample_count"]
        row: dict = {
            "month": fm,
            "insurance_type": insurance_type,
            "financial_class": financial_class,
        }
        for col in target_cols:
            row[f"{col}_lag1"] = history_window[col].iloc[-1] if len(history_window) >= 1 else 0
            row[f"{col}_lag2"] = history_window[col].iloc[-2] if len(history_window) >= 2 else 0
            row[f"{col}_lag3"] = history_window[col].iloc[-3] if len(history_window) >= 3 else 0
            row[f"{col}_lag6"] = history_window[col].iloc[-6] if len(history_window) >= 6 else 0
            row[f"{col}_lag12"] = history_window[col].iloc[-12] if len(history_window) >= 12 else 0
            row[f"{col}_roll3"] = history_window[col].tail(3).mean() if len(history_window) >= 1 else 0

        rows.append(row)
        # Append a placeholder so subsequent lags can look back at it
        history_window = pd.concat(
            [
                history_window,
                pd.DataFrame(
                    [
                        {
                            "total_insurance_paid": row["total_insurance_paid_lag1"],
                            "total_denials": row["total_denials_lag1"],
                            "sample_count": row["sample_count_lag1"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        ).tail(12)

    future_df = pd.DataFrame(rows)
    future_df = add_time_features(future_df, date_col="month")
    future_df = add_holiday_flag(future_df, date_col="month")
    return future_df


def _predict(
    bundle_path: Path,
    insurance_type: str,
    financial_class: str,
    horizon_months: int,
) -> List[dict]:
    """Generic prediction helper used by the two forecast functions below."""
    bundle = joblib.load(bundle_path)
    model: XGBRegressor = bundle["model"]
    feature_columns: list[str] = bundle["feature_columns"]

    history = pd.read_parquet(MONTHLY_FEATURES_FILE)
    future = _build_future_frame(history, insurance_type, financial_class, horizon_months)
    if future.empty:
        return []

    X = pd.get_dummies(future, columns=["insurance_type", "financial_class"], drop_first=False)
    # Align columns to the training set
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns].fillna(0)

    preds = model.predict(X)
    return [
        {
            "month": pd.to_datetime(m).strftime("%Y-%m-%d"),
            "yhat": float(max(0.0, p)),
        }
        for m, p in zip(future["month"], preds)
    ]


def forecast_insurance_paid(
    insurance_type: str, financial_class: str, horizon_months: int
) -> List[dict]:
    return _predict(XGB_INSURANCE_PAID_FILE, insurance_type, financial_class, horizon_months)


def forecast_denials(
    insurance_type: str, financial_class: str, horizon_months: int
) -> List[dict]:
    return _predict(XGB_DENIALS_FILE, insurance_type, financial_class, horizon_months)


if __name__ == "__main__":
    train_all()
