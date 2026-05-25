"""
Central configuration for the Insurance Forecast project.
All paths, constants and tunables live here so other modules
do not hard-code anything.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (folder that contains this `config/` package)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data folders
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Processed parquet files
MONTHLY_INSURANCE_PAID_FILE: Path = PROCESSED_DIR / "monthly_insurance_paid.parquet"
MONTHLY_DENIALS_FILE: Path = PROCESSED_DIR / "monthly_denials.parquet"
MONTHLY_FEATURES_FILE: Path = PROCESSED_DIR / "monthly_features.parquet"

# ---------------------------------------------------------------------------
# Model folders
# ---------------------------------------------------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "models"
PROPHET_DIR: Path = MODELS_DIR / "prophet"
XGBOOST_DIR: Path = MODELS_DIR / "xgboost"

# Concrete model file names
PROPHET_INSURANCE_PAID_FILE: Path = PROPHET_DIR / "insurance_paid_prophet.model"
PROPHET_DENIALS_FILE: Path = PROPHET_DIR / "denials_prophet.model"
XGB_INSURANCE_PAID_FILE: Path = XGBOOST_DIR / "insurance_paid_xgb.model"
XGB_DENIALS_FILE: Path = XGBOOST_DIR / "denials_xgb.model"

# ---------------------------------------------------------------------------
# Forecasting / business defaults
# ---------------------------------------------------------------------------
HOLIDAY_COUNTRY: str = "IN"            # India-style holidays
ALLOWED_HORIZONS: list[int] = [1, 3, 6]  # months forward we can forecast
DEFAULT_HORIZON: int = 3

# Columns expected in the raw CSVs
RAW_REQUIRED_COLUMNS: list[str] = [
    "patient_id",
    "insurance_type",
    "financial_class",
    "service_date",
    "charge",
    "insurance_paid",
    "denial_amount",
    "patient_responsibility",
]

# Make sure every folder exists when the module is imported
for _p in [RAW_DIR, PROCESSED_DIR, PROPHET_DIR, XGBOOST_DIR]:
    _p.mkdir(parents=True, exist_ok=True)
