"""
FastAPI entrypoint.

Run locally with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Make sure the project root is on sys.path when uvicorn is started from
# inside the api/ folder. This keeps `python -m`-style imports working.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (  # noqa: E402
    DenialForecastResponse,
    ForecastRequest,
    HealthResponse,
    InsurancePaidForecastResponse,
)
from src import inference  # noqa: E402

APP_VERSION = "1.0.0"

app = FastAPI(
    title="Insurance Claims Forecast API",
    description=(
        "Forecast insurance paid, denials and sample counts using "
        "Prophet + XGBoost (1-month, 3-month, or 6-month horizon)."
    ),
    version=APP_VERSION,
)

# Allow the API to be called from any browser-based dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": "Insurance Claims Forecast API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /forecast/insurance_paid",
            "POST /forecast/denials",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=APP_VERSION)


@app.post(
    "/forecast/insurance_paid",
    response_model=InsurancePaidForecastResponse,
    tags=["forecast"],
)
def forecast_insurance_paid(req: ForecastRequest) -> InsurancePaidForecastResponse:
    """Return next-N-month forecast for insurance paid."""
    try:
        forecasts = inference.forecast_insurance_paid(
            insurance_type=req.insurance_type,
            financial_class=req.financial_class,
            horizon=req.horizon,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Models not trained yet. Run training first. ({e})",
        )

    if not forecasts:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No model found for insurance_type='{req.insurance_type}' / "
                f"financial_class='{req.financial_class}'."
            ),
        )

    return InsurancePaidForecastResponse(
        insurance_type=req.insurance_type,
        financial_class=req.financial_class,
        horizon=req.horizon,
        forecasts=forecasts,
    )


@app.post(
    "/forecast/denials",
    response_model=DenialForecastResponse,
    tags=["forecast"],
)
def forecast_denials(req: ForecastRequest) -> DenialForecastResponse:
    """Return next-N-month forecast for denial amounts."""
    try:
        forecasts = inference.forecast_denials(
            insurance_type=req.insurance_type,
            financial_class=req.financial_class,
            horizon=req.horizon,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Models not trained yet. Run training first. ({e})",
        )

    if not forecasts:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No model found for insurance_type='{req.insurance_type}' / "
                f"financial_class='{req.financial_class}'."
            ),
        )

    return DenialForecastResponse(
        insurance_type=req.insurance_type,
        financial_class=req.financial_class,
        horizon=req.horizon,
        forecasts=forecasts,
    )
