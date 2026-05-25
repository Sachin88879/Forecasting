"""
Utility script to create synthetic claims CSVs in ``data/raw/``.
Useful so the rest of the pipeline can be run end-to-end without real PHI.

Run from the project root with:
    python -m scripts.generate_sample_data
"""

from __future__ import annotations

import sys
from pathlib import Path

import random
import csv

# Try to import optional dependencies; provide stdlib fallbacks so the script
# can still run in a minimal environment without `numpy` or `pandas`.
try:
    import numpy as np  # type: ignore[import]
except Exception:
    np = None

try:
    import pandas as pd  # type: ignore[import]
except Exception:
    pd = None


# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import RAW_DIR  # noqa: E402


if np is not None:
    RNG = np.random.default_rng(42)
else:
    class _SimpleRNG:
        def __init__(self, seed: int | None = None):
            self._r = random.Random(seed)

        def integers(self, low, high=None):
            if high is None:
                return int(self._r.randrange(low))
            return int(self._r.randrange(low, high))

        def choice(self, seq):
            return self._r.choice(seq)

        def uniform(self, a, b):
            return self._r.uniform(a, b)

        def random(self):
            return self._r.random()

    RNG = _SimpleRNG(42)

INSURANCE_TYPES = ["Private", "Government", "SelfPay", "Corporate"]
FINANCIAL_CLASSES = ["Inpatient", "Outpatient", "Emergency", "DayCare"]


def _generate_year(year: int, rows_per_month: int = 200):
    """Create one year of synthetic claim records and return a list of dicts."""
    records: list[dict] = []
    for month in range(1, 13):
        # Mild yearly seasonality + random noise. Use math.sin if numpy isn't present.
        if np is not None:
            seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 1) / 12.0)
        else:
            # lightweight fallback using math
            import math

            seasonal_factor = 1.0 + 0.15 * math.sin(2 * math.pi * (month - 1) / 12.0)

        for _ in range(int(rows_per_month * seasonal_factor)):
            day = int(RNG.integers(1, 28))
            ins = RNG.choice(INSURANCE_TYPES)
            fc = RNG.choice(FINANCIAL_CLASSES)
            charge = float(RNG.uniform(500, 10000))

            # Insurance pays more for Private/Corporate vs SelfPay
            pay_ratio = {
                "Private": 0.75,
                "Government": 0.65,
                "Corporate": 0.80,
                "SelfPay": 0.10,
            }[ins]
            paid = charge * pay_ratio * RNG.uniform(0.8, 1.0) * seasonal_factor
            denial = max(0.0, charge - paid - RNG.uniform(0, 200))
            denial = float(denial if RNG.random() < 0.2 else 0.0)
            patient_resp = max(0.0, charge - paid - denial)

            records.append(
                {
                    "patient_id": int(RNG.integers(10_000, 99_999)),
                    "insurance_type": ins,
                    "financial_class": fc,
                    "service_date": f"{year}-{month:02d}-{day:02d}",
                    "charge": round(charge, 2),
                    "insurance_paid": round(paid, 2),
                    "denial_amount": round(denial, 2),
                    "patient_responsibility": round(patient_resp, 2),
                }
            )
    return records


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in (2023, 2024, 2025):
        records = _generate_year(year)
        out = RAW_DIR / f"claims_{year}.csv"

        if pd is not None:
            df = pd.DataFrame(records)
            df.to_csv(out, index=False)
            print(f"Wrote {len(df):,} rows -> {out}")
        else:
            # Write CSV using the stdlib csv module
            if records:
                fieldnames = list(records[0].keys())
            else:
                fieldnames = [
                    "patient_id",
                    "insurance_type",
                    "financial_class",
                    "service_date",
                    "charge",
                    "insurance_paid",
                    "denial_amount",
                    "patient_responsibility",
                ]
            with open(out, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            print(f"Wrote {len(records):,} rows -> {out} (csv stdlib)")


if __name__ == "__main__":
    main()
