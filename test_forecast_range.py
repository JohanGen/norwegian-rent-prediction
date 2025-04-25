#!/usr/bin/env python3
# test_mlp_forecast.py
import joblib
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS – adjust these as needed
ROOM_KEY = "1rom"                       # one of: "1rom", "2rom", "3rom", "4rom", "5rom"
REGION   = "Hele landet"                # exact match to your Region values
YEARS    = list(range(2025, 2031))      # forecast horizon
MODEL_FP = Path(__file__).parent / "models" / f"model_{ROOM_KEY}_prod.joblib"
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1) load your trained pipeline (preprocessor + MLPRegressor)
    if not MODEL_FP.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FP}")
    pipe = joblib.load(MODEL_FP)

    # 2) build the future‐data DataFrame with the exact features your model expects
    #    (Year, Year2, Year_scaled, styringsrente, utlansrente, Region)
    future = pd.DataFrame({
        "Region":        [REGION] * len(YEARS),
        "Year":          YEARS,
        "Year2":         [y**2 for y in YEARS],
        "Year_scaled":   [(y - 2012) / (2030 - 2012) for y in YEARS],
        # ---- dummy interest rates; replace with your own projections if you have them
        "styringsrente": [4.5] * len(YEARS),
        "utlansrente":   [5.5] * len(YEARS),
    })

    # 3) predict
    preds = pipe.predict(future)

    # 4) print out
    print(f"Forecast for '{REGION}' ({ROOM_KEY}) from {YEARS[0]}–{YEARS[-1]}:")
    for y, p in zip(YEARS, preds):
        print(f"  {y}: {p:,.0f} kr/mnd")

if __name__ == "__main__":
    main()







