from __future__ import annotations
import re
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ─── Konstanter ──────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATA_XLSX   = ROOT / "data" / "Bok1.xlsx"
MODELS_DIR  = ROOT / "models"
PLOTS_DIR   = ROOT / "results" / "plots"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

ROOM_KEYS   = {
    "1rom": "1 rom",
    "2rom": "2 rom",
    "3rom": "3 rom",
    "4rom": "4 rom",
    "5rom": "5 rom"
}
TRAIN_YEARS = list(range(2012, 2023))
TEST_YEAR   = 2024
TARGET_REGION = "Hele landet"

# ─── Les og smelt data ────────────────────────────────────────────────────
def load_long_df(xlsx_path: Path) -> pd.DataFrame:
    raw   = pd.read_excel(xlsx_path, header=None)
    ncols = raw.shape[1]
    npairs = (ncols - 2)//2

    rows = []
    for i in range(1, len(raw)):
        region = raw.iat[i,0]
        room   = str(raw.iat[i,1]).strip()
        for k in range(npairs):
            yr_col, rent_col = 2+2*k, 3+2*k
            year, rent = raw.iat[i, yr_col], raw.iat[i, rent_col]
            try:
                year = int(year); rent = float(rent)
            except:
                continue
            rows.append({
                "Region":   region,
                "RoomType": room,
                "Year":     year,
                "Rent":     rent
            })

    df = pd.DataFrame(rows)
    df["Year2"]       = df["Year"]**2
    df["Year_scaled"] = (df["Year"] - 2012)/(2030-2012)
    df["Weight"]      = np.exp((df["Year"] - TEST_YEAR)/3.0)
    return df[df["Year"].between(2012, TEST_YEAR)]

# ─── Rentekurve ─────────────────────────────────────────────────────────
def load_rates_df() -> pd.DataFrame:
    data = {
        "Year":           [2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012],
        "styringsrente": [4.5,3.54,1.33,0.08,0.36,1.15,0.57,0.50,0.55,1.05,1.49,1.50,1.55],
        "utlansrente":   [5.5,4.54,2.33,1.08,1.36,2.15,1.57,1.50,1.55,2.05,2.49,2.50,2.55],
    }
    return pd.DataFrame(data)[["Year","styringsrente","utlansrente"]]

def main():
    # 1) Last inn data
    df    = load_long_df(DATA_XLSX)
    rates = load_rates_df()
    df    = df.merge(rates, on="Year", how="left")

    mae_eval = {}
    r2_eval  = {}

    for key, label in ROOM_KEYS.items():
        sub = df[df["RoomType"].str.startswith(label, na=False)]
        if sub.empty:
            print(f"⚠️ Hopper over '{label}' — ingen data.")
            continue

        # 2) Split train/test
        train = sub[sub["Year"].isin(TRAIN_YEARS)]
        test  = sub[sub["Year"] == TEST_YEAR]

        X_tr = train[["Year","Year2","Year_scaled","utlansrente","styringsrente","Region"]]
        y_tr = train["Rent"]
        w_tr = train["Weight"]

        X_te = test[["Year","Year2","Year_scaled","utlansrente","styringsrente","Region"]]
        y_te = test["Rent"]

        # 3) Bygg pipeline
        pre = ColumnTransformer([
            ("num", StandardScaler(), ["Year","Year2","Year_scaled","utlansrente"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["styringsrente","Region"]),
        ])
        xgb = XGBRegressor(
            n_estimators=600,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )
        pipe = Pipeline([("pre", pre), ("xgb", xgb)])

        # 4) Tren med vekter
        pipe.fit(X_tr, y_tr, xgb__sample_weight=w_tr)

        # 5) Evaluer
        y_pred = pipe.predict(X_te)
        mae_eval[key] = mean_absolute_error(y_te, y_pred)
        r2_eval[key]  = r2_score(y_te, y_pred)

        # 6) Lagre eval‐modell
        joblib.dump(pipe, MODELS_DIR/f"model_{key}_eval.joblib")

        # 7) Tren full data (prod‐modell)
        pipe_full = Pipeline([("pre", pre), ("xgb", xgb)])
        pipe_full.fit(
            sub[["Year","Year2","Year_scaled","utlansrente","styringsrente","Region"]],
            sub["Rent"],
            xgb__sample_weight=sub["Weight"]
        )
        joblib.dump(pipe_full, MODELS_DIR/f"model_{key}_prod.joblib")

        # ─── Kun PNG for region “Hele landet” ─────────────────────────────
        hl = sub[sub["Region"] == TARGET_REGION]
        if not hl.empty:
            X_hl = hl[["Year","Year2","Year_scaled","utlansrente","styringsrente","Region"]]
            preds_hl = pipe_full.predict(X_hl)

            plt.figure(figsize=(8,5))
            plt.plot(hl["Year"], hl["Rent"], marker="o", label="Faktisk")
            plt.plot(hl["Year"], preds_hl, marker="s", label="Modell")
            plt.title(f"{label} i {TARGET_REGION}: Faktisk vs Predikert 2012–2024")
            plt.xlabel("År")
            plt.ylabel("Leiepris (kr/mnd)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR/f"{key}_helelandet_actual_vs_pred.png")
            plt.close()

    # 8) Oppdater eval‐summary
    pd.DataFrame({"MAE":mae_eval,"R2":r2_eval}) \
      .to_csv(MODELS_DIR/"mae_eval_summary.csv")

    print("Ferdig — modeller, summary og én PNG per rom for Hele landet er oppdatert.")

if __name__ == "__main__":
    main()




