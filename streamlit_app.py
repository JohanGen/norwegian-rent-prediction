from __future__ import annotations
import re
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ──────────────────── Page config ─────────────────────────────────
st.set_page_config(page_title="Leiepris-prediktor", page_icon="🏠", layout="wide")

st.title("Deep-Learning Prosjekt Leiepris-prediktor")

class CombinedRegressor:
    def __init__(self, xgb_pipe, trend_model):
        self.xgb   = xgb_pipe
        self.trend = trend_model

    def predict(self, X):
        xgb_pred   = self.xgb.predict(X)
        arr        = X[["Year", "Year2"]].values
        trend_pred = self.trend.predict(arr)
        return xgb_pred + trend_pred

# ──────────────────── find project root ──────────────────────────
THIS = Path(__file__).resolve()
CWD  = Path.cwd().resolve()
ROOT: Path | None = None
for cand in (THIS.parent, THIS.parent.parent, CWD):
    if (cand / "data" / "leiepris_data.xlsx").exists():
        ROOT = cand
        break
if ROOT is None:
    st.error("Fant ikke data/leiepris_data.xlsx – legg filen i prosjektmappen.")
    st.stop()

DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"
RENTS_XLSX = DATA_DIR / "leiepris_data.xlsx"

ROOM_LABELS = {
    "1rom": "1 rom",
    "2rom": "2 rom",
    "3rom": "3 rom",
    "4rom": "4 rom",
    "5rom": "5 rom"
}

# ──────────────────── helpers ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_rates() -> pd.DataFrame:
    # Hard-coded for 2012–2024, then ffill out to 2030
    base = {
        2012: (1.55, 2.55),
        2013: (1.50, 2.50),
        2014: (1.49, 2.49),
        2015: (1.05, 2.05),
        2016: (0.55, 1.55),
        2017: (0.50, 1.50),
        2018: (0.57, 1.57),
        2019: (1.15, 2.15),
        2020: (0.36, 1.36),
        2021: (0.08, 1.08),
        2022: (1.33, 2.33),
        2023: (3.54, 4.54),
        2024: (4.50, 5.50),
    }
    years = list(range(2012, 2031))
    df = pd.DataFrame(
        {yr: base.get(yr, (np.nan, np.nan)) for yr in years},
        index=["styringsrente", "utlansrente"]
    ).T
    return df.ffill().bfill()

RATES_DF = load_rates()

@st.cache_data(show_spinner=False)
def load_models(kind: str) -> dict[str, CombinedRegressor]:
    out: dict[str, CombinedRegressor] = {}
    for k in ROOM_LABELS:
        p = MODEL_DIR / f"model_{k}_{kind}.joblib"
        if p.exists():
            out[k] = joblib.load(p)
    return out

@st.cache_data(show_spinner=False)
def get_regions(xlsx: Path) -> list[str]:
    raw = pd.read_excel(xlsx, header=None)
    mask = raw.apply(lambda r: r.astype(str).str.contains(r"\b\d+\s*rom\b").any(), axis=1)
    header_row = mask.idxmax() - 1
    hdrs = raw.iloc[header_row].astype(str).str.strip()
    df = raw.iloc[header_row+1:].copy(); df.columns = hdrs
    df.columns.values[:3] = ["Category","Region","RoomType"]
    df["Region"] = df["Region"].ffill()
    return sorted(df["Region"].dropna().unique())

@st.cache_data(show_spinner=False)
def load_long_df(xlsx: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, header=None)
    mask = raw.apply(lambda r: r.astype(str).str.contains(r"\b\d+\s*rom\b").any(), axis=1)
    header_row = mask.idxmax() - 1
    hdrs = raw.iloc[header_row].astype(str).str.strip()
    df = raw.iloc[header_row+1:].copy(); df.columns = hdrs
    df.columns.values[:3] = ["Category","Region","RoomType"]
    df["Region"] = df["Region"].ffill()
    years = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    df[years] = df[years].apply(pd.to_numeric, errors='coerce')
    long = df.melt(
        id_vars=["Region","RoomType"],
        value_vars=years,
        var_name="Year",
        value_name="Rent"
    ).dropna(subset=["Rent"])
    long["Year"]  = long["Year"].astype(int)
    long["Year2"] = long["Year"] ** 2
    return long

# load models & regions
MODELS  = load_models("prod")
if not MODELS:
    st.error("Ingen prod-modeller – kjør training først.")
    st.stop()
REGIONS = get_regions(RENTS_XLSX)

# ──────────────────── Historiske tall ───────────────────────────────
st.header("Historiske leiepriser")
c1, c2 = st.columns(2)
room_hist   = c1.selectbox("Antall rom (historisk)", list(ROOM_LABELS),
                           format_func=ROOM_LABELS.get, key="hist_room")
region_hist = c2.selectbox("Prissone (historisk)", REGIONS, key="hist_region")
yr_hist     = st.slider("Velg år for historiske data", 2012, 2024, 2024, key="hist_year")

LONG_DF = load_long_df(RENTS_XLSX)
mask_obs = (
    (LONG_DF.Region == region_hist) &
    (LONG_DF.RoomType.str.startswith(ROOM_LABELS[room_hist])) &
    (LONG_DF.Year == yr_hist)
)
obs = LONG_DF.loc[mask_obs, "Rent"]
if not obs.empty:
    st.metric("Observert leie", f"{int(obs.iloc[0]):,d} kr/mnd".replace(",", " "))
else:
    st.warning(f"Ingen data for {yr_hist}, {region_hist}, {ROOM_LABELS[room_hist]}")

st.markdown("---")

# ──────────────────── Prognose leiepriser ────────────────────────────
st.header("Prognose leiepriser")
c3, c4 = st.columns(2)
room_pred   = c3.selectbox("Antall rom (prognose)", list(ROOM_LABELS),
                           format_func=ROOM_LABELS.get, key="pred_room")
region_pred = c4.selectbox("Prissone (prognose)", REGIONS, key="pred_region")
yr_pred     = st.slider("Velg år for prognose", 2025, 2030, 2025, key="pred_year")

model = MODELS[room_pred]
df_pred = pd.DataFrame([{
    "Region":        region_pred,
    "Year":          yr_pred,
    "Year2":         yr_pred**2,
    "utlansrente":   RATES_DF.loc[yr_pred, "utlansrente"],
    "styringsrente": RATES_DF.loc[yr_pred, "styringsrente"],
}])

pred = model.predict(df_pred)[0]
st.metric("Prognose leie", f"{pred:,.0f} kr/mnd".replace(",", " "))

st.markdown("---")

# ──────────────────── Eval-tabell ───────────────────────────────────
@st.cache_data(show_spinner=False)
def eval_table() -> pd.DataFrame:
    f = MODEL_DIR / "mae_eval_summary.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, index_col=0)
    df.index = [ROOM_LABELS.get(i,i) for i in df.index]
    df.rename(columns={"MAE":"MAE 2024 (kr)", "R2":"R² 2024"}, inplace=True)
    return df.round({"MAE 2024 (kr)":0, "R² 2024":2})

st.subheader("Modellfeil på hold-out 2024")
st.table(eval_table())

























