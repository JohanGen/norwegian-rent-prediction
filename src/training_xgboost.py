from __future__ import annotations
import re
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Initialize Streamlit (must be first command)
st.set_page_config(page_title="Leiepris-prediktor", page_icon="🏠", layout="wide")
st.title("Deep-Learning Prosjekt Leiepris-prediktor")

class DNNRegressor:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        
    def predict(self, X):
        X_full = X.copy()
        X_full["InterestSpread"] = X_full["utlansrente"] - X_full["styringsrente"]
        X_full["RoomSize"] = X["RoomType"].apply(lambda x: 5.5 if "5" in x else float(x[0]))
        X_processed = self.preprocessor.transform(X_full)
        return self.model.predict(X_processed).flatten()

# File paths
THIS = Path(__file__).resolve()
ROOT = next((cand for cand in [THIS.parent, THIS.parent.parent, Path.cwd()] 
             if (cand / "data" / "leiepris_data.xlsx").exists()), None)

if ROOT is None:
    st.error("Fant ikke data/leiepris_data.xlsx – legg filen i prosjektmappen.")
    st.stop()

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

ROOM_LABELS = {
    "1rom": "1 rom",
    "2rom": "2 rom", 
    "3rom": "3 rom",
    "4rom": "4 rom",
    "5rom": "5 rom"
}

@st.cache_data
def load_rates() -> pd.DataFrame:
    rates = {
        2012: (1.55, 2.55), 2013: (1.50, 2.50), 2014: (1.49, 2.49),
        2015: (1.05, 2.05), 2016: (0.55, 1.55), 2017: (0.50, 1.50),
        2018: (0.57, 1.57), 2019: (1.15, 2.15), 2020: (0.36, 1.36),
        2021: (0.08, 1.08), 2022: (1.33, 2.33), 2023: (3.54, 4.54),
        2024: (4.50, 5.50)
    }
    return pd.DataFrame(
        {yr: rates.get(yr, (np.nan, np.nan)) for yr in range(2012, 2031)},
        index=["styringsrente", "utlansrente"]
    ).T.ffill().bfill()

@st.cache_data
def load_rental_data() -> pd.DataFrame:
    raw = pd.read_excel(DATA_DIR / "leiepris_data.xlsx", header=None)
    mask = raw.apply(lambda r: r.astype(str).str.contains(r"\b\d+\s*rom\b").any(), axis=1)
    header_row = mask.idxmax() - 1
    
    df = raw.iloc[header_row+1:].copy()
    df.columns = raw.iloc[header_row].astype(str).str.strip()
    df.columns.values[:3] = ["Category", "Region", "RoomType"]
    df["Region"] = df["Region"].ffill()
    
    years = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    df[years] = df[years].apply(pd.to_numeric, errors='coerce')
    
    long_df = df.melt(
        id_vars=["Region", "RoomType"],
        value_vars=years,
        var_name="Year",
        value_name="Rent"
    ).dropna(subset=["Rent"])
    
    long_df["Year"] = long_df["Year"].astype(int)
    long_df["Year2"] = long_df["Year"] ** 2
    return long_df.merge(load_rates(), left_on="Year", right_index=True)

def train_dnn_model(X_train, y_train):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[EarlyStopping(patience=20)],
        verbose=0
    )
    return model

def prepare_dnn_data(full_df, room_type):
    subset = full_df[full_df["RoomType"].str.startswith(room_type)].copy()
    subset["InterestSpread"] = subset["utlansrente"] - subset["styringsrente"]
    subset["RoomSize"] = subset["RoomType"].str.extract(r'(\d+)').astype(float)
    subset.loc[subset["RoomType"].str.contains("5 rom"), "RoomSize"] = 5.5
    
    features = subset[["Year", "Year2", "styringsrente", "utlansrente", 
                      "InterestSpread", "RoomSize", "Region"]]
    target = subset["Rent"].values
    
    X_train = features[features["Year"] < 2024]
    y_train = target[features["Year"] < 2024]
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ["Year", "Year2", "styringsrente", "utlansrente", "InterestSpread", "RoomSize"]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ["Region"])
    ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    return X_train_processed, y_train, preprocessor

@st.cache_resource
def load_models():
    models = {'dnn': {}}
    for k in ROOM_LABELS:
        dnn_path = MODEL_DIR / f"dnn_model_{k}.keras"
        preprocessor_path = MODEL_DIR / f"dnn_preprocessor_{k}.joblib"
        if dnn_path.exists() and preprocessor_path.exists():
            models['dnn'][k] = DNNRegressor(
                preprocessor=joblib.load(preprocessor_path),
                model=load_model(dnn_path)
            )
    return models

# Main App
LONG_DF = load_rental_data()
MODELS = load_models()
REGIONS = sorted(LONG_DF["Region"].unique())

# Sidebar Controls
with st.sidebar:
    st.header("Modellinnstillinger")
    if st.button("Trene nye DNN-modeller"):
        with st.spinner("Trener modeller..."):
            for room in ROOM_LABELS.values():
                try:
                    X_train, y_train, preprocessor = prepare_dnn_data(LONG_DF, room)
                    model = train_dnn_model(X_train, y_train)
                    model.save(MODEL_DIR / f"dnn_model_{room[:1]}rom.keras")
                    joblib.dump(preprocessor, MODEL_DIR / f"dnn_preprocessor_{room[:1]}rom.joblib")
                    st.success(f"Trent {room}")
                except Exception as e:
                    st.error(f"Feil: {str(e)}")
            st.rerun()

# Main Interface
tab1, tab2 = st.tabs(["Historiske priser", "Prognoser"])

with tab1:
    c1, c2 = st.columns(2)
    room_hist = c1.selectbox("Romtype", list(ROOM_LABELS.values()), key="hist_room")
    region_hist = c2.selectbox("Område", REGIONS, key="hist_region")
    yr_hist = st.slider("År", 2012, 2024, 2024, key="hist_year")
    
    obs = LONG_DF[
        (LONG_DF.Region == region_hist) &
        (LONG_DF.RoomType.str.startswith(room_hist)) &
        (LONG_DF.Year == yr_hist)
    ].Rent
    
    st.metric("Historisk leie", 
              f"{int(obs.iloc[0]):,d} kr/mnd".replace(",", " ") if not obs.empty else "Ingen data")

with tab2:
    c3, c4 = st.columns(2)
    room_pred = c3.selectbox("Romtype", list(ROOM_LABELS.values()), key="pred_room")
    region_pred = c4.selectbox("Område", REGIONS, key="pred_region")
    yr_pred = st.slider("Prognoseår", 2025, 2030, 2025, key="pred_year")
    
    df_pred = pd.DataFrame([{
        "Region": region_pred,
        "RoomType": room_pred,
        "Year": yr_pred,
        "Year2": yr_pred**2,
        "utlansrente": load_rates().loc[yr_pred, "utlansrente"],
        "styringsrente": load_rates().loc[yr_pred, "styringsrente"],
    }])
    
    if room_pred[:1]+"rom" in MODELS['dnn']:
        pred = MODELS['dnn'][room_pred[:1]+"rom"].predict(df_pred)[0]
        st.metric("Leieprognose", f"{pred:,.0f} kr/mnd".replace(",", " "))
    else:
        st.warning("Ingen trenet modell - klikk 'Trene modeller' i sidemenyen")

# Only show this if running directly
if __name__ == "__main__":
    st.warning("Kjør denne appen med: streamlit run filnavn.py")























































