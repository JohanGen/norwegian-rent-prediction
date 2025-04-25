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
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import HeNormal

initializer = HeNormal()

# Initialize Streamlit
st.set_page_config(page_title="Leiepris-prediktor", page_icon="🏠", layout="wide")
st.title("Deep-Learning Prosjekt Leiepris-prediktor")

class DNNRegressor:
    def __init__(self, preprocessor, model, r2_score=None):
        self.preprocessor = preprocessor
        self.model = model
        self.r2_score = r2_score
        
    def predict(self, X):
        X_full = X.copy()
        X_full["InterestSpread"] = X_full["utlansrente"] - X_full["styringsrente"]
        X_full["RoomSize"] = X["RoomType"].apply(lambda x: 5.5 if "5" in x else float(x[0]))
        X_processed = self.preprocessor.transform(X_full)
        return self.model.predict(X_processed).flatten()

class XGBTrendRegressor:
    def __init__(self, xgb_model, trend_model, r2_score=None):
        self.xgb_model = xgb_model
        self.trend_model = trend_model
        self.r2_score = r2_score

    def predict(self, X):
        X_prep = X.copy()
        X_prep["InterestSpread"] = X_prep["utlansrente"] - X_prep["styringsrente"]
        X_prep["RoomSize"] = X["RoomType"].apply(lambda x: 5.5 if "5" in x else float(re.search(r"\d+", x).group()))
        
        if "RoomType" in X_prep.columns:
            X_prep = X_prep.drop(columns=["RoomType"])
            
        X_prep = pd.get_dummies(X_prep, columns=["Region"])
        
        expected_cols = self.xgb_model.feature_names_in_
        for col in expected_cols:
            if col not in X_prep.columns:
                X_prep[col] = 0
        X_prep = X_prep[expected_cols]
        
        xgb_pred = self.xgb_model.predict(X_prep)
        trend_pred = self.trend_model.predict(X[["Year"]])
        return (xgb_pred + trend_pred) / 2

# File paths and configuration
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
    "5rom": "5 rom eller flere"
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
    raw = pd.read_excel(DATA_DIR / "Bok1.xlsx", header=None)

    # Find header row containing both 'Region' and 'Rom'
    potential_header_rows = raw[raw.apply(
        lambda row: row.astype(str).str.contains("Rom").any() and row.astype(str).str.contains("Region").any(),
        axis=1
    )]

    if potential_header_rows.empty:
        st.error("Fant ikke rad med kolonnenavn som inneholder både 'Region' og 'Rom'.")
        st.dataframe(raw.head(10))
        st.stop()

    header_row_idx = potential_header_rows.index[0]
    df = raw.iloc[header_row_idx + 1:].copy()
    df.columns = raw.iloc[header_row_idx].map(lambda c: str(c).strip())

    df.rename(columns={"Rom": "RoomType"}, inplace=True)
    df.columns = df.columns.map(lambda col: str(col).strip())

    year_strs = [str(y) for y in range(2012, 2025)]
    missing = [y for y in year_strs if y not in df.columns]
    if missing:
        st.error(f"Mangler årstallskolonner: {missing}")
        st.stop()

    df[year_strs] = df[year_strs].apply(pd.to_numeric, errors='coerce')

    long_df = df.melt(
        id_vars=["Region", "RoomType"],
        value_vars=year_strs,
        var_name="Year",
        value_name="Rent"
    ).dropna(subset=["Rent"])

    long_df["Year"] = long_df["Year"].astype(int)
    long_df["Year2"] = long_df["Year"] ** 2

    return long_df.merge(load_rates(), left_on="Year", right_index=True)

def train_dnn_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_initializer=initializer),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_initializer=initializer),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_initializer=initializer),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.0003), loss='mse', metrics=['mae'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=8,
        callbacks=[EarlyStopping(patience=20)],
        verbose=0
    )

    y_pred = model.predict(X_val).flatten()
    r2 = r2_score(y_val, y_pred)
    return model, r2

def train_xgboost_model(X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    return model, r2

def train_trend_model(X_train, y_train, X_val, y_val):
    model = LinearRegression()
    model.fit(X_train[['Year']], y_train)
    y_pred = model.predict(X_val[['Year']])
    r2 = r2_score(y_val, y_pred)
    return model, r2

def prepare_dnn_data(full_df, room_type):
    subset = full_df[full_df["RoomType"].str.contains(room_type.split()[0])].copy()
    subset["InterestSpread"] = subset["utlansrente"] - subset["styringsrente"]
    subset["RoomSize"] = subset["RoomType"].str.extract(r'(\d+)').astype(float)
    subset.loc[subset["RoomType"].str.contains("5 rom"), "RoomSize"] = 5.5

    features = subset[["Year", "Year2", "styringsrente", "utlansrente", 
                       "InterestSpread", "RoomSize", "Region"]]
    target = subset["Rent"]  # keep as Series

    mask = features["Year"] <= 2024
    X_train, X_val, y_train, y_val = train_test_split(
        features[mask],
        target[mask],
        test_size=0.2,
        random_state=42
    )

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ["Year", "Year2", "styringsrente", "utlansrente", "InterestSpread", "RoomSize"]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ["Region"])
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    return X_train_processed, y_train.to_numpy(), X_val_processed, y_val.to_numpy(), preprocessor

    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ["Year", "Year2", "styringsrente", "utlansrente", "InterestSpread", "RoomSize"]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ["Region"])
    ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    return X_train_processed, y_train, X_val_processed, y_val, preprocessor

def prepare_xgboost_data(full_df, room_type):
    subset = full_df[full_df["RoomType"].str.contains(room_type.split()[0])].copy()
    subset["InterestSpread"] = subset["utlansrente"] - subset["styringsrente"]
    subset["RoomSize"] = subset["RoomType"].str.extract(r'(\d+)').astype(float)
    subset.loc[subset["RoomType"].str.contains("5 rom"), "RoomSize"] = 5.5

    features = subset[["Year", "Year2", "styringsrente", "utlansrente",
                      "InterestSpread", "RoomSize", "Region"]]
    target = subset["Rent"].values

    X_train, X_val, y_train, y_val = train_test_split(
        features[features["Year"] < 2024],
        target[features["Year"] < 2024],
        test_size=0.2,
        random_state=42
    )

    X_train = pd.get_dummies(X_train, columns=["Region"])
    X_val = pd.get_dummies(X_val, columns=["Region"])

    for col in X_train.columns:
        if col not in X_val.columns:
            X_val[col] = 0
    X_val = X_val[X_train.columns]

    return X_train, y_train, X_val, y_val

@st.cache_data
def load_models():
    models = {
        'dnn': {},
        'xgb_trend': {},
        'performance': {}
    }
    
    # Load DNN models
    for room_key in ROOM_LABELS:
        dnn_path = MODEL_DIR / f"dnn_model_{room_key}.keras"
        preprocessor_path = MODEL_DIR / f"dnn_preprocessor_{room_key}.joblib"
        perf_path = MODEL_DIR / f"dnn_performance_{room_key}.joblib"
        
        if dnn_path.exists() and preprocessor_path.exists():
            try:
                r2 = joblib.load(perf_path) if perf_path.exists() else None
                models['dnn'][room_key] = DNNRegressor(
                    preprocessor=joblib.load(preprocessor_path),
                    model=load_model(dnn_path),
                    r2_score=r2
                )
                if r2:
                    models['performance'][f'dnn_{room_key}'] = r2
            except Exception as e:
                st.error(f"Feil ved lasting av DNN-modell {room_key}: {str(e)}")
    
    # Load XGBoost + Trend models
    for room_key in ROOM_LABELS:
        xgb_path = MODEL_DIR / f"xgb_model_{room_key}.joblib"
        trend_path = MODEL_DIR / f"trend_model_{room_key}.joblib"
        perf_path = MODEL_DIR / f"xgb_trend_performance_{room_key}.joblib"
        
        if xgb_path.exists() and trend_path.exists():
            try:
                xgb_model = joblib.load(xgb_path)
                trend_model = joblib.load(trend_path)
                r2 = joblib.load(perf_path) if perf_path.exists() else None
                
                models['xgb_trend'][room_key] = XGBTrendRegressor(
                    xgb_model, trend_model, r2_score=r2
                )
                if r2:
                    models['performance'][f'xgb_trend_{room_key}'] = r2
            except Exception as e:
                st.error(f"Feil ved lasting av XGB+Trend-modell {room_key}: {str(e)}")
    
    return models

# Main App
LONG_DF = load_rental_data()
REGIONS = sorted(LONG_DF["Region"].unique())

# Sidebar Controls
with st.sidebar:
    st.header("Modellinnstillinger")
    
    model_type = st.radio("Velg modelltype", 
                         ["DNN (Deep Learning)", "XGBoost + Trend"],
                         index=0)
    
    if st.button("Trene alle modeller på nytt"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (room_key, room_label) in enumerate(ROOM_LABELS.items()):
            status_text.text(f"Trener modeller for {room_label}...")
            
            # Train DNN
            try:
                X_train, y_train, X_val, y_val, preprocessor = prepare_dnn_data(LONG_DF, room_label)
                dnn_model, dnn_r2 = train_dnn_model(X_train, y_train, X_val, y_val)
                dnn_model.save(MODEL_DIR / f"dnn_model_{room_key}.keras")
                joblib.dump(preprocessor, MODEL_DIR / f"dnn_preprocessor_{room_key}.joblib")
                joblib.dump(dnn_r2, MODEL_DIR / f"dnn_performance_{room_key}.joblib")
                st.success(f"DNN for {room_label} trenet (R2: {dnn_r2:.3f})")
            except Exception as e:
                st.error(f"Feil under DNN-trening for {room_label}: {str(e)}")
            
            # Train XGBoost + Trend
            try:
                X_train, y_train, X_val, y_val = prepare_xgboost_data(LONG_DF, room_label)
                
                # Train XGBoost
                xgb_model, xgb_r2 = train_xgboost_model(X_train, y_train, X_val, y_val)
                joblib.dump(xgb_model, MODEL_DIR / f"xgb_model_{room_key}.joblib")
                
                # Train Trend model
                trend_model, trend_r2 = train_trend_model(X_train, y_train, X_val, y_val)
                joblib.dump(trend_model, MODEL_DIR / f"trend_model_{room_key}.joblib")
                
                # Combined R2 score
                combined_pred = (xgb_model.predict(X_val) + trend_model.predict(X_val[['Year']])) / 2
                combined_r2 = r2_score(y_val, combined_pred)
                joblib.dump(combined_r2, MODEL_DIR / f"xgb_trend_performance_{room_key}.joblib")
                
                st.success(f"XGB+Trend for {room_label} trenet (XGB R2: {xgb_r2:.3f}, Trend R2: {trend_r2:.3f}, Combined R2: {combined_r2:.3f})")
            except Exception as e:
                st.error(f"Feil under XGB+Trend-trening for {room_label}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(ROOM_LABELS))
        
        status_text.text("Treningsprosess fullført!")
        st.rerun()
    
    if st.button("Tøm cache og last modeller på nytt"):
        st.cache_data.clear()
        st.rerun()
    
    with st.expander("Debug-info"):
        st.write("ROOT:", ROOT)
        st.write("MODEL_DIR:", MODEL_DIR)
        st.write("Eksisterende filer:")
        if MODEL_DIR.exists():
            st.write([str(p) for p in MODEL_DIR.glob("*")])
        else:
            st.error("MODEL_DIR finnes ikke!")

# Load models after sidebar is defined
MODELS = load_models()

# Main Interface
tab1, tab2, tab3 = st.tabs(["Historiske priser", "Prognoser", "Modellresultater"])

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
    
    room_key = next(k for k, v in ROOM_LABELS.items() if v == room_pred)
    
    df_pred = pd.DataFrame([{
        "Region": region_pred,
        "RoomType": room_pred,
        "Year": yr_pred,
        "Year2": yr_pred**2,
        "utlansrente": load_rates().loc[yr_pred, "utlansrente"],
        "styringsrente": load_rates().loc[yr_pred, "styringsrente"],
    }])
    
    if model_type == "DNN (Deep Learning)":
        if room_key in MODELS['dnn']:
            pred = MODELS['dnn'][room_key].predict(df_pred)[0]
            st.metric("DNN Leieprognose", f"{pred:,.0f} kr/mnd".replace(",", " "))
            
            if MODELS['dnn'][room_key].r2_score is not None:
                st.markdown("**Modellytelse**")
                st.write(f"R² score: {MODELS['dnn'][room_key].r2_score:.3f}")
                st.progress(min(max(0, MODELS['dnn'][room_key].r2_score), 1))
        else:
            st.warning("Ingen DNN-modell - klikk 'Trene modeller' i sidemenyen")
    else:  # XGBoost + Trend
        if room_key in MODELS['xgb_trend']:
            pred = MODELS['xgb_trend'][room_key].predict(df_pred)[0]
            st.metric("XGBoost+Trend Leieprognose", f"{pred:,.0f} kr/mnd".replace(",", " "))
            
            if MODELS['xgb_trend'][room_key].r2_score is not None:
                st.markdown("**Modellytelse**")
                st.write(f"Kombinert R² score: {MODELS['xgb_trend'][room_key].r2_score:.3f}")
                st.progress(min(max(0, MODELS['xgb_trend'][room_key].r2_score), 1))
        else:
            st.warning("Ingen XGBoost+Trend-modell - klikk 'Trene modeller' i sidemenyen")

with tab3:
    st.subheader("Modellresultater for alle romtyper")
    
    # Prepare data in long format for side-by-side comparison
    r2_data = []
    for room_key, room_label in ROOM_LABELS.items():
        r2_data.append({
            'Romtype': room_label,
            'Modelltype': 'DNN',
            'R²': MODELS['performance'].get(f'dnn_{room_key}', None)
        })
        r2_data.append({
            'Romtype': room_label,
            'Modelltype': 'XGB+Trend',
            'R²': MODELS['performance'].get(f'xgb_trend_{room_key}', None)
        })
    
    r2_df = pd.DataFrame(r2_data)
    
    # Display as table
    pivot_df = r2_df.pivot(index='Romtype', columns='Modelltype', values='R²')
    st.dataframe(
        pivot_df.style.format({
            'DNN': '{:.3f}',
            'XGB+Trend': '{:.3f}'
        }, na_rep="Ikke tilgjengelig"),
        use_container_width=True
    )
    
    st.markdown("""
    **Forklaring:**
    - **R² score** (R-squared) måler hvor godt modellen passer til dataene
    - **Perfekt modell** har R² = 1
    - **Ingen forklaring** gir R² = 0
    - Negative verdier indikerer at modellen er dårligere enn en horisontal linje
    """)
    
    st.subheader("Sammenligning av modelltyper")
    if not r2_df.empty:
        import altair as alt
        
        chart = alt.Chart(r2_df).mark_bar().encode(
            x=alt.X('Romtype:N', title='Romtype'),
            y=alt.Y('R²:Q', title='R² score'),
            color=alt.Color('Modelltype:N', title='Modelltype'),
            column=alt.Column('Modelltype:N', spacing=5)
        ).properties(
            width=alt.Step(40)  # controls width of bars
        )
        
        st.altair_chart(chart)