import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(file_path: str, target_year: str = "2024"):


    # ───────────────────────────────────────────────────────── raw read ──
    raw = pd.read_excel(file_path, header=None)
    print("📊 Raw sheet shape:", raw.shape)

    # find first data row (contains “1 rom”, “2 rom”… anywhere in the row)
    mask = raw.apply(
        lambda r: r.astype(str).str.contains(r"\b\d+\s*rom\b", case=False, regex=True).any(),
        axis=1,
    )
    if not mask.any():
        raise ValueError("Could not locate a row with '<n> rom' pattern – please check file")

    data_row = mask.idxmax()          # first True index
    header_row = data_row - 1         # row above has the years

    headers = raw.iloc[header_row].astype(str).str.strip()
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = headers

    # force names for the first 3 columns
    df.columns.values[0] = "Category"
    df.columns.values[1] = "Region"
    df.columns.values[2] = "RoomType"

    # ── clean & forward‑fill merged region cells ─────────────────────────
    df["Region"] = df["Region"].ffill()                     # fill blank region rows

    # keep only rows that actually have a room‑type entry
    df = df.dropna(subset=["RoomType"]).reset_index(drop=True)

    # detect year columns and convert to numeric
    year_cols = [c for c in df.columns if c.isdigit()]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")

    # we REQUIRE target_year value, others may be NaN
    df = df.dropna(subset=[target_year])

    # ── feature engineering ─────────────────────────────────────────────
    df["RoomCount"] = df["RoomType"].astype(str).str.extract(r"(\d+)").astype(float)

    if {"2023", "2024"}.issubset(df.columns):
        df["growth_2023_2024"] = df["2024"] - df["2023"]
    else:
        df["growth_2023_2024"] = 0.0

    df = pd.get_dummies(df, columns=["Region"], prefix="Region", drop_first=True)

    feature_cols = [
        "RoomCount",
        "growth_2023_2024",
        *[c for c in df.columns if c.startswith("Region_")],
    ]

    if target_year not in df.columns:
        raise KeyError(f"Target column '{target_year}' not present – header detection failed")

    print("✅ Feature columns (", len(feature_cols), ") →", feature_cols[:8], "…")

    X = df[feature_cols].fillna(0.0).to_numpy()
    y = df[target_year].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler





