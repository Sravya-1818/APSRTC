import duckdb
import pandas as pd
import numpy as np
import datetime as dt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

# -----------------------------
# CONFIG
# -----------------------------
DB_PATH = r"D:/APSRTC/apsrtc.db"     # <-- your DB
MODEL_OUT = r"D:/APSRTC/demand_model.joblib"

TARGET = "passengers"               # forecasting target
DATE_COL = "STAR_DATE"

# -----------------------------
# FESTIVAL / EVENT CALENDAR (sample - extend later)
# -----------------------------
# You can keep adding dates here (YYYY-MM-DD). These become binary features.
FESTIVAL_DATES = set([
    "2024-01-15",  # Sankranti (example)
    "2024-04-09",  # Ugadi (example)
    "2024-10-12",  # Dussehra (example)
    "2024-11-01",  # Diwali (example)
    "2025-01-14",
    "2025-04-01",
])

def is_festival(d: pd.Timestamp) -> int:
    return int(d.strftime("%Y-%m-%d") in FESTIVAL_DATES)

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT
            DEPOT_ID,
            DEPOT_NAME,
            STAR_DATE,
            passengers,
            trips,
            revenue,
            active_vehicles
        FROM depot_daily_summary
        WHERE passengers IS NOT NULL
        ORDER BY DEPOT_ID, STAR_DATE
    """).fetchdf()
    con.close()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def add_features(df):
    df = df.sort_values(["DEPOT_ID", DATE_COL]).copy()

    # calendar features
    df["dow"] = df[DATE_COL].dt.dayofweek              # 0=Mon
    df["month"] = df[DATE_COL].dt.month
    df["weekofyear"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["day"] = df[DATE_COL].dt.day
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_month_start"] = df[DATE_COL].dt.is_month_start.astype(int)
    df["is_month_end"] = df[DATE_COL].dt.is_month_end.astype(int)

    # festival flags
    df["is_festival"] = df[DATE_COL].apply(is_festival)

    # lags per depot (very important)
    df["lag_1"]  = df.groupby("DEPOT_ID")[TARGET].shift(1)
    df["lag_7"]  = df.groupby("DEPOT_ID")[TARGET].shift(7)
    df["lag_14"] = df.groupby("DEPOT_ID")[TARGET].shift(14)

    # rolling means (shifted so no leakage)
    df["roll7_mean"]  = df.groupby("DEPOT_ID")[TARGET].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["roll14_mean"] = df.groupby("DEPOT_ID")[TARGET].shift(1).rolling(14).mean().reset_index(level=0, drop=True)

    # ratios (safe features)
    df["pax_per_trip"] = df[TARGET] / df["trips"].replace(0, np.nan)
    df["rev_per_trip"] = df["revenue"] / df["trips"].replace(0, np.nan)
    df["pax_per_vehicle"] = df[TARGET] / df["active_vehicles"].replace(0, np.nan)

    # drop very early rows that cannot have lags
    df = df.dropna(subset=[TARGET]).copy()

    return df

# -----------------------------
# BUILD MODEL PIPELINE
# -----------------------------
def build_pipeline(cat_cols, num_cols):
    # OneHotEncoder: sklearn version compatibility
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # older sklearn uses 'sparse' instead of 'sparse_output'
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop"
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.07,
        max_depth=6,
        max_iter=600,
        random_state=42
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("model", model)
    ])
    return pipe

# -----------------------------
# TRAIN / EVAL
# -----------------------------
def train_model(df):
    # choose features
    cat_cols = ["DEPOT_ID", "DEPOT_NAME", "dow", "month"]  # keep simple but powerful
    num_cols = [
        "trips", "revenue", "active_vehicles",
        "weekofyear", "day", "is_weekend", "is_month_start", "is_month_end",
        "is_festival",
        "lag_1", "lag_7", "lag_14",
        "roll7_mean", "roll14_mean",
        "pax_per_trip", "rev_per_trip", "pax_per_vehicle"
    ]

    # keep only needed cols
    df_feat = df[[DATE_COL, TARGET] + cat_cols + num_cols].copy()

    # IMPORTANT: avoid leakage by time split
    df_feat = df_feat.sort_values(DATE_COL)
    split_date = df_feat[DATE_COL].quantile(0.85)  # last 15% as test (time-based)

    train_df = df_feat[df_feat[DATE_COL] <= split_date].copy()
    test_df  = df_feat[df_feat[DATE_COL] >  split_date].copy()

    X_train = train_df[cat_cols + num_cols]
    y_train = train_df[TARGET].values

    X_test = test_df[cat_cols + num_cols]
    y_test = test_df[TARGET].values

    pipe = build_pipeline(cat_cols, num_cols)

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    # robust RMSE (no squared kw)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    return pipe, mae, rmse, cat_cols, num_cols, df_feat

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Loading data from DuckDB...")
    df_raw = load_data()
    print(f"Rows: {len(df_raw)} Depots: {df_raw['DEPOT_ID'].nunique()}")
    print(f"Date range: {df_raw[DATE_COL].min()} → {df_raw[DATE_COL].max()}")

    print("\nFeature engineering...")
    df = add_features(df_raw)

    print("\nTraining demand model...")
    pipe, mae, rmse, cat_cols, num_cols, df_feat = train_model(df)

    print("\n✅ Model trained")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")

    joblib.dump({
        "pipeline": pipe,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "target": TARGET
    }, MODEL_OUT)

    print(f"\n✅ Saved model to: {MODEL_OUT}")