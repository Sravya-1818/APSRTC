# ==========================================================
# APSRTC DEPOT DEMAND MODEL TRAINING (FINAL CLEAN VERSION)
# ==========================================================

import duckdb
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

print("Connecting to DuckDB...")

# ==========================================================
# 1. LOAD DATA
# ==========================================================

con = duckdb.connect("apsrtc.db")

df = con.execute("""
    SELECT DEPOT_ID, STAR_DATE, passengers, revenue, trips, active_vehicles
    FROM depot_daily_summary
    ORDER BY DEPOT_ID, STAR_DATE
""").fetchdf()

df["STAR_DATE"] = pd.to_datetime(df["STAR_DATE"])

print("Initial Shape:", df.shape)
print("Date Range:", df["STAR_DATE"].min(), "to", df["STAR_DATE"].max())

# ==========================================================
# 2. FESTIVAL FEATURE (Simple but Effective)
# ==========================================================

# Add your known festivals here
festival_dates = [
    "2024-10-31",  # Diwali approx
    "2024-01-14",  # Sankranti approx
    "2024-04-09",  # Ugadi approx
    "2025-01-14",
    "2025-10-20"
]

festival_dates = pd.to_datetime(festival_dates)

df["is_festival"] = df["STAR_DATE"].isin(festival_dates).astype(int)

# ==========================================================
# 3. FEATURE ENGINEERING
# ==========================================================

df = df.sort_values(["DEPOT_ID", "STAR_DATE"])

# Date features
df["day"] = df["STAR_DATE"].dt.day
df["month"] = df["STAR_DATE"].dt.month
df["year"] = df["STAR_DATE"].dt.year
df["dow"] = df["STAR_DATE"].dt.weekday
df["weekofyear"] = df["STAR_DATE"].dt.isocalendar().week.astype(int)

df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
df["is_month_start"] = df["STAR_DATE"].dt.is_month_start.astype(int)
df["is_month_end"] = df["STAR_DATE"].dt.is_month_end.astype(int)

# Lag features
df["lag_1"] = df.groupby("DEPOT_ID")["passengers"].shift(1)
df["lag_7"] = df.groupby("DEPOT_ID")["passengers"].shift(7)
df["lag_14"] = df.groupby("DEPOT_ID")["passengers"].shift(14)

df["roll7_mean"] = (
    df.groupby("DEPOT_ID")["passengers"]
    .shift(1)
    .rolling(7)
    .mean()
)

df["roll14_mean"] = (
    df.groupby("DEPOT_ID")["passengers"]
    .shift(1)
    .rolling(14)
    .mean()
)

# Efficiency ratios
df["pax_per_trip"] = df["passengers"] / df["trips"].replace(0, 1)
df["rev_per_trip"] = df["revenue"] / df["trips"].replace(0, 1)
df["pax_per_vehicle"] = df["passengers"] / df["active_vehicles"].replace(0, 1)

# Drop NA rows created by lag
df = df.dropna()

print("After Feature Engineering Shape:", df.shape)

# ==========================================================
# 4. DEFINE FEATURES
# ==========================================================

num_cols = [
    "trips", "revenue", "active_vehicles",
    "weekofyear", "day", "is_weekend",
    "is_month_start", "is_month_end",
    "is_festival",
    "lag_1", "lag_7", "lag_14",
    "roll7_mean", "roll14_mean",
    "pax_per_trip", "rev_per_trip",
    "pax_per_vehicle"
]

cat_cols = ["DEPOT_ID", "dow", "month"]

target = "passengers"

X = df[num_cols + cat_cols]
y = df[target]

# ==========================================================
# 5. BUILD PIPELINE (FIXED SPARSE ERROR)
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ]
)

model = HistGradientBoostingRegressor(
    learning_rate=0.07,
    max_depth=6,
    max_iter=600,
    random_state=42
)

pipeline = Pipeline([
    ("pre", preprocessor),
    ("model", model)
])

# ==========================================================
# 6. TIME SERIES VALIDATION
# ==========================================================

tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print("--------------------------------------------------")
print("Mean Passengers:", round(y.mean(), 2))
print("MAE:", round(mae, 2))
print("MAE %:", round((mae / y.mean()) * 100, 2), "%")
print("--------------------------------------------------")

# ==========================================================
# 7. SAVE MODEL
# ==========================================================

joblib.dump({
    "pipeline": pipeline,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "target": target
}, "demand_model.joblib")

print("Model saved as demand_model.joblib")
print("TRAINING COMPLETE.")