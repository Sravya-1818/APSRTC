# ==========================================================
# APSRTC DEPOT FORECAST GENERATOR (CORRECT VERSION)
# ==========================================================

import duckdb
import pandas as pd
import numpy as np
import joblib
import datetime as dt

print("Loading model...")

# Load trained bundle
bundle = joblib.load("demand_model.joblib")

pipeline = bundle["pipeline"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]

con = duckdb.connect("apsrtc.db")

DEPOT_ID = 22   # change depot if needed

# ----------------------------------------------------------
# LOAD HISTORICAL DATA
# ----------------------------------------------------------
df = con.execute(f"""
    SELECT DEPOT_ID, STAR_DATE, passengers, revenue, trips, active_vehicles
    FROM depot_daily_summary
    WHERE DEPOT_ID = {DEPOT_ID}
    ORDER BY STAR_DATE
""").fetchdf()

df["STAR_DATE"] = pd.to_datetime(df["STAR_DATE"])
df = df.sort_values("STAR_DATE")

# ----------------------------------------------------------
# FEATURE ENGINEERING (SAME AS TRAINING)
# ----------------------------------------------------------
df["day"] = df["STAR_DATE"].dt.day
df["month"] = df["STAR_DATE"].dt.month
df["year"] = df["STAR_DATE"].dt.year
df["dow"] = df["STAR_DATE"].dt.weekday
df["weekofyear"] = df["STAR_DATE"].dt.isocalendar().week.astype(int)

df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
df["is_month_start"] = df["STAR_DATE"].dt.is_month_start.astype(int)
df["is_month_end"] = df["STAR_DATE"].dt.is_month_end.astype(int)

# Festival flag (simple logic)
df["is_festival"] = 0

# Lag features
df["lag_1"] = df["passengers"].shift(1)
df["lag_7"] = df["passengers"].shift(7)
df["lag_14"] = df["passengers"].shift(14)

df["roll7_mean"] = df["passengers"].shift(1).rolling(7).mean()
df["roll14_mean"] = df["passengers"].shift(1).rolling(14).mean()

df["pax_per_trip"] = df["passengers"] / df["trips"].replace(0, 1)
df["rev_per_trip"] = df["revenue"] / df["trips"].replace(0, 1)
df["pax_per_vehicle"] = df["passengers"] / df["active_vehicles"].replace(0, 1)

df = df.dropna().reset_index(drop=True)

# ----------------------------------------------------------
# FORECAST NEXT 30 DAYS (RECURSIVE)
# ----------------------------------------------------------
future_days = 30
series = df.set_index("STAR_DATE")["passengers"].copy()
last_date = df["STAR_DATE"].max()

future_rows = []

for i in range(1, future_days + 1):

    next_date = last_date + dt.timedelta(days=i)

    new_row = df.iloc[-1].copy()
    new_row["STAR_DATE"] = next_date

    # Date features
    new_row["day"] = next_date.day
    new_row["month"] = next_date.month
    new_row["year"] = next_date.year
    new_row["dow"] = next_date.weekday()
    new_row["weekofyear"] = next_date.isocalendar()[1]

    new_row["is_weekend"] = 1 if next_date.weekday() in [5, 6] else 0
    new_row["is_month_start"] = 1 if next_date.day == 1 else 0
    new_row["is_month_end"] = 1 if next_date.day >= 28 else 0
    new_row["is_festival"] = 0

    # Lag logic
    new_row["lag_1"] = series.iloc[-1]
    new_row["lag_7"] = series.iloc[-7] if len(series) >= 7 else series.iloc[-1]
    new_row["lag_14"] = series.iloc[-14] if len(series) >= 14 else series.iloc[-1]

    new_row["roll7_mean"] = series.tail(7).mean()
    new_row["roll14_mean"] = series.tail(14).mean()

    # Ratios (keep last known trips & vehicles constant)
    new_row["pax_per_trip"] = new_row["lag_1"] / max(new_row["trips"], 1)
    new_row["rev_per_trip"] = new_row["revenue"] / max(new_row["trips"], 1)
    new_row["pax_per_vehicle"] = new_row["lag_1"] / max(new_row["active_vehicles"], 1)

    # Predict
    X_input = pd.DataFrame([new_row[num_cols + cat_cols]])
    prediction = pipeline.predict(X_input)[0]

    # Prevent collapse
    prediction = max(prediction, 1000)

    new_row["passengers"] = prediction

    # Update series
    series.loc[next_date] = prediction

    future_rows.append([next_date, prediction])

forecast_df = pd.DataFrame(future_rows, columns=["STAR_DATE", "passengers"])

# ----------------------------------------------------------
# DEMAND CLASSIFICATION
# ----------------------------------------------------------
p70 = df["passengers"].quantile(0.70)
p90 = df["passengers"].quantile(0.90)

def classify(x):
    if x >= p90:
        return "PEAK"
    elif x >= p70:
        return "MEDIUM"
    else:
        return "LOW"

forecast_df["demand_level"] = forecast_df["passengers"].apply(classify)

print("\nForecast Preview:\n")
print(forecast_df.head())
print("\nForecast Complete.")