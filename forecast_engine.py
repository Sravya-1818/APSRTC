import duckdb
import pandas as pd
import numpy as np
import joblib
import datetime as dt

con = duckdb.connect("apsrtc.db")

bundle = joblib.load("demand_model.joblib")
pipeline = bundle["pipeline"]
num_cols = bundle["num_cols"]
cat_cols = bundle["cat_cols"]

def get_median_pax_per_trip(depot_id):
    df = con.execute(f"""
        SELECT passengers, trips
        FROM depot_daily_summary
        WHERE DEPOT_ID = {depot_id}
    """).fetchdf()

    df["ppt"] = df["passengers"] / df["trips"]
    return df["ppt"].median()

def forecast_depot(depot_id, days=30):

    df = con.execute(f"""
        SELECT *
        FROM depot_daily_summary
        WHERE DEPOT_ID = {depot_id}
        ORDER BY STAR_DATE
    """).fetchdf()

    df["STAR_DATE"] = pd.to_datetime(df["STAR_DATE"])
    df = df.sort_values("STAR_DATE")

    # Same feature engineering as training
    df["day"] = df["STAR_DATE"].dt.day
    df["month"] = df["STAR_DATE"].dt.month
    df["year"] = df["STAR_DATE"].dt.year
    df["dow"] = df["STAR_DATE"].dt.weekday
    df["weekofyear"] = df["STAR_DATE"].dt.isocalendar().week.astype(int)

    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_month_start"] = df["STAR_DATE"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["STAR_DATE"].dt.is_month_end.astype(int)
    df["is_festival"] = 0

    df["lag_1"] = df["passengers"].shift(1)
    df["lag_7"] = df["passengers"].shift(7)
    df["lag_14"] = df["passengers"].shift(14)
    df["roll7_mean"] = df["passengers"].shift(1).rolling(7).mean()
    df["roll14_mean"] = df["passengers"].shift(1).rolling(14).mean()

    df = df.dropna().reset_index(drop=True)

    series = df.set_index("STAR_DATE")["passengers"].copy()
    last_date = df["STAR_DATE"].max()

    median_ppt = get_median_pax_per_trip(depot_id)

    future = []

    for i in range(1, days + 1):
        next_date = last_date + dt.timedelta(days=i)

        row = df.iloc[-1].copy()
        row["STAR_DATE"] = next_date

        row["day"] = next_date.day
        row["month"] = next_date.month
        row["year"] = next_date.year
        row["dow"] = next_date.weekday
        row["weekofyear"] = next_date.isocalendar()[1]

        row["is_weekend"] = int(next_date.weekday() in [5,6])
        row["is_month_start"] = int(next_date.day == 1)
        row["is_month_end"] = int(next_date.day >= 28)
        row["is_festival"] = 0

        row["lag_1"] = series.iloc[-1]
        row["lag_7"] = series.iloc[-7]
        row["lag_14"] = series.iloc[-14]
        row["roll7_mean"] = series.tail(7).mean()
        row["roll14_mean"] = series.tail(14).mean()

        X = pd.DataFrame([row[num_cols + cat_cols]])
        pred = pipeline.predict(X)[0]

        pred = max(pred, 1000)
        series.loc[next_date] = pred

        required_services = round(pred / median_ppt)
        required_vehicles = round(required_services / 4)

        future.append([
            next_date,
            pred,
            required_services,
            required_vehicles
        ])

    forecast_df = pd.DataFrame(
        future,
        columns=["date", "passengers", "required_services", "required_vehicles"]
    )

    return forecast_df