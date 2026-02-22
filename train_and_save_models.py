import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Training models...")

conn = duckdb.connect("apsrtc.db")

depot_df = conn.execute("""
    SELECT DISTINCT DEPOT_ID
    FROM route_daily
""").fetchdf()

models = {}

for depot_id in depot_df["DEPOT_ID"]:

    routes_df = conn.execute(f"""
        SELECT DISTINCT ROUTE_ID
        FROM route_daily
        WHERE DEPOT_ID = {depot_id}
    """).fetchdf()

    models[depot_id] = {}

    for route_id in routes_df["ROUTE_ID"]:

        df = conn.execute(f"""
            SELECT STAR_DATE,
                   SUM(trips) as services,
                   SUM(passengers) as passengers
            FROM route_daily
            WHERE DEPOT_ID = {depot_id}
            AND ROUTE_ID = {route_id}
            GROUP BY STAR_DATE
            ORDER BY STAR_DATE
        """).fetchdf()

        if len(df) < 10:
            continue

        df["STAR_DATE"] = pd.to_datetime(df["STAR_DATE"])
        df["dow"] = df["STAR_DATE"].dt.dayofweek
        df["month"] = df["STAR_DATE"].dt.month
        df["is_weekend"] = df["dow"].isin([5,6]).astype(int)

        features = ["dow", "month", "is_weekend"]

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )

        model.fit(df[features], df["passengers"])

        # capacity
        df["load_factor"] = df["passengers"] / (df["services"] * 55)
        df["load_factor"] = df["load_factor"].replace([np.inf,-np.inf], np.nan)
        df["load_factor"] = df["load_factor"].fillna(0.75)

        avg_load = df["load_factor"].clip(0.5,1).mean()
        effective_capacity = max(int(55 * avg_load), 45)

        models[depot_id][route_id] = {
            "model": model,
            "capacity": effective_capacity
        }

joblib.dump(models, "operational_models.pkl")

print("Models saved successfully.")