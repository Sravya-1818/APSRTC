import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from math import ceil
import joblib

print("Connecting to database...")
conn = duckdb.connect("apsrtc.db", read_only=True)

depots = conn.execute("""
    SELECT DISTINCT DEPOT_ID
    FROM route_daily
""").fetchdf()

all_models = {}

for depot_id in depots["DEPOT_ID"]:

    print(f"Training Depot {depot_id}")

    routes_df = conn.execute(f"""
        SELECT DISTINCT ROUTE_ID
        FROM route_daily
        WHERE DEPOT_ID = {depot_id}
    """).fetchdf()

    depot_models = {}

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

        if len(df) < 15:
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

        # Capacity logic (same as your current code)
        df["load_factor"] = df["passengers"] / (df["services"] * 55)
        df["load_factor"] = df["load_factor"].replace([np.inf,-np.inf], np.nan)
        df["load_factor"] = df["load_factor"].fillna(0.75)

        avg_load = df["load_factor"].clip(0.5,1).mean()
        effective_capacity = max(int(55 * avg_load), 45)

        depot_models[route_id] = {
            "model": model,
            "capacity": effective_capacity
        }

    all_models[depot_id] = depot_models

joblib.dump(all_models, "operational_models.pkl")

print("✅ All models saved as operational_models.pkl")