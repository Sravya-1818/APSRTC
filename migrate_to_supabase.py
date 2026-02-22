import duckdb
import pandas as pd
from sqlalchemy import create_engine

# ---------------------------------------------
# 1️⃣ CONNECT TO LOCAL DUCKDB
# ---------------------------------------------
duck_con = duckdb.connect("apsrtc_deploy.db")

# ---------------------------------------------
# 2️⃣ CONNECT TO SUPABASE (REPLACE PASSWORD)
# ---------------------------------------------
SUPABASE_DB_URL = "postgresql://postgres.pdrgkavwfiwunlxglgms:Sravya94924@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres?sslmode=require"

engine = create_engine(SUPABASE_DB_URL)

# ---------------------------------------------
# 3️⃣ TABLES TO MIGRATE
# ---------------------------------------------
tables = [
    "route_daily",
    "depot_daily_summary",
    "trip_level",
    "depot_vehicles",
    "depot_daily_kpi",
    "depot_route_summary",
    "service_master",
    "service_master_clean",
    "vehicle_daily_summary"
]

# ---------------------------------------------
# 4️⃣ MIGRATION PROCESS
# ---------------------------------------------
for table in tables:
    print(f"\nMigrating {table}...")

    df = duck_con.execute(f"SELECT * FROM {table}").fetchdf()

    df.to_sql(
        table,
        engine,
       if_exists="replace",
        index=False,
        method="multi",
        chunksize=5000
    )

    print(f"{table} migrated successfully ✅")

print("\n🎉 All tables migrated successfully!")