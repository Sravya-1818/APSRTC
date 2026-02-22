import duckdb

con = duckdb.connect("apsrtc_deploy.db")

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

for table in tables:
    print(f"\n========== {table} ==========")
    df = con.execute(f"DESCRIBE {table}").fetchdf()
    print(df[["column_name", "column_type"]])