from fastapi import FastAPI, Header, HTTPException
import duckdb

app = FastAPI()

# ======================================
# 🔐 YOUR API KEY (SHARE THIS ONLY)
# ======================================

API_KEY = "APSRTC_2026_SECURE_KEY_98765"

# ======================================
# 🔑 API KEY VALIDATION
# ======================================

def verify_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# ======================================
# 📊 DEPOT SUMMARY ENDPOINT
# ======================================

@app.get("/depot-summary")
def depot_summary(depot_id: int, x_api_key: str = Header(None)):

    verify_key(x_api_key)

    conn = duckdb.connect("apsrtc.db", read_only=True)

    result = conn.execute(f"""
        SELECT 
            SUM(trips) AS total_services,
            SUM(passengers) AS total_passengers,
            SUM(revenue) AS total_revenue
        FROM route_daily
        WHERE DEPOT_ID = {depot_id}
    """).fetchdf()

    return result.to_dict(orient="records")


# ======================================
# 🚌 TOP 20 ROUTES ENDPOINT
# ======================================

@app.get("/top-routes")
def top_routes(depot_id: int, x_api_key: str = Header(None)):

    verify_key(x_api_key)

    conn = duckdb.connect("apsrtc.db", read_only=True)

    result = conn.execute(f"""
        SELECT ROUTE_NAME,
               SUM(passengers) AS total_passengers,
               SUM(trips) AS total_services
        FROM route_daily
        WHERE DEPOT_ID = {depot_id}
        GROUP BY ROUTE_NAME
        ORDER BY total_passengers DESC
        LIMIT 20
    """).fetchdf()

    return result.to_dict(orient="records")