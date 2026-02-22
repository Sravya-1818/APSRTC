import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from math import ceil
from sqlalchemy import create_engine

st.set_page_config(layout="wide")

# --------------------------------------------------
# 🎨 UI STYLING (UNCHANGED)
# --------------------------------------------------

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #f8fafc, #eef2f7);
}
h1 { font-weight: 700; color: #1f2937; }
.calendar-card {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(6px);
    border-radius: 16px;
    padding: 15px;
    transition: 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.calendar-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.12);
}
.date-text { font-size: 13px; font-weight: 600; color: #6b7280; }
.service-text { font-size: 20px; font-weight: 700; color: #111827; }
[data-testid="stDataFrame"] {
    background-color: rgba(255,255,255,0.8);
    border-radius: 16px;
    padding: 10px;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.7);
    padding: 15px;
    border-radius: 14px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

st.title("30-Day Operational Forecast Calendar")

# --------------------------------------------------
# DATABASE (ONLY THIS PART CHANGED)
# --------------------------------------------------

DATABASE_URL = st.secrets["SUPABASE_DB_URL"]

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)

depot_df = pd.read_sql("""
    SELECT DISTINCT "DEPOT_ID", "DEPOT_NAME"
    FROM route_daily
    ORDER BY "DEPOT_NAME"
""", engine)

depot_map = depot_df.set_index("DEPOT_ID")["DEPOT_NAME"].to_dict()

selected_depot = st.selectbox(
    "Select Depot",
    options=list(depot_map.keys()),
    format_func=lambda x: depot_map[x]
)

# --------------------------------------------------
# LOAD ROUTES (DB REPLACED)
# --------------------------------------------------

routes_df = pd.read_sql(f"""
    SELECT DISTINCT "ROUTE_ID", "ROUTE_NAME"
    FROM route_daily
    WHERE "DEPOT_ID" = {selected_depot}
""", engine)

if routes_df.empty:
    st.warning("No routes found.")
    st.stop()

# --------------------------------------------------
# FORECAST ENGINE (UNCHANGED)
# --------------------------------------------------

future_dates = [
    datetime.today().date() + timedelta(days=i)
    for i in range(1, 31)
]

route_predictions = []

for _, route_row in routes_df.iterrows():

    route_id = route_row["ROUTE_ID"]
    route_name = route_row["ROUTE_NAME"]

    df = pd.read_sql(f"""
        SELECT "STAR_DATE",
               SUM(trips) as services,
               SUM(passengers) as passengers
        FROM route_daily
        WHERE "DEPOT_ID" = {selected_depot}
        AND "ROUTE_ID" = {route_id}
        GROUP BY "STAR_DATE"
        ORDER BY "STAR_DATE"
    """, engine)

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

    df["load_factor"] = df["passengers"] / (df["services"] * 55)
    df["load_factor"] = df["load_factor"].replace([np.inf,-np.inf], np.nan)
    df["load_factor"] = df["load_factor"].fillna(0.75)

    avg_load = df["load_factor"].clip(0.5,1).mean()
    effective_capacity = max(int(55 * avg_load), 45)

    for d in future_dates:
        dow = d.weekday()
        month = d.month
        weekend = 1 if dow in [5,6] else 0

        X = pd.DataFrame([{
            "dow": dow,
            "month": month,
            "is_weekend": weekend
        }])

        pred_passengers = int(model.predict(X)[0])
        pred_services = ceil(pred_passengers / effective_capacity)

        route_predictions.append({
            "date": d,
            "ROUTE_NAME": route_name,
            "pred_passengers": pred_passengers,
            "pred_services": pred_services
        })

forecast_df = pd.DataFrame(route_predictions)

if forecast_df.empty:
    st.warning("Not enough historical data.")
    st.stop()

depot_daily = forecast_df.groupby("date").agg({
    "pred_passengers": "sum",
    "pred_services": "sum"
}).reset_index()

p75 = depot_daily["pred_passengers"].quantile(0.75)
p90 = depot_daily["pred_passengers"].quantile(0.90)

def classify(val):
    if val >= p90:
        return "PEAK"
    elif val >= p75:
        return "HIGH"
    else:
        return "NORMAL"

depot_daily["level"] = depot_daily["pred_passengers"].apply(classify)

# --------------------------------------------------
# DISPLAY CALENDAR (UNCHANGED)
# --------------------------------------------------

st.subheader("Upcoming 30 Days")

cols_per_row = 7
rows = int(np.ceil(len(depot_daily) / cols_per_row))

index = 0

for r in range(rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        if index >= len(depot_daily):
            break

        row = depot_daily.iloc[index]

        with cols[c]:
            st.markdown(f"""
            <div class="calendar-card">
                <div class="date-text">
                    {row['date'].strftime('%d %b')}
                </div>
                <div class="service-text">
                    {int(row['pred_services'])} Services
                </div>
                <div style="font-size:13px;color:#6b7280;margin-bottom:6px;">
                    {int(row['pred_passengers']):,} passengers
                </div>
            </div>
            """, unsafe_allow_html=True)

            if row["level"] == "PEAK":
                st.error("PEAK")
            elif row["level"] == "HIGH":
                st.warning("HIGH")
            else:
                st.success("NORMAL")

        index += 1

# --------------------------------------------------
# ROUTE SPLIT (UNCHANGED)
# --------------------------------------------------

st.divider()
st.subheader("Route-wise & Service-wise Allocation")

selected_date = st.selectbox(
    "Select Date",
    depot_daily["date"].sort_values()
)

route_split = forecast_df[forecast_df["date"] == selected_date] \
    .sort_values("pred_passengers", ascending=False) \
    .reset_index(drop=True)

st.subheader("Route-wise Split")

st.dataframe(
    route_split[["ROUTE_NAME", "pred_passengers", "pred_services"]],
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# SUMMARY (UNCHANGED)
# --------------------------------------------------

st.divider()
st.subheader("30-Day Forecast Summary")

c1, c2 = st.columns(2)

c1.metric(
    "Total Forecasted Services",
    f"{int(depot_daily['pred_services'].sum()):,}"
)

c2.metric(
    "Total Forecasted Passengers",
    f"{int(depot_daily['pred_passengers'].sum()):,}"
)