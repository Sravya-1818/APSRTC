import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from math import ceil
from sqlalchemy import create_engine

st.set_page_config(layout="wide")

st.title("Route-wise 30-Day Demand Forecast")

# --------------------------------------------------
# BOX CSS (UNCHANGED)
# --------------------------------------------------

st.markdown("""
<style>
.day-box {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    text-align: center;
    margin-bottom: 10px;
}
.date-text {
    font-size: 14px;
    color: #6b7280;
}
.service-text {
    font-size: 22px;
    font-weight: 700;
    margin-top: 8px;
}
.passenger-text {
    font-size: 13px;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

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

# --------------------------------------------------
# DEPOT SELECT
# --------------------------------------------------

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
# ROUTE SELECT
# --------------------------------------------------

route_df = pd.read_sql(f"""
    SELECT DISTINCT "ROUTE_ID", "ROUTE_NAME"
    FROM route_daily
    WHERE "DEPOT_ID" = {selected_depot}
    ORDER BY "ROUTE_NAME"
""", engine)

if route_df.empty:
    st.warning("No routes available.")
    st.stop()

route_map = route_df.set_index("ROUTE_ID")["ROUTE_NAME"].to_dict()

selected_route = st.selectbox(
    "Select Route",
    options=list(route_map.keys()),
    format_func=lambda x: route_map[x]
)

# --------------------------------------------------
# LOAD ROUTE HISTORY (DB REPLACED)
# --------------------------------------------------

df = pd.read_sql(f"""
    SELECT "STAR_DATE",
           SUM(trips) as services,
           SUM(passengers) as passengers
    FROM route_daily
    WHERE "DEPOT_ID" = {selected_depot}
    AND "ROUTE_ID" = {selected_route}
    GROUP BY "STAR_DATE"
    ORDER BY "STAR_DATE"
""", engine)

if len(df) < 10:
    st.warning("Not enough historical data.")
    st.stop()

df["STAR_DATE"] = pd.to_datetime(df["STAR_DATE"])
df["dow"] = df["STAR_DATE"].dt.dayofweek
df["month"] = df["STAR_DATE"].dt.month
df["is_weekend"] = df["dow"].isin([5,6]).astype(int)

features = ["dow", "month", "is_weekend"]

# --------------------------------------------------
# TRAIN MODEL (UNCHANGED)
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

model.fit(df[features], df["passengers"])

# --------------------------------------------------
# EFFECTIVE CAPACITY (UNCHANGED)
# --------------------------------------------------

df["load_factor"] = df["passengers"] / (df["services"] * 55)
df["load_factor"] = df["load_factor"].replace([np.inf,-np.inf], np.nan)
df["load_factor"] = df["load_factor"].fillna(0.75)

avg_load = df["load_factor"].clip(0.5,1).mean()
effective_capacity = max(int(55 * avg_load), 45)

# --------------------------------------------------
# FORECAST 30 DAYS
# --------------------------------------------------

future_dates = [
    datetime.today().date() + timedelta(days=i)
    for i in range(1,31)
]

forecast_data = []

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

    forecast_data.append({
        "date": d,
        "services": pred_services,
        "passengers": pred_passengers
    })

forecast_df = pd.DataFrame(forecast_data)

# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

p75 = forecast_df["passengers"].quantile(0.75)
p90 = forecast_df["passengers"].quantile(0.90)

def classify(val):
    if val >= p90:
        return "PEAK"
    elif val >= p75:
        return "HIGH"
    else:
        return "NORMAL"

forecast_df["level"] = forecast_df["passengers"].apply(classify)

# --------------------------------------------------
# DISPLAY GRID (UNCHANGED)
# --------------------------------------------------

st.subheader("Upcoming 30 Days")

cols = 7
rows = int(np.ceil(len(forecast_df) / cols))
index = 0

for r in range(rows):
    columns = st.columns(cols)
    for c in range(cols):
        if index >= len(forecast_df):
            break

        row = forecast_df.iloc[index]

        with columns[c]:
            st.markdown(f"""
            <div class="day-box">
                <div class="date-text">{row['date'].strftime('%d %b')}</div>
                <div class="service-text">{row['services']} Services</div>
                <div class="passenger-text">{row['passengers']:,} passengers</div>
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
# SUMMARY (UNCHANGED)
# --------------------------------------------------

st.divider()

c1, c2 = st.columns(2)

c1.metric(
    "Total Services (30 Days)",
    f"{forecast_df['services'].sum():,}"
)

c2.metric(
    "Total Passengers (30 Days)",
    f"{forecast_df['passengers'].sum():,}"
)