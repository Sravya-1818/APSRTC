import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from math import ceil
from sqlalchemy import create_engine

st.set_page_config(layout="wide")

# --------------------------------------------------
# UI STYLING (UNCHANGED)
# --------------------------------------------------

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #f8fafc, #eef2f7);
}
.route-box {
    background: rgba(255,255,255,0.85);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.06);
}
.route-name {
    font-size: 18px;
    font-weight: 600;
    color: #111827;
}
.sub-text {
    font-size: 14px;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

st.title("Route Operations — Top 20 Busy Routes")

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

selected_date = st.date_input(
    "Select Forecast Date",
    datetime.today().date() + timedelta(days=1)
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

    current_services = int(df["services"].mean())

    dow = selected_date.weekday()
    month = selected_date.month
    weekend = 1 if dow in [5,6] else 0

    X = pd.DataFrame([{
        "dow": dow,
        "month": month,
        "is_weekend": weekend
    }])

    pred_passengers = int(model.predict(X)[0])
    required_services = ceil(pred_passengers / effective_capacity)

    load_percent = min(100, int((pred_passengers / (required_services * 55)) * 100))

    route_predictions.append({
        "ROUTE_NAME": route_name,
        "current_services": current_services,
        "required_services": required_services,
        "pred_passengers": pred_passengers,
        "load_percent": load_percent
    })

route_df = pd.DataFrame(route_predictions)

if route_df.empty:
    st.warning("Not enough data.")
    st.stop()

top_routes = route_df.sort_values(
    "pred_passengers",
    ascending=False
).head(20)

st.subheader("Top 20 Busy Routes")

# --------------------------------------------------
# DISPLAY LAYOUT (UNCHANGED)
# --------------------------------------------------

for _, row in top_routes.iterrows():

    delta = row["required_services"] - row["current_services"]

    with st.container():
        st.markdown('<div class="route-box">', unsafe_allow_html=True)

        st.markdown(
            f'<div class="route-name">{row["ROUTE_NAME"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="sub-text">Estimated Pax: {row["pred_passengers"]:,}</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Current Services", row["current_services"])

        if delta > 0:
            col2.metric(
                "Required Services",
                row["required_services"],
                f"+{delta}",
                delta_color="inverse"
            )
        else:
            col2.metric(
                "Required Services",
                row["required_services"],
                f"{delta}"
            )

        col3.metric("Load %", f"{row['load_percent']}%")

        st.progress(row["load_percent"] / 100)

        st.markdown('</div>', unsafe_allow_html=True)