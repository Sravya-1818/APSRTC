import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from sqlalchemy import create_engine

st.set_page_config(layout="wide")

# --------------------------------------------------
# LOAD BUS IMAGE (Top/Side View PNG)
# --------------------------------------------------

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bus_img = get_base64_image("assets/bus.png")

# --------------------------------------------------
# ADVANCED BACKGROUND + ANIMATION CSS
# --------------------------------------------------

st.markdown(f"""
<style>

/* Light grid background */
[data-testid="stAppViewContainer"] {{
    background-color: #f9fafb;
    background-image:
        linear-gradient(to right, rgba(0,0,0,0.04) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(0,0,0,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
}}

/* Moving bus animation */
.bus-container {{
    position: fixed;
    bottom: 40px;
    left: -200px;
    width: 180px;
    opacity: 0.08;
    animation: moveBus 25s linear infinite;
    z-index: 0;
}}

@keyframes moveBus {{
    0% {{ left: -200px; }}
    100% {{ left: 110%; }}
}}

/* Depot Arc Background */
.depot-arc {{
    position: fixed;
    top: 120px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 100px;
    font-weight: 800;
    color: rgba(21,128,61,0.05);
    white-space: nowrap;
    z-index: 0;
}}

/* Cards */
.card {{
    background-color: #ffffff;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    text-align: center;
    transition: 0.3s ease;
    position: relative;
    z-index: 2;
}}

.card:hover {{
    transform: translateY(-8px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.12);
}}

.card-title {{
    font-size: 14px;
    color: #6b7280;
}}

.card-value {{
    font-size: 32px;
    font-weight: 700;
    color: #15803d;
}}

.section-title {{
    font-size: 22px;
    font-weight: 600;
    margin-top: 40px;
    position: relative;
    z-index: 2;
}}

</style>

<div class="bus-container">
    <img src="data:image/png;base64,{bus_img}" width="180">
</div>

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

st.title("Overview — Depot Summary")

# --------------------------------------------------
# DEPOT SELECTION
# --------------------------------------------------

depot_df = pd.read_sql("""
    SELECT DISTINCT "DEPOT_ID", "DEPOT_NAME"
    FROM route_daily
    ORDER BY "DEPOT_NAME"
""", engine)

if depot_df.empty:
    st.warning("No depots found in database.")
    st.stop()

depot_dict = depot_df.set_index("DEPOT_ID")["DEPOT_NAME"].to_dict()

selected_depot = st.selectbox(
    "Select Depot",
    options=list(depot_dict.keys()),
    format_func=lambda x: depot_dict[x]
)

depot_name = depot_dict[selected_depot]

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------

routes = pd.read_sql(f"""
    SELECT COUNT(DISTINCT "ROUTE_ID") AS total_routes
    FROM route_daily
    WHERE "DEPOT_ID" = {selected_depot}
""", engine)

summary = pd.read_sql(f"""
    SELECT 
        COALESCE(SUM(passengers),0) AS passengers,
        COALESCE(SUM(revenue),0) AS revenue,
        COALESCE(SUM(trips),0) AS trips
    FROM depot_daily_summary
    WHERE "DEPOT_ID" = {selected_depot}
""", engine)

total_routes = int(routes["total_routes"][0])
total_passengers = int(summary["passengers"][0])
total_revenue = int(summary["revenue"][0])
total_trips = int(summary["trips"][0])

c1, c2, c3, c4 = st.columns(4)

for col, title, value in [
    (c1, "Total Routes", total_routes),
    (c2, "Total Passengers", f"{total_passengers:,}"),
    (c3, "Total Revenue", f"₹{total_revenue:,}"),
    (c4, "Total Trips", f"{total_trips:,}")
]:
    col.markdown(f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# FLEET COMPOSITION
# --------------------------------------------------

st.markdown('<div class="section-title">Fleet Composition</div>', unsafe_allow_html=True)

fleet = pd.read_sql(f"""
    SELECT "SERVICE_TYPE_NAME", COUNT(DISTINCT "ROUTE_ID") AS count
    FROM route_daily
    WHERE "DEPOT_ID" = {selected_depot}
    GROUP BY "SERVICE_TYPE_NAME"
""", engine)

if not fleet.empty:
    cols = st.columns(len(fleet))
    for i, row in fleet.iterrows():
        cols[i].markdown(f"""
        <div class="card">
            <div class="card-title">{row['SERVICE_TYPE_NAME']}</div>
            <div class="card-value">{row['count']}</div>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# PIE CHART
# --------------------------------------------------

if not fleet.empty:
    st.markdown('<div class="section-title">Service Type Distribution</div>', unsafe_allow_html=True)

    fig = px.pie(
        fleet,
        names="SERVICE_TYPE_NAME",
        values="count",
        hole=0.55
    )

    fig.update_traces(
        textinfo='percent+label',
        pull=[0.05]*len(fleet)
    )

    fig.update_layout(
        height=450,
        transition_duration=800
    )

    st.plotly_chart(fig, use_container_width=True)