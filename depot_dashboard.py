import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="APSRTC Depot Control")

# -------------------------
# CONNECT TO DB
# -------------------------
con = duckdb.connect("D:/APSRTC/apsrtc.db", read_only=True)

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("🚌 APSRTC Control")

section = st.sidebar.radio(
    "Navigation",
    ["Overview", "Operations Analytics"]
)

# -------------------------
# FILTERS
# -------------------------
col1, col2, col3 = st.columns(3)

with col1:
    depots = con.execute(
        "SELECT DISTINCT DEPOT_ID, DEPOT_NAME FROM depot_daily_kpi ORDER BY DEPOT_ID"
    ).fetchdf()
    depot_option = st.selectbox(
        "Select Depot",
        depots["DEPOT_ID"].astype(str) + " - " + depots["DEPOT_NAME"]
    )
    selected_depot = int(depot_option.split(" - ")[0])

with col2:
    start_date = st.date_input("Start Date")

with col3:
    end_date = st.date_input("End Date")

if start_date > end_date:
    st.error("Start date must be before end date")
    st.stop()

# -------------------------
# LOAD DATA (FAST)
# -------------------------
query = f"""
SELECT *
FROM depot_daily_kpi
WHERE DEPOT_ID = {selected_depot}
AND STAR_DATE BETWEEN '{start_date}' AND '{end_date}'
"""

df = con.execute(query).fetchdf()

if df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# =========================
# OVERVIEW SECTION
# =========================
if section == "Overview":

    total_revenue = df["revenue"].sum()
    total_passengers = df["passengers"].sum()
    total_trips = df["trips"].sum()
    active_vehicles = df["VEHICLE_NO"].nunique()

    revenue_per_trip = total_revenue / total_trips if total_trips else 0
    passengers_per_trip = total_passengers / total_trips if total_trips else 0

    st.markdown("## 🏢 Depot Performance Summary")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric("💰 Revenue", f"₹{total_revenue:,.0f}")
    k2.metric("👥 Passengers", f"{total_passengers:,.0f}")
    k3.metric("🚌 Trips", f"{total_trips:,.0f}")
    k4.metric("🚍 Active Vehicles", f"{active_vehicles}")
    k5.metric("₹ Revenue/Trip", f"{revenue_per_trip:,.0f}")
    k6.metric("👥 Passengers/Trip", f"{passengers_per_trip:.1f}")

# =========================
# OPERATIONS ANALYTICS
# =========================
if section == "Operations Analytics":

    st.markdown("## 📊 Service Mix")

    service_df = df.groupby("SERVICE_TYPE_NAME")["passengers"].sum().reset_index()
    fig1 = px.pie(service_df,
                  names="SERVICE_TYPE_NAME",
                  values="passengers",
                  hole=0.5)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("## 🛣 Route Intelligence")

    route_df = (
        df.groupby("ROUTE_NAME")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(route_df,
                  x="revenue",
                  y="ROUTE_NAME",
                  orientation="h")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## 🚍 Fleet Utilization")

    fleet_df = (
        df.groupby("VEHICLE_NO")["trips"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig3 = px.bar(fleet_df,
                  x="VEHICLE_NO",
                  y="trips")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("## 📈 Daily Demand Trend")

    daily_df = (
        df.groupby("STAR_DATE")["passengers"]
        .sum()
        .reset_index()
    )

    fig4 = px.line(daily_df,
                   x="STAR_DATE",
                   y="passengers",
                   markers=True)
    st.plotly_chart(fig4, use_container_width=True)