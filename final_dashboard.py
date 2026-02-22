import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingRegressor

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="APSRTC Depot Manager Dashboard", page_icon="🚌", layout="wide")

# -----------------------------
# DB CONNECTION
# -----------------------------
DB_PATH = "D:/APSRTC/apsrtc.db"
con = duckdb.connect(DB_PATH, read_only=True)

# -----------------------------
# UI STYLE (YOUR STYLE KEPT)
# -----------------------------
st.markdown("""
<style>
.kpi{
  padding:14px 16px;border-radius:14px;border:1px solid rgba(0,0,0,0.08);
  background:#fff;box-shadow:0 2px 10px rgba(0,0,0,0.04);height:92px;
}
.kpi_t{font-size:12px;opacity:.75;margin-bottom:6px}
.kpi_v{font-size:26px;font-weight:800;line-height:1.05}
.kpi_s{font-size:12px;opacity:.7;margin-top:6px}

.tile{
  border-radius:12px;padding:10px;border:1px solid rgba(0,0,0,0.10);
  min-height:118px;
}
.tile_day{font-weight:800;font-size:16px;margin-bottom:6px}
.tile_big{font-weight:900;font-size:26px}
.pill{
  display:inline-block;padding:3px 8px;border-radius:999px;font-size:11px;font-weight:700;
  border:1px solid rgba(0,0,0,0.12);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DEPOTS
# =========================================================
@st.cache_data
def get_depots():
    df = con.execute("""
        SELECT DISTINCT DEPOT_ID, DEPOT_NAME
        FROM depot_daily_summary
        ORDER BY DEPOT_NAME
    """).fetchdf()
    df["label"] = df["DEPOT_ID"].astype(str) + " - " + df["DEPOT_NAME"]
    return df

depots = get_depots()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    depot_label = st.selectbox("Select Depot", depots["label"])
    depot_id = int(depot_label.split(" - ")[0])
    depot_name = depot_label.split(" - ")[1]

# =========================================================
# LOAD HISTORY
# =========================================================
@st.cache_data
def load_history(depot_id):
    df = con.execute("""
        SELECT STAR_DATE, passengers, revenue, trips, active_vehicles
        FROM depot_daily_summary
        WHERE DEPOT_ID = ?
        ORDER BY STAR_DATE
    """, [depot_id]).fetchdf()
    df["STAR_DATE"] = pd.to_datetime(df["STAR_DATE"])
    return df

history = load_history(depot_id)

tabs = st.tabs(["🏠 Overview", "🗓 Operational Calendar"])

# =========================================================
# 1️⃣ OVERVIEW
# =========================================================
with tabs[0]:

    st.subheader("Depot Overview")

    if history.empty:
        st.error("No data found.")
        st.stop()

    last = history.iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)

    def kpi(col, title, value):
        col.markdown(f"""
        <div class="kpi">
            <div class="kpi_t">{title}</div>
            <div class="kpi_v">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    kpi(col1, "Passengers (Last Day)", f"{int(last['passengers']):,}")
    kpi(col2, "Trips", f"{int(last['trips'])}")
    kpi(col3, "Revenue", f"₹{int(last['revenue']):,}")
    kpi(col4, "Active Vehicles", f"{int(last['active_vehicles'])}")
    kpi(col5, "Passengers / Trip", f"{(last['passengers']/max(1,last['trips'])):.1f}")

    st.markdown("### Last 30 Days Trend")
    fig = px.line(history.tail(30), x="STAR_DATE", y="passengers", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 2️⃣ OPERATIONAL CALENDAR (FORECAST ONLY)
# =========================================================
with tabs[1]:

    st.subheader("Operational Calendar (Future Forecast Only)")

    # ---------------------------
    # TRAIN MODEL
    # ---------------------------
    df = history.copy()

    df["dow"] = df["STAR_DATE"].dt.dayofweek
    df["month"] = df["STAR_DATE"].dt.month

    for lag in [7, 14, 28]:
        df[f"lag_{lag}"] = df["passengers"].shift(lag)

    df = df.dropna()

    X = df[["dow", "month", "lag_7", "lag_14", "lag_28"]]
    y = df["passengers"]

    model = HistGradientBoostingRegressor(max_depth=6, max_iter=400)
    model.fit(X, y)

    # ---------------------------
    # FORECAST NEXT 365 DAYS
    # ---------------------------
    last_date = history["STAR_DATE"].max()
    series = history.set_index("STAR_DATE")["passengers"].astype(float)

    future = []

    for i in range(1, 366):
        d = last_date + pd.Timedelta(days=i)

        row = {
            "dow": d.dayofweek,
            "month": d.month,
            "lag_7": series.get(d - pd.Timedelta(days=7), series.mean()),
            "lag_14": series.get(d - pd.Timedelta(days=14), series.mean()),
            "lag_28": series.get(d - pd.Timedelta(days=28), series.mean()),
        }

        yhat = model.predict(pd.DataFrame([row]))[0]
        yhat = max(0, yhat)

        future.append((d.date(), yhat))
        series.loc[d] = yhat

    forecast = pd.DataFrame(future, columns=["date", "passengers"])

    # ---------------------------
    # SELECT MONTH
    # ---------------------------
    year = st.selectbox("Year", sorted(forecast["date"].apply(lambda x: x.year).unique()))
    month = st.selectbox("Month", list(range(1,13)))

    month_df = forecast[
        (forecast["date"].apply(lambda x: x.year) == year) &
        (forecast["date"].apply(lambda x: x.month) == month)
    ]

    if month_df.empty:
        st.warning("No forecast for this month.")
        st.stop()

    # DEPOT-SPECIFIC THRESHOLDS
    q70 = history["passengers"].quantile(0.70)
    q90 = history["passengers"].quantile(0.90)

    def label(p):
        if p >= q90:
            return "PEAK"
        elif p >= q70:
            return "HIGH"
        return "NORMAL"

    month_df["label"] = month_df["passengers"].apply(label)

    ratio = (history["passengers"] / history["trips"].replace(0,np.nan)).median()
    ratio = ratio if np.isfinite(ratio) else 45
    month_df["services"] = month_df["passengers"] / ratio

    # ---------------------------
    # CALENDAR GRID
    # ---------------------------
    first_day = dt.date(year, month, 1)
    if month == 12:
        nxt = dt.date(year+1,1,1)
    else:
        nxt = dt.date(year,month+1,1)

    days = (nxt - first_day).days
    start_weekday = (first_day.weekday()+1)%7

    cells = [None]*start_weekday + list(range(1,days+1))
    while len(cells)%7 != 0:
        cells.append(None)

    by_day = {row["date"].day: row for _,row in month_df.iterrows()}

    for week in range(len(cells)//7):
        cols = st.columns(7)
        for i in range(7):
            day = cells[week*7+i]
            if day is None or day not in by_day:
                cols[i].markdown("<div class='tile'></div>", unsafe_allow_html=True)
                continue

            r = by_day[day]

            if r["label"]=="PEAK":
                style="background:#e84b4b;color:white;"
                badge="🔴 PEAK"
            elif r["label"]=="HIGH":
                style="background:#f6e6b2;"
                badge="🟡 HIGH"
            else:
                style="background:#f5f7fb;"
                badge="⚪ NORMAL"

            cols[i].markdown(f"""
            <div class="tile" style="{style}">
              <div class="tile_day">{day} <span class="pill">{badge}</span></div>
              <div class="tile_big">{int(r['services']):,}</div>
              <div>Services</div>
              <div>{int(r['passengers']):,} pax</div>
            </div>
            """, unsafe_allow_html=True)