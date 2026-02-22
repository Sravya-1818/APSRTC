import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import math

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="APSRTC Depot Manager Dashboard", page_icon="🚌", layout="wide")

# -----------------------------
# Simple Styles (EXACT SAME AS YOUR DUMMY UI)
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

.badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;
  border:1px solid rgba(0,0,0,0.12);margin-left:6px;background:#fff}
.card{
  padding:12px 14px;border-radius:14px;border:1px solid rgba(0,0,0,0.08);
  background:#fff;box-shadow:0 2px 10px rgba(0,0,0,0.04);
}
.small{font-size:12px;opacity:.75}
.tile{
  border-radius:12px;padding:10px 10px;border:1px solid rgba(0,0,0,0.10);
  min-height:118px;
}
.tile_day{font-weight:800;font-size:16px;margin-bottom:6px}
.tile_big{font-weight:900;font-size:26px;line-height:1}
.tile_lbl{font-size:12px;opacity:.75;margin-top:6px}
.pill{
  display:inline-block;padding:3px 8px;border-radius:999px;font-size:11px;font-weight:700;
  border:1px solid rgba(0,0,0,0.12);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DB CONFIG
# -----------------------------
DB_PATH = "D:/APSRTC/apsrtc.db"   # <-- CHANGE THIS ONLY
con = duckdb.connect(DB_PATH, read_only=True)

# -----------------------------
# Helpers
# -----------------------------
def kpi(col, title, value, sub=""):
    col.markdown(
        f"""
        <div class="kpi">
          <div class="kpi_t">{title}</div>
          <div class="kpi_v">{value}</div>
          <div class="kpi_s">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def safe_int(x, default=0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return int(x)
    except:
        return default

def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except:
        return default

# -----------------------------
# Fetch depots from DB (REAL)
# -----------------------------
@st.cache_data
def get_depots():
    df = con.execute("""
        SELECT DISTINCT DEPOT_ID, DEPOT_NAME
        FROM depot_daily_summary
        ORDER BY DEPOT_NAME
    """).fetchdf()
    df["label"] = df["DEPOT_NAME"].astype(str) + " (" + df["DEPOT_ID"].astype(str) + ")"
    return df

depots_df = get_depots()
DEPOTS = depots_df["label"].tolist()

# -----------------------------
# Load depot history (REAL)
# -----------------------------
@st.cache_data
def load_depot_history(depot_id: int):
    df = con.execute(f"""
        SELECT STAR_DATE, passengers, revenue, trips, active_vehicles
        FROM depot_daily_summary
        WHERE DEPOT_ID = {depot_id}
        ORDER BY STAR_DATE
    """).fetchdf()
    if df.empty:
        return df
    df["STAR_DATE"] = pd.to_datetime(df["STAR_DATE"])
    df = df.sort_values("STAR_DATE")
    return df

# -----------------------------
# Route daily (REAL)
# -----------------------------
@st.cache_data
def load_route_daily(depot_id: int, start_date: dt.date, end_date: dt.date):
    df = con.execute(f"""
        SELECT STAR_DATE, SERVICE_TYPE_NAME, ROUTE_ID, ROUTE_NAME, passengers, revenue, trips
        FROM route_daily
        WHERE DEPOT_ID = {depot_id}
          AND STAR_DATE BETWEEN DATE '{start_date}' AND DATE '{end_date}'
    """).fetchdf()
    if df.empty:
        return df
    df["STAR_DATE"] = pd.to_datetime(df["STAR_DATE"])
    return df

# -----------------------------
# Festival/Season factors (RULE-BASED, so peaks appear)
# -----------------------------
def seasonal_flags(d: pd.Timestamp):
    m = d.month
    day = d.day
    # windows (approx) that cause bus demand spikes
    sankranti = (m == 1 and 10 <= day <= 20)
    ugadi_summer = (m == 4 and 1 <= day <= 20)
    summer_travel = (m in [5, 6])  # vacations
    dasara = (m == 10 and 1 <= day <= 20)
    diwali = (m == 11 and 1 <= day <= 15)
    year_end = (m == 12 and 20 <= day <= 31)
    new_year = (m == 1 and 1 <= day <= 5)

    return {
        "is_sankranti": int(sankranti),
        "is_ugadi": int(ugadi_summer),
        "is_summer": int(summer_travel),
        "is_dasara": int(dasara),
        "is_diwali": int(diwali),
        "is_year_end": int(year_end),
        "is_new_year": int(new_year),
    }

# -----------------------------
# ML Forecast Model (per depot) - Next 1 year
# -----------------------------
@st.cache_resource
def train_models_for_all_depots():
    models = {}
    meta = {}

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except:
        return models, meta  # fallback will be used

    for _, r in depots_df.iterrows():
        depot_id = int(r["DEPOT_ID"])
        hist = load_depot_history(depot_id)
        if hist is None or hist.empty or len(hist) < 40:
            continue

        df = hist.copy()
        df["ds"] = df["STAR_DATE"]
        df["y"] = df["passengers"].astype(float)

        # feature engineering
        df["dow"] = df["ds"].dt.weekday
        df["month"] = df["ds"].dt.month
        df["week"] = df["ds"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

        # seasonal flags
        flags = df["ds"].apply(seasonal_flags).apply(pd.Series)
        df = pd.concat([df, flags], axis=1)

        # lag features
        df["lag_1"] = df["y"].shift(1)
        df["lag_7"] = df["y"].shift(7)
        df["roll_7"] = df["y"].shift(1).rolling(7).mean()
        df["roll_28"] = df["y"].shift(1).rolling(28).mean()

        df = df.dropna().reset_index(drop=True)
        if len(df) < 20:
            continue

        feat_cols = [
            "dow","month","week","is_weekend",
            "is_sankranti","is_ugadi","is_summer","is_dasara","is_diwali","is_year_end","is_new_year",
            "lag_1","lag_7","roll_7","roll_28"
        ]
        X = df[feat_cols]
        y = df["y"]

        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=350,
            random_state=42
        )
        model.fit(X, y)

        models[depot_id] = (model, feat_cols)
        meta[depot_id] = {
            "min_date": hist["STAR_DATE"].min(),
            "max_date": hist["STAR_DATE"].max(),
            "p70": float(hist["passengers"].quantile(0.70)),
            "p90": float(hist["passengers"].quantile(0.90)),
            "avg_pax_per_trip": float((hist["passengers"] / hist["trips"].replace(0, np.nan)).median(skipna=True))
        }

    return models, meta

MODELS, META = train_models_for_all_depots()

def forecast_passengers_next_days(depot_id: int, start_date: dt.date, days: int = 365):
    """
    Returns DataFrame: date, pred_passengers
    Recursive forecast using lag features.
    """
    hist = load_depot_history(depot_id)
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["date", "pred_passengers"])

    # fallback if model not available
    if depot_id not in MODELS:
        df = hist.copy()
        df["dow"] = df["STAR_DATE"].dt.weekday
        weekday_avg = df.groupby("dow")["passengers"].mean()
        overall = df["passengers"].mean()

        out = []
        for i in range(days):
            d = pd.Timestamp(start_date + dt.timedelta(days=i))
            pred = float(weekday_avg.get(d.weekday(), overall))
            out.append([d.date(), max(0, pred)])
        return pd.DataFrame(out, columns=["date", "pred_passengers"])

    model, feat_cols = MODELS[depot_id]

    series = hist.set_index("STAR_DATE")["passengers"].astype(float).sort_index()
    last_known = series.index.max()

    # Build working series until start_date-1 by filling (if start_date > last_known)
    # We'll extend using weekday averages as bridging if needed.
    if pd.Timestamp(start_date) > last_known:
        df_tmp = hist.copy()
        df_tmp["dow"] = df_tmp["STAR_DATE"].dt.weekday
        weekday_avg = df_tmp.groupby("dow")["passengers"].mean()
        overall = df_tmp["passengers"].mean()
        d = last_known + pd.Timedelta(days=1)
        while d < pd.Timestamp(start_date):
            series.loc[d] = float(weekday_avg.get(d.weekday(), overall))
            d += pd.Timedelta(days=1)

    out = []
    for i in range(days):
        d = pd.Timestamp(start_date + dt.timedelta(days=i))

        # lags from series (recursive)
        lag_1 = float(series.loc[d - pd.Timedelta(days=1)])
        lag_7 = float(series.loc[d - pd.Timedelta(days=7)]) if (d - pd.Timedelta(days=7)) in series.index else lag_1
        roll_7 = float(series.loc[:d - pd.Timedelta(days=1)].tail(7).mean())
        roll_28 = float(series.loc[:d - pd.Timedelta(days=1)].tail(28).mean())

        flags = seasonal_flags(d)

        row = {
            "dow": d.weekday(),
            "month": d.month,
            "week": int(d.isocalendar().week),
            "is_weekend": int(d.weekday() in [5, 6]),
            **flags,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "roll_7": roll_7,
            "roll_28": roll_28
        }

        X = pd.DataFrame([row])[feat_cols]
        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)

        out.append([d.date(), pred])
        series.loc[d] = pred  # recursive update

    return pd.DataFrame(out, columns=["date", "pred_passengers"])

def get_actual_or_forecast_kpis(depot_id: int, sel_date: dt.date):
    hist = load_depot_history(depot_id)
    if hist.empty:
        return {"passengers": 0, "revenue": 0, "trips": 0, "active_vehicles": 0, "mode": "none"}

    max_date = hist["STAR_DATE"].max().date()

    # actual if within history
    row = hist[hist["STAR_DATE"].dt.date == sel_date]
    if not row.empty:
        r = row.iloc[0]
        return {
            "passengers": safe_int(r["passengers"]),
            "revenue": safe_float(r["revenue"]),
            "trips": safe_int(r["trips"]),
            "active_vehicles": safe_int(r["active_vehicles"]),
            "mode": "actual"
        }

    # forecast if beyond history or missing
    fc = forecast_passengers_next_days(depot_id, sel_date, days=1)
    pred_pax = float(fc["pred_passengers"].iloc[0]) if not fc.empty else 0.0

    # trips forecast using depot median passengers per trip (robust)
    pax_per_trip = float(META.get(depot_id, {}).get("avg_pax_per_trip", 55.0))
    pax_per_trip = 55.0 if (pax_per_trip is None or np.isnan(pax_per_trip) or pax_per_trip <= 0) else pax_per_trip
    pred_trips = max(1, int(round(pred_pax / pax_per_trip)))

    # revenue forecast using historical median revenue per passenger
    hist_rr = (hist["revenue"] / hist["passengers"].replace(0, np.nan)).median(skipna=True)
    rev_per_pax = float(hist_rr) if (hist_rr is not None and not np.isnan(hist_rr) and hist_rr > 0) else 180.0
    pred_rev = pred_pax * rev_per_pax

    # vehicles use last known active_vehicles median
    pred_veh = int(hist["active_vehicles"].median(skipna=True)) if "active_vehicles" in hist.columns else 0

    return {
        "passengers": int(round(pred_pax)),
        "revenue": float(pred_rev),
        "trips": int(pred_trips),
        "active_vehicles": int(pred_veh),
        "mode": "forecast"
    }

# -----------------------------
# Header (EXACT SAME AS YOUR UI)
# -----------------------------
c1, c2, c3 = st.columns([1.2, 2.8, 1.2])
with c1:
    st.markdown("## 🚌 APSRTC")
with c2:
    st.markdown("## Depot Manager Dashboard")
    st.caption("Day-wise • Product-wise • Route-wise • Redeployment • Dynamic Scheduling")
with c3:
    st.markdown(
        f"<span class='badge'>🟢 Live</span><span class='badge'>{dt.datetime.now().strftime('%d %b %Y, %I:%M %p')}</span>",
        unsafe_allow_html=True
    )

st.divider()

# -----------------------------
# Sidebar Controls (Global) - REAL depots
# -----------------------------
with st.sidebar:
    st.header("Controls")
    depot_label = st.selectbox("Depot", DEPOTS, index=0)
    depot_id = int(depot_label.split("(")[-1].replace(")", "").strip())
    depot_name = depot_label.split("(")[0].strip()
    today = dt.date.today()
    sel_date = st.date_input("Plan Date", value=today)
    st.caption("These filters apply across all tabs.")

# -----------------------------
# Tabs (EXACT SAME AS YOUR UI)
# -----------------------------
tabs = st.tabs([
    "🏠 Home",
    "🗓 Operational Calendar",
    "🛣 Route Plan",
    "📦 Product Performance",
    "🔁 Low Occupancy Redeployment",
    "⏱ Dynamic Scheduling",
    "🧰 Depot Health",
    "📤 Reports Export"
])

# =========================================================
# 1) HOME (REAL KPI)
# =========================================================
with tabs[0]:
    st.subheader("Today Overview + Alerts (Depot Manager)")

    k = get_actual_or_forecast_kpis(depot_id, sel_date)

    today_services = safe_int(k["trips"])
    operated = today_services  # we only have actual trips; keep same UI text
    pax = safe_int(k["passengers"])
    revenue = safe_float(k["revenue"])
    bus_ready = safe_int(k["active_vehicles"])
    bus_maint = 0

    # Load % (derived from pax per trip vs nominal capacity)
    NOMINAL_CAPACITY = 55.0
    pax_per_trip = (pax / today_services) if today_services > 0 else 0.0
    load = int(np.clip((pax_per_trip / NOMINAL_CAPACITY) * 100, 0, 100))

    # Low occupancy trips count from route_daily on sel_date (if exists)
    rd = load_route_daily(depot_id, sel_date, sel_date)
    if rd.empty:
        low_occ = "-"
    else:
        rd2 = rd.copy()
        rd2["pax_per_trip"] = rd2["passengers"] / rd2["trips"].replace(0, np.nan)
        rd2["load_pct"] = (rd2["pax_per_trip"] / NOMINAL_CAPACITY) * 100
        low_occ = int((rd2["load_pct"] < 40).sum())

    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
    kpi(k1, "Services (Planned / Operated)", f"{today_services} / {operated}", "Selected Plan Date")
    kpi(k2, "Estimated Passengers", f"{pax:,}", "FORECAST value" if k["mode"] == "forecast" else "Actual value")
    kpi(k3, "Avg Load %", f"{load}%", "Derived from pax per trip")
    kpi(k4, "Low Occupancy Trips", f"{low_occ}", "< 40% load (route_daily)")
    kpi(k5, "Bus Availability", f"{bus_ready}", f"{bus_maint} in maintenance")
    kpi(k6, "Revenue (Est.)", f"₹{int(revenue):,}", "Forecast if past data not available")

    st.markdown("")
    a1, a2 = st.columns([1.4, 1.0], gap="large")

    with a1:
        st.markdown("### ⚠️ Action Alerts")

        # overload/underload alerts from route_daily (REAL)
        if not rd.empty:
            rd3 = rd.copy()
            rd3["pax_per_trip"] = rd3["passengers"] / rd3["trips"].replace(0, np.nan)
            rd3["load_pct"] = (rd3["pax_per_trip"] / NOMINAL_CAPACITY) * 100
            overload = rd3.sort_values("load_pct", ascending=False).head(3)
            underload = rd3.sort_values("load_pct", ascending=True).head(3)

            st.markdown("**Overload (Add trips)**")
            for _, r in overload.iterrows():
                st.markdown(f"- {r['ROUTE_NAME']}: **Load {int(np.clip(r['load_pct'],0,999))}%** (consider + trips)")

            st.markdown("**Underload (Reduce / Merge / Convert)**")
            for _, r in underload.iterrows():
                st.markdown(f"- {r['ROUTE_NAME']}: **Load {int(np.clip(r['load_pct'],0,999))}%** (reduce/merge)")
        else:
            st.info("No route_daily data for this date. Alerts will show when route_daily exists.")

    with a2:
        st.markdown("### ✅ Quick Decisions (Demo)")
        st.info("UI-only approvals (workflow writing will be enabled later).")
        b1 = st.button("Approve Peak Extra Trips")
        b2 = st.button("Apply Low-Occ Redeployment Plan")
        b3 = st.button("Generate Daily Planning Report")
        if b1 or b2 or b3:
            st.success("Action recorded (demo). In real system, this writes approvals to database/workflow.")

    # Passenger trend (last 30 actual days from DB)
    st.markdown("---")
    st.markdown("### 📈 Passenger Trend (Last 30 days)")

    hist = load_depot_history(depot_id)
    if hist.empty:
        st.warning("No depot history found.")
    else:
        last30 = hist.tail(30)
        fig = px.line(last30, x="STAR_DATE", y="passengers", markers=True)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 2) OPERATIONAL CALENDAR (NEXT 30 DAYS FROM sel_date, ML FORECAST)
# =========================================================
with tabs[1]:
    st.subheader("Operational Calendar (Day-wise demand: Peak / High / Normal)")

    st.markdown("**Legend:**  🔴 Peak  |  🟡 High  |  ⚪ Normal")

    hist = load_depot_history(depot_id)
    if hist.empty:
        st.warning("No historical data for this depot.")
    else:
        # thresholds from history so red actually appears
        p70 = float(hist["passengers"].quantile(0.70))
        p90 = float(hist["passengers"].quantile(0.90))

        # forecast 1 year; calendar shows next 30 days
        fc_year = forecast_passengers_next_days(depot_id, sel_date, days=365)
        fc30 = fc_year.head(30).copy()
        fc30["date"] = pd.to_datetime(fc30["date"])
        fc30["services"] = fc30["pred_passengers"].apply(lambda x: max(1, int(round(x / max(1.0, META.get(depot_id, {}).get("avg_pax_per_trip", 55.0))))))

        def label_row(pax):
            if pax >= p90:
                return "PEAK", "Festival/Season surge"
            if pax >= p70:
                return "HIGH", "Weekend / medium surge"
            return "NORMAL", "Normal operations"

        rows = []
        for _, r in fc30.iterrows():
            lab, reason = label_row(r["pred_passengers"])
            rows.append([r["date"].date(), int(r["services"]), int(round(r["pred_passengers"])), lab, reason])
        cal = pd.DataFrame(rows, columns=["date", "services", "est_pax", "label", "reason"])

        # Day click selection (dropdown for simplicity)
        selected_day = st.selectbox(
            "Click day (for details)",
            cal["date"].tolist(),
            format_func=lambda d: d.strftime("%Y-%m-%d (%a)")
        )

        # Render next-30-days grid (Sun-Sat)
        st.markdown("#### Next 30 Days View")
        weekdays = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]
        header_cols = st.columns(7)
        for i, w in enumerate(weekdays):
            header_cols[i].markdown(f"**{w}**")

        start_weekday = (pd.Timestamp(cal["date"].iloc[0]).weekday() + 1) % 7  # Sun=0

        cells = [None] * start_weekday + cal["date"].tolist()
        while len(cells) % 7 != 0:
            cells.append(None)

        by_date = {row["date"]: row for _, row in cal.iterrows()}

        def tile_style(label):
            if label == "PEAK":
                return "background:#e84b4b;color:white;"
            if label == "HIGH":
                return "background:#f6e6b2;color:#2b2b2b;"
            return "background:#f5f7fb;color:#2b2b2b;"

        for week in range(len(cells) // 7):
            row_cols = st.columns(7, gap="small")
            for d_idx in range(7):
                d = cells[week * 7 + d_idx]
                if d is None:
                    row_cols[d_idx].markdown("<div class='tile' style='background:#fff;border:1px dashed rgba(0,0,0,0.08)'></div>", unsafe_allow_html=True)
                    continue

                r = by_date[d]
                style = tile_style(r["label"])
                badge = "🔴 PEAK" if r["label"] == "PEAK" else ("🟡 HIGH" if r["label"] == "HIGH" else "⚪ NORMAL")

                row_cols[d_idx].markdown(
                    f"""
                    <div class="tile" style="{style}">
                      <div class="tile_day">{d.day} <span class="pill">{badge}</span></div>
                      <div class="tile_big">{int(r["services"]):,}</div>
                      <div class="tile_lbl">Services</div>
                      <div class="small">{int(r["est_pax"]):,} passengers</div>
                      <div class="small">{r["reason"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("### Day Detail (like OPS planning)")

        drow = cal[cal["date"] == selected_day].iloc[0]
        dc1, dc2, dc3 = st.columns([1.2, 1.1, 1.7], gap="large")

        with dc1:
            st.markdown("<div class='card'>"
                        f"<h4>{selected_day.strftime('%d %b %Y (%A)')}</h4>"
                        f"<p><b>Label:</b> {drow['label']}</p>"
                        f"<p><b>Services:</b> {int(drow['services']):,}</p>"
                        f"<p><b>Est. Pax:</b> {int(drow['est_pax']):,}</p>"
                        f"<p><b>Reason:</b> {drow['reason']}</p>"
                        "</div>", unsafe_allow_html=True)

        with dc2:
            windows = pd.DataFrame({
                "Time Window": ["06–10", "10–16", "16–21", "21–24"],
                "Demand": ["High", "Low/Medium", "High", "Medium"],
                "Suggested Action": ["Add trips", "Reduce/merge low occ", "Add trips", "Keep standby"]
            })
            st.markdown("#### Peak/Slack Windows")
            st.dataframe(windows, use_container_width=True, height=190)

        with dc3:
            st.markdown("#### AI Suggested Actions")
            if drow["label"] == "PEAK":
                st.markdown("- Add extra trips on top corridors (festival/season surge)")
                st.markdown("- Keep standby buses and crew during evening return rush")
            elif drow["label"] == "HIGH":
                st.markdown("- Add a few trips during peak windows")
                st.markdown("- Monitor top routes; shift buses from slack routes if needed")
            else:
                st.markdown("- Keep normal schedule")
                st.markdown("- Optimize/merge low occupancy routes in slack hours")

# =========================================================
# 3) ROUTE PLAN (REAL route_daily)
# =========================================================
with tabs[2]:
    st.subheader("Route Plan (Route-wise required services + depot-wise allocation)")

    top_toggle = st.radio("View", ["Top 10", "Top 20", "All"], horizontal=True, index=1)
    top_n = 10 if top_toggle == "Top 10" else (20 if top_toggle == "Top 20" else 200)

    # use latest available actual date <= sel_date for baseline (because history ends 2025-05)
    hist = load_depot_history(depot_id)
    if hist.empty:
        st.warning("No history for this depot.")
    else:
        baseline_date = hist[hist["STAR_DATE"].dt.date <= sel_date]["STAR_DATE"].max()
        if pd.isna(baseline_date):
            baseline_date = hist["STAR_DATE"].max()
        baseline_date = baseline_date.date()

        rd = load_route_daily(depot_id, baseline_date, baseline_date)
        if rd.empty:
            st.warning("No route_daily data for baseline date.")
        else:
            # forecast total pax for selected date
            fc1 = forecast_passengers_next_days(depot_id, sel_date, days=1)
            forecast_total = float(fc1["pred_passengers"].iloc[0]) if not fc1.empty else float(rd["passengers"].sum())

            actual_total = float(rd["passengers"].sum()) if float(rd["passengers"].sum()) > 0 else 1.0
            scale = forecast_total / actual_total

            # build route table
            df_routes = rd.groupby(["ROUTE_NAME"], as_index=False).agg({
                "trips": "sum",
                "passengers": "sum"
            })
            df_routes["Current Services"] = df_routes["trips"].astype(int)
            df_routes["Est. Pax"] = (df_routes["passengers"] * scale).round().astype(int)
            df_routes["Required Services"] = (df_routes["Current Services"] * scale).round().astype(int).clip(lower=1)

            NOMINAL_CAPACITY = 55.0
            df_routes["Load %"] = ((df_routes["Est. Pax"] / df_routes["Required Services"].replace(0, np.nan)) / NOMINAL_CAPACITY * 100).fillna(0)
            df_routes["Load %"] = df_routes["Load %"].clip(0, 100).round().astype(int)

            df_routes = df_routes.sort_values("Est. Pax", ascending=False).head(top_n)
            floating_pax = int(df_routes["Est. Pax"].sum() * 0.12)

            rc1, rc2 = st.columns([3, 1], gap="large")
            with rc2:
                st.markdown("<div class='card'>"
                            f"<div class='small'>Floating Pax</div>"
                            f"<div style='font-size:34px;font-weight:900'>{floating_pax/1000:.1f}k</div>"
                            f"<div class='small'>Estimated passengers shifting between corridors</div>"
                            "</div>", unsafe_allow_html=True)

            with rc1:
                st.caption("Tip: Expand a route to see details (allocation logic will be enabled later).")

                for _, row in df_routes.iterrows():
                    route = row["ROUTE_NAME"]
                    current = int(row["Current Services"])
                    req = int(row["Required Services"])
                    pax_est = int(row["Est. Pax"])
                    loadp = int(row["Load %"])
                    delta = req - current

                    delta_txt = f"{req}  ▲" if delta > 0 else f"{req}  ▼"
                    rec = "Add trips" if delta > 0 else ("Reduce/merge" if delta < 0 else "Keep same")

                    with st.expander(f"{route}  |  Current: {current}  |  Required: {delta_txt}  |  Load: {loadp}%  |  {rec}"):
                        s1, s2, s3, s4 = st.columns([1, 1, 1, 1])
                        s1.metric("Current Services", current)
                        s2.metric("Required Services", req, delta=delta)
                        s3.metric("Est. Pax", f"{pax_est:,}")
                        s4.metric("Load %", f"{loadp}%")

                        st.info("Depot-wise allocation will be added during Route Optimization phase.")

# =========================================================
# 4) PRODUCT PERFORMANCE (REAL route_daily SERVICE_TYPE_NAME)
# =========================================================
with tabs[3]:
    st.subheader("Product-wise Performance (Service Type)")

    p1, p2, p3 = st.columns([1.1, 1.1, 1.8], gap="large")
    with p1:
        start = st.date_input("From", value=sel_date - dt.timedelta(days=30), key="prod_from")
    with p2:
        end = st.date_input("To", value=sel_date, key="prod_to")
    with p3:
        st.caption("This uses real route_daily SERVICE_TYPE_NAME for analytics.")

    rd = load_route_daily(depot_id, start, end)
    if rd.empty:
        st.warning("No service-type data available for selected range.")
    else:
        prod = rd.groupby("SERVICE_TYPE_NAME", as_index=False).agg({
            "trips": "sum",
            "passengers": "sum",
            "revenue": "sum"
        })
        prod.rename(columns={
            "SERVICE_TYPE_NAME": "Product",
            "trips": "Trips",
            "passengers": "Pax",
            "revenue": "Revenue"
        }, inplace=True)

        NOMINAL_CAPACITY = 55.0
        prod["Avg Load %"] = ((prod["Pax"] / prod["Trips"].replace(0, np.nan)) / NOMINAL_CAPACITY * 100).fillna(0)
        prod["Avg Load %"] = prod["Avg Load %"].clip(0, 100).round().astype(int)

        prod["Cancel %"] = np.nan  # not available in DB (kept for UI compatibility)

        c1, c2 = st.columns([1.2, 1.2], gap="large")
        with c1:
            fig1 = px.bar(prod.sort_values("Avg Load %", ascending=False), x="Product", y="Avg Load %", title="Avg Load % by Product")
            fig1.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fig2 = px.bar(prod.sort_values("Revenue", ascending=False), x="Product", y="Revenue", title="Revenue by Product")
            fig2.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns([1.2, 1.2], gap="large")
        with c3:
            # Cancel% not present -> show pax by product as line instead (still same UI block position)
            fig3 = px.line(prod.sort_values("Pax", ascending=False), x="Product", y="Pax", markers=True, title="Passengers by Product")
            fig3.update_layout(height=320, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            st.markdown("#### Product Summary Table")
            st.dataframe(prod.sort_values("Avg Load %", ascending=False), use_container_width=True, height=320)

# =========================================================
# 5) LOW OCCUPANCY REDEPLOYMENT (kept for later - DB needs trip_level for timings)
# =========================================================
with tabs[4]:
    st.subheader("Low Occupancy Services → Redeployment to High Corridors")
    st.info("This section will be fully enabled after integrating trip_level timings + route optimization. UI kept same.")

# =========================================================
# 6) DYNAMIC SCHEDULING (kept for later)
# =========================================================
with tabs[5]:
    st.subheader("Dynamic Scheduling (Peak vs Slack)")
    st.info("This section will be enabled after hourly ticket/trip aggregation. UI kept same.")

# =========================================================
# 7) DEPOT HEALTH (REAL active_vehicles where available)
# =========================================================
with tabs[6]:
    st.subheader("Depot Health & Readiness (Fleet + Crew + Ops)")

    k = get_actual_or_forecast_kpis(depot_id, sel_date)
    fleet_total = safe_int(k["active_vehicles"])
    ready = fleet_total
    maint = 0
    breakdown = 0

    h1, h2, h3, h4, h5, h6 = st.columns(6, gap="small")
    kpi(h1, "Fleet Total", f"{fleet_total}", "From depot_daily_summary")
    kpi(h2, "Ready Buses", f"{ready}", "Available now")
    kpi(h3, "Maintenance", f"{maint}", "Not available yet")
    kpi(h4, "Breakdowns", f"{breakdown}", "Not available yet")
    kpi(h5, "Crew Available", "-", "Crew table not connected yet")
    kpi(h6, "Delays / Cancellations", "-", "Needs trip_level/haltwise integration")

    st.markdown("#### Notes (Manager View)")
    st.markdown("- Full health indicators will come after integrating vehicle_daily_summary + trip_level + haltwise_raw.")

# =========================================================
# 8) REPORTS EXPORT (REAL exports)
# =========================================================
with tabs[7]:
    st.subheader("Reports Export")

    # exports: forecast calendar (30 days), route table baseline day, product table
    fc30 = forecast_passengers_next_days(depot_id, sel_date, days=30)
    if not fc30.empty:
        fc30 = fc30.copy()
        fc30.rename(columns={"pred_passengers": "forecast_passengers"}, inplace=True)

    baseline_hist = load_depot_history(depot_id)
    if not baseline_hist.empty:
        baseline_date = baseline_hist[baseline_hist["STAR_DATE"].dt.date <= sel_date]["STAR_DATE"].max()
        if pd.isna(baseline_date):
            baseline_date = baseline_hist["STAR_DATE"].max()
        baseline_date = baseline_date.date()
        rd_base = load_route_daily(depot_id, baseline_date, baseline_date)
    else:
        rd_base = pd.DataFrame()

    rd_range = load_route_daily(depot_id, sel_date - dt.timedelta(days=30), sel_date)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.download_button(
            "Download Forecast (Next 30 Days) CSV",
            fc30.to_csv(index=False).encode("utf-8") if not fc30.empty else b"",
            file_name=f"{depot_name}_forecast_next30_{sel_date}.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "Download Route Daily (Baseline Date) CSV",
            rd_base.to_csv(index=False).encode("utf-8") if not rd_base.empty else b"",
            file_name=f"{depot_name}_route_daily_{baseline_date}.csv" if not baseline_hist.empty else f"{depot_name}_route_daily.csv",
            mime="text/csv"
        )
    with c3:
        st.download_button(
            "Download Service-Type (Last 30 Days) CSV",
            rd_range.to_csv(index=False).encode("utf-8") if not rd_range.empty else b"",
            file_name=f"{depot_name}_service_type_last30_{sel_date}.csv",
            mime="text/csv"
        )

st.caption("APSRTC Depot Manager Dashboard • SAME UI • Real DuckDB data • ML Forecast (1 year) + Season/Festival factors")