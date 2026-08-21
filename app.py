import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import consolidate, load_master_csv, STANDARD_COLS
from forecasting import time_ordered_backtest, final_forecast, all_metrics

st.set_page_config(page_title="Life Insurance NBP — Sales Analysis & Forecasting",
                    layout="wide", page_icon="📈")

# ---------------- Sidebar: Data source ----------------
st.sidebar.title("📊 Data Source")
data_mode = st.sidebar.radio(
    "Choose data source",
    ["Use demo data (synthetic, realistic)", "Upload my own files"],
    index=0,
)

@st.cache_data
def get_demo_data():
    return load_master_csv("synthetic_nbp_master.csv")

master_df = pd.DataFrame(columns=STANDARD_COLS)

if data_mode == "Use demo data (synthetic, realistic)":
    master_df = get_demo_data()
    st.sidebar.success(f"Loaded demo dataset: {master_df['Month'].nunique()} months, "
                        f"{master_df['Insurer'].nunique()} insurers.")
else:
    uploaded = st.sidebar.file_uploader(
        "Upload monthly council files (.xls/.xlsx) or a consolidated .csv",
        type=["xls", "xlsx", "csv"], accept_multiple_files=True,
    )
    if uploaded:
        files = {f.name: f.read() for f in uploaded}
        master_df = consolidate(files)
        if master_df.empty:
            st.sidebar.error("Couldn't parse any rows from the uploaded files. "
                              "Check the file layout, or use demo data to test the app first.")
        else:
            st.sidebar.success(f"Parsed {len(uploaded)} file(s) → "
                                f"{master_df['Month'].nunique()} months, "
                                f"{master_df['Insurer'].nunique()} insurers.")
    else:
        st.sidebar.info("Upload files to begin, or switch to demo data.")

st.title("📈 Life Insurance New Business — Sales Analysis & Forecasting")
st.caption("Analyze historical new-business premium & policy trends, and forecast future sales "
           "performance across insurers and time.")

if master_df.empty:
    st.warning("No data loaded yet. Select **demo data** in the sidebar to explore the app now, "
               "or upload your own monthly files.")
    st.stop()

# ---------------- Prep ----------------
master_df["MonthDate"] = pd.to_datetime(master_df["Month"], format="%Y-%m")
industry = (master_df.groupby("MonthDate", as_index=False)
            .agg(Premium_Rs_Crore=("Premium_Rs_Crore", "sum"),
                 No_of_Policies=("No_of_Policies", "sum")))
industry = industry.sort_values("MonthDate")
industry = industry.set_index("MonthDate").asfreq("MS").reset_index()
industry[["Premium_Rs_Crore", "No_of_Policies"]] = industry[["Premium_Rs_Crore", "No_of_Policies"]].interpolate()

tabs = st.tabs(["🔎 Overview & EDA", "🧪 Backtesting", "🔮 Forecast", "💡 Insights",
                 "📄 Assumptions & Limitations"])

# ================= TAB 1: EDA =================
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    latest = industry.iloc[-1]
    prev_year = industry[industry["MonthDate"] == latest["MonthDate"] - pd.DateOffset(years=1)]
    yoy = ((latest["Premium_Rs_Crore"] / prev_year["Premium_Rs_Crore"].values[0]) - 1) * 100 \
        if len(prev_year) else np.nan
    c1.metric("Latest Month", latest["MonthDate"].strftime("%b %Y"))
    c2.metric("Industry NBP (₹ crore)", f"{latest['Premium_Rs_Crore']:,.0f}",
              f"{yoy:+.1f}% YoY" if not np.isnan(yoy) else None)
    c3.metric("Policies Issued", f"{latest['No_of_Policies']:,.0f}")
    c4.metric("Active Insurers", master_df["Insurer"].nunique())

    st.subheader("Industry Trend & Seasonality")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=industry["MonthDate"], y=industry["Premium_Rs_Crore"],
                              mode="lines+markers", name="Industry NBP (₹ crore)"))
    fig.update_layout(height=400, xaxis_title="Month", yaxis_title="Premium (₹ crore)")
    st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Seasonality (avg by calendar month)")
        seasonal = industry.copy()
        seasonal["MonthNum"] = seasonal["MonthDate"].dt.month
        seasonal_avg = seasonal.groupby("MonthNum")["Premium_Rs_Crore"].mean().reset_index()
        seasonal_avg["MonthName"] = pd.to_datetime(seasonal_avg["MonthNum"], format="%m").dt.strftime("%b")
        fig2 = px.bar(seasonal_avg, x="MonthName", y="Premium_Rs_Crore")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
    with colB:
        st.subheader("Insurer Market Share (latest month)")
        latest_month = master_df["MonthDate"].max()
        share = master_df[master_df["MonthDate"] == latest_month]
        fig3 = px.pie(share, names="Insurer", values="Premium_Rs_Crore", hole=0.4)
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Insurer Contribution Over Time")
    top_insurers = (master_df.groupby("Insurer")["Premium_Rs_Crore"].sum()
                     .sort_values(ascending=False).head(8).index.tolist())
    trend_df = master_df[master_df["Insurer"].isin(top_insurers)]
    fig4 = px.area(trend_df, x="MonthDate", y="Premium_Rs_Crore", color="Insurer")
    fig4.update_layout(height=420)
    st.plotly_chart(fig4, use_container_width=True)

# ================= TAB 2: Backtesting =================
with tabs[1]:
    st.subheader("Time-Ordered Backtest — Candidate Model Comparison")
    st.caption("Rolling-origin backtest: each fold trains only on past months and forecasts "
               "forward, exactly like a real deployment — no shuffling, no leakage.")

    target_metric = st.selectbox("Backtest target", ["Premium_Rs_Crore", "No_of_Policies"], key="bt_target")
    horizon = st.slider("Forecast horizon per fold (months)", 1, 6, 3)
    min_train = st.slider("Minimum training window (months)", 12, max(12, len(industry) - horizon - 1), 18)

    series = industry.set_index("MonthDate")[target_metric]
    with st.spinner("Running rolling-origin backtest across all candidate models..."):
        bt_results = time_ordered_backtest(series, horizon=horizon, min_train=min_train)

    if bt_results.empty:
        st.warning("Not enough history for this horizon/training-window combination. Reduce the horizon or minimum training window.")
    else:
        st.dataframe(bt_results.style.highlight_min(subset=["MAE", "RMSE", "WAPE_%"], color="#d4f4dd")
                     .highlight_min(subset=["Bias_%"], color="#f4f4d4"),
                     use_container_width=True)
        best_model = bt_results.iloc[0]["Model"]
        st.success(f"✅ Best performing model by WAPE: **{best_model}**")
        st.caption("MAE / RMSE / WAPE — lower is better. Bias — closer to 0% is better "
                   "(positive = over-forecasting, negative = under-forecasting).")

# ================= TAB 3: Forecast =================
with tabs[2]:
    st.subheader("Forward-Looking Forecast")
    fc_target = st.selectbox("Forecast target", ["Premium_Rs_Crore", "No_of_Policies"], key="fc_target")
    fc_horizon = st.slider("Forecast horizon (months ahead)", 1, 12, 6)

    series = industry.set_index("MonthDate")[fc_target]
    with st.spinner("Fitting SARIMA and Holt-Winters models..."):
        fc = final_forecast(series, horizon=fc_horizon)

    hist_df = industry[["MonthDate", fc_target]].rename(columns={fc_target: "Actual"})
    hist_df["Type"] = "Actual"

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=hist_df["MonthDate"], y=hist_df["Actual"],
                               mode="lines", name="Actual", line=dict(color="#1f77b4")))
    future_dates = pd.to_datetime(fc["Month"], format="%Y-%m")
    fig5.add_trace(go.Scatter(x=future_dates, y=fc["Forecast_SARIMA"],
                               mode="lines+markers", name="Forecast (SARIMA)",
                               line=dict(color="#d62728", dash="dash")))
    fig5.add_trace(go.Scatter(x=future_dates, y=fc["Forecast_HoltWinters"],
                               mode="lines", name="Forecast (Holt-Winters)",
                               line=dict(color="#2ca02c", dash="dot")))
    fig5.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(fc["Upper_80"]) + list(fc["Lower_80"][::-1]),
        fill="toself", fillcolor="rgba(214,39,40,0.15)", line=dict(width=0),
        name="80% Confidence Interval", showlegend=True))
    fig5.update_layout(height=450, xaxis_title="Month", yaxis_title=fc_target)
    st.plotly_chart(fig5, use_container_width=True)

    st.dataframe(fc.style.format({
        "Forecast_SARIMA": "{:,.0f}", "Lower_80": "{:,.0f}", "Upper_80": "{:,.0f}",
        "Forecast_HoltWinters": "{:,.0f}"}), use_container_width=True)

    csv = fc.to_csv(index=False).encode()
    st.download_button("⬇️ Download forecast as CSV", csv, "nbp_forecast.csv", "text/csv")

# ================= TAB 4: Insights =================
with tabs[3]:
    st.subheader("Auto-Generated Business Insights")
    last12 = industry.tail(12)
    prev12 = industry.iloc[-24:-12] if len(industry) >= 24 else pd.DataFrame()
    yoy_growth = ((last12["Premium_Rs_Crore"].sum() / prev12["Premium_Rs_Crore"].sum()) - 1) * 100 \
        if len(prev12) else None

    top_insurer_latest = (master_df[master_df["MonthDate"] == master_df["MonthDate"].max()]
                           .sort_values("Premium_Rs_Crore", ascending=False).iloc[0])
    peak_month = (industry.assign(MonthNum=industry["MonthDate"].dt.month)
                  .groupby("MonthNum")["Premium_Rs_Crore"].mean().idxmax())
    peak_month_name = pd.to_datetime(str(peak_month), format="%m").strftime("%B")

    growing = (master_df.groupby("Insurer").apply(
        lambda d: d.sort_values("MonthDate")["Premium_Rs_Crore"].pct_change().mean() * 100
    ).sort_values(ascending=False))

    bullets = []
    if yoy_growth is not None:
        direction = "grew" if yoy_growth >= 0 else "declined"
        bullets.append(f"Trailing 12-month industry NBP **{direction} {abs(yoy_growth):.1f}%** YoY.")
    bullets.append(f"**{top_insurer_latest['Insurer']}** leads market share in the latest month "
                    f"with ₹{top_insurer_latest['Premium_Rs_Crore']:,.0f} crore.")
    bullets.append(f"**{peak_month_name}** is consistently the strongest month for new business "
                    f"— plan capacity and campaigns around this peak.")
    if len(growing) > 0:
        bullets.append(f"**{growing.index[0]}** shows the strongest average month-on-month growth "
                        f"momentum among all insurers.")

    for b in bullets:
        st.markdown(f"- {b}")

    st.subheader("Growth Leaderboard (avg MoM % change)")
    st.dataframe(growing.reset_index().rename(
        columns={0: "Avg_MoM_Growth_%", "Premium_Rs_Crore": "Avg_MoM_Growth_%"}
    ).round(2), use_container_width=True)

# ================= TAB 5: Assumptions =================
with tabs[4]:
    st.subheader("Assumptions & Limitations")
    st.markdown("""
**Data**
- Demo mode uses a **synthetic dataset** shaped like the real Life Insurance Council monthly
  NBP files (insurer × month × premium × policy count), with realistic trend, March fiscal-year-end
  seasonality, and random shocks — for pipeline testing and demoing before real data is loaded.
- Real data should be downloaded month-by-month from lifeinscouncil.org and uploaded via the
  sidebar; the parser auto-detects HTML-as-.xls council exports, genuine Excel files, and CSVs.
- Subtotal/"Total"/"Industry"/"Private Total" rows are automatically dropped during cleaning to
  avoid double-counting against individual insurer rows.

**Modeling**
- Backtesting uses a **rolling-origin, time-ordered split** — never trains on future data.
- Percentage-based metrics (WAPE, Bias) are used instead of plain MAPE because individual-insurer
  monthly values can be zero or very small, which makes MAPE unstable.
- SARIMA and Holt-Winters both assume the recent historical pattern (trend + seasonality)
  broadly continues; they do not model regulatory changes, new product launches, or macro shocks.
- With fewer than ~24 months of history, seasonal components are disabled automatically and the
  models fall back to simpler trend-only estimates — accuracy will be lower until more history
  is available.

**Forecast use**
- Forecasts are directional planning inputs, not guarantees — always read alongside the
  confidence interval, not just the point forecast.
""")
