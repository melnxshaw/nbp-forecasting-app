import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import consolidate, load_master_csv, ENRICHED_COLS, totals_series, category_mix
from forecasting import time_ordered_backtest, final_forecast

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

master_df = pd.DataFrame(columns=ENRICHED_COLS)

if data_mode == "Use demo data (synthetic, realistic)":
    master_df = get_demo_data()
    st.sidebar.success(f"Loaded demo dataset: {master_df['Month'].nunique()} months, "
                        f"{master_df['Insurer'].nunique()} insurers.")
else:
    uploaded = st.sidebar.file_uploader(
        "Upload monthly council files (.xls — real HTML export) or CSVs "
        "(pre-cleaned or already-consolidated)",
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
st.caption("Analyze historical new-business premium & policy trends across insurers and "
           "premium categories, and forecast future sales performance.")

if master_df.empty:
    st.warning("No data loaded yet. Select **demo data** in the sidebar to explore the app now, "
               "or upload your own monthly files.")
    st.stop()

# ---------------- Prep ----------------
totals_df = totals_series(master_df)
mix_df = category_mix(master_df)

if totals_df.empty:
    st.error("Data loaded, but no 'Total' rows were found per insurer/month — check the "
             "uploaded file layout in the Assumptions tab for the expected format.")
    st.stop()

totals_df["MonthDate"] = pd.to_datetime(totals_df["Month"], format="%Y-%m")
mix_df["MonthDate"] = pd.to_datetime(mix_df["Month"], format="%Y-%m")

industry = (totals_df.groupby("MonthDate", as_index=False)
            .agg(Premium_Rs_Crore=("Premium_Rs_Crore", "sum"),
                 No_of_Policies=("No_of_Policies", "sum")))
industry = industry.sort_values("MonthDate")
industry = industry.set_index("MonthDate").asfreq("MS").reset_index()
industry[["Premium_Rs_Crore", "No_of_Policies"]] = industry[["Premium_Rs_Crore", "No_of_Policies"]].interpolate()

n_months = industry["MonthDate"].nunique()

tabs = st.tabs(["🔎 Overview & EDA", "🧩 Premium Category Mix", "🧪 Backtesting", "🔮 Forecast",
                 "💡 Insights", "📄 Assumptions & Limitations"])

# ================= TAB 1: EDA =================
with tabs[0]:
    if n_months < 2:
        st.info(f"Only **{n_months} month** of data loaded so far. Trend/seasonality charts "
                "need multiple months — upload more monthly files to unlock them. Showing "
                "what's available below.")

    c1, c2, c3, c4 = st.columns(4)
    latest = industry.iloc[-1]
    prev_year_row = industry[industry["MonthDate"] == latest["MonthDate"] - pd.DateOffset(years=1)]
    yoy = ((latest["Premium_Rs_Crore"] / prev_year_row["Premium_Rs_Crore"].values[0]) - 1) * 100 \
        if len(prev_year_row) else np.nan
    c1.metric("Latest Month", latest["MonthDate"].strftime("%b %Y"))
    c2.metric("Industry NBP (₹ crore)", f"{latest['Premium_Rs_Crore']:,.0f}",
              f"{yoy:+.1f}% YoY" if not np.isnan(yoy) else None)
    c3.metric("Policies Issued", f"{latest['No_of_Policies']:,.0f}")
    c4.metric("Active Insurers", totals_df["Insurer"].nunique())

    st.subheader("Industry Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=industry["MonthDate"], y=industry["Premium_Rs_Crore"],
                              mode="lines+markers", name="Industry NBP (₹ crore)"))
    fig.update_layout(height=400, xaxis_title="Month", yaxis_title="Premium (₹ crore)")
    st.plotly_chart(fig, use_container_width=True)

    if n_months >= 12:
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
            latest_month = totals_df["MonthDate"].max()
            share = totals_df[totals_df["MonthDate"] == latest_month]
            fig3 = px.pie(share, names="Insurer", values="Premium_Rs_Crore", hole=0.4)
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.subheader("Insurer Market Share (latest month)")
        latest_month = totals_df["MonthDate"].max()
        share = totals_df[totals_df["MonthDate"] == latest_month]
        fig3 = px.pie(share, names="Insurer", values="Premium_Rs_Crore", hole=0.4)
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Insurer Contribution Over Time")
    top_insurers = (totals_df.groupby("Insurer")["Premium_Rs_Crore"].sum()
                     .sort_values(ascending=False).head(8).index.tolist())
    trend_df = totals_df[totals_df["Insurer"].isin(top_insurers)]
    fig4 = px.area(trend_df, x="MonthDate", y="Premium_Rs_Crore", color="Insurer")
    fig4.update_layout(height=420)
    st.plotly_chart(fig4, use_container_width=True)

# ================= TAB 2: Premium Category Mix =================
with tabs[1]:
    st.subheader("Premium Mix by Business Category")
    st.caption("Individual vs Group, Single vs Non-Single premium — the category breakdown "
               "your brief specifically called out.")

    cat_totals = (mix_df.groupby(["MonthDate", "BusinessType"], as_index=False)
                  .agg(Premium_Rs_Crore=("Premium_Current_Month", "sum")))

    fig_mix = px.area(cat_totals, x="MonthDate", y="Premium_Rs_Crore", color="BusinessType")
    fig_mix.update_layout(height=450, xaxis_title="Month", yaxis_title="Premium (₹ crore)")
    st.plotly_chart(fig_mix, use_container_width=True)

    latest_month_mix = mix_df["MonthDate"].max()
    latest_mix = (mix_df[mix_df["MonthDate"] == latest_month_mix]
                  .groupby("BusinessType", as_index=False)
                  .agg(Premium_Rs_Crore=("Premium_Current_Month", "sum")))
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(latest_mix, names="BusinessType", values="Premium_Rs_Crore",
                          title=f"Category Split — {latest_month_mix.strftime('%b %Y')}", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("Insurer × Category (latest month)")
        pivot = (mix_df[mix_df["MonthDate"] == latest_month_mix]
                 .pivot_table(index="Insurer", columns="BusinessType",
                              values="Premium_Current_Month", aggfunc="sum", fill_value=0))
        st.dataframe(pivot.style.format("{:,.1f}"), use_container_width=True, height=350)

# ================= TAB 3: Backtesting =================
with tabs[2]:
    st.subheader("Time-Ordered Backtest — Candidate Model Comparison")
    st.caption("Rolling-origin backtest: each fold trains only on past months and forecasts "
               "forward — no shuffling, no leakage.")

    if n_months < 15:
        st.warning(f"Only **{n_months} months** loaded. Backtesting needs a reasonable training "
                   "history + horizon (ideally 18+ months) to produce meaningful folds. "
                   "Results below may be limited or unavailable until more months are uploaded.")

    target_metric = st.selectbox("Backtest target", ["Premium_Rs_Crore", "No_of_Policies"], key="bt_target")
    horizon = st.slider("Forecast horizon per fold (months)", 1, 6, min(3, max(1, n_months // 4)))
    max_min_train = max(3, n_months - horizon - 1)
    min_train = st.slider("Minimum training window (months)", 3, max_min_train, min(6, max_min_train))

    series = industry.set_index("MonthDate")[target_metric]
    with st.spinner("Running rolling-origin backtest across all candidate models..."):
        bt_results = time_ordered_backtest(series, horizon=horizon, min_train=min_train)

    if bt_results.empty:
        st.warning("Not enough history for this horizon/training-window combination. "
                   "Reduce the horizon or minimum training window, or upload more months.")
    else:
        st.dataframe(bt_results.style.highlight_min(subset=["MAE", "RMSE", "WAPE_%"], color="#d4f4dd")
                     .highlight_min(subset=["Bias_%"], color="#f4f4d4"),
                     use_container_width=True)
        best_model = bt_results.iloc[0]["Model"]
        st.success(f"✅ Best performing model by WAPE: **{best_model}**")
        st.caption("MAE / RMSE / WAPE — lower is better. Bias — closer to 0% is better "
                   "(positive = over-forecasting, negative = under-forecasting).")

# ================= TAB 4: Forecast =================
with tabs[3]:
    st.subheader("Forward-Looking Forecast")
    if n_months < 6:
        st.warning(f"Only **{n_months} months** loaded — forecasts with this little history "
                   "will be low-confidence trend projections rather than reliable seasonal "
                   "forecasts. Upload more months for a stronger forecast.")

    fc_target = st.selectbox("Forecast target", ["Premium_Rs_Crore", "No_of_Policies"], key="fc_target")
    max_horizon = 12 if n_months >= 12 else max(1, n_months)
    fc_horizon = st.slider("Forecast horizon (months ahead)", 1, max_horizon, min(3, max_horizon))

    series = industry.set_index("MonthDate")[fc_target]
    with st.spinner("Fitting SARIMA and Holt-Winters models..."):
        fc = final_forecast(series, horizon=fc_horizon)

    hist_df = industry[["MonthDate", fc_target]].rename(columns={fc_target: "Actual"})

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

# ================= TAB 5: Insights =================
with tabs[4]:
    st.subheader("Auto-Generated Business Insights")
    bullets = []

    if n_months >= 24:
        last12 = industry.tail(12)
        prev12 = industry.iloc[-24:-12]
        yoy_growth = ((last12["Premium_Rs_Crore"].sum() / prev12["Premium_Rs_Crore"].sum()) - 1) * 100
        direction = "grew" if yoy_growth >= 0 else "declined"
        bullets.append(f"Trailing 12-month industry NBP **{direction} {abs(yoy_growth):.1f}%** YoY.")

    top_insurer_latest = (totals_df[totals_df["MonthDate"] == totals_df["MonthDate"].max()]
                           .sort_values("Premium_Rs_Crore", ascending=False).iloc[0])
    bullets.append(f"**{top_insurer_latest['Insurer']}** leads market share in the latest month "
                    f"with ₹{top_insurer_latest['Premium_Rs_Crore']:,.0f} crore.")

    if n_months >= 12:
        peak_month = (industry.assign(MonthNum=industry["MonthDate"].dt.month)
                      .groupby("MonthNum")["Premium_Rs_Crore"].mean().idxmax())
        peak_month_name = pd.to_datetime(str(peak_month), format="%m").strftime("%B")
        bullets.append(f"**{peak_month_name}** is consistently the strongest month for new "
                        f"business — plan capacity and campaigns around this peak.")

    if totals_df["Insurer"].nunique() > 1 and totals_df["MonthDate"].nunique() > 1:
        growing = (totals_df.groupby("Insurer").apply(
            lambda d: d.sort_values("MonthDate")["Premium_Rs_Crore"].pct_change().mean() * 100
        ).dropna().sort_values(ascending=False))
        if len(growing) > 0:
            bullets.append(f"**{growing.index[0]}** shows the strongest average month-on-month "
                            f"growth momentum among all insurers.")

    latest_month_for_mix = mix_df["MonthDate"].max()
    top_category = (mix_df[mix_df["MonthDate"] == latest_month_for_mix]
                     .groupby("BusinessType")["Premium_Current_Month"].sum().idxmax())
    bullets.append(f"**{top_category}** is the largest premium category in the latest month — "
                    "see the Premium Category Mix tab for the full breakdown.")

    for b in bullets:
        st.markdown(f"- {b}")

    if totals_df["Insurer"].nunique() > 1 and totals_df["MonthDate"].nunique() > 1:
        st.subheader("Growth Leaderboard (avg MoM % change)")
        st.dataframe(growing.reset_index().rename(columns={0: "Avg_MoM_Growth_%"}).round(2),
                     use_container_width=True)

# ================= TAB 6: Assumptions =================
with tabs[5]:
    st.subheader("Assumptions & Limitations")
    st.markdown("""
**Data**
- Real council files download as `.xls` but are actually **HTML tables** — the parser detects
  this automatically (no manual conversion needed).
- The "Detailed New Business Performance" report gives, per insurer: 5 premium categories
  (Individual Single/Non-Single, Group Single/Non-Single, Group Yearly Renewable) plus a Total
  row, each with current-month, YTD, same-month-last-year, YTD-last-year, and YTD variation %.
- The dashboard's core time series uses the **Total** row per insurer per month; the Premium
  Category Mix tab uses the 5 sub-category rows.
- Pre-cleaned CSVs (columns: insurer, business_type, premium_current_month, ..., month) are
  also accepted directly — the year is inferred from the filename (e.g. `nbp_2025_master.csv`
  → year 2025), since the source month column has no year attached.
- Demo mode uses a **synthetic dataset** (Total-level only, no category breakdown) for testing
  the pipeline before real data is available.

**Modeling**
- Backtesting uses a **rolling-origin, time-ordered split** — never trains on future data.
- With fewer than ~24 months of history, seasonal components are disabled automatically and
  models fall back to simpler trend-only estimates — accuracy improves as more real months
  are uploaded.
- WAPE/Bias are used instead of plain MAPE because individual insurer-month values can be
  zero or very small, which makes MAPE unstable.

**Forecast use**
- Forecasts are directional planning inputs, not guarantees — always read alongside the
  confidence interval, not just the point forecast.
""")
