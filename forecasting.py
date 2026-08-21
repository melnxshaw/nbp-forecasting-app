"""
Forecasting models + time-ordered backtesting for monthly NBP series.

Models:
  - Naive (last value carried forward)
  - Seasonal Naive (same month last year)
  - Moving Average (trailing window)
  - Holt-Winters / ETS (trend + additive seasonality)
  - SARIMA (seasonal ARIMA)

Metrics (matching the hackathon brief): MAE, RMSE, WAPE, Bias
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------- Metrics ----------

def mae(actual, pred):
    return float(np.mean(np.abs(np.array(actual) - np.array(pred))))

def rmse(actual, pred):
    return float(np.sqrt(np.mean((np.array(actual) - np.array(pred)) ** 2)))

def wape(actual, pred):
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual - pred)) / denom * 100)

def bias(actual, pred):
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return float(np.sum(pred - actual) / denom * 100)  # % over(+)/under(-) forecast

def all_metrics(actual, pred):
    return {"MAE": mae(actual, pred), "RMSE": rmse(actual, pred),
            "WAPE_%": wape(actual, pred), "Bias_%": bias(actual, pred)}


# ---------- Models ----------
# Each model fn(train: pd.Series, horizon: int) -> np.array of length horizon

def model_naive(train, horizon):
    return np.repeat(train.iloc[-1], horizon)

def model_seasonal_naive(train, horizon, season=12):
    if len(train) < season:
        return model_naive(train, horizon)
    vals = train.values
    return np.array([vals[-season + (i % season)] for i in range(horizon)])

def model_moving_average(train, horizon, window=3):
    window = min(window, len(train))
    avg = train.iloc[-window:].mean()
    return np.repeat(avg, horizon)

def model_holt_winters(train, horizon, season=12):
    try:
        seasonal = "add" if len(train) >= 2 * season else None
        model = ExponentialSmoothing(
            train, trend="add",
            seasonal=seasonal, seasonal_periods=season if seasonal else None,
            initialization_method="estimated"
        ).fit(optimized=True)
        fc = model.forecast(horizon)
        return np.clip(fc.values, 0, None)
    except Exception:
        return model_seasonal_naive(train, horizon, season)

def model_sarima(train, horizon, season=12, order=(1, 1, 1), seasonal_order=None):
    try:
        so = seasonal_order or (1, 1, 1, season if len(train) >= 2 * season else 0)
        model = SARIMAX(train, order=order, seasonal_order=so,
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        fc = fit.get_forecast(horizon)
        mean = fc.predicted_mean.values
        ci = fc.conf_int(alpha=0.2)  # 80% CI
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
        return np.clip(mean, 0, None), np.clip(lower, 0, None), np.clip(upper, 0, None)
    except Exception:
        fc = model_holt_winters(train, horizon, season)
        return fc, fc * 0.85, fc * 1.15


MODEL_REGISTRY = {
    "Naive": model_naive,
    "Seasonal Naive": model_seasonal_naive,
    "Moving Average (3mo)": model_moving_average,
    "Holt-Winters (ETS)": model_holt_winters,
}


# ---------- Backtesting ----------

def time_ordered_backtest(series: pd.Series, horizon=3, min_train=18, step=1):
    """
    Rolling-origin backtest. Returns a DataFrame of metrics per model,
    averaged across all backtest folds (time-ordered, no shuffling/leakage).
    """
    results = {name: {"MAE": [], "RMSE": [], "WAPE_%": [], "Bias_%": []}
               for name in MODEL_REGISTRY}

    n = len(series)
    fold_starts = list(range(min_train, n - horizon + 1, step))
    if not fold_starts:
        return pd.DataFrame()

    for start in fold_starts:
        train = series.iloc[:start]
        test = series.iloc[start:start + horizon]
        if len(test) < horizon:
            continue
        for name, fn in MODEL_REGISTRY.items():
            try:
                pred = fn(train, horizon)
                m = all_metrics(test.values, pred)
                for k, v in m.items():
                    if not np.isnan(v):
                        results[name][k].append(v)
            except Exception:
                continue

    rows = []
    for name, metrics in results.items():
        row = {"Model": name}
        for k, vals in metrics.items():
            row[k] = round(np.mean(vals), 2) if vals else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("WAPE_%")


def final_forecast(series: pd.Series, horizon=6, season=12):
    """
    Produces the final forward-looking forecast using SARIMA (with CI),
    plus a Holt-Winters comparison line.
    """
    sarima_mean, sarima_lo, sarima_hi = model_sarima(series, horizon, season)
    hw = model_holt_winters(series, horizon, season)

    future_idx = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    out = pd.DataFrame({
        "Month": future_idx.strftime("%Y-%m"),
        "Forecast_SARIMA": sarima_mean,
        "Lower_80": sarima_lo,
        "Upper_80": sarima_hi,
        "Forecast_HoltWinters": hw,
    })
    return out
