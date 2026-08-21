"""
Data ingestion & cleaning pipeline for REAL Life Insurance Council NBP data.

The real monthly file (downloaded as .xls, actually HTML) contains a
"Detailed New Business Performance" table shaped like this, per insurer:

    1   ACKO LIFE INSURANCE COMPANY LIMITED        <- insurer header row (no values)
        Individual Single Premium        0.00  0.00  0.00  0.00  0.00%   0  0  0  0  0.00%
        Individual Non Single Premium     0.35  2.73  0.04  0.04  6237%   200 1494 11 13 ...
        Group Single Premium              ...
        Group Non Single Premium          ...
        Group Yearly Renewable Premium    ...
        Total                             4.36  48.63 4.37  21.72 123.93% 200 1502 12 20 ...
        (blank separator row)
    2   ADITYA BIRLA SUN LIFE INSURANCE...

Columns per row (fixed position): Sr.no | Particulars |
  Premium: For-the-month(curr yr) | Upto-the-month/YTD(curr yr) |
           For-the-month(prev yr) | Upto-the-month/YTD(prev yr) | YTD Variation %
  Policies: same 5-column pattern

This gives premium-CATEGORY-level granularity (Individual Single/Non-Single,
Group Single/Non-Single, Group Yearly Renewable, Total) — richer than a
simple insurer-total series, and matches the brief's "premium categories"
requirement directly.

This module also accepts an already-cleaned CSV using simplified column
names (as produced by manual pre-processing), and a plain aggregated CSV
(Month, Insurer, Premium_Rs_Crore, No_of_Policies) for backward compatibility
with demo/synthetic data.
"""
import io
import re
import numpy as np
import pandas as pd

# ---------------- Schemas ----------------

ENRICHED_COLS = [
    "Year", "Month", "MonthName", "Insurer", "BusinessType",
    "Premium_Current_Month", "Premium_Current_YTD",
    "Premium_Previous_Month", "Premium_Previous_YTD", "Premium_YTD_Variation_Pct",
    "Policies_Current_Month", "Policies_Current_YTD",
    "Policies_Previous_Month", "Policies_Previous_YTD", "Policies_YTD_Variation_Pct",
]

SIMPLE_COLS = ["Month", "Insurer", "Premium_Rs_Crore", "No_of_Policies"]  # legacy/demo format

KNOWN_BUSINESS_TYPES = {
    "individual single premium", "individual non single premium",
    "group single premium", "group non single premium",
    "group yearly renewable premium", "total",
}

MONTH_NAME_TO_NUM = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june",
     "july", "august", "september", "october", "november", "december"], start=1)}

TITLE_PATTERN = re.compile(r"Period ended\s+([A-Za-z]+)-(\d{4})", re.I)
YEAR_IN_FILENAME = re.compile(r"(20\d{2})")


def _to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("%", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _find_detailed_table(tables):
    """The detailed per-category table is the largest table with >=10 columns."""
    candidates = [t for t in tables if t.shape[1] >= 10]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t.shape[0])


def parse_detailed_html(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parses one real council 'Detailed New Business Performance' monthly file."""
    try:
        tables = pd.read_html(io.BytesIO(file_bytes))
    except Exception:
        return pd.DataFrame(columns=ENRICHED_COLS)

    table = _find_detailed_table(tables)
    if table is None:
        return pd.DataFrame(columns=ENRICHED_COLS)

    table = table.reset_index(drop=True)

    # find month/year from the title text (usually row 0, may repeat across cols)
    year, month_num, month_name = None, None, None
    title_blob = " ".join(str(v) for v in table.head(3).values.flatten())
    m = TITLE_PATTERN.search(title_blob)
    if m:
        month_name = m.group(1).strip().lower()
        year = int(m.group(2))
        month_num = MONTH_NAME_TO_NUM.get(month_name)
    if year is None:
        yf = YEAR_IN_FILENAME.search(filename)
        year = int(yf.group(1)) if yf else None

    rows = []
    current_insurer = None
    for _, r in table.iterrows():
        vals = r.tolist()
        col1 = str(vals[1]).strip() if len(vals) > 1 and pd.notna(vals[1]) else ""
        rest_na = all(pd.isna(v) for v in vals[2:12]) if len(vals) >= 12 else True

        if col1 == "" and rest_na:
            continue  # blank separator row

        if col1.lower() not in KNOWN_BUSINESS_TYPES:
            # this is an insurer header row (or a stray title/label row)
            if rest_na and col1 and not col1.lower().startswith(("particulars", "s.no", "detailed", "summary")):
                current_insurer = col1
            continue

        if current_insurer is None:
            continue

        rows.append({
            "Year": year, "Month": f"{year}-{month_num:02d}" if year and month_num else None,
            "MonthName": month_name.title() if month_name else None,
            "Insurer": current_insurer,
            "BusinessType": col1.title() if col1.lower() != "total" else "Total",
            "Premium_Current_Month": _to_float(vals[2]),
            "Premium_Current_YTD": _to_float(vals[3]),
            "Premium_Previous_Month": _to_float(vals[4]),
            "Premium_Previous_YTD": _to_float(vals[5]),
            "Premium_YTD_Variation_Pct": _to_float(vals[6]),
            "Policies_Current_Month": _to_float(vals[7]),
            "Policies_Current_YTD": _to_float(vals[8]),
            "Policies_Previous_Month": _to_float(vals[9]),
            "Policies_Previous_YTD": _to_float(vals[10]),
            "Policies_YTD_Variation_Pct": _to_float(vals[11]) if len(vals) > 11 else np.nan,
        })

    return pd.DataFrame(rows, columns=ENRICHED_COLS)


def _map_precleaned_csv(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Maps a manually-precleaned CSV (insurer, business_type, premium_current_month, ...,
    month [name only, no year]) into the enriched schema, inferring Year from filename."""
    colmap = {
        "insurer": "Insurer", "business_type": "BusinessType",
        "premium_current_month": "Premium_Current_Month", "premium_current_ytd": "Premium_Current_YTD",
        "premium_previous_month": "Premium_Previous_Month", "premium_previous_ytd": "Premium_Previous_YTD",
        "premium_ytd_variation": "Premium_YTD_Variation_Pct",
        "policies_current_month": "Policies_Current_Month", "policies_current_ytd": "Policies_Current_YTD",
        "policies_previous_month": "Policies_Previous_Month", "policies_previous_ytd": "Policies_Previous_YTD",
        "policies_ytd_variation": "Policies_YTD_Variation_Pct",
        "month": "MonthName",
    }
    df = df.rename(columns=colmap)
    yf = YEAR_IN_FILENAME.search(filename)
    year = int(yf.group(1)) if yf else None
    df["Year"] = year
    df["MonthName"] = df["MonthName"].astype(str).str.strip().str.title()
    df["Month"] = df["MonthName"].map(
        lambda mn: f"{year}-{MONTH_NAME_TO_NUM.get(mn.lower(), 0):02d}" if year and mn.lower() in MONTH_NAME_TO_NUM else None
    )
    df["BusinessType"] = df["BusinessType"].astype(str).str.strip().str.title()
    df.loc[df["BusinessType"].str.lower() == "total", "BusinessType"] = "Total"
    for c in ENRICHED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[ENRICHED_COLS]


def consolidate(files: dict) -> pd.DataFrame:
    """
    files: dict of {filename: bytes}
    Auto-detects: real council HTML-as-.xls, pre-cleaned simplified CSV,
    already-enriched CSV, or legacy simple aggregated CSV.
    Returns one clean consolidated enriched DataFrame across all months.
    """
    frames = []
    for filename, content in files.items():
        fl = filename.lower()
        if fl.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception:
                continue
            cols_lower = set(c.lower() for c in df.columns)
            if set(ENRICHED_COLS).issubset(df.columns):
                frames.append(df[ENRICHED_COLS])
            elif {"insurer", "business_type", "premium_current_month", "month"}.issubset(cols_lower):
                frames.append(_map_precleaned_csv(df, filename))
            elif set(SIMPLE_COLS).issubset(df.columns):
                # legacy/demo simple schema -> upgrade to enriched with BusinessType="Total"
                simple = df[SIMPLE_COLS].copy()
                simple["BusinessType"] = "Total"
                simple["Premium_Current_Month"] = simple["Premium_Rs_Crore"]
                simple["Policies_Current_Month"] = simple["No_of_Policies"]
                for c in ["Premium_Current_YTD", "Premium_Previous_Month", "Premium_Previous_YTD",
                          "Premium_YTD_Variation_Pct", "Policies_Current_YTD",
                          "Policies_Previous_Month", "Policies_Previous_YTD", "Policies_YTD_Variation_Pct"]:
                    simple[c] = np.nan
                simple["Year"] = pd.to_datetime(simple["Month"], format="%Y-%m").dt.year
                simple["MonthName"] = pd.to_datetime(simple["Month"], format="%Y-%m").dt.strftime("%B")
                frames.append(simple[ENRICHED_COLS])
            else:
                continue
        else:
            frames.append(parse_detailed_html(content, filename))

    if not frames:
        return pd.DataFrame(columns=ENRICHED_COLS)

    master = pd.concat(frames, ignore_index=True)
    master = clean_master(master)
    return master


def clean_master(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Insurer"] = (
        df["Insurer"].astype(str).str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"(?i)^life insurance corporation.*", "LIC", regex=True)
        .str.replace(r"(?i)\s*(company)?\s*limited$", "", regex=True)
        .str.strip()
    )
    num_cols = [c for c in ENRICHED_COLS if c.startswith(("Premium_", "Policies_"))]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Month", "Insurer", "BusinessType"])
    df = df.drop_duplicates(subset=["Month", "Insurer", "BusinessType"], keep="last")
    df = df.sort_values(["Month", "Insurer", "BusinessType"]).reset_index(drop=True)
    return df


def load_master_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if set(SIMPLE_COLS).issubset(df.columns) and not set(ENRICHED_COLS).issubset(df.columns):
        simple = df[SIMPLE_COLS].copy()
        simple["BusinessType"] = "Total"
        simple["Premium_Current_Month"] = simple["Premium_Rs_Crore"]
        simple["Policies_Current_Month"] = simple["No_of_Policies"]
        for c in ["Premium_Current_YTD", "Premium_Previous_Month", "Premium_Previous_YTD",
                  "Premium_YTD_Variation_Pct", "Policies_Current_YTD",
                  "Policies_Previous_Month", "Policies_Previous_YTD", "Policies_YTD_Variation_Pct"]:
            simple[c] = np.nan
        simple["Year"] = pd.to_datetime(simple["Month"], format="%Y-%m").dt.year
        simple["MonthName"] = pd.to_datetime(simple["Month"], format="%Y-%m").dt.strftime("%B")
        df = simple[ENRICHED_COLS]
    return clean_master(df)


def totals_series(master: pd.DataFrame) -> pd.DataFrame:
    """Extract insurer-month TOTAL rows -> the simple (Month, Insurer, Premium, Policies)
    view that feeds EDA/backtesting/forecasting, same as before."""
    t = master[master["BusinessType"] == "Total"].copy()
    t = t.rename(columns={
        "Premium_Current_Month": "Premium_Rs_Crore",
        "Policies_Current_Month": "No_of_Policies",
    })
    return t[["Month", "Insurer", "Premium_Rs_Crore", "No_of_Policies"]].reset_index(drop=True)


def category_mix(master: pd.DataFrame) -> pd.DataFrame:
    """Excludes Total rows -> the 5 premium-category breakdown for mix analysis."""
    c = master[master["BusinessType"] != "Total"].copy()
    return c
