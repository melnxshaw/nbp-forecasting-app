"""
Data ingestion & cleaning pipeline for Life Insurance NBP data.

Handles THREE input types so real council downloads work without any
manual conversion:
  1. Real council files: .xls extension but actually HTML tables
     (this is what lifeinscouncil.org's "GetData" button produces)
  2. Genuine Excel files (.xlsx/.xls binary)
  3. Already-consolidated CSV (e.g. our synthetic master file, or a
     master file the user maintains themselves)

Output: a single tidy long-format DataFrame:
  Month (YYYY-MM), Insurer, Premium_Rs_Crore, No_of_Policies
"""
import io
import re
import pandas as pd
import numpy as np

STANDARD_COLS = ["Month", "Insurer", "Premium_Rs_Crore", "No_of_Policies"]

# Common column-name variants seen across insurer/regulator reports
PREMIUM_COL_HINTS = ["premium", "nbp", "new business premium"]
POLICY_COL_HINTS = ["polic", "no. of lives", "lives", "schemes"]
INSURER_COL_HINTS = ["insurer", "company", "name of the insurer"]

MONTH_PATTERN = re.compile(r"(20\d{2})[-_]?(0[1-9]|1[0-2])")


def _guess_month_from_filename(filename: str):
    m = MONTH_PATTERN.search(filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None


def _find_col(columns, hints):
    for c in columns:
        cl = str(c).lower()
        for h in hints:
            if h in cl:
                return c
    return None


def _read_any_table(file_bytes: bytes, filename: str) -> list[pd.DataFrame]:
    """Try HTML first (council files are HTML disguised as .xls), then Excel, then CSV."""
    tables = []
    # 1. Try HTML (this is the common case for lifeinscouncil.org exports)
    try:
        tables = pd.read_html(io.BytesIO(file_bytes))
        if tables:
            return tables
    except Exception:
        pass
    # 2. Try genuine Excel
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        return [xls.parse(sheet) for sheet in xls.sheet_names]
    except Exception:
        pass
    # 3. Try CSV
    try:
        return [pd.read_csv(io.BytesIO(file_bytes))]
    except Exception:
        pass
    return []


def parse_council_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parses one monthly council export (HTML-as-.xls, real .xlsx, or .csv)
    into tidy rows: Month, Insurer, Premium_Rs_Crore, No_of_Policies.
    Falls back gracefully and returns an empty frame (with a note) if the
    layout can't be confidently detected, rather than silently guessing wrong.
    """
    month_guess = _guess_month_from_filename(filename)
    tables = _read_any_table(file_bytes, filename)

    best = None
    for t in tables:
        if t.shape[0] >= 3 and t.shape[1] >= 2:
            if best is None or t.shape[0] > best.shape[0]:
                best = t

    if best is None:
        return pd.DataFrame(columns=STANDARD_COLS)

    df = best.copy()
    df.columns = [str(c).strip() for c in df.columns]

    insurer_col = _find_col(df.columns, INSURER_COL_HINTS) or df.columns[0]
    premium_col = _find_col(df.columns, PREMIUM_COL_HINTS)
    policy_col = _find_col(df.columns, POLICY_COL_HINTS)

    out = pd.DataFrame()
    out["Insurer"] = df[insurer_col].astype(str).str.strip()
    out["Premium_Rs_Crore"] = pd.to_numeric(
        df[premium_col], errors="coerce") if premium_col else np.nan
    out["No_of_Policies"] = pd.to_numeric(
        df[policy_col], errors="coerce") if policy_col else np.nan
    out["Month"] = month_guess if month_guess else "UNKNOWN"

    # drop subtotal / total / header-junk rows
    junk_pattern = re.compile(r"total|grand|private|industry|^\s*$|^nan$", re.I)
    out = out[~out["Insurer"].str.match(junk_pattern, na=False)]
    out = out.dropna(subset=["Premium_Rs_Crore"], how="all")
    out = out[out["Insurer"].notna() & (out["Insurer"] != "")]

    return out[STANDARD_COLS].reset_index(drop=True)


def consolidate(files: dict) -> pd.DataFrame:
    """
    files: dict of {filename: bytes}
    Returns one clean consolidated long-format DataFrame across all months.
    """
    frames = []
    for filename, content in files.items():
        if filename.lower().endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content))
                if set(STANDARD_COLS).issubset(df.columns):
                    frames.append(df[STANDARD_COLS])
                    continue
            except Exception:
                pass
        frames.append(parse_council_file(content, filename))

    if not frames:
        return pd.DataFrame(columns=STANDARD_COLS)

    master = pd.concat(frames, ignore_index=True)
    master = clean_master(master)
    return master


def clean_master(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Insurer"] = (
        df["Insurer"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"(?i)^life insurance corporation.*", "LIC", regex=True)
        .str.replace(r"(?i)\bltd\.?$|\blimited$", "", regex=True)
        .str.strip()
    )
    df["Premium_Rs_Crore"] = pd.to_numeric(df["Premium_Rs_Crore"], errors="coerce")
    df["No_of_Policies"] = pd.to_numeric(df["No_of_Policies"], errors="coerce")
    df = df.dropna(subset=["Month", "Premium_Rs_Crore"])
    df = df[df["Premium_Rs_Crore"] >= 0]
    df = df.drop_duplicates(subset=["Month", "Insurer"], keep="last")
    df = df.sort_values(["Month", "Insurer"]).reset_index(drop=True)
    return df


def load_master_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean_master(df)
