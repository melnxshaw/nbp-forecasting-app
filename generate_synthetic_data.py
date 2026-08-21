"""
Generates synthetic monthly Life Insurance New Business Premium (NBP) data,
shaped like the real Life Insurance Council (lifeinscouncil.org) monthly files:
one row per insurer, columns for Premium (Rs crore) and No. of Policies,
for each month. This lets the full pipeline / app be built and tested today,
and swapped for real downloaded data later using the same column structure.

Real-world patterns baked in (so forecasting is realistic, not trivial):
- LIC dominates premium share but is volatile month to month
- Private insurers show steadier YoY growth
- Strong March spike every year (Indian tax-saving / fiscal year-end buying)
- Mild dip in April-May (post fiscal-year lull)
- Overall upward trend across years with random shocks
"""
import numpy as np
import pandas as pd

np.random.seed(42)

INSURERS = [
    "LIC", "SBI Life", "HDFC Life", "ICICI Prudential Life", "Axis Max Life",
    "Bajaj Allianz Life", "Kotak Mahindra Life", "Tata AIA Life", "PNB MetLife",
    "Aditya Birla Sun Life", "Star Union Dai-ichi", "Canara HSBC Life",
    "Pramerica Life", "Bharti AXA Life", "Shriram Life"
]

# base monthly premium (Rs crore) and rough market character per insurer
BASE_PREMIUM = {
    "LIC": 14000, "SBI Life": 2600, "HDFC Life": 2200, "ICICI Prudential Life": 1500,
    "Axis Max Life": 900, "Bajaj Allianz Life": 850, "Kotak Mahindra Life": 600,
    "Tata AIA Life": 550, "PNB MetLife": 350, "Aditya Birla Sun Life": 500,
    "Star Union Dai-ichi": 180, "Canara HSBC Life": 220, "Pramerica Life": 90,
    "Bharti AXA Life": 120, "Shriram Life": 100,
}
GROWTH_RATE = {  # annual growth trend, LIC slower/volatile, private insurers grow faster
    "LIC": 0.02, "SBI Life": 0.14, "HDFC Life": 0.12, "ICICI Prudential Life": 0.10,
    "Axis Max Life": 0.13, "Bajaj Allianz Life": 0.08, "Kotak Mahindra Life": 0.11,
    "Tata AIA Life": 0.15, "PNB MetLife": 0.07, "Aditya Birla Sun Life": 0.09,
    "Star Union Dai-ichi": 0.06, "Canara HSBC Life": 0.10, "Pramerica Life": 0.05,
    "Bharti AXA Life": 0.04, "Shriram Life": 0.06,
}
AVG_TICKET_SIZE_LAKH = {  # used to derive policy counts from premium (varies by insurer)
    "LIC": 1.1, "SBI Life": 1.6, "HDFC Life": 1.9, "ICICI Prudential Life": 1.7,
    "Axis Max Life": 1.5, "Bajaj Allianz Life": 1.3, "Kotak Mahindra Life": 1.8,
    "Tata AIA Life": 1.6, "PNB MetLife": 1.2, "Aditya Birla Sun Life": 1.4,
    "Star Union Dai-ichi": 1.0, "Canara HSBC Life": 1.3, "Pramerica Life": 0.9,
    "Bharti AXA Life": 1.0, "Shriram Life": 0.8,
}

MONTH_SEASONALITY = {  # multiplicative seasonal factor by calendar month
    1: 0.92, 2: 0.98, 3: 1.55, 4: 0.72, 5: 0.85, 6: 0.95,
    7: 1.00, 8: 0.98, 9: 1.05, 10: 1.02, 11: 0.95, 12: 1.05,
}

def generate(start="2021-04", end="2026-07"):
    months = pd.period_range(start=start, end=end, freq="M")
    rows = []
    for insurer in INSURERS:
        base = BASE_PREMIUM[insurer]
        g = GROWTH_RATE[insurer]
        ticket = AVG_TICKET_SIZE_LAKH[insurer]
        for i, m in enumerate(months):
            years_elapsed = i / 12.0
            trend = base * ((1 + g) ** years_elapsed)
            season = MONTH_SEASONALITY[m.month]
            noise = np.random.normal(1.0, 0.06)
            # occasional shock (regulatory change, covid-like dip, etc.)
            shock = 1.0
            if np.random.rand() < 0.03:
                shock = np.random.choice([0.75, 1.25])
            premium = max(trend * season * noise * shock, 1.0)
            policies = int(max((premium * 100) / (ticket * np.random.normal(1.0, 0.08)), 1))
            rows.append({
                "Month": m.to_timestamp().strftime("%Y-%m"),
                "Insurer": insurer,
                "Premium_Rs_Crore": round(premium, 2),
                "No_of_Policies": policies,
            })
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv("/home/claude/nbp_project/synthetic_nbp_master.csv", index=False)
    print(df.shape)
    print(df.head(10))
    print(df["Month"].min(), df["Month"].max())
