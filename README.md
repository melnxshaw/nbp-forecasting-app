# Life Insurance New Business — Sales Analysis & Forecasting

A working, deployable analytics app for the hackathon use case: analyze historical
life insurance new-business premium & policy data, uncover trends/seasonality, and
forecast future performance with backtested accuracy metrics (MAE, RMSE, WAPE, Bias).

**Live features:**
- Upload real monthly files from lifeinscouncil.org (auto-detects HTML-as-.xls exports,
  real Excel, or CSV) — or explore instantly with a built-in realistic demo dataset
- EDA: industry trend, seasonality by month, insurer market share, top-insurer contribution
- Time-ordered (rolling-origin) backtesting comparing Naive / Seasonal Naive / Moving
  Average / Holt-Winters models on MAE, RMSE, WAPE, Bias
- Forward forecast (SARIMA + Holt-Winters) with 80% confidence interval, downloadable as CSV
- Auto-generated business insights (YoY growth, peak season, fastest-growing insurer)
- Documented assumptions & limitations tab

---

## Project files
```
nbp_project/
├── app.py                      # Streamlit app (main entry point)
├── data_pipeline.py            # Ingestion & cleaning (handles real council files)
├── forecasting.py              # Models, metrics, backtesting
├── generate_synthetic_data.py  # Builds the demo dataset
├── synthetic_nbp_master.csv    # Pre-generated demo data (15 insurers x 64 months)
├── requirements.txt
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`. Select **"Use demo data"** in the sidebar to explore
immediately, or **"Upload my own files"** to drop in real downloaded council files.

## Getting real data (manual, one-time per month)
1. Go to https://www.lifeinscouncil.org/industry%20information/nbp.aspx
2. Pick Year + Month → click **GetData** → it downloads (HTML disguised as `.xls`)
3. Rename each download so months don't overwrite each other, e.g. `NBP_2025_04.xls`
4. Upload all files together in the app sidebar — the parser reads them directly,
   no manual conversion needed

## Deploy for free (so you have a live link to show judges)

**Option A — Streamlit Community Cloud (recommended, ~5 min, free)**
1. Create a new GitHub repo, push these files (`app.py`, `data_pipeline.py`,
   `forecasting.py`, `synthetic_nbp_master.csv`, `requirements.txt`)
   ```bash
   git init
   git add .
   git commit -m "NBP sales analysis & forecasting app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io → **New app** → connect your GitHub → select the repo
   → main file path `app.py` → **Deploy**
3. You'll get a public URL like `https://<repo-name>.streamlit.app` — share that with judges

**Option B — Hugging Face Spaces (also free)**
1. Create a Space → SDK: Streamlit
2. Upload the same files (Spaces auto-installs `requirements.txt`)
3. Public URL is generated automatically

Either option gives you a real, live, working link — not just a local demo.

## Assumptions & limitations
See the in-app "Assumptions & Limitations" tab — it's part of the deliverable
(reproducibility with documented assumptions was explicitly required in the brief).
