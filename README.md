# 📓 Grant's Notebook

A live portfolio dashboard built with Streamlit. Prices update every 15 seconds during market hours via Yahoo Finance.

## Quick Start

### 1. Deploy to Streamlit Cloud (recommended)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account → select this repo → set `app.py` as the main file
4. Deploy — you'll get a public URL like `https://your-app.streamlit.app`

### 2. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works

| Component | Description |
|---|---|
| `app.py` | Main Streamlit dashboard — fetches live prices, displays KPIs, tables, and charts |
| `daily_snapshot.py` | Runs at 6 PM ET via GitHub Actions — records closing prices and portfolio balance |
| `data/holdings.csv` | Portfolio holdings (edit this when buying/selling) |
| `data/daily_closes.csv` | Historical daily closing prices (auto-appended) |
| `data/portfolio_balance.csv` | Historical portfolio balance (auto-appended) |

## Updating the portfolio

To add or remove a stock, edit `data/holdings.csv`. Columns:

```
date_bought, ticker, name, shares, buy_price
```

## Daily automation

The GitHub Actions workflow (`.github/workflows/daily_snapshot.yml`) runs every weekday at 6 PM ET and:

1. Fetches closing prices for all tickers
2. Appends to `data/daily_closes.csv`
3. Computes total portfolio value and appends to `data/portfolio_balance.csv`
4. Commits and pushes the updated CSV files

You can also trigger it manually from the Actions tab in GitHub.
