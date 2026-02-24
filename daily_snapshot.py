"""
daily_snapshot.py — Runs at 6 PM ET every weekday via GitHub Actions.

1. Fetches closing prices for all tickers.
2. Appends a row to data/daily_closes.csv.
3. Computes total portfolio value and appends to data/portfolio_balance.csv.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path
import sys

DATA_DIR = Path(__file__).parent / "data"
HOLDINGS_PATH = DATA_DIR / "holdings.csv"
CLOSES_PATH = DATA_DIR / "daily_closes.csv"
BALANCE_PATH = DATA_DIR / "portfolio_balance.csv"


def main():
    today_str = datetime.today().strftime("%Y-%m-%d")

    # ── Load holdings ───────────────────────────────────────────────────────
    holdings = pd.read_csv(HOLDINGS_PATH)
    all_tickers = holdings["ticker"].tolist()
    stock_tickers = [t for t in all_tickers if t != "Foreign Stock"]

    # ── Fetch closing prices ────────────────────────────────────────────────
    print(f"Fetching closing prices for {len(stock_tickers)} tickers...")
    data = yf.download(stock_tickers, period="1d", interval="1d", progress=False, threads=True)

    if data.empty:
        print("ERROR: yfinance returned no data. Market may be closed today.")
        sys.exit(1)

    prices = {}
    close_data = data["Close"]
    if isinstance(close_data, pd.Series):
        # Single ticker edge case
        prices[stock_tickers[0]] = float(close_data.dropna().iloc[-1])
    else:
        for t in stock_tickers:
            if t in close_data.columns:
                series = close_data[t].dropna()
                if not series.empty:
                    prices[t] = round(float(series.iloc[-1]), 2)

    if not prices:
        print("ERROR: Could not fetch any prices.")
        sys.exit(1)

    print(f"Fetched prices for {len(prices)} tickers.")

    # ── Handle Foreign Stock ────────────────────────────────────────────────
    foreign_row = holdings[holdings["ticker"] == "Foreign Stock"]
    if not foreign_row.empty:
        prices["Foreign Stock"] = float(foreign_row["buy_price"].iloc[0])

    # ── Append to daily_closes.csv ──────────────────────────────────────────
    closes_df = pd.read_csv(CLOSES_PATH, parse_dates=["date"])

    # Don't duplicate if already run today
    if today_str in closes_df["date"].astype(str).values:
        print(f"Warning: {today_str} already exists in daily_closes.csv. Updating row.")
        closes_df = closes_df[closes_df["date"].astype(str) != today_str]

    # Build new row with all column tickers
    new_row = {"date": today_str}
    for col in closes_df.columns:
        if col == "date":
            continue
        new_row[col] = prices.get(col, np.nan)

    closes_df = pd.concat([closes_df, pd.DataFrame([new_row])], ignore_index=True)
    closes_df = closes_df.sort_values("date").reset_index(drop=True)
    closes_df.to_csv(CLOSES_PATH, index=False)
    print(f"Appended closes for {today_str} to daily_closes.csv")

    # ── Compute portfolio balance and append ────────────────────────────────
    total_value = 0.0
    for _, row in holdings.iterrows():
        ticker = row["ticker"]
        shares = row["shares"]
        price = prices.get(ticker, 0)
        total_value += shares * price

    total_value = round(total_value)

    balance_df = pd.read_csv(BALANCE_PATH, parse_dates=["date"])

    if today_str in balance_df["date"].astype(str).values:
        print(f"Warning: {today_str} already in portfolio_balance.csv. Updating.")
        balance_df = balance_df[balance_df["date"].astype(str) != today_str]

    balance_df = pd.concat([
        balance_df,
        pd.DataFrame([{"date": today_str, "balance": total_value}])
    ], ignore_index=True)
    balance_df = balance_df.sort_values("date").reset_index(drop=True)
    balance_df.to_csv(BALANCE_PATH, index=False)
    print(f"Appended balance ${total_value:,} for {today_str}")

    print("Daily snapshot complete.")


if __name__ == "__main__":
    main()
