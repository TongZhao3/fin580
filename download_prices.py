"""
Download daily historical price data for multi-asset ETFs.
Assets: SPY, TLT, GLD, UUP
Period: 2020-01-01 to 2025-01-01
"""

import yfinance as yf
import pandas as pd

# ── Configuration ──────────────────────────────────────────────
TICKERS = ["SPY", "TLT", "GLD", "UUP"]
START = "2020-01-01"
END = "2025-01-01"
OUTPUT = "prices.csv"

# ── Download ───────────────────────────────────────────────────
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=False)

# Keep only Adjusted Close
adj_close = raw["Adj Close"]

# Clean up: drop rows where all values are NaN, round to 2 decimals
adj_close = adj_close.dropna(how="all").round(2)

# Ensure column order matches TICKERS
adj_close = adj_close[TICKERS]

# ── Save ───────────────────────────────────────────────────────
adj_close.to_csv(OUTPUT)
print(f"✅ Saved {len(adj_close)} trading days → {OUTPUT}")
print(f"   Date range: {adj_close.index[0].date()} to {adj_close.index[-1].date()}")
print()
print(adj_close.head(10))
print("...")
print(adj_close.tail(5))
