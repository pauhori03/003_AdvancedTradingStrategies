# data.py
# This file handles all data preparation steps:
#  - Download daily adjusted prices from Yahoo Finance
#  - Clean missing values and align all assets
#  - Split the full dataset into Train / Test / Validation periods

import warnings
warnings.filterwarnings("ignore")

from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf



# Configuration

# Selected assets from different industries

TICKERS = [
    "JPM","BAC","C","GS","MS","WFC","USB","PNC","BK","SCHW",
    "XOM","CVX","COP","PSX","BP","SHEL","TTE","EQNR","PBR","ENB",
    "AAPL","MSFT","GOOG","AMZN","META","NVDA","TSLA","NFLX","ORCL","INTC",
    "UNH","PFE","JNJ","ABBV","MRK","BMY","LLY","AMGN","CVS","MDT",
    "NKE","DIS","MCD","SBUX","TGT","COST","PG","CL","KMB","KO"
]

# 15-year window
today = date.today()
START = "2010-01-01"
END   = None  #  Today by default in yfinance

# Small fill limits to handle holidays or short gaps
FFILL_LIMIT = 5
BFILL_LIMIT = 5

def download_adj_close(tickers, start, end):
    """
    Downloads daily adjusted close prices for all tickers.
    Returns a DataFrame with one column per asset.
    """
    raw = yf.download(
        tickers, start=start, end=end, interval="1d",
        progress=False, group_by="ticker",
        auto_adjust=False
    )

   # Extract only the "Adj Close" values from the multi-index
    if isinstance(raw.columns, pd.MultiIndex):
        cols = {}
        for t in tickers:
            try:
                cols[t] = raw[(t, "Adj Close")]
            except Exception:
                # If a ticker is missing, just skip it
                pass
        prices = pd.DataFrame(cols)
    else:
        # Fallback if columns are flat
        prices = raw.get("Adj Close")
        if prices is None:
            raise ValueError("Could not find 'Adj Close' in the download.")

    # Sort by date and remove duplicates
    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices[~prices.index.duplicated(keep="first")]
    return prices

def clean_align_panel(df, ffill_limit=5, bfill_limit=5):
    """
    Cleans and aligns the dataset:
      - Fills small gaps forward/backward
      - Drops remaining missing values
      - Removes flat or invalid series
    """
    clean = df.ffill(limit=ffill_limit).bfill(limit=bfill_limit)
    clean = clean.dropna(how="any")
    keep = [c for c in clean.columns if clean[c].std(skipna=True) > 0]
    return clean[keep].astype(float)

def chronological_split(df, train_ratio=0.6, test_ratio=0.2):
    """
    Splits the data chronologically:
      60% Train, 20% Test, 20% Validation.
    """
    n = len(df)
    i_tr = int(n * train_ratio)
    i_te = int(n * (train_ratio + test_ratio))
    return df.iloc[:i_tr].copy(), df.iloc[i_tr:i_te].copy(), df.iloc[i_te:].copy()

if __name__ == "__main__":
    panel_raw = download_adj_close(TICKERS, START, END)

    panel = clean_align_panel(panel_raw, FFILL_LIMIT, BFILL_LIMIT)
    train, test, valid = chronological_split(panel, 0.6, 0.2)

    print("Panel (clean):", panel.shape, "|", panel.index.min().date(), "→", panel.index.max().date())
    print("Train:", train.shape, train.index.min().date(), "→", train.index.max().date())
    print("Test :", test.shape,  test.index.min().date(),  "→", test.index.max().date())
    print("Valid:", valid.shape, valid.index.min().date(), "→", valid.index.max().date())
