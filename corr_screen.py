# corr_screen.py
# Analyze pair correlations in TRAIN set to find candidate pairs.
# Keep pairs with mean rolling correlation ≥ threshold (default 0.70)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import combinations
from tabulate import tabulate

# Parameters (used when running this script directly)
USE_LOG = True       # correlations on log-prices are common in long-horizon analysis
CORR_WIN = 252       # ~1 trading year window
CORR_THRES = 0.70    # minimum mean correlation threshold


def rolling_mean_corr(s1, s2, window):
    """
    Calculate rolling correlation between two price series
    and return its mean (ignoring NaNs).
    Returns NaN if there are no valid values.
    """
    c = s1.rolling(window=window).corr(s2)
    c = c.dropna()
    if len(c) == 0:
        return np.nan
    else:
        return float(c.mean())


def screen_pairs_by_corr(train_df, window=252, threshold=0.70, use_log=True):
    """
    Compute mean rolling correlation for all possible column combinations.
    Returns two DataFrames:
      - candidates: pairs that pass the threshold
      - corr_df: all calculated correlations (unscreened)
    """
    # Optionally use log-prices
    if use_log:
        lvl = np.log(train_df)
    else:
        lvl = train_df.copy()

    # Create all unique combinations of asset pairs
    cols = lvl.columns.tolist()
    pairs = list(combinations(cols, 2))

    results = []
    for a, b in pairs:
        # Calculate the mean rolling correlation for this pair
        mc = rolling_mean_corr(lvl[a], lvl[b], window)
        results.append({
            "asset1": a,
            "asset2": b,
            "mean_rolling_corr": mc
        })

    # Build the full correlation DataFrame
    corr_df = pd.DataFrame(results)
    corr_df = corr_df.dropna()
    corr_df = corr_df.sort_values("mean_rolling_corr", ascending=False)

    # Keep only pairs above the correlation threshold
    candidates = corr_df[corr_df["mean_rolling_corr"] >= threshold]
    candidates = candidates.reset_index(drop=True)

    return candidates, corr_df


if __name__ == "__main__":
    # Example run: load data and print top correlated pairs
    from data import download_adj_close, clean_align_panel, chronological_split, TICKERS, START, END

    panel = download_adj_close(TICKERS, START, END)
    panel = clean_align_panel(panel)
    train, test, valid = chronological_split(panel, 0.6, 0.2)

    candidates, corr_full = screen_pairs_by_corr(train, CORR_WIN, CORR_THRES, USE_LOG)

    print(f"\nCorrelation screen on TRAIN (mean rolling ≥ {CORR_THRES:.2f}) — {len(candidates)} candidates found:\n")
    print(tabulate(candidates.head(20), headers="keys", tablefmt="simple", floatfmt=".3f"))
