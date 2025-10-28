# plots.py
# Minimal plotting helpers for the project.
# Requirements:
#   - matplotlib
#   - pandas
#
# Inputs expected (from your pipeline artifacts dicts):
#   art = {
#     "equity": pd.Series,      # equity curve
#     "beta": pd.Series,        # hedge ratio over time
#     "spread": pd.Series,      # Kalman spread
#     "signal": pd.Series,      # {-1,0,+1}
#     "trades": pd.DataFrame,   # backtest logs (not used directly here)
#     "summary": dict
#   }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- utilities ----------

def _rolling_z(x: pd.Series, window: int = 60, min_periods: int = 30, sd_floor: float = 1e-8) -> pd.Series:
    """Rolling Z-score (same idea as in signals.py)."""
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std().clip(lower=sd_floor)
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _segments_from_signal(signal: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Turn a {-1,0,+1} signal into trade segments:
      [(entry_dt, exit_dt, side), ...], side in {-1,+1}
    Entry = 0 -> nonzero; Exit = nonzero -> 0.
    If a segment is open at the end, it's ignored for 'per-trade' stats.
    """
    sig = signal.astype(int)
    entries, exits = [], []
    prev = 0
    for dt, s in sig.items():
        if prev == 0 and s != 0:         # entry
            entries.append(dt)
        elif prev != 0 and s == 0:       # exit
            exits.append(dt)
        prev = s

    # pair up entries and exits
    segs = []
    for en_dt, ex_dt in zip(entries, exits):
        s = int(signal.loc[en_dt])       # side at entry
        s = 1 if s > 0 else -1
        segs.append((en_dt, ex_dt, s))
    return segs


def _trade_returns_from_equity(equity: pd.Series, signal: pd.Series) -> pd.Series:
    """
    Approx per-trade returns using equity at entry/exit boundaries.
    return_i = (Equity_exit - Equity_entry) / Equity_entry
    """
    segs = _segments_from_signal(signal)
    rets = []
    idx = []
    for en_dt, ex_dt, _side in segs:
        if en_dt not in equity.index or ex_dt not in equity.index:
            continue
        e0 = float(equity.loc[en_dt])
        e1 = float(equity.loc[ex_dt])
        if e0 != 0 and np.isfinite(e0) and np.isfinite(e1):
            r = (e1 / e0) - 1.0
            rets.append(r)
            idx.append(ex_dt)
    return pd.Series(rets, index=idx, name="trade_return")


# ---------- plots ----------

def plot_spread_with_zbands(
    art: dict,
    entry_z: float,
    exit_z: float,
    z_window: int = 60,
    title: str = "Spread & Z-bands"
):
    """
    Two-panel figure:
      Top  : spread with rolling mean and shaded positions
      Bottom: rolling Z-score with entry/exit thresholds
    """
    spread = art["spread"].copy()
    signal = art["signal"].copy().astype(int)

    z = _rolling_z(spread, window=z_window)
    mu = spread.rolling(z_window, min_periods=z_window // 2).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2.0, 1.0]})

    # --- Top: spread ---
    ax1.plot(spread.index, spread.values, label="Spread", linewidth=1.2)
    if mu.notna().any():
        ax1.plot(mu.index, mu.values, label=f"Rolling mean ({z_window})", linewidth=1.0)
    ax1.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)

    in_long = signal == -1   # long spread
    in_short = signal == +1  # short spread
    if in_long.any():
        ax1.fill_between(spread.index, spread.min(), spread.max(), where=in_long, color="C2", alpha=0.07, label="Long spread")
    if in_short.any():
        ax1.fill_between(spread.index, spread.min(), spread.max(), where=in_short, color="C3", alpha=0.07, label="Short spread")

    ax1.set_title(title)
    ax1.set_ylabel("Spread")
    ax1.legend(loc="best")

    # --- Bottom: Z-score ---
    ax2.plot(z.index, z.values, label="Z-score", linewidth=1.0)
    ax2.axhline(+entry_z, color="C3", linestyle="--", linewidth=0.9, label=f"+entry ({entry_z})")
    ax2.axhline(-entry_z, color="C2", linestyle="--", linewidth=0.9, label=f"-entry ({entry_z})")
    ax2.axhline(+exit_z,  color="gray", linestyle=":",  linewidth=0.9, label=f"+exit ({exit_z})")
    ax2.axhline(-exit_z,  color="gray", linestyle=":",  linewidth=0.9, label=f"-exit ({exit_z})")
    ax2.axhline(0.0,      color="k",   linestyle="-",  linewidth=0.8)

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Z")
    ax2.legend(loc="best")

    fig.tight_layout()


def plot_beta(art: dict, title: str = "Hedge ratio β_t"):
    """Hedge ratio over time."""
    beta = art["beta"].copy()
    plt.figure(figsize=(11, 4))
    plt.plot(beta.index, beta.values, linewidth=1.2, label="β_t")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Beta")
    plt.legend(loc="best")
    plt.tight_layout()


def plot_equity_curves(eq_train: pd.Series = None, eq_test: pd.Series = None, eq_valid: pd.Series = None, title: str = "Equity curves"):
    """Overlay equity curves for train/test/valid (only plots the ones provided)."""
    plt.figure(figsize=(11, 4.8))
    if isinstance(eq_train, pd.Series) and len(eq_train) > 0:
        plt.plot(eq_train.index, eq_train.values, label="TRAIN", linewidth=1.4)
    if isinstance(eq_test, pd.Series) and len(eq_test) > 0:
        plt.plot(eq_test.index, eq_test.values, label="TEST", linewidth=1.4)
    if isinstance(eq_valid, pd.Series) and len(eq_valid) > 0:
        plt.plot(eq_valid.index, eq_valid.values, label="VALID", linewidth=1.4)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend(loc="best")
    plt.tight_layout()


def plot_trade_return_hist(art: dict, bins: int = 30, title: str = "Per-trade return distribution"):
    """
    Histogram of per-trade returns inferred from equity at entry/exit boundaries.
    Shows mean/median and a 0% reference line.
    """
    equity = art["equity"]
    signal = art["signal"].astype(int)

    tr = _trade_returns_from_equity(equity, signal)
    if tr.empty:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No completed trades to plot", ha="center", va="center")
        plt.axis("off")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(tr.values, bins=bins, alpha=0.9)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.0, label="0%")
    plt.axvline(tr.mean(), color="C1", linestyle="-", linewidth=1.2, label=f"Mean={tr.mean():.2%}")
    plt.axvline(tr.median(), color="C2", linestyle="-.", linewidth=1.2, label=f"Median={tr.median():.2%}")
    plt.title(title + f"  |  n={len(tr)}")
    plt.xlabel("Return per trade")
    plt.ylabel("Frequency")
    plt.legend(loc="best")
    plt.tight_layout()
