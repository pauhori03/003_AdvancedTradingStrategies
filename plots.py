# plots.py
# Minimal plotting helpers for the project.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_equity_curves(
    eq_train: pd.Series = None,
    eq_test: pd.Series = None,
    eq_valid: pd.Series = None,
    title: str = "Equity curves"
):
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
    plt.show()


def plot_kalman_beta(kalman_df: pd.DataFrame):
    """Plot dynamic hedge ratio (beta_t) from Kalman Filter."""
    plt.figure(figsize=(10, 4))
    plt.plot(kalman_df["beta"], color="orange", label="Kalman β_t")
    plt.title("Dynamic Hedge Ratio (Kalman Filter)")
    plt.xlabel("Date")
    plt.ylabel("β_t")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_signals(px_x: pd.Series, px_y: pd.Series, signal: pd.Series):
    """Overlay entry/exit signals on both assets' price series."""
    plt.figure(figsize=(10, 4))
    
    # Plot prices
    plt.plot(px_x, label="Asset X", color="dodgerblue")
    plt.plot(px_y, label="Asset Y", color="coral", alpha=0.8)

    # Mark signals
    long_signals = signal[signal == -1].index   # when strategy goes long spread
    short_signals = signal[signal == 1].index   # when strategy goes short spread

    # --- Señales sobre Asset Y ---
    plt.scatter(long_signals, px_y.loc[long_signals],
                color="green", marker="^", s=70, label="Long spread (Buy Y)")
    plt.scatter(short_signals, px_y.loc[short_signals],
                color="red", marker="v", s=70, label="Short spread (Sell Y)")

    # --- Señales reflejadas en Asset X ---
    plt.scatter(long_signals, px_x.loc[long_signals],
                color="lime", marker="^", s=55, edgecolor="black", alpha=0.6, label="Buy X")
    plt.scatter(short_signals, px_x.loc[short_signals],
                color="darkred", marker="v", s=55, edgecolor="black", alpha=0.6, label="Sell X")

    # Labels and aesthetics
    plt.title("Pair Prices with Trading Signals (Both Legs)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spread_signals(px_x: pd.Series, px_y: pd.Series, beta: pd.Series,
                        entry_z: float, exit_z: float, z_window: int = 60):
    """Plot the spread with entry/exit thresholds and signals."""

    # Dynamic spread with Kalman beta_t
    spread = px_y - beta * px_x

    # Rolling statistics
    rolling_mean = spread.rolling(z_window).mean()
    rolling_std = spread.rolling(z_window).std()

    upper_band = rolling_mean + entry_z * rolling_std
    lower_band = rolling_mean - entry_z * rolling_std
    exit_upper = rolling_mean + exit_z * rolling_std
    exit_lower = rolling_mean - exit_z * rolling_std

    plt.figure(figsize=(11, 5))
    plt.plot(spread.index, spread.values, color="skyblue", lw=1.0, label="Spread")

    # Entry bands
    plt.plot(upper_band, "r--", lw=1.0, label=f"+{entry_z} STD Entry")
    plt.plot(lower_band, "r--", lw=1.0, label=f"-{entry_z} STD Entry")

    # Exit bands
    plt.plot(exit_upper, "orange", lw=1.0, linestyle="--", alpha=0.7, label=f"+{exit_z} STD Exit")
    plt.plot(exit_lower, "orange", lw=1.0, linestyle="--", alpha=0.7, label=f"-{exit_z} STD Exit")

    # Entry markers
    entry_high = spread[spread > upper_band]
    entry_low = spread[spread < lower_band]
    plt.scatter(entry_high.index, entry_high.values, marker="v", color="green",
                s=60, label=f"Entry (+STD)")
    plt.scatter(entry_low.index, entry_low.values, marker="^", color="green",
                s=60, label=f"Entry (-STD)")

    # Exit markers
    exit_high = spread[(spread < exit_upper) & (spread > rolling_mean)]
    exit_low = spread[(spread > exit_lower) & (spread < rolling_mean)]
    plt.scatter(exit_high.index, exit_high.values, marker="v", color="orange",
                s=50, label=f"Exit (+STD)")
    plt.scatter(exit_low.index, exit_low.values, marker="^", color="orange",
                s=50, label=f"Exit (-STD)")

    plt.title(f"Spread Over Time with Trading Thresholds (entry={entry_z}, exit={exit_z})")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

