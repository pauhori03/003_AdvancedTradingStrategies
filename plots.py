# plots.py
# Minimal plotting helpers for the project.

import matplotlib.pyplot as plt
import pandas as pd

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
    """Overlay entry/exit signals on the price series."""
    plt.figure(figsize=(10, 4))
    plt.plot(px_x, label="Asset X", color="dodgerblue")
    plt.plot(px_y, label="Asset Y", color="coral", alpha=0.8)

    # Mark signals
    buy_signals = signal[signal == -1].index
    sell_signals = signal[signal == 1].index
    plt.scatter(buy_signals, px_y.loc[buy_signals], color="green", label="Long spread", marker="^")
    plt.scatter(sell_signals, px_y.loc[sell_signals], color="red", label="Short spread", marker="v")

    plt.title("Pair Prices with Trading Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
