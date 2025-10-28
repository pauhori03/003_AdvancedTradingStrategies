# signals.py
# Creates trading signals from the spread using a rolling Z-score.
# Rules:
#   +1 = short the spread (sell Y, buy X)
#   -1 = long the spread  (buy Y, sell X)
#    0 = flat (no position) when spread is near its mean



import numpy as np
import pandas as pd


def rolling_zscore(
    x: pd.Series,
    window: int = 60,
    min_periods: int = 30,
    sd_floor: float = 1e-8  # minimum std to avoid divide-by-zero
) -> pd.Series:
    """
    Calculates a rolling Z-score = (x - mean) / std.
    Adds a small lower limit so the standard deviation never becomes zero.
    """
    mu = x.rolling(window=window, min_periods=min_periods).mean()
    sd = x.rolling(window=window, min_periods=min_periods).std()
    sd = sd.clip(lower=sd_floor)
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def generate_signals(
    spread: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    z_window: int = 60,
    min_periods: int = 30,
    cooldown: int = 5,     # wait a few days before opening a new trade
    min_hold: int = 3      # stay in a position at least this long
) -> pd.Series:
    """
    Turns the spread into trading signals based on its Z-score.
    - Go short when the spread moves far above the mean.
    - Go long when it moves far below the mean.
    - Close (flat) when it comes back close to the mean.
    Cooldown avoids re-entering immediately after an exit.
    """
    # 1) Rolling Z-score
    z = rolling_zscore(spread, window=z_window, min_periods=min_periods)

    # 2) Build the signal step by step
    sig = pd.Series(0, index=z.index, dtype=int)

    last_side = 0        # -1, 0, +1
    last_change = None   # index (integer) where side last changed
    cd_left = 0          # cooldown bars remaining

    for i, dt in enumerate(z.index):
        zt   = z.iloc[i]
        prev = z.iloc[i-1] if i > 0 else 0.0

        # Reduce cooldown each step
        if cd_left > 0:
            cd_left -= 1

        # --- Exit rule: go flat when near mean ---
        if last_side != 0 and abs(zt) < exit_z:
            if last_change is None or (i - last_change) >= min_hold:
                last_side = 0
                last_change = i
                cd_left = cooldown

        # --- Entry rules (only if not cooling down) ---
        if last_side == 0 and cd_left == 0:
            # Cross above lower bound → short spread
            if (prev <= -entry_z) and (zt > -entry_z):
                last_side = +1
                last_change = i
            # Cross below upper bound → long spread
            elif (prev >= entry_z) and (zt < entry_z):
                last_side = -1
                last_change = i

        sig.iloc[i] = last_side

    return sig
