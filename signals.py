# signals.py
# Generate trading signals from the Kalman spread using a rolling Z-score.
# Policy:
#   +1 = short spread  (short Y, long beta*X)  when  Z >  entry_z
#   -1 = long  spread  ( long Y, short beta*X) when  Z < -entry_z
#    0 = flat          (exit)                  when |Z| < exit_z


import numpy as np
import pandas as pd


def rolling_zscore(
    x: pd.Series,
    window: int = 60,
    min_periods: int = 30,
    sd_floor: float = 1e-8  # <-- NEW: minimum std to avoid divide-by-zero
) -> pd.Series:
    """
    Compute a rolling Z-score: (x - rolling_mean) / rolling_std.
    Adds a small floor to the std so we never divide by ~0.
    """
    mu = x.rolling(window=window, min_periods=min_periods).mean()
    sd = x.rolling(window=window, min_periods=min_periods).std()
    sd = sd.clip(lower=sd_floor)  # <-- NEW: guard against zero/near-zero std
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def generate_signals(
    spread: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    z_window: int = 60,
    min_periods: int = 30,
    cooldown: int = 5,     # do not re-enter for N bars after closing a trade
    min_hold: int = 3      # keep a position at least N bars before allowing exit
) -> pd.Series:
    """
    Convert the Kalman spread into {-1, 0, +1} signals using *crossovers*:
      +1 (short spread) when Z crosses UP through -entry_z
      -1 (long  spread) when Z crosses DOWN through +entry_z
       0 (flat)        when |Z| < exit_z
    Cooldown avoids re-entries right after an exit; min_hold avoids micro-flips.
    """
    # 1) Rolling Z-score of the spread
    z = rolling_zscore(spread, window=z_window, min_periods=min_periods)

    # 2) Build a persistent, stateful signal
    sig = pd.Series(0, index=z.index, dtype=int)

    last_side = 0        # -1, 0, +1
    last_change = None   # index (integer) where side last changed
    cd_left = 0          # cooldown bars remaining

    for i, dt in enumerate(z.index):
        zt   = z.iloc[i]
        prev = z.iloc[i-1] if i > 0 else 0.0

        # Decrement cooldown if any
        if cd_left > 0:
            cd_left -= 1

        # --- Exit rule: inside the neutral band -> go flat (but respect min_hold) ---
        if last_side != 0 and abs(zt) < exit_z:
            if last_change is None or (i - last_change) >= min_hold:
                last_side = 0
                last_change = i
                cd_left = cooldown

        # --- Entry rules: ONLY on crossovers, and only if not in cooldown ---
        if last_side == 0 and cd_left == 0:
            # Cross UP through -entry_z  => short spread (+1)
            if (prev <= -entry_z) and (zt > -entry_z):
                last_side = +1
                last_change = i
            # Cross DOWN through +entry_z => long spread (-1)
            elif (prev >= entry_z) and (zt < entry_z):
                last_side = -1
                last_change = i

        sig.iloc[i] = last_side

    return sig
