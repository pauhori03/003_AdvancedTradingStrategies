# metrics.py
# Purpose:
#   - Compute performance metrics for equity curves and basic trade stats

from __future__ import annotations
import numpy as np
import pandas as pd

def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def sharpe(equity: pd.Series, rf_daily: float = 0.0) -> float:
    r = daily_returns(equity) - rf_daily
    sd = r.std()
    return 0.0 if sd == 0 else float(np.sqrt(252) * r.mean() / sd)

def sortino(equity: pd.Series) -> float:
    r = daily_returns(equity)
    downside = r[r < 0]
    sd = downside.std()
    return 0.0 if sd == 0 else float(np.sqrt(252) * r.mean() / sd)

def calmar(equity: pd.Series) -> float:
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = max(1e-9, len(equity) / 252.0)
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0
    mdd = abs(max_drawdown(equity))
    return 0.0 if mdd == 0 else float(cagr / mdd)

def trade_statistics(trades: pd.DataFrame) -> dict:
    # Count regime changes in 'signal' as a proxy for round-trips
    s = trades["signal"].astype(int)
    flips = (s != s.shift(1)).fillna(s != 0)
    return {
        "flips": int(flips.sum()),
        "avg_commission": float(trades["commission"].mean()) if len(trades) else 0.0,
        "total_commission": float(trades["commission"].sum()) if len(trades) else 0.0
    }
