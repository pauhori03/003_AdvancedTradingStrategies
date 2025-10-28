# optimize.py

# Tests different parameter combinations for the strategy
# Each combination (entry_z, exit_z, q, r) is evaluated as:
#   1) Run Kalman filter → beta_t and spread
#   2) Generate trading signals based on Z-scores
#   3) Run backtest (includes fees and borrow)
#   4) Score each result using Calmar ratio

import numpy as np
import pandas as pd
from itertools import product

from kalman_filter import run_kalman
from signals import generate_signals
from backtest import run_backtest


def _safe_calmar(equity: pd.Series) -> float:
    """Compute Calmar ratio robustly. Return -inf if not computable."""
    if not isinstance(equity, pd.Series) or len(equity) < 3:
        return float("-inf")
    if not np.isfinite(equity).all() or (equity <= 0).any():
        return float("-inf")

    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = abs(float(dd.min())) if len(dd) else 0.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = max(1e-9, len(equity) / 252.0)
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0

    if mdd <= 0 or not np.isfinite(cagr) or not np.isfinite(mdd):
        return float("-inf")
    return float(cagr / mdd)


def _empty_artifacts() -> dict:
    """Return an empty artifact dict so callers never crash."""
    return {
        "score": float("-inf"),
        "equity": pd.Series(dtype=float),
        "beta": pd.Series(dtype=float),
        "spread": pd.Series(dtype=float),
        "signal": pd.Series(dtype=int),
        "trades": pd.DataFrame(),
        "summary": {}
    }


#  evaluation of one Θ on TRAIN 

def evaluate_theta(
    train_x: pd.Series,
    train_y: pd.Series,
    entry_z: float,
    exit_z: float,
    q: float,
    r: float,
    cash_start: float = 1_000_000.0,
    cash_alloc: float = 0.80,
    z_window: int = 60
) -> dict:
    """
    Tests one combination of parameters on the training data.
    Steps:
      1. Use Kalman filter to estimate dynamic hedge ratio and spread
      2. Generate buy/sell signals using Z-score levels
      3. Backtest to see how equity evolves
      4. Evaluate performance using Calmar ratio
    """
    # Clean and align both price series
    df_xy = pd.concat([train_x.rename("X"), train_y.rename("Y")], axis=1)
    df_xy = df_xy.replace([np.inf, -np.inf], np.nan).dropna()
    if df_xy.empty:
        return _empty_artifacts()

    x = df_xy["X"]
    y = df_xy["Y"]

    # Run Kalman filter
    kres = run_kalman(x, y, q=q, r=r, use_log=True)

    # Basic validation
    if ("spread" not in kres) or ("beta" not in kres):
        return _empty_artifacts()
    if not np.isfinite(kres["spread"]).all() or float(kres["spread"].var()) < 1e-12:
        return _empty_artifacts()
    if not np.isfinite(kres["beta"]).all():
        return _empty_artifacts()

    # Generate signals
    sig = generate_signals(
        kres["spread"],
        entry_z=entry_z,
        exit_z=exit_z,
        z_window=z_window
    )

    # Combine all and drop missing values
    df_all = pd.concat(
        [x.rename("px_x"),
         y.rename("px_y"),
         kres["beta"].rename("beta"),
         sig.rename("signal")],
        axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if df_all.empty:
        return _empty_artifacts()

    # Backtest
    equity, _positions, trades, beta_series, summary = run_backtest(
        px_x=df_all["px_x"],
        px_y=df_all["px_y"],
        beta=df_all["beta"],
        signal=df_all["signal"],
        cash_start=cash_start,
        cash_alloc=cash_alloc
    )

    # Guard equity
    if equity.empty or not np.isfinite(equity).all():
        return _empty_artifacts()

    # Score with Calmar ratio
    score = _safe_calmar(equity)

    return {
        "score": score,
        "equity": equity,
        "beta": beta_series,
        "spread": pd.Series(dtype=float),
        "signal": df_all["signal"],
        "trades": trades,
        "summary": summary
    }


# Full Grid Search on TRAIN

def grid_search_train(
    train_x: pd.Series,
    train_y: pd.Series,
    # Z thresholds: slightly milder to produce trades without overtrading
    grid_entry=(2.0, 2.2, 2.5),
    grid_exit=(0.5, 0.6, 0.7),
    # Kalman noises: larger r for stability, q in a modest range
    grid_q=(1e-8, 1e-7, 1e-6),
    grid_r=(1e-2, 2e-2, 5e-2),
    cash_start: float = 1_000_000.0,
    cash_alloc: float = 0.80,
    z_window: int = 60
) -> dict:
    """
    Runs a grid search over multiple parameter combinations.
    Finds the one with the best Calmar ratio on the training set.
    """
    best = None
    best_artifacts = None

    for ez, xz, qv, rv in product(grid_entry, grid_exit, grid_q, grid_r):
        out = evaluate_theta(
            train_x, train_y,
            entry_z=ez, exit_z=xz, q=qv, r=rv,
            cash_start=cash_start, cash_alloc=cash_alloc, z_window=z_window
        )
        if (best is None) or (out["score"] > best["score"]):
            best = {"entry_z": ez, "exit_z": xz, "q": qv, "r": rv, "score": out["score"]}
            best_artifacts = out

   # Default result if nothing worked
    if best is None:
        best = {"entry_z": 2.0, "exit_z": 0.5, "q": 1e-5, "r": 1e-2, "score": float("-inf")}
        best_artifacts = _empty_artifacts()

    return {"best": best, "artifacts": best_artifacts}
