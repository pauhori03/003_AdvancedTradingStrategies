# cointegration.py
# Purpose:
#   - Confirm cointegration among screened pairs using Engle–Granger on TRAIN only
#   - Project rules:
#       (1) Each price series is NON-stationary (ADF p > 0.05)
#       (2) Residual from OLS(S1 ~ 1 + S2) IS stationary (ADF p_res < 0.05)


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tabulate import tabulate
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Defaults for __main__
USE_LOG = True
ALPHA   = 0.05
CORR_WIN = 252      # for reporting mean corr in table (same window as screen)
TOP_K   = 10


def engle_granger_check(df_two_cols: pd.DataFrame, alpha: float = 0.05, use_log: bool = True) -> dict:
    """
    Engle–Granger cointegration check:
      (1) Individual series are NON-stationary in levels: ADF p > alpha
      (2) Residual of OLS(S1 ~ 1 + S2) IS stationary: ADF p_res < alpha

    Returns a dict with ADF p-values, OLS params (w0, w1), and pass boolean.
    """
    if df_two_cols.shape[1] != 2:
        raise ValueError("Expect exactly 2 columns (two assets).")

    col1, col2 = df_two_cols.columns.tolist()
    df = df_two_cols.dropna().astype(float)

    s1, s2 = df[col1].copy(), df[col2].copy()
    if use_log:
        s1, s2 = np.log(s1), np.log(s2)

    p1 = adfuller(s1.values, regression="c", autolag="AIC")[1]
    p2 = adfuller(s2.values, regression="c", autolag="AIC")[1]

    X = sm.add_constant(s2.values)
    model = sm.OLS(s1.values, X).fit()
    w0, w1 = model.params
    resid = s1.values - (w0 + w1 * s2.values)
    p_res = adfuller(resid, regression="n", autolag="AIC")[1]

    cond_nonstat = (p1 > alpha) and (p2 > alpha)
    cond_res_stat = (p_res < alpha)

    return {
        "pair": (col1, col2),
        "ADF_p_S1": float(p1),
        "ADF_p_S2": float(p2),
        "ADF_p_residuals": float(p_res),
        "w0": float(w0),
        "w1": float(w1),
        "pass": bool(cond_nonstat and cond_res_stat)
    }


def rank_passing_pairs(
    train_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    alpha: float = 0.05,
    use_log: bool = True,
    top_k: int = 10,
    corr_win_for_report: int = 252
) -> pd.DataFrame:
    """
    Run Engle–Granger on correlation-screened candidates only.
    Rank passing pairs by (low ADF_p_res first, then high mean rolling corr).
    Returns a DataFrame with top_k passing pairs and metrics.
    """
    rows = []
    for _, r in candidates_df.iterrows():
        a, b = r["asset1"], r["asset2"]
        out = engle_granger_check(train_df[[a, b]], alpha=alpha, use_log=use_log)
        if out["pass"]:
            mean_rc = float(train_df[a].rolling(corr_win_for_report).corr(train_df[b]).mean())
            out["mean_rolling_corr_train"] = mean_rc
            rows.append(out)

    if not rows:
        return pd.DataFrame(columns=[
            "pair", "mean_rolling_corr_train", "ADF_p_S1", "ADF_p_S2",
            "ADF_p_residuals", "w0", "w1", "pass"
        ])

    passed = pd.DataFrame(rows)
    passed = passed.sort_values(["ADF_p_residuals", "mean_rolling_corr_train"],
                                ascending=[True, False])
    return passed.head(top_k).reset_index(drop=True)


if __name__ == "__main__":
    # Standalone path: build TRAIN, run corr screen, then Engle–Granger ranking
    from data import download_adj_close, clean_align_panel, chronological_split, TICKERS, START, END
    from corr_screen import screen_pairs_by_corr, USE_LOG as USE_LOG_CORR, CORR_WIN, CORR_THRES

    panel = clean_align_panel(download_adj_close(TICKERS, START, END))
    train, test, valid = chronological_split(panel, 0.6, 0.2)

    candidates, _ = screen_pairs_by_corr(train, CORR_WIN, CORR_THRES, USE_LOG_CORR)
    top_passing = rank_passing_pairs(train, candidates, alpha=ALPHA, use_log=USE_LOG, top_k=TOP_K, corr_win_for_report=CORR_WIN)

    if top_passing.empty:
        print("\nNo pair passed Engle–Granger at α=0.05. "
              "Consider lowering CORR_THRES or raising ALPHA.")
    else:
        print(f"\nEngle–Granger PASS on TRAIN (top {len(top_passing)}):")

        cols = ["pair", "mean_rolling_corr_train", "ADF_p_S1", "ADF_p_S2",
                "ADF_p_residuals", "w0", "w1", "pass"]
        dfp = top_passing[cols].copy()

        # Make 'pair' printable
        dfp["pair"] = dfp["pair"].apply(lambda p: f"{p[0]}–{p[1]}")
        for c in ["mean_rolling_corr_train", "ADF_p_S1", "ADF_p_S2", "w0", "w1"]:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce").round(3)
        dfp["ADF_p_residuals"] = pd.to_numeric(dfp["ADF_p_residuals"], errors="coerce").round(4)

        # Print table 
        print(tabulate(dfp, headers="keys", tablefmt="github"))


