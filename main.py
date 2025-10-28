# main.py
# Orchestrates the full pipeline for ONE selected pair:
#   - Optimize Θ on TRAIN (grid search)
#   - Evaluate the best Θ on TEST and VALID
#   - Print metrics and save optional artifacts

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from data import download_adj_close, clean_align_panel, chronological_split, TICKERS, START, END
from optimize import grid_search_train, evaluate_theta
from metrics import sharpe, sortino, calmar, max_drawdown, daily_returns, trade_statistics
from plots import (
    plot_equity_curves,
    plot_kalman_beta,
    plot_signals,
    plot_spread_signals
)

# Pair to trade (from cointegration.py results)
X, Y = "BK", "SCHW"  # The Bank of New York Mellon Corporation and Charles Schwab Corporation

# ====== 1) Build panel & splits ======
panel = clean_align_panel(download_adj_close(TICKERS, START, END))
train, test, valid = chronological_split(panel, 0.6, 0.2)
train_x, train_y = train[X], train[Y]
test_x,  test_y  = test[X],  test[Y]
valid_x, valid_y = valid[X], valid[Y]

# ====== 2) Optimize Θ on TRAIN ======
gs = grid_search_train(train_x, train_y)
best = gs["best"]             # dict: entry_z, exit_z, q, r, score (Calmar on TRAIN)
art_train = gs["artifacts"]   # in-sample artifacts for plots (beta/spread/signal/equity/trades/summary)

print("\n[TRAIN] Best Θ found:", best)

# ====== 3) Evaluate Θ* on TEST and VALID ======
cash_start_test = float(art_train["equity"].iloc[-1])

art_test = evaluate_theta(
    test_x, test_y,
    best["entry_z"], best["exit_z"], best["q"], best["r"],
    cash_start=cash_start_test,          #
    cash_alloc=0.80
)

cash_start_valid = float(art_test["equity"].iloc[-1])

art_valid = evaluate_theta(
    valid_x, valid_y,
    best["entry_z"], best["exit_z"], best["q"], best["r"],
    cash_start=cash_start_valid,        
    cash_alloc=0.80
)


# ====== 4) Print metrics ======
def report(tag, art):
    eq = art["equity"]
    print(f"\n[{tag}] Metrics")
    print(f"  Final equity : {eq.iloc[-1]:,.2f}")
    print(f"  Total return : {eq.iloc[-1] / eq.iloc[0] - 1.0: .2%}")
    print(f"  Sharpe       : {sharpe(eq):.3f}")
    print(f"  Sortino      : {sortino(eq):.3f}")
    print(f"  Calmar       : {calmar(eq):.3f}")
    print(f"  MaxDD        : {max_drawdown(eq):.2%}")
    ts = trade_statistics(art["trades"])
    print(f"  Trades       : {ts['flips']} flips (proxy), "
          f"Total commission: {ts['total_commission']:,.2f}")

report("TRAIN (in-sample, FYI)", art_train)
report("TEST (OOS)", art_test)
report("VALID (OOS final)", art_valid)



# Merge full history for plots
# Combine full price and model series (train + test + valid)
px_x_all = pd.concat([train_x, test_x, valid_x])
px_y_all = pd.concat([train_y, test_y, valid_y])

beta_all = pd.concat([
    art_train["beta"],
    art_test["beta"],
    art_valid["beta"]
])

signal_all = pd.concat([
    art_train["signal"],
    art_test["signal"],
    art_valid["signal"]
])

# Merge equity as well for consistency
eq_all = pd.concat([
    art_train["equity"],
    art_test["equity"],
    art_valid["equity"]
])


# (a) Equity TRAIN/TEST/VALID
plot_equity_curves(
    eq_train=art_train["equity"],
    eq_test=art_test["equity"],
    eq_valid=art_valid["equity"],
    title="Equity — TRAIN / TEST / VALID"
)

# (b) Kalman β_t (VALID)
plot_kalman_beta(pd.DataFrame({"beta": beta_all}))

# (c) Signals — full timeline
plot_signals(px_x_all, px_y_all, signal_all)

# (d) Spread with thresholds — full timeline
plot_spread_signals(
    px_x_all,
    px_y_all,
    beta_all,
    entry_z=best["entry_z"],
    exit_z=best["exit_z"],
    z_window=60
)

# --- Combinar todos los trades del histórico ---
trades_all = pd.concat([
    art_train["trades"].assign(period="TRAIN"),
    art_test["trades"].assign(period="TEST"),
    art_valid["trades"].assign(period="VALID")
])

# --- Proxy de retornos por trade (simple pero informativo) ---
trades_all["net_exposure"] = np.abs(trades_all["notional_x"]) + np.abs(trades_all["notional_y"])
trades_all["gross_pnl"] = trades_all["notional_y"] - trades_all["notional_x"] - trades_all["commission"]
trades_all["return_pct"] = trades_all["gross_pnl"] / trades_all["net_exposure"] * 100

# --- Métricas básicas ---
mean_ret = trades_all["return_pct"].mean()
std_ret = trades_all["return_pct"].std()
n_trades = len(trades_all)

print(f"\nTrade Statistics (Full History)")
print(f"  Total trades: {n_trades}")
print(f"  Mean return per trade: {mean_ret:.2f}%")
print(f"  Std. dev of returns: {std_ret:.2f}%")

# --- Histograma global ---
plt.figure(figsize=(8, 4))
plt.hist(trades_all["return_pct"], bins=30, color="skyblue", edgecolor="black")
plt.title("Distribution of Returns per Trade (Full Historical Data)")
plt.xlabel("Return per trade (%)")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- Histograma comparativo por periodo ---
plt.figure(figsize=(9, 4.8))

colors = {"TRAIN": "cornflowerblue", "TEST": "orange", "VALID": "seagreen"}

for period, df in trades_all.groupby("period"):
    plt.hist(df["return_pct"], bins=30, alpha=0.5, label=f"{period} ({len(df)} trades)", color=colors[period])

# Promedio global
plt.axvline(mean_ret, color="red", linestyle="--", linewidth=1.5, label=f"Global Mean = {mean_ret:.2f}%")

# Estética del gráfico
plt.title("Distribution of Returns per Trade — by Period")
plt.xlabel("Return per trade (%)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
