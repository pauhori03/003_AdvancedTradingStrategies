# main.py
# Orchestrates the full pipeline for ONE selected pair:
#   - Optimize Θ on TRAIN (grid search)
#   - Evaluate the best Θ on TEST and VALID
#   - Print metrics and save optional artifacts

import pandas as pd
import matplotlib.pyplot as plt
from data import download_adj_close, clean_align_panel, chronological_split, TICKERS, START, END
from optimize import grid_search_train, evaluate_theta
from metrics import sharpe, sortino, calmar, max_drawdown, daily_returns, trade_statistics
from plots import (
    plot_spread_with_zbands,
    plot_beta,
    plot_equity_curves,
    plot_trade_return_hist,
)

# Pair to trade (from cointegration.py results)
X, Y = "PEP", "NSRGY"  # PepsiCo vs. Nestle

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
art_test  = evaluate_theta(test_x,  test_y, best["entry_z"], best["exit_z"], best["q"], best["r"],
                           cash_start=1_000_000.0, cash_alloc=0.50)
art_valid = evaluate_theta(valid_x, valid_y, best["entry_z"], best["exit_z"], best["q"], best["r"],
                           cash_start=1_000_000.0, cash_alloc=0.50)


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

# 1) Spread + Z-bands (usa los Θ óptimos encontrados en TRAIN)
entry_z = best["entry_z"]
exit_z  = best["exit_z"]

plot_spread_with_zbands(art_train, entry_z=entry_z, exit_z=exit_z, z_window=60,
                        title="TRAIN — Spread & Z-bands")
plot_beta(art_train, title="TRAIN — Hedge ratio β_t")

plot_spread_with_zbands(art_test, entry_z=entry_z, exit_z=exit_z, z_window=60,
                        title="TEST — Spread & Z-bands")
plot_beta(art_test, title="TEST — Hedge ratio β_t")

plot_spread_with_zbands(art_valid, entry_z=entry_z, exit_z=exit_z, z_window=60,
                        title="VALID — Spread & Z-bands")
plot_beta(art_valid, title="VALID — Hedge ratio β_t")

# 2) Equity curves overlay
plot_equity_curves(art_train["equity"], art_test["equity"], art_valid["equity"],
                   title="Equity — TRAIN / TEST / VALID")

# 3) Trade return distribution (muestra por periodo)
plot_trade_return_hist(art_train, title="TRAIN — Per-trade returns")
plot_trade_return_hist(art_test,  title="TEST — Per-trade returns")
plot_trade_return_hist(art_valid, title="VALID — Per-trade returns")

plt.show()  # <- show all figures at the end


