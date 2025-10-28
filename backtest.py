# backtest.py
# Step 4 — Simple backtest engine for pairs (market-neutral)
# - Inputs: prices X/Y, dynamic hedge beta_t, and signal in {-1,0,+1}
# - Costs:
#     * Commission: 0.125% per leg on every trade
#     * Borrow: 0.25% annual (charged daily) on short notional
# - Sizing: invest 80% of equity when in a position
# - Returns: equity curve (Series), positions (DataFrame), trades (DataFrame), summary (dict)

import numpy as np
import pandas as pd

def run_backtest(
    px_x: pd.Series,
    px_y: pd.Series,
    beta: pd.Series,
    signal: pd.Series,
    cash_start: float = 1_000_000.0,
    cash_alloc: float = 0.80,
    fee_rate: float = 0.00125,      # 0.125%
    borrow_annual: float = 0.0025,  # 0.25% annual
    days_in_year: int = 252
):
    # Numeric safeguards
    price_floor = 1e-6   # never divide by (almost) zero price
    beta_cap    = 50.0   # cap hedge ratio to a reasonable range

    # Basic checks and common index
    assert list(px_x.index) == list(px_y.index) == list(beta.index) == list(signal.index)
    idx = px_x.index

    # State
    cash = cash_start
    qty_x = 0.0
    qty_y = 0.0

    # Logs
    equity_vals = []
    pos_rows = []
    trade_rows = []

    EPS_TRADE = 1e-8  # ignore tiny re-hedges due to float noise
    

    for t, dt in enumerate(idx):
        s = int(signal.iloc[t])     # -1, 0, +1
        b = float(beta.iloc[t])     # hedge ratio
        b = float(np.clip(b, -beta_cap, beta_cap))
        x = max(float(px_x.iloc[t]), price_floor)     # price X
        y = max(float(px_y.iloc[t]), price_floor)     # price Y

        # Borrow cost on yesterday's short leg
        short_notional = 0.0
        if qty_y < 0:
            short_notional += qty_y * y
        if qty_x < 0:
            short_notional += qty_x * x
        if short_notional != 0:
            cash -= abs(short_notional) * (borrow_annual / days_in_year)

        # Target position based on signal (self-financed, fixed gross = deploy)
        equity_pre = cash + qty_x * x + qty_y * y
        deploy = cash_alloc * equity_pre
        # split gross between legs in proportion 1 : |b|
        den = (abs(b) + 1.0)
        k = deploy / den                     # dollars assigned to the Y leg (in abs value)

        if s == 0:
            tgt_x, tgt_y = 0.0, 0.0

        elif s == +1:
            # short spread: short Y, long beta*X
            # |notional_y| = k, |notional_x| = k*|b|
            tgt_y = - k / y
            tgt_x = + (k * b) / x            # sign(b) respected automatically

        else:  # s == -1
            # long spread: long Y, short beta*X
            tgt_y = + k / y
            tgt_x = - (k * b) / x


        # Trades to reach target
        dqx = tgt_x - qty_x
        dqy = tgt_y - qty_y
        if abs(dqx) < EPS_TRADE: dqx = 0.0
        if abs(dqy) < EPS_TRADE: dqy = 0.0

        notional_x = dqx * x
        notional_y = dqy * y

        # Commission on both legs
        commission = abs(notional_x) * fee_rate + abs(notional_y) * fee_rate
        cash -= commission

        # Cash impact of trades (buy reduces cash, sell increases)
        cash -= (notional_x + notional_y)

        # Update positions
        qty_x += dqx
        qty_y += dqy

        # End-of-day equity
        equity = cash + qty_x * x + qty_y * y
        equity_vals.append(equity)

        # Logs
        trade_rows.append({
            "date": dt, "signal": s,
            "d_qty_x": dqx, "d_qty_y": dqy,
            "price_x": x, "price_y": y,
            "notional_x": notional_x, "notional_y": notional_y,
            "commission": commission
        })
        pos_rows.append({"date": dt, "qty_x": qty_x, "qty_y": qty_y})

    # Pack outputs
    equity_series = pd.Series(equity_vals, index=idx, name="equity")
    positions = pd.DataFrame(pos_rows).set_index("date")
    trades = pd.DataFrame(trade_rows).set_index("date")

    summary = {
        "final_equity": float(equity_series.iloc[-1]),
        "total_return": float(equity_series.iloc[-1] / equity_series.iloc[0] - 1.0),
        "n_trades": int((trades["d_qty_x"].abs() + trades["d_qty_y"].abs() > 0).sum()),
        "total_commission": float(trades["commission"].sum())
    }

    return equity_series, positions, trades, summary