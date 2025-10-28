# kalman_filter.py
# Minimal 2D Kalman filter to estimate dynamic hedge ratio (beta_t) and intercept (alpha_t).
# State:      x_t = [beta_t, alpha_t]^T
# Transition: x_t = x_{t-1} + w_{t-1},  w ~ N(0, Q = q * I)
# Measure:    0 = y_t - beta_t * x_t - alpha_t + v_t,  v ~ N(0, R = r)

import numpy as np
import pandas as pd

def run_kalman(
    px_x: pd.Series,
    px_y: pd.Series,
    q: float = 1e-6,      # process-noise variance (how fast beta/alpha can move)
    r: float = 1e-2,      # observation-noise variance (confidence in spread)
    x0_beta: float = 1.0,
    x0_alpha: float = 0.0,
    p0: float = 1e-2,     # initial covariance scale
    eps: float = 1e-12,   # tiny floor to avoid divisions by ~0
    use_log: bool = True,
    beta_cap: float = 10.0  # safety cap for beta
) -> pd.DataFrame:
    """Return DataFrame with columns ['beta','alpha','spread'] aligned to px_x.index."""
    assert list(px_x.index) == list(px_y.index)
    idx = px_x.index

    # --- Initialization (this is your "KalmanConfig") ---
    state = np.array([x0_beta, x0_alpha], dtype=float)  # [beta, alpha]
    p_cov = np.eye(2) * max(p0, eps)                     # state covariance
    q_cov = np.eye(2) * max(q, eps)                      # process covariance
    r_var = float(max(r, eps))                           # measurement variance (scalar)
    f_mat = np.eye(2)                                    

    out_beta, out_alpha, out_spread = [], [], []
    last_good_state = state.copy()

    # --- Recursion over time ---
    for xv, yv in zip(px_x.values, px_y.values):
        # Optionally stabilize with log-prices (consistent con tu proyecto)
        x_val = float(xv)
        y_val = float(yv)
        if use_log:
            x_val = np.log(max(x_val, 1e-8))
            y_val = np.log(max(y_val, 1e-8))

        # 1) Predict
        x_pred = f_mat @ state
        p_pred = f_mat @ p_cov @ f_mat.T + q_cov

        # 2) Observation model for z_t = 0 with h_t = [-x_t, -1]
        h_row = np.array([[-x_val, -1.0]], dtype=float)  # shape (1,2)
        resid = y_val - (x_pred[0] * x_val + x_pred[1])  # predicted residual
        if not np.isfinite(resid):
            resid = 0.0  # guard

        # 3) Innovation & Kalman gain (guard innovation variance)
        s_innov = float(h_row @ p_pred @ h_row.T + r_var)
        if (not np.isfinite(s_innov)) or (s_innov < eps):
            s_innov = eps
        k_gain = (p_pred @ h_row.T) / s_innov            # (2x1)

        # 4) Update (add tiny ridge so P stays positive-definite)
        x_upd = x_pred + k_gain.flatten() * resid
        p_upd = (np.eye(2) - k_gain @ h_row) @ p_pred
        p_upd += np.eye(2) * 1e-12

        # Fallback if numbers explode
        if not (np.all(np.isfinite(x_upd)) and np.all(np.isfinite(p_upd))):
            state = last_good_state.copy()
            p_cov = p_cov + np.eye(2) * 1e-9
        else:
            state = x_upd
            state[0] = float(np.clip(state[0], -beta_cap, beta_cap))  # clip beta
            p_cov = p_upd
            last_good_state = state.copy()

        # Output series
        beta, alpha = float(state[0]), float(state[1])
        spread = y_val - (beta * x_val + alpha)
        if not np.isfinite(spread):
            spread = 0.0

        out_beta.append(beta)
        out_alpha.append(alpha)
        out_spread.append(spread)

    return pd.DataFrame({"beta": out_beta, "alpha": out_alpha, "spread": out_spread}, index=idx)

