# 003_AdvancedTradingStrategies

# Advanced Trading Strategies — Kalman Filter Pairs Trading

This project implements a **pairs trading strategy** using the **Kalman Filter** for dynamic hedge ratio estimation.  
It combines econometric analysis (cointegration tests) with sequential Bayesian updating to maintain a market-neutral position between two correlated assets.

The workflow includes:
- Data download and cleaning
- Correlation and cointegration screening
- Kalman filter–based dynamic hedge estimation
- Signal generation (Z-score thresholds)
- Backtesting with transaction and borrowing costs
- Performance evaluation (Sharpe, Sortino, Calmar, Max Drawdown)

## Folder Structure

003_AdvancedTradingStrategies/
│
├── .gitignore                                # Ignore virtual environments, caches, local backups
├── 003_ADVANCED_TRADING_STRATEGIES_PAPER.pdf # Final written report (executive summary)
├── APPENDIX_RESULTS.xlsx                     # Appendix with detailed metrics and trade results
├── LICENSE                                   # MIT License file
├── README.md                                 # Project documentation
├── backtest.py                               # Backtesting engine: positions, trades, and equity evolution
├── cointegration.py                          # Engle–Granger cointegration testing for pair validation
├── corr_screen.py                            # Correlation screening for pair candidates
├── data.py                                   # Data download, cleaning, and alignment
├── kalman_filter.py                          # Kalman Filter module for dynamic hedge ratio β_t estimation
├── main.py                                   # Pipeline orchestration (TRAIN / TEST / VALID runs)
├── metrics.py                                # Performance metrics (Sharpe, Sortino, Calmar, Max Drawdown)
├── optimize.py                               # Parameter grid search for optimal entry/exit thresholds
├── plots.py                                  # Visualization utilities (equity, β_t, signals)
├── requirements.txt                          # Python dependencies for environment setup
├── signals.py                                # Signal generation based on spread Z-score logic
├── equity.png                                # Sample equity curve (TRAIN / TEST / VALID visualization)
└── APPENDIX_RESULTS.xlsx                     # Quantitative appendix used for written report

## Methodology Summary

1. **Pair Selection**
   - High-correlation pairs screened using Pearson correlation.
   - Cointegration tested with the Engle–Granger method.
   - Selected pair: *PepsiCo (PEP) vs Nestlé (NSRGY)*.

2. **Dynamic Hedge Estimation**
   - Kalman Filter estimates time-varying β_t.
   - State-space model:
     - Observation: y_t = β_t x_t + ε_t
     - Transition: β_t = β_{t-1} + η_t
   - Q and R matrices tuned during training.

3. **Signal Generation**
   - Z-score of spread used to define entry/exit thresholds.
   - Long spread: z < -entry_z  
     Short spread: z > +entry_z  
     Exit when |z| < exit_z

4. **Backtesting**
   - Market-neutral positions (long one leg, short the other).
   - Commission: 0.125% per leg  
     Borrow cost: 0.25% annualized

5. **Evaluation**
   - Metrics: Sharpe, Sortino, Calmar, Max Drawdown
   - Segmented into Train / Test / Validation

## How to Run

Follow these steps to execute the full trading strategy pipeline:

### 1. Setup the Environment
Create and activate a virtual environment (recommended):

python3 -m venv .venv
source .venv/bin/activate    # On macOS/Linux
.venv\Scripts\activate     # On Windows

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Project
python main.py

## Outputs

The program will:

- Train and optimize parameters on the TRAIN dataset
- Evaluate performance on TEST and VALIDATION datasets
- Print metrics in the console (Sharpe, Sortino, Calmar, MaxDD, etc.)
- Generate and display plots:
- Equity curve (TRAIN / TEST / VALID)
- Kalman βₜ dynamics
- Entry/exit signal visualization
- Spread signals





