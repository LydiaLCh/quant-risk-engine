"""
risk/var_engine.py
------------------
Value at Risk: Historical Simulation and Parametric (variance-covariance) methods.

Pass 1 target: historical simulation VaR on real-ish return data. 99% 1-day VaR ~2-3%.
Pass 2 target: MC VaR using core/monte_carlo.py — compare with historical.
Pass 3 stress: use 2008/2020 crisis window — watch VaR severely underestimate tail risk.
"""

import numpy as np
from typing import Optional


def historical_var(returns: np.ndarray, confidence: float = 0.99,
                   horizon: int = 1) -> float:
    """
    Historical Simulation VaR.

    Steps:
    1. Sort daily returns ascending.
    2. VaR = the (1 - confidence) quantile of the loss distribution.
    3. Scale to horizon using the square-root-of-time rule.

    Parameters
    ----------
    returns    : 1D array of daily P&L returns (as decimals, e.g. -0.02 for -2%)
    confidence : VaR confidence level (default 0.99 = 99%)
    horizon    : holding period in days (default 1)

    Returns
    -------
    VaR as a positive number representing the loss threshold.
    """
    losses = -returns  # flip sign: losses are positive
    var_1day = float(np.quantile(losses, confidence))
    return var_1day * np.sqrt(horizon)


def parametric_var(returns: np.ndarray, confidence: float = 0.99,
                   horizon: int = 1) -> float:
    """
    Parametric (Variance-Covariance) VaR.

    Assumes returns are normally distributed: VaR = mu - z * sigma.
    For 99% confidence, z = 2.326. For 95%, z = 1.645.

    Parameters
    ----------
    returns    : 1D array of daily returns
    confidence : confidence level
    horizon    : holding period in days

    Returns
    -------
    VaR as a positive number.
    """
    from scipy.stats import norm
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(confidence)
    var_1day = -(mu - z * sigma)
    return float(var_1day * np.sqrt(horizon))


def mc_var(S0: float, K: float, r: float, sigma: float, T: float,
           confidence: float = 0.99, n_paths: int = 10_000,
           seed: Optional[int] = 42) -> dict:
    """
    Monte Carlo VaR using the GBM simulator from core/monte_carlo.py.
    Pass 2 connection: replaces historical simulation with full simulation.

    Simulates 1-day returns under GBM and computes VaR from the simulated
    loss distribution. Compare with historical_var to see model divergence
    under fat-tailed regimes.

    Returns both MC VaR and the simulated return distribution.
    """
    from core.monte_carlo import simulate_gbm
    dt = 1 / 252  # 1 trading day
    paths = simulate_gbm(S0, r, sigma, dt, n_steps=1, n_paths=n_paths, seed=seed)
    simulated_returns = (paths[:, -1] - S0) / S0
    var = historical_var(simulated_returns, confidence, horizon=1)
    return {
        "var":      var,
        "returns":  simulated_returns,
        "mean":     simulated_returns.mean(),
        "std":      simulated_returns.std(),
    }


def rolling_var(returns: np.ndarray, window: int = 250,
                confidence: float = 0.99, method: str = "historical") -> np.ndarray:
    """
    Rolling VaR over a sliding window. Useful for backtesting (Pass 3).

    Parameters
    ----------
    window : rolling window in days (250 = ~1 trading year)
    method : 'historical' or 'parametric'
    """
    n = len(returns)
    var_series = np.full(n, np.nan)
    for i in range(window, n):
        window_returns = returns[i - window:i]
        if method == "historical":
            var_series[i] = historical_var(window_returns, confidence)
        else:
            var_series[i] = parametric_var(window_returns, confidence)
    return var_series


def count_breaches(returns: np.ndarray, var_series: np.ndarray) -> dict:
    """
    Backtest VaR: count how often actual losses exceed the VaR estimate.
    Regulatory expectation: <1% of days for 99% VaR (Basel traffic light test).

    Returns
    -------
    dict with breach_count, breach_rate, and breach_indices.
    """
    losses = -returns
    breaches = losses > var_series
    breach_count = int(np.nansum(breaches))
    valid_days = int(np.sum(~np.isnan(var_series)))
    return {
        "breach_count":   breach_count,
        "breach_rate":    breach_count / valid_days if valid_days > 0 else 0,
        "breach_indices": np.where(breaches)[0].tolist(),
        "valid_days":     valid_days,
    }


# ---------------------------------------------------------------------------
# Pass 3: Where This Model Breaks
# ---------------------------------------------------------------------------
# 1. Crisis windows (2008, 2020): historical VaR trained on calm data will
#    underestimate losses in a crisis by a factor of 3-5x. The 250-day window
#    simply doesn't contain enough extreme events.
#
# 2. Normality assumption (parametric): real returns have fat tails (kurtosis > 3)
#    and negative skew. A normal distribution massively underestimates the
#    probability of -5%, -10% days. Use t-distribution or EVT instead.
#
# 3. Square-root-of-time scaling: VaR(h) = VaR(1) * sqrt(h) assumes iid returns.
#    Real returns have autocorrelation (momentum, vol clustering). This scaling
#    is baked into Basel but is known to be wrong.
#
# 4. Non-subadditivity: VaR(A+B) can exceed VaR(A) + VaR(B) — it's not a
#    coherent risk measure. This is the core reason regulators moved to ES (CVaR).
