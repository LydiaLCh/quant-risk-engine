"""
portfolio/optimiser.py
----------------------
Mean-variance portfolio optimisation (Markowitz).

Pass 1 target: min-variance portfolio + efficient frontier plotted.
Pass 3 stress: use small dataset / highly correlated assets — covariance instability.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.optimize import minimize


def portfolio_stats(weights: np.ndarray, returns: np.ndarray,
                    cov_matrix: np.ndarray, annualise: bool = True) -> Tuple[float, float, float]:
    """
    Compute expected return, volatility, and Sharpe ratio for a set of weights.

    Parameters
    ----------
    weights    : asset weights (must sum to 1)
    returns    : 1D array of mean daily returns per asset
    cov_matrix : daily covariance matrix
    annualise  : scale to annual figures (x252 for return, x sqrt(252) for vol)

    Returns
    -------
    (expected_return, volatility, sharpe_ratio)
    """
    scale = 252 if annualise else 1
    port_return = float(np.dot(weights, returns) * scale)
    port_vol    = float(np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(scale))
    sharpe      = port_return / port_vol if port_vol > 0 else 0.0
    return port_return, port_vol, sharpe


def min_variance_portfolio(returns_matrix: np.ndarray,
                           allow_short: bool = False) -> dict:
    """
    Find the minimum variance portfolio weights.

    Parameters
    ----------
    returns_matrix : 2D array of shape (n_days, n_assets)
    allow_short    : if False, enforce long-only (weights >= 0)

    Returns
    -------
    dict with weights, expected_return, volatility, sharpe.
    """
    n_assets = returns_matrix.shape[1]
    mean_returns = returns_matrix.mean(axis=0)
    cov = np.cov(returns_matrix, rowvar=False)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = None if allow_short else [(0, 1)] * n_assets
    w0 = np.ones(n_assets) / n_assets

    result = minimize(
        lambda w: portfolio_stats(w, mean_returns, cov)[1],  # minimise vol
        w0, method="SLSQP", bounds=bounds, constraints=constraints
    )

    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")

    weights = result.x
    ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov)

    return {
        "weights":          weights.tolist(),
        "expected_return":  ret,
        "volatility":       vol,
        "sharpe":           sharpe,
    }


def max_sharpe_portfolio(returns_matrix: np.ndarray, rf: float = 0.05,
                         allow_short: bool = False) -> dict:
    """
    Find the tangency (maximum Sharpe ratio) portfolio.

    Parameters
    ----------
    rf : annualised risk-free rate for Sharpe calculation
    """
    n_assets = returns_matrix.shape[1]
    mean_returns = returns_matrix.mean(axis=0)
    cov = np.cov(returns_matrix, rowvar=False)

    def neg_sharpe(w):
        ret, vol, _ = portfolio_stats(w, mean_returns, cov)
        return -(ret - rf) / vol if vol > 0 else 0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = None if allow_short else [(0, 1)] * n_assets
    w0 = np.ones(n_assets) / n_assets

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    weights = result.x
    ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov)

    return {
        "weights":         weights.tolist(),
        "expected_return": ret,
        "volatility":      vol,
        "sharpe":          sharpe - rf / vol if vol > 0 else 0,
    }


def efficient_frontier(returns_matrix: np.ndarray, n_points: int = 50,
                       allow_short: bool = False) -> dict:
    """
    Trace the efficient frontier by minimising vol for each target return level.

    Returns arrays of (volatility, return) pairs for plotting.
    """
    n_assets = returns_matrix.shape[1]
    mean_returns = returns_matrix.mean(axis=0)
    cov = np.cov(returns_matrix, rowvar=False)

    # Return range: from min-var portfolio to max-return portfolio
    min_ret = min_variance_portfolio(returns_matrix)["expected_return"]
    max_ret = float(mean_returns.max() * 252)
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_vols   = []
    frontier_rets   = []
    frontier_weights = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_stats(w, mean_returns, cov)[0] - t},
        ]
        bounds = None if allow_short else [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(
            lambda w: portfolio_stats(w, mean_returns, cov)[1],
            w0, method="SLSQP", bounds=bounds, constraints=constraints
        )
        if result.success:
            w = result.x
            ret, vol, _ = portfolio_stats(w, mean_returns, cov)
            frontier_vols.append(vol)
            frontier_rets.append(ret)
            frontier_weights.append(w.tolist())

    return {
        "volatilities":    frontier_vols,
        "returns":         frontier_rets,
        "weights":         frontier_weights,
    }


# ---------------------------------------------------------------------------
# Pass 3: Where This Model Breaks
# ---------------------------------------------------------------------------
# 1. Covariance matrix instability: with n_assets > n_days/5, the sample
#    covariance matrix becomes ill-conditioned or singular. Optimiser weights
#    become extreme and unstable. Fix: shrinkage (Ledoit-Wolf), factor models.
#
# 2. Estimation error in returns: mean returns estimated from historical data
#    have huge standard errors. A 1% error in expected return shifts the
#    optimal weights dramatically. The optimiser amplifies estimation noise.
#
# 3. Highly correlated assets: when assets are nearly perfectly correlated,
#    the covariance matrix is near-singular. The optimiser finds extreme long/short
#    positions that cancel out. Garbage in, garbage out.
#
# 4. No transaction costs or turnover constraints: real portfolios add L1/L2
#    penalties for turnover and position limits. This frontier assumes frictionless
#    rebalancing — which becomes expensive in practice.
