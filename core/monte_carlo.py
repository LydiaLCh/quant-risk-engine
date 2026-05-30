"""
core/monte_carlo.py
-------------------
Monte Carlo simulation engine using Geometric Brownian Motion (GBM).

Pass 1 target: European call price converges to BS as N → ∞.
Pass 2 target: supply exposure profiles to risk/cva_engine.py.
Pass 3 stress: use N=100 — observe slow convergence, motivate variance reduction.
"""

import numpy as np
from typing import Callable, Optional


def simulate_gbm(S0: float, mu: float, sigma: float, T: float,
                 n_steps: int, n_paths: int,
                 seed: Optional[int] = None) -> np.ndarray:
    """
    Simulate GBM paths using the exact log-normal discretisation.

        S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Parameters
    ----------
    S0      : initial price
    mu      : drift (use risk-free rate r for risk-neutral pricing)
    sigma   : annualised volatility
    T       : total time horizon in years
    n_steps : number of time steps
    n_paths : number of simulated paths
    seed    : random seed for reproducibility

    Returns
    -------
    paths : np.ndarray of shape (n_paths, n_steps + 1)
             paths[:, 0] == S0 for all paths
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    Z = np.random.standard_normal((n_paths, n_steps))
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )
    return paths


def price_european(S0: float, K: float, T: float, r: float, sigma: float,
                   option_type: str = "call", n_steps: int = 252,
                   n_paths: int = 10_000, seed: Optional[int] = 42,
                   antithetic: bool = False) -> dict:
    """
    Price a European option via Monte Carlo.

    Parameters
    ----------
    antithetic : bool
        Use antithetic variates for variance reduction.
        For each Z, also simulate -Z. Halves variance roughly.

    Returns
    -------
    dict with keys: price, std_error, ci_low, ci_high
    """
    if antithetic:
        half = n_paths // 2
        paths_pos = simulate_gbm(S0, r, sigma, T, n_steps, half, seed)
        if seed is not None:
            np.random.seed(seed)
        dt = T / n_steps
        Z = np.random.standard_normal((half, n_steps))
        paths_neg = np.zeros((half, n_steps + 1))
        paths_neg[:, 0] = S0
        for t in range(1, n_steps + 1):
            paths_neg[:, t] = paths_neg[:, t - 1] * np.exp(
                (r - 0.5 * sigma ** 2) * dt - sigma * np.sqrt(dt) * Z[:, t - 1]
            )
        terminal = np.concatenate([paths_pos[:, -1], paths_neg[:, -1]])
    else:
        paths = simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed)
        terminal = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(terminal - K, 0)
    else:
        payoffs = np.maximum(K - terminal, 0)

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    se = discounted.std() / np.sqrt(len(discounted))

    return {
        "price":    price,
        "std_error": se,
        "ci_low":   price - 1.96 * se,
        "ci_high":  price + 1.96 * se,
    }


def price_asian(S0: float, K: float, T: float, r: float, sigma: float,
                option_type: str = "call", n_steps: int = 252,
                n_paths: int = 10_000, seed: Optional[int] = 42) -> dict:
    """
    Price an Asian (arithmetic average) option — path-dependent, no BS closed form.

    Payoff: max(avg(S) - K, 0) for a call.
    """
    paths = simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed)
    avg_prices = paths[:, 1:].mean(axis=1)  # exclude S0

    if option_type == "call":
        payoffs = np.maximum(avg_prices - K, 0)
    else:
        payoffs = np.maximum(K - avg_prices, 0)

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    se = discounted.std() / np.sqrt(n_paths)

    return {
        "price":     price,
        "std_error": se,
        "ci_low":    price - 1.96 * se,
        "ci_high":   price + 1.96 * se,
    }


def exposure_profile(S0: float, K: float, T: float, r: float, sigma: float,
                     n_steps: int = 52, n_paths: int = 5_000,
                     seed: Optional[int] = 42) -> dict:
    """
    Compute Expected Positive Exposure (EPE) profile over time.
    Used by risk/cva_engine.py to calculate CVA.

    Returns time grid and EPE at each time step.
    """
    paths = simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed)
    dt = T / n_steps
    time_grid = np.linspace(0, T, n_steps + 1)

    # Exposure at each time step = max(intrinsic value, 0) — simplified
    epe = np.maximum(paths - K, 0).mean(axis=0)

    return {"time": time_grid, "epe": epe, "dt": dt}


# ---------------------------------------------------------------------------
# Pass 3: Where This Model Breaks
# ---------------------------------------------------------------------------
# 1. N=100 paths: standard error ∝ 1/√N. With N=100, SE ≈ 10x larger than
#    N=10,000. The 95% CI will barely contain the true price. Variance
#    reduction (antithetic, control variates) is not optional for real use.
#
# 2. GBM assumes constant vol: same weakness as BS. Fat tails, vol clustering,
#    and jumps (Merton jump-diffusion) are not captured.
#
# 3. Discretisation error: even the exact log-normal scheme has step-size error
#    for path-dependent payoffs (barriers, Asians). Finer n_steps = more accurate
#    but slower. This is a compute/accuracy trade-off desks manage daily.
#
# 4. Seed sensitivity: always set a seed for reproducibility in tests. In
#    production, quasi-random (Sobol) sequences are used instead of pseudo-random.
