"""
core/greeks.py
--------------
Analytical Greeks for European options under Black-Scholes.

Pass 1 target: all 5 Greeks implemented and tested. Delta plot = sigmoid.
Pass 2 target: feed into portfolio/hedging.py for delta-neutral rebalancing.
Pass 3 stress: use large time steps to see gamma exposure accumulate.
"""

import math
import numpy as np
from scipy.stats import norm
from core.black_scholes import d1, d2


def delta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """
    Delta — sensitivity of option price to underlying price.

    Long call: 0 < delta < 1  (positive, approaches 1 deep ITM)
    Long put:  -1 < delta < 0 (negative, approaches -1 deep ITM)
    Intuition: how much the option moves per $1 move in S.
    """
    _d1 = d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(_d1)
    return norm.cdf(_d1) - 1


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Gamma — rate of change of delta (same for calls and puts).

    Gamma is always positive for long options.
    Peaks at-the-money and near expiry — where the delta changes fastest.
    Intuition: curvature of the option price curve; how fast delta changes.
    """
    _d1 = d1(S, K, T, r, sigma)
    return norm.pdf(_d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Vega — sensitivity to implied volatility (same for calls and puts).

    Returned per 1-unit move in sigma (i.e., per 100% vol move).
    In practice: divide by 100 to get vega per 1% vol move.
    Long options always have positive vega — you benefit from higher vol.
    """
    _d1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(_d1) * math.sqrt(T)


def theta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """
    Theta — sensitivity to time passage (time decay), per year.

    Divide by 365 for daily theta (what you lose each calendar day).
    Long options always have negative theta — time works against buyers.
    """
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(_d1) * sigma) / (2 * math.sqrt(T))
    if option_type == "call":
        term2 = -r * K * math.exp(-r * T) * norm.cdf(_d2)
        return term1 + term2
    term2 = r * K * math.exp(-r * T) * norm.cdf(-_d2)
    return term1 + term2


def rho(S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = "call") -> float:
    """
    Rho — sensitivity to risk-free interest rate.

    Calls have positive rho (higher rates → higher call value).
    Puts have negative rho (higher rates → lower put value).
    Less important than delta/gamma for short-dated options.
    """
    _d2 = d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * math.exp(-r * T) * norm.cdf(_d2)
    return -K * T * math.exp(-r * T) * norm.cdf(-_d2)


def all_greeks(S: float, K: float, T: float, r: float, sigma: float,
               option_type: str = "call") -> dict:
    """Return all 5 Greeks as a dict."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "vega":  vega(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "rho":   rho(S, K, T, r, sigma, option_type),
    }


def delta_surface(S_range: np.ndarray, K: float, T: float, r: float,
                  sigma: float, option_type: str = "call") -> np.ndarray:
    """Vectorised delta over a range of underlying prices. Should look like a sigmoid."""
    return np.array([delta(s, K, T, r, sigma, option_type) for s in S_range])


# ---------------------------------------------------------------------------
# Pass 3: Where This Model Breaks
# ---------------------------------------------------------------------------
# 1. Gamma spikes near expiry and ATM: a gamma-hedged book still requires
#    continuous re-hedging. In practice, discrete rebalancing (daily/weekly)
#    leaves residual gamma risk proportional to (rebalance interval)^2.
#
# 2. Vega is model-dependent: BS vega assumes flat vol. Real books are hedged
#    against the vol surface (vanna, volga) — BS vega alone is insufficient.
#
# 3. Theta/gamma duality: for a delta-hedged position, daily P&L ≈
#    ½Γ(ΔS)² - Θ·Δt. If realised vol > implied vol, you collect; if not,
#    you pay. This is the core of volatility trading.
