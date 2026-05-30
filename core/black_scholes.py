"""
core/black_scholes.py
---------------------
Black-Scholes European option pricer.

Pass 1 target: bs_call / bs_put working and verified against put-call parity.
Pass 3 stress: price near-expiry and deep-OTM options to expose the volatility smile.
"""

import math
from scipy.stats import norm


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """d1 term in the Black-Scholes formula."""
    if T <= 0:
        raise ValueError("Time to expiry T must be > 0")
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """d2 = d1 - sigma * sqrt(T). N(d2) = risk-neutral prob of exercise."""
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European call price.

    Parameters
    ----------
    S     : current underlying price
    K     : strike price
    T     : time to expiry in years
    r     : continuously compounded risk-free rate
    sigma : annualised volatility

    Returns
    -------
    Call option price (float)
    """
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(_d1) - K * math.exp(-r * T) * norm.cdf(_d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European put price (via put-call parity).

    C - P = S - K*e^(-rT)  =>  P = C - S + K*e^(-rT)
    """
    return bs_call(S, K, T, r, sigma) - S + K * math.exp(-r * T)


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                option_type: str = "call", tol: float = 1e-6, max_iter: int = 200) -> float:
    """
    Implied volatility via Newton-Raphson bisection fallback.

    Finds sigma such that bs_call/put(sigma) == market_price.
    """
    # bounds
    low, high = 1e-6, 10.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = bs_call(S, K, T, r, mid) if option_type == "call" else bs_put(S, K, T, r, mid)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2


# ---------------------------------------------------------------------------
# Pass 3: Where This Model Breaks
# ---------------------------------------------------------------------------
# 1. Volatility smile / skew: BS assumes constant sigma, but real implied vols
#    vary by strike (smile) and expiry (term structure). Deep OTM puts trade at
#    much higher implied vol than ATM — the model cannot explain this.
#
# 2. Near-expiry instability: as T→0 the d1/d2 terms blow up. Gamma and vega
#    spike; delta becomes a step function. Continuous hedging is impossible.
#
# 3. Log-normal assumption: real returns have fat tails and negative skew.
#    BS underestimates the probability of large moves — exactly the events
#    that risk management exists to handle.
#
# 4. Constant risk-free rate: rates are stochastic. For long-dated options
#    this is a material error (see: Hull-White, Black-Karasinski).
#
# 5. No dividends / transaction costs: the model assumes a frictionless world.
#    Real hedging costs erode P&L significantly.
