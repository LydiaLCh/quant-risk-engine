"""
risk/expected_shortfall.py
--------------------------
Expected Shortfall (CVaR) — the expected loss given that the loss exceeds VaR.

ES is sub-additive, captures tail risk, and is the FRTB/Basel IV standard (at 97.5%).
Key assertion: ES >= VaR always. Test this.

Pass 1 target: ES from simulated losses. Assert ES > VaR in unit tests.
Pass 2 target: ES sits on top of var_engine.py — same return arrays, deeper metric.
Pass 3 stress: compare ES and VaR on 2008 returns — ES grows much faster in crises.
"""

import numpy as np
from risk.var_engine import historical_var


def expected_shortfall(returns: np.ndarray, confidence: float = 0.975,
                       horizon: int = 1) -> float:
    """
    Historical Expected Shortfall (CVaR).

    ES = average of losses that exceed the VaR threshold.
    Basel IV uses 97.5% ES (not 99% VaR) as the internal models standard.

    Parameters
    ----------
    returns    : 1D array of daily returns (decimals)
    confidence : ES confidence level (FRTB standard = 0.975)
    horizon    : holding period in days

    Returns
    -------
    ES as a positive number.
    """
    losses = -returns
    threshold = np.quantile(losses, confidence)
    tail_losses = losses[losses >= threshold]
    if len(tail_losses) == 0:
        return float(threshold)
    es_1day = float(tail_losses.mean())
    return es_1day * np.sqrt(horizon)


def parametric_es(returns: np.ndarray, confidence: float = 0.975,
                  horizon: int = 1) -> float:
    """
    Parametric ES under normality assumption.

    ES = mu + sigma * phi(z) / (1 - confidence)
    where phi is the standard normal PDF and z is the confidence quantile.
    """
    from scipy.stats import norm
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(confidence)
    es_1day = -(mu - sigma * norm.pdf(z) / (1 - confidence))
    return float(es_1day * np.sqrt(horizon))


def var_es_comparison(returns: np.ndarray,
                      var_confidence: float = 0.99,
                      es_confidence: float = 0.975) -> dict:
    """
    Side-by-side VaR vs ES comparison on the same return series.

    Both are comparable under FRTB: 99% VaR ≈ 97.5% ES for normal returns.
    Under fat tails / crisis data, ES will be materially higher.

    Returns
    -------
    dict with var, es, ratio (ES/VaR), and tail_size.
    """
    losses = -returns
    var = historical_var(returns, var_confidence)
    es = expected_shortfall(returns, es_confidence)
    tail = losses[losses >= np.quantile(losses, es_confidence)]

    return {
        "var_99":      var,
        "es_975":      es,
        "es_var_ratio": es / var if var > 0 else None,
        "tail_size":   len(tail),
        "tail_mean":   float(tail.mean()),
        "tail_max":    float(tail.max()),
    }


def stressed_es(returns: np.ndarray, stress_window: tuple,
                confidence: float = 0.975) -> dict:
    """
    Compute ES over a stressed historical window (e.g., 2008 or 2020 crash).

    FRTB requires a Stressed ES (SES) using the worst 12-month window
    for the bank's current portfolio.

    Parameters
    ----------
    stress_window : (start_idx, end_idx) indices into the returns array

    Returns
    -------
    dict comparing full-period ES vs stressed-window ES.
    """
    full_es = expected_shortfall(returns, confidence)
    stressed_returns = returns[stress_window[0]:stress_window[1]]
    stress_es = expected_shortfall(stressed_returns, confidence)
    return {
        "full_period_es": full_es,
        "stressed_es":    stress_es,
        "stress_multiplier": stress_es / full_es if full_es > 0 else None,
    }


# ---------------------------------------------------------------------------
# Pass 3: Where This Model Breaks
# ---------------------------------------------------------------------------
# 1. ES is harder to backtest than VaR: you cannot directly observe whether
#    the average tail loss was correct — you need Acerbi-Szekely tests or
#    similar. Regulators acknowledge this but prefer ES for its coherence.
#
# 2. Estimation error in the tail: ES uses only the worst (1 - confidence)%
#    of observations. With 250 days of data and 97.5% confidence, that's
#    only ~6 observations. The estimate is noisy.
#
# 3. Stressed ES amplification: in 2008, a 97.5% ES on a 12-month crisis
#    window can be 5-10x the normal-period ES. This is intentional under
#    FRTB — it ensures capital buffers reflect worst-case, not average, risk.
#
# 4. Sub-additivity holds in theory but can fail in practice with finite
#    samples and heavy-tailed distributions. Always verify ES(A+B) <= ES(A) + ES(B).
