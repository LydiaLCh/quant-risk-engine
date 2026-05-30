"""
portfolio/backtest.py
---------------------
VaR backtesting: P&L attribution and breach counting.

Regulatory requirement (Basel traffic light test):
  - < 5 breaches in 250 days (green zone) → model acceptable
  - 5-9 breaches (yellow zone) → capital add-on
  - 10+ breaches (red zone) → model rejected

Pass 2 connection: uses var_engine rolling VaR and ES.
Pass 3 stress: run on 2008/2020 data — count how many breaches occur.
"""

import numpy as np
from risk.var_engine import rolling_var, count_breaches, historical_var
from risk.expected_shortfall import expected_shortfall


def basel_traffic_light(breach_count: int) -> dict:
    """
    Basel III traffic light test for VaR model validation.

    Parameters
    ----------
    breach_count : number of VaR breaches over a 250-day backtest window

    Returns
    -------
    dict with zone, action required, and capital multiplier.
    """
    if breach_count <= 4:
        zone = "🟢 Green"
        action = "Model acceptable — no add-on required"
        multiplier = 3.0
    elif breach_count <= 9:
        zone = "🟡 Yellow"
        action = f"Capital add-on required (multiplier increases with breach count)"
        multiplier = 3.0 + 0.2 * (breach_count - 4)
    else:
        zone = "🔴 Red"
        action = "Model rejected — must switch to standardised approach"
        multiplier = 4.0

    return {
        "breach_count":  breach_count,
        "zone":          zone,
        "action":        action,
        "multiplier":    multiplier,
        "capital_charge": f"VaR × {multiplier}",
    }


def full_backtest(returns: np.ndarray, window: int = 250,
                  var_confidence: float = 0.99) -> dict:
    """
    Full VaR backtest over a return series.

    Steps:
    1. Compute rolling VaR (historical simulation).
    2. Count actual breaches.
    3. Apply Basel traffic light test.
    4. Compare realised P&L distribution vs VaR forecast.

    Parameters
    ----------
    returns    : 1D array of daily returns (full period)
    window     : rolling window for VaR estimation
    var_confidence : confidence level

    Returns
    -------
    Comprehensive backtest report dict.
    """
    var_series = rolling_var(returns, window, var_confidence, "historical")
    breach_result = count_breaches(returns, var_series)
    traffic = basel_traffic_light(breach_result["breach_count"])

    losses = -returns
    valid_mask = ~np.isnan(var_series)
    valid_losses = losses[valid_mask]
    valid_var = var_series[valid_mask]

    return {
        "total_days":      len(returns),
        "backtest_days":   breach_result["valid_days"],
        "var_confidence":  var_confidence,
        "window":          window,
        "mean_var":        float(np.nanmean(var_series)),
        "max_loss":        float(losses.max()),
        "breach_count":    breach_result["breach_count"],
        "breach_rate":     breach_result["breach_rate"],
        "expected_rate":   1 - var_confidence,
        "traffic_light":   traffic,
        "breach_indices":  breach_result["breach_indices"],
    }


def pnl_attribution(returns: np.ndarray, var_series: np.ndarray) -> dict:
    """
    P&L attribution relative to VaR.

    Categorises each day's P&L as:
    - Normal profit (positive P&L)
    - Normal loss (loss < VaR)
    - VaR breach (loss > VaR)
    - Extreme breach (loss > 2x VaR)
    """
    losses = -returns
    valid = ~np.isnan(var_series)
    l = losses[valid]
    v = var_series[valid]

    normal_profit  = int(np.sum(l < 0))
    normal_loss    = int(np.sum((l >= 0) & (l < v)))
    var_breach     = int(np.sum((l >= v) & (l < 2 * v)))
    extreme_breach = int(np.sum(l >= 2 * v))

    return {
        "normal_profit":   normal_profit,
        "normal_loss":     normal_loss,
        "var_breach":      var_breach,
        "extreme_breach":  extreme_breach,
        "total_days":      len(l),
    }


# ---------------------------------------------------------------------------
# Pass 3: Where This Model Breaks
# ---------------------------------------------------------------------------
# 1. 2008 backtesting: in Q4 2008, most banks had 15-30 VaR breaches in 250
#    days — far into the Red Zone. This triggered capital add-ons at exactly
#    the wrong time (when banks needed capital most). Basel 2.5 and then
#    FRTB addressed this via stressed ES to ensure capital is adequate pre-crisis.
#
# 2. Clustering of breaches: VaR breaches are not uniformly distributed.
#    They cluster during crises (vol clustering). The Basel test ignores this;
#    Christoffersen's independence test specifically checks for clustering.
#
# 3. P&L comparison problem: the "clean P&L" used for backtesting should
#    exclude fees, reserves, and intraday trading. Banks sometimes use "hypothetical
#    P&L" (freeze the book at close) — FRTB mandates this explicitly.
#
# 4. The traffic light system has perverse incentives: a bank close to the
#    Yellow/Red boundary may be tempted to underestimate VaR to avoid the
#    capital add-on. Regulators use independent model validation to counter this.
