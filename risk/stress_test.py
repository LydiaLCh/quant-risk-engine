"""
risk/stress_test.py
-------------------
Stress testing and scenario analysis.

Pass 2 target: apply rates shock via yield_curve.py, recalculate VaR on shocked portfolio.
This is the core of ICAAP / FRTB scenario analysis.

Pass 3 stress: compare stressed vs base VaR — quantify the gap regulators care about.
"""

import numpy as np
from typing import List, Tuple, Dict
from core.yield_curve import apply_parallel_shift, bond_price
from risk.var_engine import historical_var, parametric_var
from risk.expected_shortfall import expected_shortfall


# ---------------------------------------------------------------------------
# Predefined historical scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "2008_crisis": {
        "description": "Global Financial Crisis — Oct 2008 shock",
        "equity_shock":   -0.20,   # -20% equity markets
        "rates_shift_bps": 150,    # +150bps credit spread widening
        "vol_multiplier":  3.0,    # 3x vol spike
    },
    "2020_covid": {
        "description": "COVID-19 crash — March 2020",
        "equity_shock":   -0.34,   # -34% S&P500 peak-to-trough
        "rates_shift_bps": -100,   # -100bps as central banks cut rates
        "vol_multiplier":  4.0,    # VIX spiked to 85
    },
    "rates_up_200": {
        "description": "Regulatory standard: +200bps parallel rates shock",
        "equity_shock":   -0.05,   # mild equity drag
        "rates_shift_bps": 200,    # ICAAP standard shock
        "vol_multiplier":  1.5,
    },
    "rates_down_100": {
        "description": "Rates down 100bps (low-rate environment)",
        "equity_shock":   0.02,
        "rates_shift_bps": -100,
        "vol_multiplier":  1.2,
    },
}


def apply_equity_shock(portfolio_value: float, shock: float) -> float:
    """Apply a multiplicative equity shock. shock=-0.20 means -20%."""
    return portfolio_value * (1 + shock)


def apply_rates_shock(zero_curve: List[Tuple[float, float]],
                      scenario_name: str) -> List[Tuple[float, float]]:
    """
    Apply a named scenario's rates shift to a zero curve.
    Returns the shocked curve.
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Choose from: {list(SCENARIOS.keys())}")
    shift_bps = SCENARIOS[scenario_name]["rates_shift_bps"]
    return apply_parallel_shift(zero_curve, shift_bps)


def stressed_portfolio_var(base_returns: np.ndarray,
                           scenario_name: str,
                           confidence: float = 0.99) -> dict:
    """
    Compute VaR and ES on a portfolio after applying a named stress scenario.

    Simplification: scales the return distribution by the scenario's vol multiplier
    and shifts the mean by the equity shock. In a real system, you'd reprice
    every position under the shocked curves.

    Pass 2 connection: calls var_engine and expected_shortfall on shocked returns.
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    scenario = SCENARIOS[scenario_name]
    vol_mult = scenario["vol_multiplier"]
    eq_shock = scenario["equity_shock"]

    # Scale returns: shift mean by equity shock, multiply vol
    shocked_returns = base_returns * vol_mult + eq_shock / 252  # daily equivalent

    base_var = historical_var(base_returns, confidence)
    stressed_var = historical_var(shocked_returns, confidence)
    base_es = expected_shortfall(base_returns)
    stressed_es = expected_shortfall(shocked_returns)

    return {
        "scenario":         scenario_name,
        "description":      scenario["description"],
        "base_var":         base_var,
        "stressed_var":     stressed_var,
        "var_increase":     stressed_var - base_var,
        "var_multiplier":   stressed_var / base_var if base_var > 0 else None,
        "base_es":          base_es,
        "stressed_es":      stressed_es,
        "es_multiplier":    stressed_es / base_es if base_es > 0 else None,
    }


def bond_portfolio_stress(bonds: List[Dict], zero_curve: List[Tuple[float, float]],
                          scenario_name: str) -> dict:
    """
    Stress test a bond portfolio by repricing under a shocked zero curve.

    Parameters
    ----------
    bonds : list of dicts with keys: face, coupon_rate, maturity, quantity
    zero_curve : base zero curve as list of (maturity, rate) tuples

    Returns
    -------
    dict with base_value, stressed_value, pnl, and pnl_pct.
    """
    shocked_curve = apply_rates_shock(zero_curve, scenario_name)

    base_value = 0.0
    stressed_value = 0.0
    for bond in bonds:
        qty = bond.get("quantity", 1)
        base_value += qty * bond_price(
            bond["face"], bond["coupon_rate"], bond["maturity"], zero_curve
        )
        stressed_value += qty * bond_price(
            bond["face"], bond["coupon_rate"], bond["maturity"], shocked_curve
        )

    pnl = stressed_value - base_value
    return {
        "scenario":       scenario_name,
        "base_value":     base_value,
        "stressed_value": stressed_value,
        "pnl":            pnl,
        "pnl_pct":        pnl / base_value * 100 if base_value > 0 else None,
    }


def run_all_scenarios(base_returns: np.ndarray, confidence: float = 0.99) -> List[dict]:
    """Run all predefined scenarios and return a sorted summary."""
    results = []
    for name in SCENARIOS:
        result = stressed_portfolio_var(base_returns, name, confidence)
        results.append(result)
    return sorted(results, key=lambda x: x["stressed_var"], reverse=True)


# ---------------------------------------------------------------------------
# Pass 3: Where This Model Breaks
# ---------------------------------------------------------------------------
# 1. Return-scaling is a crude approximation: real stress testing reprices
#    every position under shocked curves (full revaluation). Scaling returns
#    by vol_multiplier ignores non-linear payoffs (options, structured products).
#
# 2. Scenario correlations break down in crises: in 2008, equities and credit
#    both crashed while rates fell — correlations went to +1 or -1 suddenly.
#    Static correlation matrices from normal periods are useless in stress.
#
# 3. Liquidity is ignored: a stressed VaR assumes you can exit positions at
#    current prices. In a crisis, bid-ask spreads widen dramatically and
#    liquidity disappears. FRTB's Liquidity Horizons (10, 20, 60 days) address this.
#
# 4. Reverse stress testing: instead of asking "what happens under scenario X?",
#    real desks ask "what scenario would cause us to fail?". This is more
#    informative but computationally harder.
