"""
risk/cva_engine.py
------------------
Credit Valuation Adjustment (CVA) — simplified implementation.

CVA = cost of counterparty default risk on an OTC derivative.
Reduces the value of a derivative by the expected loss from the counterparty defaulting.

CVA = integral over time of: DF(t) * EE(t) * PD(t) * LGD

Pass 2 connection: uses exposure profiles from core/monte_carlo.py.
Pass 3 stress: crank up PD or LGD and observe CVA impact on book value.
"""

import numpy as np
from typing import Optional
from core.monte_carlo import exposure_profile


def compute_cva(S0: float, K: float, T: float, r: float, sigma: float,
                pd_annual: float = 0.01, lgd: float = 0.6,
                n_steps: int = 52, n_paths: int = 5_000,
                seed: Optional[int] = 42) -> dict:
    """
    Compute CVA using Monte Carlo exposure profiles.

    CVA ≈ LGD * sum_t [ DF(t) * EE(t) * PD(t, t+dt) ]

    where:
      DF(t)       = discount factor at time t
      EE(t)       = Expected Exposure at time t (from MC simulation)
      PD(t, t+dt) = marginal default probability over [t, t+dt]
      LGD         = Loss Given Default (1 - Recovery Rate)

    Parameters
    ----------
    pd_annual : annual probability of default (e.g., 0.01 = 1% p.a.)
    lgd       : loss given default (e.g., 0.6 = 60%; recovery = 40%)

    Returns
    -------
    dict with cva, percentage of clean price, and exposure profile.
    """
    # Get EPE profile from Monte Carlo
    profile = exposure_profile(S0, K, T, r, sigma, n_steps, n_paths, seed)
    time_grid = profile["time"]
    epe = profile["epe"]
    dt = profile["dt"]

    # Marginal default probability per period (from annual PD, assuming constant hazard)
    hazard_rate = -np.log(1 - pd_annual)  # h such that P(default in 1yr) = pd_annual
    marginal_pd = (1 - np.exp(-hazard_rate * dt))  # PD over each interval dt

    # Discount factors
    discount_factors = np.exp(-r * time_grid)

    # CVA = LGD * sum[ DF(t) * EE(t) * marginal_PD ]
    cva = lgd * float(np.sum(discount_factors[1:] * epe[1:] * marginal_pd))

    # Clean price (undiscounted, unhedged call approximation)
    from core.black_scholes import bs_call
    clean_price = bs_call(S0, K, T, r, sigma)

    return {
        "cva":              cva,
        "clean_price":      clean_price,
        "cva_adjusted_price": clean_price - cva,
        "cva_pct_of_price": cva / clean_price * 100 if clean_price > 0 else None,
        "pd_annual":        pd_annual,
        "lgd":              lgd,
        "time_grid":        time_grid.tolist(),
        "epe_profile":      epe.tolist(),
    }


def dva(S0: float, K: float, T: float, r: float, sigma: float,
        own_pd_annual: float = 0.005, lgd: float = 0.6,
        n_steps: int = 52, n_paths: int = 5_000,
        seed: Optional[int] = 42) -> dict:
    """
    Debt Valuation Adjustment (DVA) — the benefit from your OWN default risk.

    DVA = CVA from the counterparty's perspective on your liabilities.
    Controversial: recognising a gain from deteriorating own credit quality
    is counter-intuitive and discouraged under IFRS 13.

    Returns DVA using the same framework as CVA.
    """
    result = compute_cva(S0, K, T, r, sigma, own_pd_annual, lgd, n_steps, n_paths, seed)
    return {
        "dva":               result["cva"],
        "own_pd_annual":     own_pd_annual,
        "lgd":               lgd,
        "dva_pct_of_price":  result["cva_pct_of_price"],
    }


def bilateral_cva(S0: float, K: float, T: float, r: float, sigma: float,
                  counterparty_pd: float = 0.01, own_pd: float = 0.005,
                  lgd: float = 0.6, n_steps: int = 52,
                  n_paths: int = 5_000) -> dict:
    """
    Bilateral CVA = CVA - DVA.

    This is the net XVA adjustment: the true cost of counterparty risk
    accounting for both parties' default risk.
    """
    cva_result = compute_cva(S0, K, T, r, sigma, counterparty_pd, lgd, n_steps, n_paths, seed=42)
    dva_result  = dva(S0, K, T, r, sigma, own_pd, lgd, n_steps, n_paths, seed=42)

    bilateral = cva_result["cva"] - dva_result["dva"]
    clean = cva_result["clean_price"]

    return {
        "cva":              cva_result["cva"],
        "dva":              dva_result["dva"],
        "bilateral_cva":   bilateral,
        "clean_price":     clean,
        "adjusted_price":  clean - bilateral,
        "net_pct":         bilateral / clean * 100 if clean > 0 else None,
    }


# ---------------------------------------------------------------------------
# Pass 3: Where This Model Breaks
# ---------------------------------------------------------------------------
# 1. Wrong-way risk: this model assumes independence between the counterparty's
#    default probability and the exposure. In reality, if the counterparty
#    defaults when the trade is most in-your-favour (e.g., a commodity producer
#    defaulting when commodity prices are low), correlation makes CVA much larger.
#
# 2. Constant hazard rate: real PD term structures are steeply upward-sloping
#    (short-term PD << long-term PD). Using a flat hazard rate underestimates
#    CVA for short-dated trades and overestimates for long-dated ones.
#
# 3. DVA controversy: when your own credit deteriorates, DVA creates a "gain"
#    that inflates reported P&L. In 2011, several banks reported large profits
#    from DVA as their credit spreads widened during the eurozone crisis.
#    IFRS 9 now requires OCA (own credit adjustment) rather than full DVA.
#
# 4. No collateral/CSA: most institutional OTC derivatives are collateralised
#    under a Credit Support Annex (CSA). Collateral dramatically reduces
#    exposure and CVA. This model assumes zero collateral — a conservative overestimate.
