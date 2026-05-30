"""
portfolio/hedging.py
--------------------
Delta and gamma hedging rebalancing loop.

Pass 1 target: discrete delta rebalancing loop + P&L tracker. P&L near zero for small moves.
Pass 2 target: use Greeks from core/greeks.py. P&L ≈ ½ × Γ × (ΔS)² — verify this.
Pass 3 stress: use weekly rebalancing — watch gamma exposure accumulate.
"""

import numpy as np
from typing import Optional, List
from core.black_scholes import bs_call
from core.greeks import delta as bs_delta, gamma as bs_gamma


def delta_hedge_simulation(S0: float, K: float, T: float, r: float, sigma: float,
                            n_steps: int = 252, seed: Optional[int] = 42) -> dict:
    """
    Simulate discrete delta hedging of a short call position.

    Strategy:
      - Sell 1 call at t=0 (receive premium)
      - At each rebalancing step, hold -delta units of the underlying
        to remain delta-neutral
      - Track cumulative P&L: should be near zero for perfect hedging

    The residual P&L is the gamma P&L: ½ × Γ × (ΔS)²
    If realised vol > implied vol: positive P&L (collect).
    If realised vol < implied vol: negative P&L (pay).

    Parameters
    ----------
    n_steps : number of rebalancing periods (252 = daily, 52 = weekly, 12 = monthly)

    Returns
    -------
    dict with S_path, delta_path, pnl_path, cumulative_pnl, and gamma_pnl_approx
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)

    # Simulate underlying path via GBM
    S = np.zeros(n_steps + 1)
    S[0] = S0
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal()
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Initialise: sell 1 call, hold -delta_0 shares
    pnl = np.zeros(n_steps + 1)
    deltas = np.zeros(n_steps + 1)
    gammas = np.zeros(n_steps + 1)

    # Cash account: receive call premium + invest
    call_premium = bs_call(S[0], K, T, r, sigma)
    cash = call_premium  # cash from selling the call

    tau = T  # time remaining
    delta_t = bs_delta(S[0], K, tau, r, sigma, "call")
    share_holding = -delta_t  # short delta_t shares to delta-hedge
    cash -= share_holding * S[0]  # buy/sell shares at t=0

    deltas[0] = delta_t
    gammas[0] = bs_gamma(S[0], K, tau, r, sigma)

    for t in range(1, n_steps + 1):
        tau = max(T - times[t], 1e-8)

        # Grow cash at risk-free rate
        cash *= np.exp(r * dt)

        # New delta
        new_delta = bs_delta(S[t], K, tau, r, sigma, "call") if tau > 1e-6 else float(S[t] > K)
        new_gamma = bs_gamma(S[t], K, tau, r, sigma) if tau > 1e-6 else 0.0

        # Rebalance: buy/sell (new_delta - old_delta) shares
        trade = -(new_delta - delta_t)
        cash -= trade * S[t]

        share_holding = -new_delta
        delta_t = new_delta
        deltas[t] = new_delta
        gammas[t] = new_gamma

    # Final P&L: liquidate share holding, settle call payoff
    cash += share_holding * S[-1]  # sell remaining shares
    cash -= max(S[-1] - K, 0)      # pay call payoff if exercised

    # Gamma P&L approximation: sum of ½ Γ (ΔS)²  - Θ·Δt
    dS = np.diff(S)
    gamma_pnl = 0.5 * gammas[:-1] * dS**2

    return {
        "S_path":           S.tolist(),
        "delta_path":       deltas.tolist(),
        "gamma_path":       gammas.tolist(),
        "final_pnl":        float(cash),
        "gamma_pnl_approx": float(gamma_pnl.sum()),
        "n_steps":          n_steps,
        "rebalance_freq":   f"every {T/n_steps*252:.1f} trading day(s)",
    }


def gamma_pnl_formula(gamma: float, dS: float) -> float:
    """
    The gamma P&L formula: ½ × Γ × (ΔS)²

    This is the core of P&L attribution on a delta-hedged options desk.
    If you collect more gamma P&L than you pay in theta, you profit.
    This is 'long gamma' trading.
    """
    return 0.5 * gamma * dS ** 2


def compare_rebalancing_frequencies(S0: float, K: float, T: float,
                                     r: float, sigma: float,
                                     freqs: List[int] = [252, 52, 12, 4],
                                     seed: int = 42) -> List[dict]:
    """
    Run the delta hedge simulation at different rebalancing frequencies.

    Pass 3 stress: weekly/monthly rebalancing leaves large residual P&L
    because gamma exposure accumulates between rebalances.

    Parameters
    ----------
    freqs : list of n_steps to simulate (252=daily, 52=weekly, 12=monthly, 4=quarterly)
    """
    results = []
    for n in freqs:
        result = delta_hedge_simulation(S0, K, T, r, sigma, n_steps=n, seed=seed)
        results.append({
            "n_steps":    n,
            "freq_label": result["rebalance_freq"],
            "final_pnl":  result["final_pnl"],
            "gamma_pnl":  result["gamma_pnl_approx"],
        })
    return results


# ---------------------------------------------------------------------------
# Pass 3: Where This Model Breaks
# ---------------------------------------------------------------------------
# 1. Weekly rebalancing: gamma exposure accumulates over 5 trading days.
#    The residual P&L is ½ Γ (ΔS_week)² — for a volatile stock this can
#    easily be 10-50x the daily equivalent. Real desks hedge daily or intraday.
#
# 2. Transaction costs: each rebalance involves trading costs (bid-ask spread,
#    market impact). More frequent rebalancing reduces gamma risk but increases
#    transaction costs. The optimal frequency balances these two.
#
# 3. Discrete vs continuous: the BS model assumes continuous rebalancing.
#    Any discretisation introduces hedging error. This error is path-dependent —
#    a large single move is worse than many small moves of the same total size.
#
# 4. Gamma sign: if you are SHORT gamma (sold options), every large move
#    hurts you. Short gamma positions are the primary source of tail risk
#    for options market-makers who don't hedge frequently enough.
