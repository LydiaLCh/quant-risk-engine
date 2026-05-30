"""
core/yield_curve.py
-------------------
Zero curve bootstrapping, duration, convexity, and DV01.

Pass 1 target: bootstrap a zero curve from bond prices; spot rates increase with maturity.
Pass 2 target: feed into risk/stress_test.py for rates shock scenarios.
Pass 3 stress: apply inverted curve — check duration/convexity still work correctly.
"""

import numpy as np
from typing import List, Tuple


def bootstrap_zero_curve(maturities: List[float],
                         coupon_rates: List[float],
                         prices: List[float],
                         face: float = 100.0) -> List[Tuple[float, float]]:
    """
    Bootstrap a zero (spot) rate curve from coupon bond prices.

    Works sequentially: uses already-known zero rates to strip coupons,
    solving for the next zero rate at each maturity.

    Parameters
    ----------
    maturities   : list of bond maturities in years (must be sorted ascending)
    coupon_rates : annual coupon rates as decimals (e.g., 0.05 for 5%)
    prices       : dirty prices of each bond
    face         : face value (default 100)

    Returns
    -------
    List of (maturity, zero_rate) tuples
    """
    zero_curve = []

    for i, (T, c, P) in enumerate(zip(maturities, coupon_rates, prices)):
        coupon = face * c  # annual coupon payment
        n_periods = int(round(T))  # assume annual coupons for simplicity

        # Subtract present value of intermediate coupons using known zero rates
        pv_coupons = 0.0
        for t in range(1, n_periods):
            # Find the zero rate for this intermediate maturity (linear interpolation)
            z = _interp_zero(t, zero_curve)
            pv_coupons += coupon * np.exp(-z * t)

        # Final cash flow: coupon + face
        final_cf = coupon + face
        remaining_pv = P - pv_coupons

        # Solve: remaining_pv = final_cf * exp(-z_T * T)
        if remaining_pv <= 0:
            raise ValueError(f"Negative remaining PV at maturity {T} — check inputs.")
        z_T = -np.log(remaining_pv / final_cf) / T
        zero_curve.append((T, z_T))

    return zero_curve


def _interp_zero(t: float, zero_curve: List[Tuple[float, float]]) -> float:
    """Linear interpolation of zero rate at time t from known curve."""
    if not zero_curve:
        return 0.0
    mats = [z[0] for z in zero_curve]
    rates = [z[1] for z in zero_curve]
    return float(np.interp(t, mats, rates))


def bond_price(face: float, coupon_rate: float, T: float,
               zero_curve: List[Tuple[float, float]]) -> float:
    """
    Price a bond using the bootstrapped zero curve.
    Assumes annual coupon payments.
    """
    coupon = face * coupon_rate
    n = int(round(T))
    pv = 0.0
    for t in range(1, n + 1):
        cf = coupon if t < n else coupon + face
        z = _interp_zero(t, zero_curve)
        pv += cf * np.exp(-z * t)
    return pv


def macaulay_duration(face: float, coupon_rate: float, T: float,
                      ytm: float) -> float:
    """
    Macaulay duration: weighted average time of cash flows.
    Weights = PV(CF_t) / Price.
    """
    coupon = face * coupon_rate
    n = int(round(T))
    pv_total = 0.0
    weighted_time = 0.0
    for t in range(1, n + 1):
        cf = coupon if t < n else coupon + face
        pv = cf * np.exp(-ytm * t)
        pv_total += pv
        weighted_time += t * pv
    return weighted_time / pv_total


def modified_duration(face: float, coupon_rate: float, T: float,
                      ytm: float) -> float:
    """
    Modified duration = Macaulay duration / (1 + ytm).
    Gives the % price change per unit change in yield.
    """
    mac = macaulay_duration(face, coupon_rate, T, ytm)
    return mac / (1 + ytm)


def convexity(face: float, coupon_rate: float, T: float, ytm: float) -> float:
    """
    Convexity — second-order correction to duration.
    Bonds gain more from yield falls than they lose from rises (positive convexity).
    """
    coupon = face * coupon_rate
    n = int(round(T))
    pv_total = 0.0
    conv = 0.0
    for t in range(1, n + 1):
        cf = coupon if t < n else coupon + face
        pv = cf * np.exp(-ytm * t)
        pv_total += pv
        conv += t * (t + 1) * pv
    return conv / (pv_total * (1 + ytm) ** 2)


def dv01(face: float, coupon_rate: float, T: float, ytm: float) -> float:
    """
    DV01 (Dollar Value of 1bp): price change for a 1bp (0.0001) move in yield.
    The trader's daily risk metric for rates books.
    """
    price_up   = sum(
        (face * coupon_rate if t < int(T) else face * coupon_rate + face)
        * np.exp(-(ytm + 0.0001) * t)
        for t in range(1, int(T) + 1)
    )
    price_down = sum(
        (face * coupon_rate if t < int(T) else face * coupon_rate + face)
        * np.exp(-(ytm - 0.0001) * t)
        for t in range(1, int(T) + 1)
    )
    return (price_down - price_up) / 2


def apply_parallel_shift(zero_curve: List[Tuple[float, float]],
                         shift_bps: float) -> List[Tuple[float, float]]:
    """
    Apply a parallel shift (in bps) to the entire zero curve.
    Used by stress_test.py for +200bps / -100bps shock scenarios.
    """
    shift = shift_bps / 10_000
    return [(mat, rate + shift) for mat, rate in zero_curve]


# ---------------------------------------------------------------------------
# Pass 3: Where This Model Breaks
# ---------------------------------------------------------------------------
# 1. Inverted curve: bootstrapping assumes rates increase with maturity (normal
#    curve). An inverted curve (2s10s negative) is valid but requires the
#    interpolation to handle declining rates — check for negative forwards.
#
# 2. Annual coupon assumption: real bonds pay semi-annually. The bootstrap
#    needs frequency adjustment. DV01 and duration are materially different
#    for semi-annual vs annual when ytm is high.
#
# 3. Duration linearity: duration ≈ price sensitivity only for small yield moves.
#    For a +200bps shock, convexity correction is essential:
#    ΔP/P ≈ -ModDur * Δy + ½ * Convexity * (Δy)²
#
# 4. No credit spread: this curve assumes risk-free rates. Real desk curves
#    include OIS, LIBOR/SOFR basis, and issuer spread — all bootstrapped separately.
