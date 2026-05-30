"""Tests for core/black_scholes.py — Pass 1 verification."""
import pytest
from core.black_scholes import bs_call, bs_put, implied_vol


S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2  # ATM, 1yr, 5% rate, 20% vol


def test_call_positive():
    assert bs_call(S, K, T, r, sigma) > 0


def test_put_positive():
    assert bs_put(S, K, T, r, sigma) > 0


def test_put_call_parity():
    """C - P = S - K*e^(-rT)"""
    import math
    C = bs_call(S, K, T, r, sigma)
    P = bs_put(S, K, T, r, sigma)
    lhs = C - P
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8, f"Put-call parity failed: {lhs} != {rhs}"


def test_deep_itm_call_approaches_intrinsic():
    """Deep ITM call ≈ S - K*e^(-rT)."""
    import math
    C = bs_call(200, 100, 1.0, 0.05, 0.2)
    intrinsic = 200 - 100 * math.exp(-0.05)
    assert abs(C - intrinsic) < 1.0


def test_deep_otm_call_near_zero():
    """Deep OTM call has nearly zero value."""
    C = bs_call(50, 200, 1.0, 0.05, 0.2)
    assert C < 0.01


def test_implied_vol_roundtrip():
    """Recover sigma from bs_call price via implied_vol."""
    price = bs_call(S, K, T, r, sigma)
    iv = implied_vol(price, S, K, T, r, "call")
    assert abs(iv - sigma) < 1e-4, f"Implied vol {iv:.4f} != {sigma}"


def test_zero_time_raises():
    with pytest.raises(ValueError):
        bs_call(S, K, 0, r, sigma)
