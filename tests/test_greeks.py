"""Tests for core/greeks.py — Pass 1 verification."""
import pytest
from core.greeks import delta, gamma, vega, theta, rho, all_greeks

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def test_call_delta_between_0_and_1():
    d = delta(S, K, T, r, sigma, "call")
    assert 0 < d < 1


def test_put_delta_between_minus1_and_0():
    d = delta(S, K, T, r, sigma, "put")
    assert -1 < d < 0


def test_call_put_delta_sum():
    """Delta(call) - Delta(put) = 1 (from put-call parity differentiation)."""
    dc = delta(S, K, T, r, sigma, "call")
    dp = delta(S, K, T, r, sigma, "put")
    assert abs(dc - dp - 1.0) < 1e-8


def test_gamma_positive():
    """Gamma is always positive for long options."""
    assert gamma(S, K, T, r, sigma) > 0


def test_vega_positive():
    """Vega is positive for long options — higher vol = higher price."""
    assert vega(S, K, T, r, sigma) > 0


def test_theta_negative_call():
    """Theta is negative for long calls — time works against buyers."""
    assert theta(S, K, T, r, sigma, "call") < 0


def test_rho_positive_call():
    """Rho positive for calls — higher rates benefit calls."""
    assert rho(S, K, T, r, sigma, "call") > 0


def test_rho_negative_put():
    """Rho negative for puts."""
    assert rho(S, K, T, r, sigma, "put") < 0


def test_all_greeks_keys():
    g = all_greeks(S, K, T, r, sigma, "call")
    assert set(g.keys()) == {"delta", "gamma", "vega", "theta", "rho"}
