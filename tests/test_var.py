"""Tests for risk/var_engine.py and risk/expected_shortfall.py."""
import numpy as np
import pytest
from risk.var_engine import historical_var, parametric_var
from risk.expected_shortfall import expected_shortfall

np.random.seed(0)
RETURNS = np.random.normal(0, 0.01, 1000)  # synthetic daily returns, 1% daily vol


def test_historical_var_positive():
    var = historical_var(RETURNS, confidence=0.99)
    assert var > 0


def test_parametric_var_positive():
    var = parametric_var(RETURNS, confidence=0.99)
    assert var > 0


def test_es_exceeds_var():
    """Key assertion: ES >= VaR always (coherence property)."""
    var = historical_var(RETURNS, confidence=0.99)
    es = expected_shortfall(RETURNS, confidence=0.99)
    assert es >= var, f"ES {es:.4f} should be >= VaR {var:.4f}"


def test_es_exceeds_var_at_different_confidences():
    for conf in [0.90, 0.95, 0.975, 0.99]:
        var = historical_var(RETURNS, confidence=conf)
        es = expected_shortfall(RETURNS, confidence=conf)
        assert es >= var, f"ES < VaR at confidence={conf}"


def test_higher_confidence_higher_var():
    var95 = historical_var(RETURNS, confidence=0.95)
    var99 = historical_var(RETURNS, confidence=0.99)
    assert var99 >= var95


def test_var_scales_with_horizon():
    """VaR(10 days) should be roughly sqrt(10) * VaR(1 day)."""
    var1 = historical_var(RETURNS, confidence=0.99, horizon=1)
    var10 = historical_var(RETURNS, confidence=0.99, horizon=10)
    ratio = var10 / var1
    assert abs(ratio - np.sqrt(10)) < 0.5, f"Scaling ratio {ratio:.2f} far from sqrt(10)"


def test_crisis_var_higher():
    """Simulated crisis returns should give higher VaR than normal returns."""
    crisis_returns = np.random.normal(-0.02, 0.04, 500)  # negative mean, high vol
    normal_var = historical_var(RETURNS, 0.99)
    crisis_var = historical_var(crisis_returns, 0.99)
    assert crisis_var > normal_var
