"""Tests for core/monte_carlo.py — Pass 1 convergence checks."""
import pytest
import numpy as np
from core.monte_carlo import simulate_gbm, price_european
from core.black_scholes import bs_call

S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def test_gbm_paths_shape():
    paths = simulate_gbm(S0, 0.05, sigma, T, n_steps=252, n_paths=1000)
    assert paths.shape == (1000, 253)


def test_gbm_initial_price():
    paths = simulate_gbm(S0, r, sigma, T, n_steps=50, n_paths=500)
    assert np.allclose(paths[:, 0], S0)


def test_mc_converges_to_bs():
    """With N=50,000 paths, MC price should be within 1% of BS price."""
    bs = bs_call(S0, K, T, r, sigma)
    mc = price_european(S0, K, T, r, sigma, n_paths=50_000, seed=0)
    assert abs(mc["price"] - bs) / bs < 0.01, (
        f"MC price {mc['price']:.4f} too far from BS {bs:.4f}"
    )


def test_mc_ci_contains_bs():
    """95% CI should contain the true (BS) price most of the time."""
    bs = bs_call(S0, K, T, r, sigma)
    mc = price_european(S0, K, T, r, sigma, n_paths=10_000, seed=42)
    assert mc["ci_low"] <= bs <= mc["ci_high"], (
        f"BS price {bs:.4f} outside MC CI [{mc['ci_low']:.4f}, {mc['ci_high']:.4f}]"
    )


def test_antithetic_reduces_variance():
    """Antithetic variates should give a tighter standard error."""
    base = price_european(S0, K, T, r, sigma, n_paths=5000, seed=1, antithetic=False)
    anti = price_european(S0, K, T, r, sigma, n_paths=5000, seed=1, antithetic=True)
    assert anti["std_error"] < base["std_error"]


def test_put_price_positive():
    result = price_european(S0, K, T, r, sigma, option_type="put", n_paths=5000)
    assert result["price"] > 0
