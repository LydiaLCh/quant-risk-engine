"""Tests for portfolio/optimiser.py and portfolio/hedging.py."""
import numpy as np
import pytest
from portfolio.optimiser import portfolio_stats, min_variance_portfolio, efficient_frontier
from portfolio.hedging import delta_hedge_simulation, gamma_pnl_formula

np.random.seed(42)
# Synthetic 3-asset return matrix (500 days)
RETURNS = np.random.multivariate_normal(
    mean=[0.0003, 0.0005, 0.0002],
    cov=[[0.0001, 0.00003, 0.00001],
         [0.00003, 0.00015, 0.00002],
         [0.00001, 0.00002, 0.00008]],
    size=500
)


def test_min_var_weights_sum_to_one():
    result = min_variance_portfolio(RETURNS)
    assert abs(sum(result["weights"]) - 1.0) < 1e-6


def test_min_var_weights_non_negative():
    result = min_variance_portfolio(RETURNS, allow_short=False)
    assert all(w >= -1e-8 for w in result["weights"])


def test_min_var_lower_vol_than_equal_weight():
    """Min-var portfolio should have lower vol than equal-weight benchmark."""
    n = RETURNS.shape[1]
    ew = np.ones(n) / n
    mean_ret = RETURNS.mean(axis=0)
    cov = np.cov(RETURNS, rowvar=False)
    _, ew_vol, _ = portfolio_stats(ew, mean_ret, cov)
    result = min_variance_portfolio(RETURNS)
    assert result["volatility"] <= ew_vol + 1e-6


def test_efficient_frontier_increasing_return():
    """Frontier returns should be non-decreasing with vol."""
    ef = efficient_frontier(RETURNS, n_points=10)
    rets = ef["returns"]
    assert all(rets[i] <= rets[i+1] + 1e-6 for i in range(len(rets)-1))


def test_gamma_pnl_formula():
    """½ Γ (ΔS)² is always non-negative for long gamma."""
    assert gamma_pnl_formula(0.05, 2.0) == pytest.approx(0.1)
    assert gamma_pnl_formula(0.05, -2.0) == pytest.approx(0.1)  # symmetric


def test_delta_hedge_returns_dict():
    result = delta_hedge_simulation(100, 100, 1.0, 0.05, 0.2, n_steps=52, seed=0)
    assert "final_pnl" in result
    assert "delta_path" in result
    assert "gamma_path" in result


def test_delta_hedge_pnl_small_for_daily():
    """Daily rebalancing should produce near-zero final P&L (within reasonable bound)."""
    result = delta_hedge_simulation(100, 100, 1.0, 0.05, 0.2, n_steps=252, seed=42)
    # P&L should be small relative to the option premium (~10)
    assert abs(result["final_pnl"]) < 3.0, f"Large hedging error: {result['final_pnl']:.4f}"
