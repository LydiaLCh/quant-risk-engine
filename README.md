# Quant Risk Engine

A modular Python implementation of core quantitative risk concepts, built iteratively
across three passes as a learning and portfolio project ahead of a risk internship at
JP Morgan Chase (June–August 2026).

## Design Philosophy

Most people build separate scripts — a Black-Scholes file here, a VaR notebook there.
The problem: concepts in risk aren't isolated. VaR uses Monte Carlo. Monte Carlo prices
derivatives. Derivatives need the Greeks. The Greeks come from Black-Scholes.
Black-Scholes breaks under stress — which is exactly what Basel/FRTB is designed to catch.

This project treats the full concept map as a single interconnected system. Each module
exposes clean interfaces so other modules can call it. By the time all modules are built,
the result is a mini risk engine — the kind of thing that actually runs on a trading desk.

**Where models fail is documented explicitly.** Every module ends with a
`Where This Model Breaks` section. Understanding model limitations is the actual job.

---

## Repository Structure

```
quant-risk-engine/
│
├── core/
│   ├── black_scholes.py      ← pricing engine (foundation layer)
│   ├── monte_carlo.py        ← GBM simulation engine
│   ├── greeks.py             ← analytical Greeks (calls core/black_scholes)
│   └── yield_curve.py        ← bootstrapping, duration, convexity, DV01
│
├── risk/
│   ├── var_engine.py         ← Historical, Parametric, MC VaR
│   ├── expected_shortfall.py ← ES/CVaR (FRTB standard at 97.5%)
│   ├── stress_test.py        ← scenario analysis (calls yield_curve + var_engine)
│   └── cva_engine.py         ← CVA/DVA using MC exposure profiles
│
├── portfolio/
│   ├── optimiser.py          ← mean-variance, efficient frontier, Sharpe
│   ├── hedging.py            ← delta/gamma hedge rebalancing loop
│   └── backtest.py           ← VaR breach counting, Basel traffic light test
│
├── regulation/
│   └── frtb_summary.md       ← FRTB, Basel IV, DORA plain English summary
│
├── notebooks/                ← Jupyter walkthroughs (one per module)
│
├── tests/                    ← pytest unit tests with quantitative assertions
│
└── README.md
```

---

## The Three Passes

### Pass 1 — Understand & Implement (30 May – 8 Jun)
Get a working implementation of each concept. Correctness over elegance.
- `core/black_scholes.py` — verified against put-call parity ✓
- `core/greeks.py` — all 5 Greeks, delta sigmoid plot ✓
- `core/monte_carlo.py` — converges to BS price as N→∞ ✓
- `core/yield_curve.py` — bootstrapped zero curve, DV01 ✓
- `risk/var_engine.py` — historical + parametric VaR ✓
- `risk/expected_shortfall.py` — ES ≥ VaR always asserted ✓
- `portfolio/optimiser.py` — min-variance + efficient frontier ✓
- `portfolio/hedging.py` — delta rebalancing loop ✓

### Pass 2 — Connect the Modules (8–15 Jun)
Make the modules talk to each other. Key connections:
- `var_engine` → `monte_carlo`: MC-simulated VaR vs historical VaR
- `greeks` → `hedging` → P&L: gamma P&L formula verified (½ Γ ΔS²)
- `stress_test` → `var_engine`: +200bps shock on bond portfolio VaR
- `cva_engine` → `monte_carlo`: CVA from simulated exposure profiles

### Pass 3 — Stress, Break & Document (15–22 Jun)
Deliberately break each model. Document failures as interview talking points.
- Black-Scholes: volatility smile exposed near expiry / deep OTM
- Historical VaR: crisis window (2008, 2020) causes severe underestimation
- Delta hedge: weekly rebalancing accumulates large gamma exposure
- Portfolio optimiser: small/correlated datasets → covariance instability
- MC pricer: N=100 paths → slow convergence motivates variance reduction

---

## Running the Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Key assertions in the test suite:
- **Put-call parity** holds to 8 decimal places
- **Implied vol roundtrip**: recover σ from BS price via bisection
- **MC converges to BS**: within 1% with N=50,000 paths
- **ES ≥ VaR always**: tested at 90%, 95%, 97.5%, 99% confidence
- **Min-var portfolio** has lower vol than equal-weight benchmark
- **Delta hedge P&L** is small relative to option premium (daily rebalancing)

---

## Quick Start

```python
from core.black_scholes import bs_call, bs_put
from core.greeks import all_greeks
from core.monte_carlo import price_european
from risk.var_engine import historical_var
from risk.expected_shortfall import expected_shortfall

# Price a 1-year ATM call
price = bs_call(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

# Get all Greeks
greeks = all_greeks(100, 100, 1.0, 0.05, 0.20, "call")

# MC price with antithetic variates
mc = price_european(100, 100, 1.0, 0.05, 0.20, antithetic=True, n_paths=10_000)

# Historical VaR and ES on return series
import numpy as np
returns = np.random.normal(0, 0.01, 500)
var = historical_var(returns, confidence=0.99)
es  = expected_shortfall(returns, confidence=0.975)
assert es >= var  # always
```

---

## Dependencies

```
numpy
scipy
matplotlib
jupyter
pytest
```

---

## Status

Built across 3 weeks (30 May – 22 June 2026). Updated with insights from
internship at JP Morgan Chase as the summer progresses.

Where models fail is documented explicitly in every module's
`Where This Model Breaks` section — because understanding model limitations
is the actual job.
