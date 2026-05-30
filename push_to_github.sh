#!/bin/bash
# ============================================================
# push_to_github.sh
# Run this from the quant-risk-engine/ folder on your machine.
# It will create the GitHub repo and push everything in one go.
# ============================================================

TOKEN=""

# 1. Get your GitHub username
echo "Fetching GitHub username..."
USERNAME=$(curl -s -H "Authorization: token $TOKEN" https://api.github.com/user | python3 -c "import sys,json; print(json.load(sys.stdin)['login'])")
echo "Username: $USERNAME"

# 2. Create the repo (public)
echo "Creating repository quant-risk-engine..."
curl -s -X POST \
  -H "Authorization: token $TOKEN" \
  -H "Content-Type: application/json" \
  https://api.github.com/user/repos \
  -d '{
    "name": "quant-risk-engine",
    "description": "Modular quant risk engine: Black-Scholes, Monte Carlo, VaR, ES, CVA, portfolio optimisation. Built across 3 passes for JPMC risk internship prep.",
    "private": false,
    "auto_init": false
  }' | python3 -c "import sys,json; d=json.load(sys.stdin); print('Repo created:', d.get('html_url', d.get('message','')))"

# 3. Initialise git and push
echo "Initialising git..."
git init
git add .
git commit -m "Initial commit: full quant risk engine (Pass 1-3 structure)

Modules:
- core/: Black-Scholes, Greeks, Monte Carlo (GBM), Yield Curves
- risk/: VaR (Historical + Parametric + MC), Expected Shortfall, Stress Tests, CVA/DVA
- portfolio/: Mean-Variance Optimiser, Delta/Gamma Hedging, VaR Backtester
- regulation/: FRTB, Basel IV, DORA plain English summary
- tests/: pytest suite with quantitative assertions (put-call parity, ES>=VaR, convergence)

Built in 3 passes (30 May - 22 Jun 2026):
  Pass 1: implement each concept in isolation
  Pass 2: connect modules (VaR->MC, Greeks->Hedging, CVA->MC, StressTest->VaR)
  Pass 3: stress, break, and document model failures as interview talking points"

git branch -M main
git remote add origin https://$USERNAME:$TOKEN@github.com/$USERNAME/quant-risk-engine.git
git push -u origin main

echo ""
echo "Done! Repo live at: https://github.com/$USERNAME/quant-risk-engine"
echo "Add this link to your CV and LinkedIn before 22nd June."
