# FRTB, Basel IV & DORA — Plain English Summary

> Written as interview prep. These are the talking points for Day 1 at JPMC.

---

## Basel III → Basel IV: What Changed and Why

### Basel III (post-2008)
Introduced after the Global Financial Crisis to rebuild capital buffers.

**Three pillars:**
1. **Minimum capital** — CET1 ≥ 4.5%, Tier 1 ≥ 6%, Total Capital ≥ 8%
2. **Supervisory review** — ICAAP: banks self-assess capital adequacy, regulators challenge
3. **Market discipline** — public disclosure of risk exposures and capital structure

**Liquidity ratios (new in Basel III):**
- **LCR** (Liquidity Coverage Ratio): hold enough HQLA to survive 30-day stress
- **NSFR** (Net Stable Funding Ratio): fund long-term assets with stable long-term liabilities

### Basel IV (finalised 2017, implementing 2025–2028)
Not a new framework — a revision of Basel III's capital floor and credit risk standards.

**Key change: Output floor at 72.5%**
Banks using internal models (IRB, IMA) cannot produce RWA below 72.5% of the
standardised approach. This prevents banks from gaming internal models to minimise capital.

**Other Basel IV changes:**
- Revised standardised approach for credit risk (more risk-sensitive)
- Revised operational risk framework (replaces AMA)
- FRTB replaces Basel 2.5 market risk framework (see below)

---

## FRTB — Fundamental Review of the Trading Book

The most significant change to market risk capital in a generation.

### The Core Problem FRTB Fixes
Basel 2.5 used **99% VaR** over a **10-day horizon** as the capital metric.
Three problems:
1. VaR is not sub-additive — diversification can appear to reduce risk when it doesn't
2. VaR doesn't capture tail risk (what's past the 99th percentile)
3. The 10-day scaling (√10 × 1-day VaR) assumes iid returns — wrong in crises

### FRTB's Solution: Expected Shortfall at 97.5%
Replace VaR with **ES at 97.5% confidence** — captures the expected loss *given* a breach.
ES is:
- Sub-additive (coherent)
- Tail-sensitive
- The FRTB standard for Internal Models Approach (IMA)

### Liquidity Horizons (FRTB innovation)
Not all positions can be unwound in 10 days. FRTB assigns horizons by asset class:
| Asset Class | Liquidity Horizon |
|-------------|------------------|
| Large-cap equities | 10 days |
| IG credit | 20 days |
| HY credit / EM | 40–60 days |
| Exotic options | 60 days |

Capital = ES scaled to these horizons. Illiquid positions get penalised.

### P&L Attribution Test
FRTB requires banks to prove their risk model explains actual P&L.
If the model's "risk-theoretical P&L" diverges too much from actual P&L → model rejected.
This is why model validation teams exist at every major bank.

### IMA vs SA
- **IMA** (Internal Models Approach): use your own ES model — lower capital but higher regulatory burden
- **SA** (Standardised Approach): use the regulator's formula — simpler but always more conservative
- Desks can switch between IMA and SA. Basel IV's output floor limits how much IMA saves.

### FRTB Timeline (UK/EU)
- EU: Jan 2025 reporting, Jan 2026 capital requirements
- UK PRA: aligning to similar timeline

---

## DORA — Digital Operational Resilience Act

EU regulation effective Jan 2025. Applies to all financial entities and their ICT providers.

### What It Does
Forces banks to prove they can withstand, respond to, and recover from ICT disruptions.
"Operational risk meets cyber risk meets outsourcing risk."

### Key Requirements
1. **ICT risk management**: documented framework, board accountability
2. **Incident reporting**: major ICT incidents reported to regulators within 4 hours (initial) and 72 hours (full)
3. **Digital operational resilience testing**: annual penetration testing (TLPT for systemic firms)
4. **Third-party risk**: contracts with cloud providers / fintech must include DORA clauses
5. **Information sharing**: participate in threat intelligence sharing

### Why It Matters for Risk Teams
- JPMC's risk infrastructure runs on ICT systems — any disruption = potential VaR model failure
- DORA requires banks to have a "Register of Information" of all ICT third-party dependencies
- Quant teams are affected: model validation frameworks must be resilient under DORA
- ML/AI in risk models is under regulatory scrutiny — DORA and the EU AI Act overlap here

### The ML/AI Angle
Regulators are asking: if your VaR model uses ML, can you explain it? Can you backtest it under DORA's resilience standards? Can you prove it doesn't introduce new operational risk?

This is an active area — expect questions at JPMC about how AI is changing traditional risk roles.

---

## Interview Talking Points (2 minutes each)

**"Walk me through FRTB in plain English."**
> Basel 2.5 used 99% VaR, which doesn't capture tail risk and can be gamed. FRTB replaces it with 97.5% ES across liquidity-adjusted horizons — so a 60-day illiquid position gets a 60-day ES, not a scaled 10-day one. There's also a P&L attribution test that means your model has to actually explain your book's returns. The output floor at 72.5% stops internal models from producing unrealistically low capital numbers.

**"What's the difference between VaR and ES and why does it matter?"**
> VaR tells you the threshold you won't exceed with probability p. ES tells you the expected loss given that you do exceed it. ES is coherent — it's sub-additive, so portfolio ES ≤ sum of component ES. VaR isn't. In a crisis, VaR can understate losses by 3-5x because it ignores what's in the tail. That's why Basel IV moved to ES.

**"What is DORA and why should a quant risk analyst care?"**
> DORA is the EU's operational resilience regulation for financial firms. It means every ICT system the bank relies on — including risk models — has to have documented resilience, incident reporting, and third-party oversight. For quant risk, it's relevant because AI/ML models in production systems are now under DORA's scope, and regulators want to understand what happens to your VaR model if a cloud provider goes down.
