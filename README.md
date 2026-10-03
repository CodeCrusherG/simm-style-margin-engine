# SIMM-Style Initial Margin Engine

A from-scratch Python engine that prices a multi-asset derivatives portfolio, computes risk sensitivities by bump-and-reprice, aggregates them into an initial margin number using the **structure of the ISDA SIMM methodology**, and back-tests the margin against 10-day full-revaluation losses.

> **Read this first:** the SIMM parameters (risk weights, correlations, gammas, psi) in `simm.py` are **illustrative placeholders**, not the ISDA-calibrated set. The aggregation mechanics follow the published methodology, but the margin numbers are not real SIMM numbers. Curvature and base-correlation margin are not implemented. The risk-factor history used in the back-test is **simulated** (tenor-level rate curves and CDS spreads are not freely available); the mechanics are real, the result is a demonstration.

## What it does

1. **Pricers** (`pricers.py`): interest rate swap, fixed-rate bond, equity option (Black-Scholes), CDS (flat hazard rate / credit triangle), FX forward.
2. **Sensitivities** (`sensitivities.py`): bump-and-reprice for IR delta (12 SIMM-style tenors, per currency), equity delta and vega, credit (CDS spread) delta and FX delta.
3. **SIMM-style aggregation** (`simm.py`):
   - weighted sensitivity `WS = RW * s`
   - intra-bucket `K_b = sqrt(sum rho_ij WS_i WS_j)` with tenor correlation for IR
   - inter-bucket aggregation with `S_b = clip(sum WS, -K_b, K_b)` and gamma
   - risk-class aggregation with a psi matrix, computed **per product class** (RatesFX, Credit, Equity) and summed
4. **Back-test** (`im_backtest.py`): simulated fat-tailed, correlated factor history with a stress regime; non-overlapping 10-day windows; full revaluation P&L; Kupiec test. Compares the SIMM-style IM (static) with a historical-VaR IM calibrated on the first 60% of windows and tested out of sample.
5. **Report** (`run_simm.py`): Excel workbook (sensitivities, IM breakdown, back-test, window P&L) and a plot.

## Run

```bash
pip install -r requirements.txt
python run_simm.py
python -m unittest -v
```

## Demo results (simulated factors, illustrative parameters)

Portfolio: USD 10y payer swap, USD 5y receiver swap, 7y bond, long call and short put on two equities, two CDS positions, one USDINR forward.

| Model | IM | Test windows | Exceptions | Worst loss / IM |
|---|---|---|---|---|
| SIMM-style IM (static) | about 1.75m | 100 | 0 | 0.97 |
| Historical-VaR IM (calibrated) | about 0.46m | 100 | 14 | 3.6 |

The stress regime was built into the simulated history, so this shows the mechanism, not a market finding: a margin model calibrated only on calm data fails when volatility jumps, whereas a stressed, parametric approach like SIMM holds up (at the cost of higher margin). The SIMM-style margin is also close to its limit in the worst window (0.97), and with illustrative parameters it should not be read as accurate.

![IM back-test](im_backtest_plot.png)

## Tests

18 unit tests cover: put-call parity, par-rate swap and bond pricing, CDS at market spread, sign of PV01 and vega, bump-and-reprice delta vs Black-Scholes delta, single-sensitivity margin equal to RW times sensitivity, offsetting trades netting to zero margin, diversification, sub-additivity, linear scaling in notional, window construction and Kupiec.

## Limitations

- Illustrative SIMM parameters; no curvature, base-correlation, concentration or commodity risk class.
- Simulated factor history; static portfolio over each margin period.
- Integer maturities for swaps and bonds, annual fixed legs, flat hazard rate per CDS name, no CSA or netting-set structure, no FX delta bucketing by currency.
- Equity and FX returns could be replaced with real data; rate-curve and CDS history would need a licensed source.
