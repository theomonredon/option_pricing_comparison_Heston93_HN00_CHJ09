# Option Pricing Comparison — Heston 1993 / HN 2000 / CHJ 2009

Empirical comparison of three Fourier-inversion option pricing models on
S&P 500 single-name option chains, with a rolling daily out-of-sample
backtest broken down by volatility regime.

> *README is a placeholder until the batch has run and figures are available.*

## Quick start

```bash
pip install -r requirements.txt

# 1. Place option-chain parquets under data/<TICKER>/<YYYY>/<MM>/<YYYY-MM-DD>_{call|put}.parquet
# 2. Edit config.yaml — tickers, dates, test months
# 3. Validate
python scripts/validate_data.py

# 4. Heavy batch (~70h for 6 tickers x 6 months x 3 models)
python scripts/run_batch.py

# 5. Open the analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

The batch is **resumable** — re-running `run_batch.py` skips already-computed
`(date, ticker, model)` triples.

## Project layout

```
src/
  models/            Heston 1993, HN 2000, CHJ 2009 with a common BasePricer
  preprocessing/     loaders + filter chain + IV extraction
  calibration/       polymorphic optimiser + loss functions
  analysis/          metrics + plotting

scripts/
  validate_data.py   pre-flight check
  run_batch.py       batch in-sample + OOS rolling

notebooks/
  analysis.ipynb     11-section report, loads results parquets

config.yaml          tickers / dates / filters / regimes / OOS horizons
results/             parquet outputs + figures (gitignored except figures/)
data/                option chains (gitignored)
```

## Methodology

### Models

| Model | Type | State variable |
|-------|------|----------------|
| Heston (1993) | Continuous SV, single-factor | $V_t \in \mathbb{R}$ |
| Heston-Nandi (2000) | GARCH discrete, closed-form | $h_t \in \mathbb{R}$ |
| Christoffersen-Heston-Jacobs (2009) | Continuous SV, two-factor | $(V_{1,t}, V_{2,t}) \in \mathbb{R}^2$ |

All three share the same affine characteristic function structure, so the
**same Fourier-inversion pricer** is used for the three.

### Filtering

bid > 0, mid ≥ 0.10, relative spread ≤ 25%, DTE ∈ [7, 365], moneyness ∈ [0.7, 1.3], volume ≥ 1, mid ≥ intrinsic.

### Calibration

Loss = IV-vega RMSE (Christoffersen-Jacobs 2004). L-BFGS-B with log-scale
re-parameterisation on positive parameters.

### Out-of-sample protocol

For each trading day $d$ in a test month, calibrate on the panel at $d$,
then price the panels at $d+1$, $d+2$, $d+5$ with the fixed parameters
(for HN, update $h_t$ by continuing the GARCH filter on realised returns).

Six test months are chosen across three volatility regimes (calm / normal /
stressed), classified automatically by SPY 21-day realised volatility.

### Comparison axes

1. **In-sample fit** — heatmap and time-series of IV-vega RMSE
2. **Granular fit** — error broken down by maturity × moneyness buckets
3. **Parameter stability** — CV and lag-1 autocorrelation of calibrated params
4. **Out-of-sample error** — distribution per (model, horizon)
5. **Regime breakdown** — does the model rank change under stress?
6. **Computational cost** — median calibration time per model
