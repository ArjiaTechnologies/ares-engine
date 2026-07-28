# ARES Engine

ARES Engine is an open-source, end-to-end ETH machine-learning research and paper-trading pipeline. It ingests exchange OHLCV data, rejects bad feeds, builds backward-looking market features, creates future-movement labels, trains sequence models, runs chronological walk-forward tests with costs, exports immutable model bundles, and promotes a challenger only when it clears hard gates and beats the incumbent.

**ARES is not a magic money printer.** A clean pipeline can still produce a useless strategy. The code is built to make failure visible instead of hiding it behind a pretty equity curve.

This repository is a clean-room, from-scratch implementation of the architecture described in the supplied Whiplash technical dossier. It contains no Whiplash source code, trained weights, private datasets, credentials, or claimed performance results.

The public product and repository name is **ARES Engine**. The Python distribution name is `ares-eth-engine`, while the import package remains `ares_engine` and the command remains `ares`. That separation avoids colliding with unrelated projects already using Ares/ARES names.

## What is implemented

- Coinbase as the primary CCXT feed and Kraken as an independent sanity-check feed.
- Cursor-complete paginated UTC OHLCV ingestion, resume support, deduplication, closed-candle filtering, and Parquet.
- DuckDB views over Parquet for local analysis without copying the canonical dataset.
- Schema, duplicate, missing-candle, requested-range coverage, UTC-grid, OHLC invariant, metadata, volume, freshness, newest-timestamp alignment, and cross-venue p95/latest-candle divergence gates.
- EMA, close-to-EMA, Wilder RSI, Bollinger levels/position, log returns, realized volatility, and volume state.
- k-ahead directional labels with a dead zone and a triple-barrier alternative.
- Fixed-length sequences with training-fold-only `StandardScaler` fitting.
- TensorFlow/Keras LSTM and compact residual causal TCN candidates.
- Optuna search over labels, lookback, model family, capacity, dropout, learning rate, and thresholds.
- Purged expanding-window validation, delayed execution, fees, slippage, and doubled-cost stress tests.
- Return, Sharpe, drawdown, turnover, exposure, hit rate, and trade-count metrics.
- Immutable bundles containing `model.keras`, `scaler.joblib`, feature/config/metrics/provenance JSON, and a SHA-256 manifest.
- Staged all-venue validation before any canonical commit, atomic per-file writes, and fail-fast ingestion/cycle locks.
- Score-gated champion/challenger promotion and read-only paper signals.
- Fail-closed paper inference for malformed, stale, mismatched, missing, bulk-divergent, or latest-candle-divergent data across every configured venue.
- Serialized APScheduler cycles, Docker, GitHub Actions CI/release automation, CodeQL, tests, contribution and security policies.

The architecture and artifact names map directly to the target technical dossier; see [`docs/dossier_mapping.md`](docs/dossier_mapping.md).

## Deliberate hardening

Some common research habits are trash and ARES refuses them:

- **Random train/test splits:** invalid for ordered market data. ARES uses chronological folds.
- **Global scaling:** leaks future distribution information. ARES fits each scaler on its training fold.
- **Optimizing fee assumptions:** dishonest. Costs are fixed assumptions and separately stress-tested.
- **Guessing intrabar barrier order:** OHLC cannot tell which barrier hit first. ARES labels same-bar dual hits neutral.
- **Committing a half-validated batch:** corrupting. ARES stages every venue, validates the full set, and commits nothing when any gate fails.
- **Producing signals from bad data:** dangerous. ARES blocks inference and writes no signal when a configured data gate fails.
- **Promoting the newest model:** reckless. A challenger must pass gates and exceed the champion score.
- **Calling a backtest “profit proof”:** false. ARES calls it research and paper validation.

## Quick start

Python 3.11, 3.12, or 3.13 is the supported release runtime. TensorFlow is the reference backend:

```bash
git clone https://github.com/ArjiaTechnologies/ares-engine.git
cd ares-engine
uv sync --extra dev --extra ml
uv run ares doctor
uv run ares demo
uv run ares demo --ml
uv run ares verify-offline
```

A Keras 3/Torch compatibility backend is available for machines where a TensorFlow wheel is unavailable:

```bash
uv sync --extra dev --extra ml-torch
ARES_KERAS_BACKEND=torch uv run ares verify-offline
```

`ares demo` uses deterministic synthetic data. `ares verify-offline` goes further: it exercises quality checks, model training, walk-forward validation, bundle export, manifest verification, promotion, bundle reload, freshness enforcement, and paper inference.

## Verification status

Two verification records exist:

- The original July 24, 2026 snapshot (47 non-ML tests, one Keras ML test, synthetic lifecycle): [`docs/verification.md`](docs/verification.md).
- The July 27, 2026 Fable 5 third-party audit of this exact source tree: 186 tests (145 non-ML, 41 ML) passing on Python 3.11, 3.12, and 3.13, independent reference implementations for backtest/feature/label math, adversarial leakage tests, hostile-bundle and promotion-race batteries, exact-shape venue simulators, and reproduced-then-fixed defects. See [`docs/audits/FABLE_5_AUDIT.md`](docs/audits/FABLE_5_AUDIT.md).

**Audit limitation, stated plainly:** live bounded public-endpoint ingestion was not executed in the audit sandbox because outbound requests to Coinbase and Kraken were blocked with HTTP 403 at the sandbox proxy. Exact-shape simulations and adversarial local-server tests passed, but they do not establish real endpoint compatibility. Run `ares verify-public-ingestion` (or `scripts/verify_public_ingestion.py`) from an unrestricted machine and review its artifacts; until then the audit verdict is CONDITIONAL, not PASS. The smoke run's median return under doubled costs was negative: all of this is **software verification, not evidence of an edge**.

## Live public-data workflow

```bash
# 1. Pull Coinbase and Kraken ETH/USD candles and fail on bad data.
uv run ares ingest --config configs/default.yaml

# 2. Inspect the persisted quality report.
uv run ares quality --config configs/default.yaml

# 3. Run the configured LSTM through walk-forward validation.
uv run ares validate --config configs/default.yaml

# 4. Search LSTM/TCN and label/threshold candidates.
uv run ares search --config configs/default.yaml

# 5. Train and export the selected configuration as a challenger.
uv run ares train --config artifacts/best_config.yaml --name ares_candidate_001

# 6. Promote only if it passes gates and beats the current champion.
uv run ares promote artifacts/ares_candidate_001 --config artifacts/best_config.yaml

# 7. Emit a read-only paper signal from the champion.
uv run ares paper --config artifacts/best_config.yaml
```

Exchange APIs impose different per-request limits and historical-retention rules. ARES advances an explicit cursor until the requested time boundary is complete, stores only closed candles, resumes from the last persisted timestamp, and judges every configured secondary venue on aligned overlap, freshness, and latest-candle divergence. Every venue is staged and checked before canonical files change; gate-failing or truncated batches are reported but never committed, and at least one independent validation venue is mandatory. It does not pretend that an exchange can provide infinite minute history in one request.

## Data layout

```text
data/
  raw/<exchange>/eth-usd/<timeframe>.parquet
  quality/latest.json
  quality/<source>.json
  paper/signals.jsonl
  ares.duckdb
artifacts/
  best_params.json
  best_meta.json
  best_config.yaml
  ares_<timestamp>/
    model.keras
    scaler.joblib
    feature_spec.json
    config.json
    metrics.json
    provenance.json
    manifest.json
  champion.json
```

Generated data, databases, model bundles, and secrets are ignored by Git. Commit code and configuration, not private keys or giant market datasets.

## Validation score and gates

The robust score combines median fold Sharpe, median return, mean AUC, worst drawdown, and Sharpe instability. It is a ranking mechanism, not an economic truth. Hard gates remain separate and can reject a high-scoring but dangerous candidate for drawdown, turnover, trade count, weak returns, or cost fragility.

Thresholds, fees, slippage, lookback, label settings, feature columns, and the scaler travel with the model bundle. Inference refuses feature, symbol, timeframe, freshness, or data-quality mismatches instead of silently feeding garbage into a model.

## Scheduler

```bash
uv run ares cycle quick --config configs/default.yaml
uv run ares cycle deep --config configs/default.yaml
uv run ares scheduler --config configs/default.yaml
```

A quick cycle ingests data and emits a champion paper signal. A deep cycle ingests, runs Optuna when configured, revalidates the winner, trains, exports, and attempts promotion. A filesystem lock prevents quick/deep or ingestion runs from overlapping and trampling the same canonical state. Run deep cycles only after the configuration and resource budget are intentional; blindly retraining models is expensive noise.

## Query Parquet through DuckDB

```sql
SELECT exchange, min(timestamp), max(timestamp), count(*)
FROM ohlcv
GROUP BY exchange;
```

Open `data/ares.duckdb` after ingestion. The `ohlcv` object is a view over Parquet, so the source files remain canonical.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -m "not ml"

uv sync --extra dev --extra ml
uv run pytest -m ml
```

See [`docs/architecture.md`](docs/architecture.md), [`docs/research_protocol.md`](docs/research_protocol.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), and [`DISCLAIMER.md`](DISCLAIMER.md).

## Known research limits

ARES v0.1 does not have an audited live return record. Optuna and model selection reuse the configured walk-forward folds, so those scores are selection estimates rather than a locked, untouched final holdout; early stopping also monitors the same fold validation window that is later scored, which further inflates fold estimates slightly. OHLC bars cannot reveal intrabar event order. Fee and slippage values are assumptions, not guarantees. Exchange history can be incomplete (validators with capped history depth are only required to cover overlap, freshness, and alignment, never the primary's full start), and market regimes can change after every test passes.

Before capital is even discussed, a candidate needs a locked post-search holdout, a forward paper period, drift monitoring, latency and fill modeling, position and loss limits, kill switches, reconciliation, incident procedures, and independent review. Skipping those steps is not aggressive; it is sloppy.

## Scope boundary

ARES stops at paper signals. It does not place orders, manage exchange balances, custody keys, or claim audited live profitability. Those omissions are intentional. Order execution belongs in a separately threat-modeled service with least-privilege credentials and hard operational controls.

## License

MIT. See [`LICENSE`](LICENSE).
