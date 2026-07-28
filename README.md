# ARES Engine

> An open-source machine-learning trading research and paper-inference engine for ETH market-data experimentation.

ARES ingests public exchange candles, rejects inconsistent data, builds backward-looking features, trains sequence models, runs chronological cost-aware validation, exports verified bundles, and emits read-only paper signals. It does not place orders, manage balances, require exchange credentials, or claim profitability.

The Python distribution is `ares-eth-engine`; the import package is `ares_engine`; the CLI is `ares`. The implementation follows the supplied Whiplash technical dossier where practical, but that dossier is a specification—not source code or evidence of correctness.

## Safety boundary

- Public market data only by default; configured credential fields are rejected by the public-ingestion audit.
- No order-routing methods or live-capital path.
- Paper inference blocks malformed, stale, open-candle, mismatched, divergent, or non-finite feeds.
- Backtest insolvency is terminal: equity becomes zero, stays zero, and the candidate fails gates.
- Bundles are verified and loaded from a private re-hashed snapshot. `joblib` and Keras deserialization still require a trusted publisher; SHA-256 detects change but does not authenticate authorship.
- Research scores are model-selection estimates, not investment advice or evidence of an edge.

See [DISCLAIMER.md](DISCLAIMER.md), [SECURITY.md](SECURITY.md), and [docs/research_protocol.md](docs/research_protocol.md).

## Implemented system

- Credential-free Coinbase primary and Kraken validation OHLCV ingestion through CCXT.
- Measured transport telemetry: HTTP requests, `fetch_ohlcv` calls, pages, raw/normalized/deduplicated rows, cursor progression, retries, statuses, empty and short pages.
- Full-range, schema, UTC-grid, continuity, OHLC, volume, freshness, alignment, and cross-venue divergence gates.
- Immutable multi-venue generations under `data/snapshots/<generation>/`, verified manifests, and one atomic `CURRENT` pointer. Readers capture a generation once and cannot observe mixed venue state.
- Parquet as canonical storage and a DuckDB view bound to the captured generation.
- EMA, Wilder RSI, Bollinger, log-return, realized-volatility, volume-state, and range features.
- k-ahead dead-zone and triple-barrier labels; same-candle dual barrier hits are neutral.
- Fixed-length sequences, purged expanding-window folds, and training-fold-only scaling.
- TensorFlow/Keras LSTM and causal residual TCN models.
- Optuna study isolation over the full normalized OHLCV payload and every material research configuration.
- Delayed long/flat/short backtesting with fees, slippage, doubled-cost stress, drawdown, turnover, exposure, hit rate, trade counts, and bankruptcy reporting.
- Immutable seven-file bundles, hostile-file checks, runtime shape validation, manifest-anchored promotion, and fail-closed paper inference.
- Process locks for ingestion, scheduler cycles, promotion, and paper-signal logs.
- Python 3.11–3.13 CI, ML/package/Docker jobs, CodeQL when private GitHub Code Security is available, dependency review where supported, and a manual-only public-ingestion workflow. Release automation is intentionally disabled.

The detailed component map is in [docs/architecture.md](docs/architecture.md) and [docs/dossier_mapping.md](docs/dossier_mapping.md).

## Quick start

Python 3.11, 3.12, and 3.13 are supported. TensorFlow is the reference ML backend.

```bash
git clone https://github.com/ArjiaTechnologies/ares-engine.git
cd ares-engine
uv sync --extra dev --extra ml
uv run ares doctor
uv run ares demo --bars 500
uv run ares verify-offline --config configs/smoke.yaml --bars 2000
```

On a platform without a TensorFlow wheel, Keras with Torch can be used as a compatibility backend:

```bash
uv sync --extra dev --extra ml-torch
ARES_KERAS_BACKEND=torch uv run ares verify-offline --config configs/smoke.yaml --bars 2000
```

## Public-data workflow

```bash
uv run ares ingest --config configs/default.yaml
uv run ares quality --config configs/default.yaml
uv run ares validate --config configs/default.yaml
uv run ares search --config configs/default.yaml
uv run ares train --config artifacts/best_config.yaml --name ares_candidate_001
uv run ares promote artifacts/ares_candidate_001 --config artifacts/best_config.yaml
uv run ares paper --config artifacts/best_config.yaml
```

The bounded evidence workflow contacts only public endpoints and writes independently re-openable evidence:

```bash
uv run ares verify-public-ingestion \
  --primary coinbase --validation kraken \
  --symbol ETH/USD --timeframe 1h \
  --start 2026-07-10T18:00:00Z --end 2026-07-25T18:00:00Z \
  --page-limit 60 --retries 3 \
  --output artifacts/public-ingestion-audit

uv run ares validate-ingestion-report artifacts/public-ingestion-audit
```

Choose a closed historical interval ending at least 48 hours before execution. The command runs the request twice in isolated storage and validates the resulting traces, reports, Parquet contents, manifests, and hashes. The authoritative execution evidence is recorded in [docs/audits/sol/SOL_LIVE_INGESTION_EVIDENCE.md](docs/audits/sol/SOL_LIVE_INGESTION_EVIDENCE.md).

## Canonical data layout

```text
data/
  CURRENT
  snapshots/
    <generation-id>/
      raw/<exchange>/eth-usd/<timeframe>.parquet
      quality/latest.json
      quality/<source>.json
      ares.duckdb
      manifest.json
  rejected/<run-id>/
  paper/signals.jsonl
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

All canonical readers resolve one captured generation ID. Publication builds and verifies the complete generation, flushes its files, and then atomically replaces `CURRENT`. A crash before replacement leaves the old generation active; a crash after replacement exposes only the complete new generation. Recovery is cleanup, not a consistency requirement.

## Scheduler

```bash
uv run ares cycle quick --config configs/default.yaml
uv run ares cycle deep --config configs/default.yaml
uv run ares scheduler --config configs/default.yaml
```

A quick cycle ingests and may emit a champion paper signal. A deep cycle ingests, searches when configured, validates, trains, exports, and attempts promotion. Filesystem locks prevent overlapping mutation.

## Development verification

```bash
uv sync --extra dev --extra ml
uv run python -m compileall -q src tests
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run pytest --cov=ares_engine --cov-branch --cov-report=term-missing
uv run pip-audit
uv build
uv run twine check dist/*
```

The current authoritative audit is [docs/audits/sol/SOL_AUDIT.md](docs/audits/sol/SOL_AUDIT.md). Historical Fable 5 material is preserved under `docs/audits/archive/fable5-untrusted/` as superseded AI-assisted review notes, not independent certification.

## Research limitations

ARES does not automate a nested untouched post-search holdout. Search and early stopping use configured chronological validation windows, so final performance claims require a separately locked dataset and a forward paper period. OHLC bars cannot establish intrabar ordering. Fees, slippage, latency, and liquidity are assumptions. Public exchange history can be incomplete, and market regimes change.

Before any capital use, a separate execution service would need least-privilege credentials, risk and loss limits, kill switches, reconciliation, observability, incident procedures, independent security review, and an audited forward record. None of that is part of ARES.

## License

MIT. See [LICENSE](LICENSE).
