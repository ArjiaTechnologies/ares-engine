# Changelog

All notable changes to ARES Engine are documented here.

## Unreleased — Fable 5 audit hardening (2026-07-27)

- Fixed a torn multi-venue commit path: canonical Parquet replacement is now a
  journaled two-phase commit with automatic roll-forward recovery at the next
  locked ingestion; staging files can no longer match canonical globs, and the
  DuckDB view is built from an explicit hidden-file-free list.
- Applied start-coverage requirements to the primary venue only, so
  depth-capped validators (Kraken serves roughly the newest 720 candles) no
  longer make long-range ingestion permanently impossible.
- Excluded NaN/infinite objective values from Optuna winner selection and
  scoped study identity by a dataset fingerprint so stale trials from older
  data can never be selected after the dataset changes.
- Closed the verify-then-load race in bundle loading via snapshot re-hashing;
  garbage champion pointers now fail closed with a typed error.
- Paper inference refuses feeds containing the in-progress candle in either
  the primary or any validation feed when drop_open_candle is configured.
- Corrected the `[ml]` extra for non-x86_64 platforms, shipped default
  configs inside the package so the installed wheel works outside a checkout,
  validated impossible `--bars` values up front, and reported actual new
  canonical rows.
- CI now runs ruff format and strict mypy; the release workflow is guarded to
  the canonical repository; the Docker image runs as a non-root user.
- Added `ares verify-public-ingestion`, `ares validate-ingestion-report`, and
  `scripts/verify_public_ingestion.py` for credential-free bounded live
  ingestion evidence outside restricted sandboxes.
- Added 136 tests: independent reference implementations for backtest,
  features, and labels; adversarial leakage, hostile-bundle, promotion-race,
  wire-simulator, scheduler, and harness batteries. Test count 50 -> 186.

## 0.1.0 — 2026-07-24

- Recreated the Whiplash-style ETH ML lifecycle as an open-source Python package.
- Added cursor-complete Coinbase/Kraken ingestion, staged all-venue validation before commit, requested-range coverage, ordered atomic Parquet/DuckDB storage, hard data-quality gates, mandatory independent validation feeds, newest-timestamp alignment, all-configured-venue freshness checks, and p95/latest-candle divergence checks.
- Added EMA, Wilder RSI, Bollinger, return, volatility, range, and volume features.
- Added k-ahead dead-zone and triple-barrier labels, fixed-length sequences, LSTM/TCN models, Optuna search, purged walk-forward validation, and cost-aware backtesting.
- Added hash/size/metadata-verified bundles, manifest-anchored champion/challenger promotion, fail-closed multi-venue paper inference, fail-fast ingestion/cycle/promotion locks, finite-score and artifacts-root promotion checks, and APScheduler quick/deep cycles with optional Optuna search.
- Added tests, Docker, Python 3.11-3.13 GitHub Actions CI, tag-driven GitHub releases, CodeQL, contributor/security documentation, and deterministic offline lifecycle verification.
- Kept the ARES Engine brand while using the collision-resistant `ares-eth-engine` Python distribution name.
