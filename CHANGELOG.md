# Changelog

All notable changes to ARES Engine are documented here.

## 0.1.0 — 2026-07-24

- Recreated the Whiplash-style ETH ML lifecycle as an open-source Python package.
- Added cursor-complete Coinbase/Kraken ingestion, staged all-venue validation before commit, requested-range coverage, ordered atomic Parquet/DuckDB storage, hard data-quality gates, mandatory independent validation feeds, newest-timestamp alignment, all-configured-venue freshness checks, and p95/latest-candle divergence checks.
- Added EMA, Wilder RSI, Bollinger, return, volatility, range, and volume features.
- Added k-ahead dead-zone and triple-barrier labels, fixed-length sequences, LSTM/TCN models, Optuna search, purged walk-forward validation, and cost-aware backtesting.
- Added hash/size/metadata-verified bundles, manifest-anchored champion/challenger promotion, fail-closed multi-venue paper inference, fail-fast ingestion/cycle/promotion locks, finite-score and artifacts-root promotion checks, and APScheduler quick/deep cycles with optional Optuna search.
- Added tests, Docker, Python 3.11-3.13 GitHub Actions CI, tag-driven GitHub releases, CodeQL, contributor/security documentation, and deterministic offline lifecycle verification.
- Kept the ARES Engine brand while using the collision-resistant `ares-eth-engine` Python distribution name.
