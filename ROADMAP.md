# Roadmap

## v0.1 — research engine

- Coinbase primary and Kraken cross-check ingestion
- Parquet/DuckDB storage and data gates
- feature, label, sequence, LSTM/TCN, Optuna, walk-forward, and backtest pipeline
- immutable bundles, promotion, fail-closed paper inference, scheduler, CI, and Docker

## v0.2 — stronger research controls

- automated locked post-search holdout and benchmark baselines
- recent-window challenger replay before promotion
- probability calibration and threshold stability reports
- MLflow experiment logging and River-based drift monitors
- dataset manifests with partition-level hashes
- richer cost, spread, and latency sensitivity surfaces

## v0.3 — operational paper service

- Coinbase public WebSocket ingestion with REST reconciliation
- durable event queue, idempotent signal processing, observability, and alerting
- model rollback, stale-champion policies, and scheduled drift review
- reproducible container images and signed release artifacts

Live order execution is not on this roadmap until a separate security and risk-control design is reviewed. Bolting exchange keys onto the research process would be negligent.
