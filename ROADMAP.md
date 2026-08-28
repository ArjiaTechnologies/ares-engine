# Roadmap

## v0.1 — research engine

- Coinbase primary and Kraken cross-check ingestion
- Parquet/DuckDB storage and data gates
- feature, label, sequence, LSTM/TCN, Optuna, walk-forward, and backtest pipeline
- immutable bundles, promotion, fail-closed paper inference, scheduler, CI, and Docker

## v0.2 — stronger research controls

- automated locked post-search holdout and benchmark baselines — first bounded
  milestone implemented through explicit `ares locked-holdout`; further
  independent replication and forward paper evidence remain required
- recent-window challenger replay before promotion — first bounded milestone
  implemented through `ares prepare-challenger`; exact unseen-window source and
  bundle evidence is required and anchored by promotion, while independent
  forward paper replication remains required
- probability calibration and threshold stability reports — first bounded
  diagnostic milestone implemented in walk-forward validation; out-of-fold
  reliability and nearby decision-threshold sensitivity are recorded without
  silently calibrating the model or weakening promotion gates
- MLflow experiment logging and River-based drift monitors
- dataset manifests with partition-level hashes
- richer cost, spread, and latency sensitivity surfaces

## v0.3 — operational paper service

- Coinbase public WebSocket ingestion with REST reconciliation
- durable event queue, idempotent signal processing, observability, and alerting
- model rollback, stale-champion policies, and scheduled drift review
- reproducible container images and signed release artifacts

Live order execution is not on this roadmap until a separate security and risk-control design is reviewed. Bolting exchange keys onto the research process would be negligent.
