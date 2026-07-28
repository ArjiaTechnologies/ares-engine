# Dossier-to-ARES implementation map

The Whiplash ETH ML dossier is an architectural reference. It is not source code, a performance record, or proof of correctness.

| Dossier concept | ARES implementation |
|---|---|
| Paginated exchange OHLCV | `ares_engine.data.providers.CCXTOHLCVProvider` with explicit cursor advancement and measured transport telemetry |
| Cleanup, Parquet, and SQL analysis | `ares_engine.data.ingest`, immutable generations in `data.storage`, generation-bound DuckDB views |
| EMA, RSI, Bollinger, return, volatility, volume, range | `ares_engine.features` |
| k-ahead and triple-barrier labels | `ares_engine.labels` |
| Rolling sequences and train-only scaling | `ares_engine.dataset` and validation-fold scaler fitting |
| LSTM and compact TCN | `ares_engine.models` |
| Configuration search | `ares_engine.search` with material study identity |
| Walk-forward evaluation | `ares_engine.validation` |
| Delayed cost-aware simulation | `ares_engine.backtest`, including terminal bankruptcy |
| Model/scaler/spec/config/metrics/provenance artifact | `ares_engine.bundles` with exact manifests and snapshot loading |
| Challenger and champion | `ares_engine.training` and locked `ares_engine.promotion` |
| Paper loader | fail-closed `ares_engine.live`; no order execution |
| Scheduled cycles | `ares_engine.scheduler` |
| End-to-end proof | `ares verify-offline`, `ares verify-public-ingestion`, and the Sol audit artifacts |

ARES adds independently verified controls beyond the dossier: mandatory validation venues, complete-generation atomic visibility, strict evidence recomputation, real request traces, UTC-grid/range/freshness/alignment/divergence gates, purged boundaries, insolvent-trial exclusion, hostile bundle handling, promotion and paper-log process locks, non-root Docker, and private least-privilege CI. MLflow and River remain optional dependencies rather than active correctness controls.
