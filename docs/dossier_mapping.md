# Dossier-to-ARES implementation map

| Dossier layer or artifact | ARES implementation |
|---|---|
| Paginated exchange OHLCV | `ares_engine.data.providers.CCXTOHLCVProvider` with an advancing cursor that runs to the requested time boundary |
| Timestamp cleanup and Parquet | `ares_engine.data.ingest`, `ares_engine.data.storage` |
| EMA, RSI, Bollinger, returns, volatility | `ares_engine.features` |
| k-ahead dead-zone and triple-barrier labels | `ares_engine.labels` |
| Fixed-length rolling sequences | `ares_engine.dataset` |
| Train-fold-only scaling | `fit_scaler` inside every validation fold |
| TensorFlow/Keras LSTM | `ares_engine.models.build_model` |
| Compact residual TCN candidate | `ares_engine.models.build_model` |
| Optuna model/configuration search | `ares_engine.search` |
| Walk-forward validation | `ares_engine.validation` |
| Delayed, fee/slippage-aware backtest | `ares_engine.backtest` |
| `model.keras`, scaler, feature spec, thresholds, costs | `ares_engine.bundles` |
| Challenger gates and champion promotion | `ares_engine.training`, `ares_engine.promotion` |
| Paper/live loader | `ares_engine.live` |
| Scheduled quick/deep cycles | `ares_engine.scheduler` |
| Concrete end-to-end verification | `ares verify-offline`, `docs/verification.*` |

ARES adds controls that the dossier describes only partially or not at all: staged all-venue validation before canonical commit, requested-range coverage, ordered atomic Parquet replacement, fail-fast ingestion/cycle/promotion locks, mandatory independent validation venues, all-configured-venue metadata/freshness/newest-timestamp checks, p95 and latest-candle price gates, true UTC-grid validation, purged fold boundaries, refusal to select an Optuna trial that failed hard gates, size/hash/metadata bundle verification, rejection of symlinks and unlisted bundle files, champion-manifest anchoring, latest-complete-row inference, read-only paper output, CI, release automation, tests, Docker, security guidance, and explicit claim discipline.

The dossier described MLflow and River as environment-level experiment/drift extensions rather than core execution paths. ARES keeps them in the optional `tracking` dependency group; first-class integration remains roadmap work.
