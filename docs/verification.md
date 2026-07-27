# Verification record

ARES was exercised end to end on July 24, 2026 with 2,500 deterministic hourly synthetic candles for each of two venues. The run covered data-quality checks, cross-venue comparison, feature and label construction, purged walk-forward training, cost-aware backtesting, bundle export, hash and metadata verification, first-champion promotion, bundle reload, primary/secondary freshness checks, latest-candle divergence checks, and paper-signal generation.

## Result

- 47 non-ML tests passed.
- 1 Keras ML smoke test passed.
- Both synthetic venue feeds passed schema, continuity, UTC-grid, OHLC, volume, and freshness checks.
- Cross-venue p95 close divergence was 3.999 bps; latest aligned divergence was 0.560 bps.
- Two walk-forward folds executed.
- The seven-file `.keras` bundle exported, verified, promoted, reloaded, and generated a paper signal.
- The paper probability was 0.414881, producing a short research signal (`-1`) under the smoke thresholds.
- Parquet/DuckDB integration was not executed in this sandbox because `pyarrow` and `duckdb` were unavailable. The GitHub Actions matrix installs the declared base dependencies and runs that path on Python 3.11, 3.12, and 3.13.

## The result is not a trading claim

The smoke configuration deliberately uses permissive gates so the entire software lifecycle can be tested quickly. Its median return under doubled costs was **-1.472%**, and not every stressed fold was positive. Calling this profitable would be dishonest. The run proves that the pipeline executes and that the controls are wired together; it does not prove a durable market edge.

The machine-readable snapshot is in [`verification.json`](verification.json). Reproduce the lifecycle with:

```bash
uv run ares verify-offline --config configs/smoke.yaml
```
