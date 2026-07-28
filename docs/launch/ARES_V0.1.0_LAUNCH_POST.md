# ARES Engine v0.1.0 is open source

ARES Engine v0.1.0 is now open source: an ML trading research and paper-inference engine, not a signal-selling bot.

The hard work was not adding another indicator. It was making three failure-prone boundaries explicit:

1. **Consistent multi-exchange data.** ARES pulls public Coinbase and Kraken OHLCV through CCXT, measures real requests, pages, rows, retries, cursor movement, and duplicates, then rejects incomplete, stale, malformed, off-grid, misaligned, or materially divergent inputs. A complete immutable generation becomes visible through one atomic pointer, so readers cannot mix venues from different runs.
2. **Evaluation that resists self-deception.** LSTM and causal TCN candidates use Optuna, purged walk-forward validation, training-fold-only scaling, one-bar-delayed execution, fees, slippage, doubled-cost stress, and terminal bankruptcy semantics. Full data and material configuration define the study identity. This is model selection—not a locked nested final holdout or proof of profitability.
3. **Artifacts and inference that fail closed.** Manifest-verified bundles bind the model, scaler, features, configuration, and evidence. Promotion is atomic; paper inference refuses tampered, mismatched, non-finite, stale, open, divergent, or insufficient inputs.

ARES makes false confidence harder. It provides paper-only output, contains no live-order path, makes no profitability claim, and is not investment advice.

Repository: <https://github.com/ArjiaTechnologies/ares-engine>

```bash
git clone https://github.com/ArjiaTechnologies/ares-engine.git
cd ares-engine
uv sync --extra dev --extra ml
uv run ares doctor
uv run ares demo --bars 500
```

Select an already-closed 15-day interval ending 72 hours ago, then verify real public endpoints:

```bash
eval "$(uv run python - <<'PY'
from datetime import UTC, datetime, timedelta
end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=72)
start = end - timedelta(days=15)
print(f'ARES_START={start:%Y-%m-%dT%H:%M:%SZ}')
print(f'ARES_END={end:%Y-%m-%dT%H:%M:%SZ}')
PY
)"
uv run ares verify-public-ingestion \
  --primary coinbase --validation kraken \
  --symbol ETH/USD --timeframe 1h \
  --start "$ARES_START" --end "$ARES_END" \
  --page-limit 60 --retries 3 \
  --output artifacts/public-ingestion-audit
```

Clone it.
Run the public-ingestion verification.
Try to break it.
Open an issue with the evidence.
