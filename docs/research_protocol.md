# Research protocol

This protocol is the minimum standard for treating an ARES result as research rather than curve-fitting theater.

## 1. Freeze the question

Write down the symbol, timeframe, data sources, date range, features, label family, execution delay, fee/slippage assumptions, gates, and primary comparison metric before running a search. Do not change the target after seeing results without starting a new experiment.

## 2. Admit only sane data

The primary and every configured validation venue must cover the declared range and pass schema, numeric, OHLC, duplicate, continuity, UTC-grid, metadata, and freshness checks. At least one independent validation venue is mandatory. Every aligned overlap must pass both the configured distributional and latest-candle divergence gates, and the newest primary candle must align with the newest candle from every validator. Stage the full venue set first and commit nothing unless all checks pass. Save source hashes and quality reports. Never forward-fill missing market candles and pretend they traded.

## 3. Prevent leakage

Use chronological folds. Purge at least the label horizon before each validation window. Fit scalers only on the training fold. Build every feature from current or past bars. Apply a one-bar execution delay. Any shortcut here invalidates the result.

## 4. Search without moving reality

Search model and decision parameters, not historical fee assumptions. Record every trial, including failures. Keep hard gates separate from the scalar ranking score. A failed trial is ineligible for selection even when every other trial also fails. Reject models that survive only because costs, turnover, or drawdown were hidden.

## 5. Lock a final evaluation

ARES v0.1 provides purged walk-forward model selection but does not automate a nested untouched final holdout. Before making a performance claim, freeze the selected configuration and evaluate it once on data excluded from search. Repeatedly checking that holdout turns it into another training set.

## 6. Forward paper test

Run the verified bundle on newly arriving closed candles. Require fresh primary and every configured secondary feed. Require the latest timestamp to have a complete, finite feature window; do not silently fall back to an older row. Log every probability and signal, including flat signals and blocked runs. Do not silently delete outages or bad days.

## 7. Promotion criteria

A challenger must pass every configured gate, retain its complete manifest and config-consistent metadata, and exceed the champion by the required score margin. Promotion is an atomic pointer change anchored to the challenger manifest; bundles remain immutable. A new model is not better merely because it is new.

## 8. Operational discipline

Do not run overlapping ingestion, training, or promotion processes against the same state directory. ARES uses fail-fast locks, but operators still need monitoring, backups, retention, and incident procedures. Lock contention is a failed run, not permission to bypass the lock.

## 9. Claims

Allowed claim: “This configuration passed the documented historical and paper-validation protocol under stated assumptions.”

Disallowed claim: “ARES is profitable,” unless supported by an independently audited live record. Backtests, synthetic tests, and screenshots are not proof.
