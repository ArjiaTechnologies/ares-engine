# Research protocol

This is the minimum standard for an ARES result. Passing it demonstrates a controlled experiment, not profitability.

## 1. Pre-register the experiment

Freeze symbol, timeframe, venues, time range, features, label family, sequence and purge settings, execution delay, costs, search space, gates, model settings, seeds, and comparison metric. Any material change creates a new Optuna study identity.

## 2. Admit only complete data

Require primary requested-range coverage and validation-venue overlap/freshness. Reject invalid schema, non-finite numbers, duplicates, missing or off-grid candles, OHLC violations, metadata mismatch, stale timestamps, venue misalignment, or excessive aligned-price divergence. Never fill missing candles as if trades occurred.

Publish only a complete immutable multi-venue generation. Save generation and evidence manifests. A rejected ingestion must not affect `CURRENT`.

## 3. Prevent leakage

Use chronological expanding folds, purge at least the label horizon, keep sequences inside allowed boundaries, and fit scalers only on training rows. Features use current/past bars only. Labels may use the declared future horizon only. Execute a signal one bar after its probability. Mutation of future data must not change earlier features or training transforms.

## 4. Search honestly

Never tune historical fees downward. Record failed, pruned, non-finite, bankrupt, and gate-failed trials; none may win. Do not call search-fold scores a final holdout. Reproduce a winning trial from its stored metadata before export.

## 5. Lock final evaluation

ARES does not yet automate nested final holdout selection. Freeze the chosen configuration and evaluate it once on data excluded from all search and early stopping. Repeated inspection contaminates that holdout.

## 6. Forward paper test

Load only trusted verified bundles. Use newly completed candles and every configured validation venue. Block and log operational failures; never fall back silently to an older feature row. Protect the signal log with the process lock. A paper signal is not an order.

## 7. Promotion

The challenger must be solvent, finite, gate-passing, internally consistent, contained in the artifacts root, and better than the incumbent by the configured margin. Reverify it under the promotion lock and atomically write a manifest-anchored champion pointer. Never mutate an old bundle.

## 8. Claims

Allowed: “This configuration passed the documented historical and paper-validation protocol under the stated assumptions.”

Not allowed without an independently audited live record: “ARES is profitable,” “institutional grade,” “production trading ready,” or any guaranteed-return claim.
