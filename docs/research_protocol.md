# Research protocol

This is the minimum standard for an ARES result. Passing it demonstrates a controlled experiment, not profitability.

## 1. Pre-register the experiment

Freeze symbol, timeframe, venues, time range, features, label family, sequence and purge settings, execution delay, costs, search space, gates, model settings, seeds, and comparison metric. Any material change creates a new Optuna study identity.

## 2. Admit only complete data

Require primary requested-range coverage and validation-venue overlap/freshness. Reject invalid schema, non-finite numbers, duplicates, missing or off-grid candles, OHLC violations, metadata mismatch, stale timestamps, venue misalignment, or excessive aligned-price divergence. Never fill missing candles as if trades occurred.

Publish only a complete immutable multi-venue generation. Save generation and evidence manifests. A rejected ingestion must not affect `CURRENT`.

## 3. Prevent leakage

Use chronological expanding folds, purge at least the label horizon, keep sequences inside allowed boundaries, and fit scalers only on training rows. Features use current/past bars only. Labels may use the declared future horizon only. Execute a signal one bar after its probability. Mutation of future data must not change earlier features or training transforms.

Walk-forward fitting uses **training-loss early stopping**. Neither evaluation
features nor labels reach fitting or weight selection. Its scaler still uses only
training rows. This intentionally replaces the earlier outer-fold `val_auc`
stopping protocol; old and new scores must not share a resumed study. Study
identity v3 binds the explicit fitting protocol, so earlier studies are retained
as history and do not contribute trials to the new study. Promotion rejects score
comparisons between historical outer-validation-selected bundles and the new
training-loss protocol. A larger score is not an override: comparable evidence
requires separate re-evaluation, with existing bundles and receipts preserved.

Walk-forward evidence records reliability bins, Brier score, calibration error,
and nearby symmetric long/short threshold perturbations. Calibration covers only
directional labels and compares a constant 0.5 probability and each fold's own
training positive rate. No calibrator is fitted. Single-class and small samples
are explicitly limited; zero-denominator skill is unavailable. There is no claim
that descriptive bins establish calibration or statistical significance.

Threshold diagnostics retain the exact configured pair. Infeasible or numerically
duplicate perturbations are reported as skipped; partial coverage is not a full
sensitivity result. Every evaluated point and the cash, buy-and-hold and always-short
benchmarks share timestamps, fees, slippage, execution delay and the configured
stress multiplier. Each fold starts flat; there is no invented terminal exit cost.
Malformed/insufficient inputs and non-finite derived evidence are rejected.

`diagnostic_provenance` binds the fitting protocol, effective configuration, ordered
OHLCV source, split rows, training/evaluation arrays and predictions with SHA-256.
Threshold evidence additionally binds timestamps/returns/predictions per fold.
Hashes identify canonical supplied values and settings; they do not authenticate source data
or replace a source commit and dependency lock. Calibration and threshold reports
disclose repeated observations when evaluation folds overlap: pooled calibration
counts each fold observation, threshold summaries use fold medians, and neither
treats overlapping folds as independent samples.

Optuna still selects models using these **research** folds. The reports cannot
establish independent performance, choose a threshold, bypass a gate or consume
the final holdout/recent replay partition. Locked holdout, replay, costs, drawdown,
solvency and independent forward-paper evidence remain separate requirements.

For a deterministic example with fixture probabilities (no trained model or market
claim), run `python scripts/calibration_demo.py`. The boundary pair `0.52/0.48`
produces an explicit partial sweep and cost comparisons. Run
`python scripts/calibration_demo.py --insufficient` to see a rejected one-bar
example (exit 2). Both work offline after installing the package.

## 4. Search honestly

Never tune historical fees downward. Record failed, pruned, non-finite, bankrupt, and gate-failed trials; none may win. Do not call search-fold scores a final holdout. Reproduce a winning trial from its stored metadata before export.

## 5. Lock final evaluation

Run `ares locked-holdout --config configs/default.yaml --holdout-bars 720` to
precommit the final chronological partition before any Optuna trial. The
workflow hashes the complete dataset, isolated research partition, quarantined
holdout, and base configuration; its embargo is at least the maximum label
horizon permitted anywhere in the search space.

Only the earlier research partition reaches search. After a passing trial is
selected, its configuration is frozen; scaler fitting, training labels, and
the independent early-stopping window remain in research history. The selected
candidate and deterministic cash, buy-and-hold, and causal-momentum baselines
are evaluated on the exact held-out window under the same delayed execution,
fees, slippage, solvency gate, and doubled-cost stress.

The commitment is atomically marked `EVALUATION_STARTED` before final inference.
A failed, interrupted, or completed evaluation is consumed and cannot be
replayed through the same commitment. Dataset/configuration mutation, short
embargoes, unexpected study identity, invalid data, and insolvency fail closed.
The separate legacy search command does not create a locked-holdout claim.
This is trusted-local historical evidence, not authenticated authorship,
profitability, live-order authority, or a substitute for a forward paper test.

## 6. Forward paper test

Load only trusted verified bundles. Use newly completed candles and every configured validation venue. Block and log operational failures; never fall back silently to an older feature row. Protect the signal log with the process lock. A paper signal is not an order.

## 7. Promotion

Use `ares prepare-challenger` for a promotable candidate. It reserves the exact configured latest bars before search, scaling, fitting, or early stopping, then replays the frozen verified bundle on that unseen window. The report binds the research cutoff, challenger manifest, complete source hash, latest source timestamp, normal-cost metrics, doubled-cost metrics, and recomputed gates.

The challenger must be solvent, finite, validation-gate-passing, replay-gate-passing, internally consistent, contained in the artifacts root, and better than the incumbent by the configured margin. `ares promote` requires `--replay-report` under the canonical configuration, rejects overlap, stale/cross-bundle/weakened reports, re-verifies both bundle and report under the promotion lock, and atomically writes a manifest-and-replay-anchored champion pointer. Never mutate an old bundle or replay report. The standalone `search` and `train` commands are research primitives and do not independently establish replay eligibility.

## 8. Claims

Allowed: “This configuration passed the documented historical and paper-validation protocol under the stated assumptions.”

Not allowed without an independently audited live record: “ARES is profitable,” “institutional grade,” “production trading ready,” or any guaranteed-return claim.
