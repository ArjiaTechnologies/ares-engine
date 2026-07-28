# Changelog

All notable changes to ARES Engine are documented here.

## Unreleased — Sol independent hardening (2026-07-28)

- Replaced eventual multi-file roll-forward with immutable, manifest-verified multi-venue generations and one atomic `CURRENT` pointer. CLI, scheduler, DuckDB, training, promotion, and paper paths capture one generation per operation.
- Rebuilt public-ingestion evidence as a strict v2 format. The validator reopens and recomputes evidence instead of trusting success flags or checksums over arbitrary bytes.
- Instrumented real provider and HTTP boundaries with per-run/per-venue request, page, cursor, retry, status, raw-row, normalization, deduplication, overlap, empty-page, and short-page telemetry.
- Expanded Optuna study identity to canonical full OHLCV plus feature, label, sequence, fold, scaling, cost, threshold, gate, model, training, seed, code, and schema configuration.
- Defined terminal bankruptcy: equity clamps to zero, cannot recover through sign-flipped arithmetic, and bankrupt candidates fail validation/search.
- Hardened bundle verification and loading against malformed manifests, unlisted/nested/symlinked/hard-linked/oversized/non-finite payloads, metadata and runtime-shape mismatch, and verify/load races.
- Hardened promotion, champion pointers, paper inference, paper-log locking, configuration path components, and non-finite inputs.
- Added adversarial, reference, crash, concurrency, property-based, and real-provider tests. The Sol local suite contains 245 passing tests with 90% total branch-aware coverage and at least 90% for every designated critical module.
- Executed credential-free live Coinbase/Kraken ETH/USD 1h ingestion twice over 360 closed candles per venue; independent evidence validation passed.
- Built and ran the Linux ARM64 ML container as non-root with Python 3.11.15 and TensorFlow 2.21.0, including the complete offline lifecycle and graceful shutdown.
- Added Python 3.11–3.13, ML, package, installed-wheel, dependency-audit, Docker, CodeQL, dependency-review, and manual-only public-ingestion workflows with least-privilege permissions and pinned actions.
- Raised the PyArrow minimum to patched 23.0.1 after clean CI exposed `PYSEC-2026-113` in the previously permitted 21.0.0 resolution.
- Disabled release automation until a separate authorized public-release task.
- Archived the Fable 5 review as superseded, untrusted historical material and made the Sol audit authoritative.

## 0.1.0 source snapshot — 2026-07-24

- Initial research and paper-inference engine: public Coinbase/Kraken ingestion, Parquet/DuckDB storage, features, labels, LSTM/TCN models, Optuna, walk-forward validation, cost-aware backtesting, bundles, promotion, scheduling, tests, Docker, and documentation.
- This entry describes the preserved source snapshot; no release tag or GitHub release is created by the Sol hardening task.
