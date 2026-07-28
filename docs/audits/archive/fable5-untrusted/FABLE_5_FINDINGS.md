# Fable 5 audit — findings register

Every finding lists severity, location (file:lines at the baseline commit
`b7736ac`, tree `e48e6ad8…`), reproduction, observed/expected behavior, root
cause, impact, fix, regression test, and fix commit. "Reproduced" means the
defective behavior was demonstrated on the unmodified baseline before repair.

---

## F-001 — Torn multi-venue canonical commit on crash — **High** (fixed)

- **Location:** `src/ares_engine/data/ingest.py:197-206` (baseline)
- **Reproduction:** monkeypatch `commit_staged`/`write_market` to raise after the
  first venue's rename; run `ingest_market_data` with two fake venues
  (`tests/test_ingest_atomicity.py::test_interrupted_commit_is_rolled_forward_before_next_ingestion`).
- **Observed:** kraken's canonical Parquet updated, coinbase's stale; no journal,
  no repair mechanism; canonical state mixed across venues until some later
  ingestion happened to overwrite both.
- **Expected:** a failed ingestion must never leave mixed old/new canonical state.
- **Root cause:** per-venue `write_market` renames were sequential and unjournaled.
- **Impact:** crash mid-commit corrupted the multi-venue invariant. Downstream
  gates (latest-candle alignment) blocked paper signals, so bad signals were not
  produced, but the canonical store itself was torn.
- **Fix:** journaled two-phase commit: stage all venues (fsynced), persist a
  rename journal, rename, delete journal; `recover_pending_commit` rolls an
  interrupted journal forward under the ingestion lock before any new fetch.
- **Regression tests:** `test_interrupted_commit_is_rolled_forward_before_next_ingestion`,
  `test_recovery_runs_automatically_at_next_locked_ingestion`,
  `test_staging_failure_leaves_no_canonical_or_temp_residue`.
- **Commit:** `7bcc62c`.

## F-002 — Default long-range ingestion permanently impossible — **High** (fixed)

- **Location:** `src/ares_engine/data/ingest.py:140-141` (baseline)
- **Reproduction:** wire-simulator venue with a Kraken-style total history depth
  cap serving as validator while the primary covers the configured 2023 start
  (`tests/test_ccxt_wire_ingestion.py::test_depth_limited_validator_is_accepted_when_overlap_suffices`
  fails on baseline code).
- **Observed:** `incomplete_start_coverage` for the validator fails the whole
  ingestion forever; with `configs/default.yaml` (since=2023-01-01, Kraken
  validator) no first ingestion can ever commit, and reruns never improve.
- **Expected:** validators exist for overlap/freshness/price sanity; only the
  primary needs full requested-range coverage.
- **Root cause:** `expected_start` was applied uniformly to every venue.
- **Impact:** the primary documented workflow (`ares ingest` with the default
  config) could never succeed against real venues with capped history depth.
  (The real Kraken OHLC depth cap of ~720 candles is asserted from its public
  API documentation and modeled in the simulator; it was not verifiable live
  from this sandbox.)
- **Fix:** start coverage binds the primary only; validators keep end-coverage,
  freshness, newest-candle alignment, and `min_cross_venue_overlap` gates.
- **Regression tests:** depth-limited validator accepted; depth-limited PRIMARY
  still fails closed in both venue dialects.
- **Commit:** `7bcc62c`.

## F-003 — Non-finite Optuna objective can win selection — **High** (fixed)

- **Location:** `src/ares_engine/search.py:86-104` (baseline)
- **Reproduction:** stub `run_walk_forward` returning summaries scored
  `nan, inf, 0.25, -0.5`; run `run_search`
  (`tests/test_search_adversarial.py::test_nan_and_infinite_scores_can_never_win`,
  red on baseline: optuna logs "Best is trial 1 with value: inf").
- **Observed:** the `inf` trial was selected; its parameters were exported to
  `best_params.json`/`best_config.yaml`.
- **Expected:** NaN/infinite metrics cannot rank as winners.
- **Root cause:** winner filter checked state/`passed`/non-None but not finiteness.
- **Impact:** a degenerate fold metric (e.g. runaway synthetic return) could
  crown a garbage configuration for export and promotion.
- **Fix:** `math.isfinite(float(trial.value))` required for eligibility.
- **Commit:** `d0b4958`.

## F-004 — Stale-vintage Optuna trials win after data changes — **Medium** (fixed)

- **Location:** `src/ares_engine/search.py:71-78` (baseline)
- **Reproduction:** two `run_search` calls with the same study name/storage on
  different datasets; the first (old-data) trial scored 5.0, the second run's
  honest score was 0.1; selection returned the 5.0 trial
  (reproduced in-session; regression:
  `test_changed_dataset_cannot_inherit_stale_trials`).
- **Root cause:** `load_if_exists=True` with a fixed study name; selection scans
  all study trials regardless of the data they were evaluated on.
- **Impact:** every scheduler deep cycle after the first silently mixed data
  vintages during selection.
- **Fix:** study identity scoped by a dataset fingerprint
  (rows/start/end/close-hash/symbol/timeframe): identical data resumes,
  changed data starts fresh.
- **Commit:** `d0b4958`.

## F-005 — DuckDB canonical view poisoned by crashed temp files — **Medium** (fixed)

- **Location:** `src/ares_engine/data/storage.py:56-67,87,109` (baseline)
- **Reproduction:** place a `.1h.crashed.parquet` staging-style file (exactly what
  an interrupted `write_market` leaves) beside a canonical file; `sync_duckdb`;
  the `ohlcv` view returned 6 rows where 3 were real, including poisoned prices
  (reproduced in-session; regression:
  `test_crashed_temp_file_cannot_poison_duckdb_view`).
- **Root cause:** temp files used the canonical `.parquet` suffix, and both
  `rglob` and the SQL glob match dot-prefixed files.
- **Fix:** `.parquet.tmp` staging suffix, explicit sorted non-hidden file list
  for the view, orphan sweep under the ingestion lock.
- **Commit:** `7bcc62c`.

## F-006 — `[ml]` extra unresolvable off x86_64 — **Medium** (fixed)

- **Location:** `pyproject.toml:39-42` (baseline)
- **Reproduction:** `python -m pip install -e ".[dev,ml]"` on linux-aarch64 →
  "No matching distribution found for tensorflow-cpu" (log preserved in the
  command log).
- **Root cause:** `tensorflow-cpu` ships wheels only for x86_64 Linux/Windows;
  the marker assumed everything non-Darwin has it.
- **Impact:** the mandated install path failed on ARM Linux hosts and arm64
  Docker builds of the `ml` stage.
- **Fix:** platform-machine-aware markers; plain `tensorflow` elsewhere
  (verified: TF 2.21.0 aarch64 installs and passes the full ML suite).
- **Commit:** `176260e`.

## F-007 — Paper inference accepted in-progress candles via direct API — **Medium** (fixed)

- **Location:** `src/ares_engine/live.py:84-152` (baseline)
- **Identified by:** call-path review (the CLI/scheduler path was safe because
  ingestion drops open candles; the public `generate_paper_signal` contract was
  not). Behavior confirmed by the new tests failing against the baseline logic:
  a frame containing the currently-forming candle produced a signal from
  partial data.
- **Fix:** with `drop_open_candle` configured, any row at or after the bar
  containing the as-of reference in the primary or any secondary feed raises
  `DataQualityError` before inference.
- **Regression tests:** in-progress candle in primary and in secondary feeds.
- **Commit:** `7356245`.

## F-008 — Verify-then-load race in bundle loading — **Medium** (fixed)

- **Location:** `src/ares_engine/bundles.py:196-219` (baseline)
- **Demonstration:** instrumented `verify_bundle` that tampers `scaler.joblib`
  immediately after verification returns; baseline `load_bundle` would
  deserialize the tampered bytes
  (`test_modification_between_verify_and_load_fails_closed`).
- **Fix:** every manifest-listed file is copied into a private snapshot and
  re-hashed against the verified manifest before keras/joblib deserialization;
  post-verification modification now raises `BundleIntegrityError`.
- **Commit:** `eb07a96`.

## F-009 — mypy --strict broken; CI never ran it — **Medium** (fixed)

- **Location:** 17 files; `.github/workflows/ci.yml` (baseline)
- **Reproduction:** `mypy src` → 31 errors (missing pandas/yaml stubs plus
  genuine defects: `float(None)` reachability and a FrozenTrial/Trial type
  mismatch in `search.py`, an invalid `SecondaryInput` alias in `live.py`,
  a shadowed variable in `quality.py`).
- **Fix:** stubs added to dev extra, mypy overrides for untyped libraries,
  all annotations repaired without behavior change; CI now runs
  `ruff format --check` and `mypy src`.
- **Commits:** `c26d59d`, `b2e1cb8`, `176260e` (dev deps).

## F-010 — Installed wheel unusable outside a checkout — **Low** (fixed)

- **Location:** `src/ares_engine/cli.py` config defaults (baseline)
- **Reproduction:** wheel-only venv, `ares demo` from an empty directory →
  BadParameter (configs/smoke.yaml missing).
- **Fix:** default configs shipped inside the package with a filesystem-first
  resolver; drift pinned by a byte-equality test. Verified: full CLI + demo
  green from outside the checkout on a wheel-only install.
- **Commit:** `1b8849f`.

## F-011 — Offline lifecycle accepted impossible --bars values — **Low** (fixed)

- **Location:** `src/ares_engine/cli.py` verify-offline/demo (baseline)
- **Reproduction:** `ares verify-offline --bars 1800` with the smoke config →
  late confusing "needs at least 1746 complete sequence samples; only 1741".
- **Fix:** exact `minimum_required_bars` arithmetic (warm-up + lookback +
  train + purge + validation), early refusal with the required number; the
  arithmetic is pinned to empirical warm-up and boundary sample counts.
- **Commit:** `8fff629`.

## F-012 — Docker root execution; unguarded release job; ruff drift — **Low** (fixed)

- Docker now creates and runs as non-root `ares` (uid 1000) with writable
  data/artifact dirs; the tag-driven release job is restricted to the canonical
  repository so forks/mirrors cannot auto-release; ruff 0.14 (inside the
  project's own `>=0.9,<1` pin) lint errors and 19 unformatted files fixed.
- **Commits:** `b2e1cb8`, `4381192`.

## F-013 — new_rows reported fetch size, not additions — **Informational** (fixed)

- `IngestionResult.exchanges[].new_rows` now reports actual canonical additions;
  pinned by `test_new_rows_reports_actual_additions_not_fetch_size`. Commit `7bcc62c`.

## F-014 — Garbage champion pointer raised raw JSONDecodeError — **Informational** (fixed)

- Now `BundleIntegrityError("Champion pointer is not valid JSON")`; commit `e500aeb`.

## F-015 — Dead code and unused severity level — **Informational** (documented)

- `ares paper`'s `data_quality_passed` failure branch is unreachable (inference
  raises instead); `Severity` literal "warning" is never used — every quality
  issue is a hard error. Left as-is deliberately; noted for a future cleanup.

## F-016 — Non-advancing-cursor guard has two failure dialects — **Informational** (documented + tested)

- Through real CCXT, repeated pages are since-filtered into an empty batch:
  pagination stops early and the coverage gate blocks the run (no exception).
  The explicit `non-advancing cursor` error fires only when duplicate rows
  survive filtering. Both paths are fail-closed and both are now tested.

## F-017 — No locked final holdout — **Informational** (already disclosed)

- Search and selection reuse the configured walk-forward folds; early stopping
  monitors the same fold validation window that is later scored. The README and
  research protocol already disclose this; wording extended. Scores are
  selection estimates, not untouched-holdout results.
