# Fable 5 audit — requirements matrix

Status legend: **EXECUTED** (ran here, evidence exists) · **EXECUTED-SIM**
(ran here against exact-shape simulators; not live) · **STATIC** (reviewed,
not executable in this sandbox) · **BLOCKED-ENV** (impossible here; portable
path delivered) · **N/A** (does not exist in this design; documented).

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Archive safety (traversal/absolute/symlinks) | EXECUTED | command log §1 |
| 2 | Source identity: inner SHA-256, tree `e48e6ad8…`, 70 files | EXECUTED | command log §1 |
| 3 | Read every tracked file; architecture map | EXECUTED | audit §2; 5,601 baseline lines read |
| 4 | Clean install `pip install -e ".[dev,ml]"` | EXECUTED | failed on baseline (F-006), passes after fix |
| 5 | compileall / ruff check / ruff format --check / mypy src | EXECUTED | all clean at final commit |
| 6 | pytest with branch coverage; skips explained | EXECUTED | 186 passed; 0 skipped on 3.11/3.12/3.13 with TF installed; the single conditional skip (`test_ml_smoke`) only triggers without any Keras runtime |
| 7 | Python 3.11 / 3.12 / 3.13 validation | EXECUTED | full suite on all three (source-built interpreters) |
| 8 | build sdist+wheel; twine check | EXECUTED | both PASSED |
| 9 | Wheel-only CLI outside checkout; config discovery; packaged files | EXECUTED | F-010 fixed; 14 commands + offline demo green |
| 10 | Docker build & run | STATIC + BLOCKED-ENV | no daemon; Dockerfile hardened (non-root), layer review clean |
| 11 | CI workflows: YAML, permissions, matrix, releases, no PyPI publish | STATIC + EXECUTED (commands locally) | ci/codeql/release reviewed; exact CI commands green locally; uv resolution proven; release job repo-guarded; no PyPI publishing exists |
| 12 | Live bounded Coinbase+Kraken ETH/USD 1h multi-page ingestion | BLOCKED-ENV | proxy 403 evidence; portable `ares verify-public-ingestion` + handoff delivered |
| 13 | Pagination advance / short pages / repeated cursor / coverage | EXECUTED-SIM | real CCXT vs wire sims; both repeated-cursor dialects tested |
| 14 | Dedup, sorting, UTC grid, valid OHLCV | EXECUTED | sim ingestion + quality gates + reference tests |
| 15 | Idempotent reruns; canonical bytes identical | EXECUTED | bit-identical file hashes across reruns |
| 16 | Partial venue failure cannot partially replace canonical state | EXECUTED | staged validation + journaled commit + crash injection |
| 17 | Temp-file hygiene; DuckDB views reference intended files | EXECUTED | F-005 reproduced and fixed; regression test |
| 18 | Cross-venue newest-alignment, divergence on aligned candles, freshness on completed candles | EXECUTED | quality tests + sim scenarios + open-candle gates |
| 19 | Failure injection: venue failure orders, empty/duplicate/overlap/short pages, malformed rows, rate limit, timeout, interruption, concurrency | EXECUTED / EXECUTED-SIM | wire-sim battery + crash injection + real subprocess locks |
| 20 | Feature math vs hand calculations (EMA/RSI/Bollinger/returns/vol/volume/range; warm-up; NaN; ordering; parity) | EXECUTED | 19 reference tests, loop-based independent implementation |
| 21 | k-ahead labels: horizon, dead zone boundaries, tail NaN, alignment | EXECUTED | reference tests incl. strict-boundary fixtures |
| 22 | Triple-barrier: barriers, expiry, exact touch, same-candle ambiguity→neutral, own-candle exclusion, no intrabar guessing | EXECUTED | hand fixtures + randomized reference agreement |
| 23 | Leakage: scalers train-only; folds chronological; purge ≥ horizon; sequence boundaries; no future at time t | EXECUTED | bitwise future-mutation invariance; instrumented scaler capture; purge arithmetic proof |
| 24 | Search does not use a final untouched holdout | N/A + documented | no such holdout exists in the design; disclosed in README/protocol/findings F-017 |
| 25 | Backtest vs independent reference: delay, costs, reversals, Sharpe, drawdown, exposure, hit rate, trades, stress | EXECUTED | 17 scenario tests + 25 randomized, tolerance 1e-11 |
| 26 | Deterministic bounded Keras lifecycle: seeds, shapes, ES, reload parity, CPU | EXECUTED | lifecycle tests; leakage test asserts run-to-run bitwise determinism |
| 27 | TCN: causal padding, dilation, residuals, reload parity | EXECUTED | structural + lifecycle tests |
| 28 | Optuna: failed/pruned/gate-failed/NaN/inf cannot win; direction; penalty signs; metadata; rebuild; bounded | EXECUTED | adversarial suite; F-003/F-004 reproduced & fixed |
| 29 | Hostile bundles fail closed (all listed classes) | EXECUTED | 21-test battery incl. hard links, coherent tamper, race |
| 30 | Reload prediction parity within justified tolerance | EXECUTED | atol 1e-7 (same machine/backend; Keras float32 round-trip) |
| 31 | Promotion: all scenarios incl. crashes, corrupt parties, concurrency, permissions | EXECUTED | promotion suite + 2-process race |
| 32 | Paper inference refuses every listed invalid input; newest completed row; no fallback; exact scaler/order/thresholds; read-only | EXECUTED | fail-closed battery + F-007 fix |
| 33 | No order APIs / credentials / hidden execution anywhere | EXECUTED | full-tree scan + runtime forbidden-method instrumentation |
| 34 | Scheduler: overlap locks, visible failures, no promote-after-fail, idempotency, lock release, log secret-safety | EXECUTED | scheduler suite (synthetic time) |
| 35 | Documentation truthfulness separation | EXECUTED | README/CHANGELOG/audit docs; claim discipline kept |
| 36 | verify-public-ingestion command + script + artifact validation + harness tests | EXECUTED | harness + 10 regression tests; fixture runs cannot claim live |
| 37 | Environment-block evidence (curl/requests/ccxt/DNS/proxy) | EXECUTED | command log §9 |
