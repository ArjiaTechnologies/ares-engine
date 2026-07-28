# Sol findings

All listed Critical/High findings are resolved. Line references identify the corrected implementation on `sol/hardening`.

## SOL-001 — forged live-ingestion success

| Field | Detail |
|---|---|
| Severity | High |
| File and line | `src/ares_engine/public_audit.py:998` |
| Reproduction | Constructed a non-fixture report with `live_public_endpoints_reached=false`, `overall_passed=true`, checksum-consistent fake Parquet, and ran the official validator. |
| Observed | The Fable validator could accept report assertions/checksums without establishing valid Parquet, real traces, or recomputed gates. |
| Expected | A live success must be proven from exact, readable, internally consistent artifacts and real per-venue network activity. |
| Root cause | Success fields and file hashes were treated as evidence rather than untrusted inputs. |
| Impact | A fabricated audit directory could be presented as real exchange compatibility. |
| Fix | Strict v2 schemas/exact file sets; manifest and checksum checks; Parquet reopen/schema/content/range/quality recomputation; two-run trace and hash checks; recomputed overall verdict. |
| Regression test | `test_forged_live_report_and_fake_parquet_are_rejected` and `tests/test_public_audit_harness.py` corruption battery. |
| Fix commit | `00fe9cc` |

## SOL-002 — mixed canonical generation visible after crash

| Field | Detail |
|---|---|
| Severity | High |
| File and line | `src/ares_engine/data/storage.py:378` |
| Reproduction | Interrupted the Fable journal between venue renames and observed that readers could address old/new files independently until later recovery. |
| Observed | Roll-forward gave eventual convergence, not atomic reader visibility. |
| Expected | Readers see the complete old generation or complete new generation, never a mix. |
| Root cause | Canonical identity was spread across several mutable pathnames. |
| Impact | Training, validation, or inference could consume internally inconsistent venues. |
| Fix | Complete immutable generation, verified manifest/DuckDB, fsync, atomic `CURRENT`, and capture-once readers. Recovery only removes abandoned state. |
| Regression test | `tests/test_ingest_atomicity.py`, `tests/test_storage_security.py`, concurrent-reader and subprocess-fault tests. |
| Fix commit | `00fe9cc` |

## SOL-003 — catastrophic short loss could sign-flip equity

| Field | Detail |
|---|---|
| Severity | High |
| File and line | `src/ares_engine/backtest.py:104` |
| Reproduction | Applied a short position to a one-bar +200% asset return, then a later loss. |
| Observed | Fable behavior allowed equity below zero and subsequent multiplication could make it positive. |
| Expected | Insolvency is terminal, cannot improve scores, and fails candidate gates. |
| Root cause | Unbounded multiplicative compounding lacked explicit limited-liability/insolvency semantics. |
| Impact | Return, Sharpe, drawdown, hit rate, and search ranking could reward a bankrupt strategy. |
| Fix | Clamp to zero, persist terminal bankruptcy, stop compounding, evaluate pre-bankruptcy path, expose metrics, and require solvency. |
| Regression test | `test_catastrophic_short_return_causes_terminal_bankruptcy` plus exact/randomized reference tests. |
| Fix commit | `8dbac5d` |

## SOL-004 — page counts were inferred, not observed

| Field | Detail |
|---|---|
| Severity | Medium |
| File and line | `src/ares_engine/data/providers.py:22`, `src/ares_engine/public_audit.py:118` |
| Reproduction | Wrapped a provider that executed once over a window whose date/limit arithmetic suggested five pages. |
| Observed | Fable audit wrapper could report estimated pages unrelated to actual calls. |
| Expected | Per-run/per-venue counters come from actual transport/provider boundaries and reset between runs. |
| Root cause | The reporting wrapper lacked provider-native telemetry. |
| Impact | Pagination and endpoint-contact claims were unsubstantiated. |
| Fix | Added actual HTTP, fetch, page, row, cursor, retry, status, rate-limit, empty/short page telemetry and strict trace-summary validation. |
| Regression test | `test_request_pages_are_measured_not_estimated`, wire ingestion tests, and real evidence traces. |
| Fix commit | `00fe9cc` |

## SOL-005 — stale Optuna studies could cross material changes

| Field | Detail |
|---|---|
| Severity | Medium |
| File and line | `src/ares_engine/search.py:72` |
| Reproduction | Changed only `high` or RSI period while holding rows/start/end/close constant. |
| Observed | Fable fingerprint remained identical. |
| Expected | Every material data or experiment change receives a distinct deterministic study identity. |
| Root cause | Identity covered only row/time/close summaries and a few labels. |
| Impact | Old trials could be selected for incompatible data/configuration. |
| Fix | Canonical full normalized row serialization plus venue, feature, label, sequence/model, fold/scaling, backtest, threshold/search, gate, seed, and schema identity. |
| Regression test | `test_full_ohlcv_and_material_config_are_in_study_identity` and `tests/test_search_adversarial.py`. |
| Fix commit | `8dbac5d` |

## SOL-006 — bundle checks did not close all load-time integrity gaps

| Field | Detail |
|---|---|
| Severity | Medium |
| File and line | `src/ares_engine/bundles.py:262`, `src/ares_engine/bundles.py:291` |
| Reproduction | Mutated files after verification; rehashed coherent metadata; substituted scaler/model shapes; introduced links, extra/nested/oversized/non-finite payloads. |
| Observed | Hashing alone could not prove metadata/runtime compatibility or close a verify/load race. |
| Expected | Any hostile or raced payload fails before unverified bytes influence inference. |
| Root cause | Verification and deserialization were separate trust steps and runtime shapes were not anchored comprehensively. |
| Impact | Incorrect or changed model/scaler objects could load under superficially valid manifests. |
| Fix | Exact payload/type/link/size checks, finite JSON, private re-hashed snapshot loading, and feature/scaler/model input/output validation. |
| Regression test | `tests/test_bundle_security.py` including model/scaler mismatch and concurrent modification. |
| Fix commit | `8dbac5d` |

## SOL-007 — unsafe configuration components and non-finite values

| Field | Detail |
|---|---|
| Severity | Medium |
| File and line | `src/ares_engine/config.py:16`, `src/ares_engine/config.py:60` |
| Reproduction | Supplied `../kraken`, path separators, or NaN gate thresholds. |
| Observed | Earlier configuration accepted values capable of escaping intended storage paths or contaminating comparisons. |
| Expected | Identifiers are safe components and every numeric configuration is finite. |
| Root cause | Empty-string normalization was mistaken for structural validation; Pydantic infinity/NaN was allowed. |
| Impact | Filesystem path manipulation or fail-open/undefined numerical behavior. |
| Fix | Restricted CCXT exchange/symbol syntax and globally disabled non-finite configuration values. |
| Regression test | `test_exchange_identifiers_cannot_escape_storage_paths`, `test_non_finite_configuration_is_rejected`. |
| Fix commit | `00fe9cc` |

## SOL-008 — malformed evidence could crash the validator

| Field | Detail |
|---|---|
| Severity | Medium |
| File and line | `src/ares_engine/public_audit.py:768`, `src/ares_engine/public_audit.py:998` |
| Reproduction | Replaced trace venue payloads with scalars and supplied invalid report date/timeframe values. |
| Observed | Intermediate Sol versions raised exceptions rather than returning validation problems. |
| Expected | Hostile evidence always fails closed with a nonzero CLI result and enumerated problems. |
| Root cause | Nested types and parsed-range validity were used before complete defensive checks. |
| Impact | Denial of validation and ambiguous automation outcomes. |
| Fix | Guard every nested payload and isolate date/timeframe parsing before semantic checks. |
| Regression test | Malformed trace, invalid timeframe, and invalid range cases in `tests/test_public_audit_harness.py`. |
| Fix commit | `00fe9cc` |

## SOL-009 — CI lifecycle smoke requested too few bars

| Field | Detail |
|---|---|
| Severity | Medium |
| File and line | `.github/workflows/ci.yml:48`, `.github/workflows/ci.yml:100` |
| Reproduction | Compared workflow `--bars 1500` with the smoke configuration's calculated minimum of about 1,805. |
| Observed | The intended ML/Docker lifecycle would fail before testing the lifecycle. |
| Expected | CI supplies a valid deterministic dataset and then tests behavior. |
| Root cause | Workflow fixture size was not updated with validation requirements. |
| Impact | False CI failures and missing lifecycle evidence. |
| Fix | Use 2,000 bars; add complete Python, ML, package, dependency, Docker, CodeQL, dependency-review, and manual-live workflows; disable release. |
| Regression test | Local host and real Docker 2,000-bar lifecycle; private Actions run pending in initial draft. |
| Fix commit | `34472f5` |

## SOL-010 — concurrent paper processes could interleave the signal log

| Field | Detail |
|---|---|
| Severity | Low |
| File and line | `src/ares_engine/live.py:243` |
| Reproduction | Held the signal-log lock while a second paper inference attempted append. |
| Observed | Earlier code relied on a simple append without an explicit process-level ownership boundary/durability flush. |
| Expected | One durable complete JSON record or a fail-closed lock error. |
| Root cause | Paper logging was treated as an incidental side effect. |
| Impact | Concurrent records could be ambiguous or lose durability during a crash. |
| Fix | Fail-fast `FileLock`, one JSON line under lock, flush and fsync. |
| Regression test | `test_signal_log_lock_contention_fails_closed` and no-log/happy-path tests. |
| Fix commit | `8dbac5d` |

## SOL-011 — audit and repository presentation overstated evidence

| Field | Detail |
|---|---|
| Severity | Low |
| File and line | `README.md:1`, `docs/audits/archive/fable5-untrusted/README.md:1` |
| Reproduction | Compared Fable audit/README claims with its command log and residual risks. |
| Observed | “Third-party,” Docker/CI PASS, journal atomicity, real endpoint compatibility, and 16/17 commit language conflicted. |
| Expected | Claims are limited to executed evidence and historical AI review is labeled accurately. |
| Root cause | Documentation inherited review assertions without cross-checking execution records. |
| Impact | Readers could over-trust incomplete software assurance. |
| Fix | Archive warning, authoritative Sol evidence, corrected architecture/data layout/release status, no performance language. |
| Regression test | Tracked-document review and claim searches. |
| Fix commit | Documentation commit containing the Sol audit; exact hash recorded in the final command log. |

## SOL-012 — obsolete journal helpers remained after architecture replacement

| Field | Detail |
|---|---|
| Severity | Informational |
| File and line | `src/ares_engine/data/storage.py` (removed code) |
| Reproduction | Searched all call sites after generation storage was introduced. |
| Observed | Legacy temporary suffix and storage helpers were unreachable. |
| Expected | One publication architecture with no misleading dead recovery path. |
| Root cause | Incremental replacement left compatibility code with zero references. |
| Impact | Maintenance ambiguity and risk of accidental reintroduction. |
| Fix | Removed only after repository-wide reference proof. |
| Regression test | Full suite, import/compile checks, and repository reference search. |
| Fix commit | `00fe9cc` |
