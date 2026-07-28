# Sol independent audit

## Verdict

**CONDITIONAL** — all local, quantitative, live-ingestion, packaging, Docker, and supported private GitHub Actions requirements pass. GitHub CodeQL and Dependency Review cannot execute their licensed analysis on this private repository because GitHub Code Security is not enabled; their workflows are pinned and activate automatically if that capability becomes available. This is a non-critical external limitation. There are no unresolved Critical or High source defects.

Audit date: 2026-07-28

Audited branch: `sol/hardening`

Evidence authority: independently reproduced execution, references, source/artifacts, reviewed tests, then documentation. Fable 5 and the dossier were treated as untrusted leads.

## Source identity

| Item | Verified identity |
|---|---|
| Original source tree | `e48e6ad8e45643ca06e6fff21cb27d5044c04a87` |
| Original inner archive SHA-256 | `0a18e017996746764325b18185f4de320898e41c84bbc91ab695b1979b52a7b0` |
| Fable baseline commit | `b7736ac8d15207537d71fb1e8c64fa4e8dd2b844` |
| Fable final commit | `a25dba4bfad05e79e7488674cf699fbd45687746` |
| Fable final tree | `cd2790ceffc0b77a4838a40b60b2caf44ea467cb` |
| Sol data/evidence repair | `00fe9cc` |
| Sol research/inference repair | `8dbac5d` |
| Sol CI repair | `34472f5` |
| Sol documentation/archive | `1cd3fed` |
| Sol duplicate-run repair | `da0df74` |
| Sol private-GHAS workflow repair | `ad19c4a`, `8b52bb3` |
| Sol patched PyArrow floor | `bf92447` |

Every supplied Fable checksum matched. `git bundle verify`, `git fsck --full`, history inspection, the complete baseline-to-Fable diff, and both reconstructed source archives were independently checked. The supplied correction-prompt filename was not present; the complete attached request was available and used. The dossier rendered as six pages and was used only as architectural reference.

## Independent review

Every tracked Python source, test, configuration, workflow, Docker file, script, packaging file, and document was read. The review covered configuration, providers, pagination, normalization, quality, Parquet/DuckDB, generation publication, features, labels, sequences, folds, scaling, LSTM/TCN, Optuna, backtesting, bundles, promotion, paper inference, scheduling, locking, CLI, public evidence, packaging, Docker, CI, and release automation. See [architecture](../../architecture.md) and [findings](SOL_FINDINGS.md).

Searches and adversarial tests covered leakage, silent exceptions, bounded retry behavior, unsafe deserialization, TOCTOU, process races, timestamp assumptions, NaN/infinity, partial writes, credential/order surfaces, machine paths, dead code, and unsupported documentation claims.

## Known blocker disposition

| Blocker | Reproduced result | Corrected evidence |
|---|---|---|
| Forged live evidence | Fable validator accepted success fields/checksums without proving real live Parquet/traces | Strict v2 validator reopens Parquet and recomputes exact files, schemas, rows, timestamps, quality, traces, manifests, hashes, gates, and success; extensive corruption battery passes |
| Estimated pages | Fable wrapper inferred page count rather than observing the provider boundary | Provider telemetry records actual HTTP/fetch/page/raw/normalized/dedup/cursor/retry/status fields per venue and run; live run measured six pages each |
| Torn canonical state | Fable journal could expose a mixed multi-venue state until later recovery | Immutable complete generations plus atomic `CURRENT`; subprocess crash, concurrent reader/writer, concurrent ingester, abandoned cleanup, DuckDB, CLI, scheduler tests pass |
| Incomplete Optuna identity | Fable identity omitted most OHLCV and material configuration | Canonical full content/config/schema identity changes for every material mutation and is stable for semantic identity |
| Sign-flip insolvency | An adverse short move over 100% could make equity negative and later positive | Equity clamps permanently to zero, reports bankruptcy, fails solvency, and cannot win search/promotion |
| Unsupported audit language | Fable mixed commit counts, claimed third-party status, and overstated Docker/CI/storage evidence | Fable material archived with an explicit superseded/untrusted notice; current documentation reports only executed evidence |

## Verification summary

- Local CPython 3.12.13: 245 tests passed, 0 skipped, 32 dependency warnings, 90% total branch-aware coverage.
- Critical modules: backtest 100%, storage 97%, public evidence 97%, promotion 100%, paper inference 98% branch-aware coverage.
- Compileall, Ruff lint, Ruff format, and strict mypy over 24 source files: clean.
- Dependency audit: no known vulnerabilities after clean CI exposed and the audit pinned out vulnerable PyArrow 21.0.0; the local package itself is correctly skipped because it is not published to PyPI.
- Source distribution and wheel built; `twine check` passed; every CLI help path, version, doctor, and demo passed from an installed wheel outside the checkout.
- Live public endpoints: Coinbase and Kraken ETH/USD 1h, 360 closed candles per venue, six actual pages per venue, two isolated identical runs, strict validator `valid: true`.
- Docker: real Linux ARM64 ML image build/run passed as UID 1000; Python 3.11.15 and TensorFlow 2.21.0; offline lifecycle and graceful `SIGTERM` exit passed.
- GitHub Actions: CI run `30388958288` passed all seven jobs; Dependency Review workflow `30388958215` passed with an explicit unsupported-private skip; CodeQL workflow `30388958147` passed with an explicit unsupported-private skip. The pinned CodeQL v4 analysis job remains ready for GitHub Code Security.

## Quantitative conclusions

Independent loop-based and hand-calculated references cover feature/label math and backtest entry, exit, reversal, flat, costs, turnover, exposure, trades, hit rate, Sharpe, drawdown, gaps, zero variance, empty/single trades, non-finite rejection, and bankruptcy. Hypothesis properties confirm implementation/reference parity and terminal insolvency.

Leakage tests show that future mutations do not change earlier features or training scalers, validation rows do not enter transforms, horizons and sequences stay within allowed fold boundaries, and paper inference uses the newest completed information. One material research limitation remains: the configured folds support search/model selection, not an automated untouched nested final holdout.

## Scope and claims

This audit establishes the reviewed software behavior under the recorded environments. It does not establish profitability, investment suitability, exchange uptime, bundle publisher authenticity, or readiness for live capital. ARES remains paper-only and has no order path.

Machine-readable results: [SOL_TEST_RESULTS.json](SOL_TEST_RESULTS.json). Command-level evidence: [SOL_COMMAND_LOG.md](SOL_COMMAND_LOG.md). Residual risks: [SOL_RESIDUAL_RISKS.md](SOL_RESIDUAL_RISKS.md).
