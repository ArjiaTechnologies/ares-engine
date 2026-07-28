# Sol requirements matrix

| Requirement | Status | Evidence |
|---|---|---|
| Supplied checksums, Git bundle, commits, and trees | PASS | `SOL_AUDIT.md`, command log |
| Full tracked-file and Fable-hunk review | PASS | Audit scope and findings |
| Forged evidence rejected | PASS | 23 public-audit tests plus blocker reproduction |
| Actual request/page telemetry | PASS | Provider wire tests and real traces |
| Atomic multi-venue canonical visibility | PASS | 16 atomicity, 5 storage-security, staging/storage tests |
| Complete Optuna study identity | PASS | Search adversarial and blocker tests |
| Terminal bankruptcy | PASS | Exact and Hypothesis backtest references |
| Fable contradictions corrected/archived | PASS | `archive/fable5-untrusted/README.md` |
| Features and labels independently checked | PASS | 19 reference tests plus feature/label/leakage suites |
| Leakage controls | PASS | Future-mutation, fold, scaler, sequence, inference tests |
| Backtest independently checked | PASS | 21 reference tests; backtest module 100% |
| LSTM/TCN lifecycle and CPU reload parity | PASS | ML lifecycle/smoke and hostile-bundle suite |
| Optuna failed/pruned/non-finite/bankrupt/gate-failed exclusion | PASS | Search adversarial tests |
| Bundle and promotion attack battery | PASS | Bundle/promotion/concurrency suites |
| Paper inference fail-closed battery | PASS | Paper and defensive suites; live module 98% |
| Real Coinbase/Kraken public ingestion | PASS | `SOL_LIVE_INGESTION_EVIDENCE.md` |
| Python 3.11–3.13 | PASS | 195 non-ML tests plus static checks on each interpreter; 50 ML tests on 3.11 |
| Compile, Ruff, format, strict mypy | PASS | Command log |
| Critical module branch coverage ≥90% | PASS | 100/97/97/100/98% |
| Build, Twine, installed wheel outside checkout | PASS | Command log |
| Dependency audit | PASS | No known vulnerabilities |
| Docker build/run/non-root/writable/offline/stop | PASS | `SOL_DOCKER_EVIDENCE.md` |
| Least-privilege CI and manual live job | PASS | CI run `30388958288`; workflow source and `SOL_CI_EVIDENCE.md` |
| CodeQL and Dependency Review | EXTERNAL LIMITATION | Pinned workflows pass and record explicit skips; private GitHub Code Security/GHAS is unavailable |
| Release workflow disabled | PASS | `.github/workflows/release.yml` |
| Private repository and no pre-push exposure | PASS | GitHub metadata checked before any source push |
| Draft PR and supported checks green | PASS | PR #1; CI/conditional security workflows green |
| Private merge | PENDING | Performed only after final evidence-only checks are green |
| No tag/release/PyPI/live orders | PASS | GitHub/package/source inspection |

Current matrix verdict: **CONDITIONAL** solely because CodeQL and Dependency Review require a private GitHub Code Security capability that is not enabled. All supported required checks are green; the private merge follows the final evidence-only run.
