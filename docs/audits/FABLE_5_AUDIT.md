# Fable 5 audit of ARES Engine v0.1.0

- **Auditor:** Claude (Fable 5), acting as independent reconstruction/execution/repair auditor
- **Date:** 2026-07-27
- **Verdict:** **CONDITIONAL** (see §7 — a single environment-dependent requirement remains)

## 1. Source identity (verified exactly)

| Item | Expected | Observed |
|---|---|---|
| Uploaded wrapper archive SHA-256 | n/a (repackaged) | `cc86e2b38cc1136421c574a32dc37c0bb45f8d6c4bfedee4d1579f01c7d47a13` |
| Inner source archive SHA-256 | `0a18e017996746764325b18185f4de320898e41c84bbc91ab695b1979b52a7b0` | identical |
| Restored Git tree | `e48e6ad8e45643ca06e6fff21cb27d5044c04a87` | identical |
| Tracked files | 70 | 70 |
| Recorded original release commit | `9fee3e6084d98bfee1009e8a3e48d4d0b7c37ba1` | recorded (identity carried by tree hash) |
| Baseline commit (this repo) | — | `b7736ac` on `main` |
| Audit branch | — | `audit/fable-5-hardening` (16 focused commits) |

The wrapper's base64 payload was decoded and safety-checked independently; the
bundled bootstrap script and workflow were treated as untrusted data and never
executed. No traversal, absolute paths, symlinks, or non-regular entries in
either archive.

## 2. What was executed (evidence, not claims)

- **Static:** `compileall`, `ruff check`, `ruff format --check`, `mypy --strict`
  (0 errors after repair), on every tracked source.
- **Tests:** 186 tests (145 non-ML + 41 ML) — all passing on CPython 3.11.15,
  3.12.13, and 3.13.14 (aarch64, built from source for this sandbox), with
  TensorFlow 2.21.0. Branch coverage ~85% (baseline: 50 tests, 76%).
- **Independent references:** the backtest engine, EMA/Wilder-RSI/Bollinger/
  volatility/volume/range features, and both label families match deliberately
  loop-based pure-Python reimplementations exactly (hand-computed equity paths;
  25 randomized backtest cases; boundary/equality/ambiguity fixtures).
- **Leakage:** radical rewrites of all data after a fold's legal horizon leave
  feature tensors, labels, fitted scaler statistics, fold metrics, and per-fold
  validation probabilities **bitwise identical**; scaler inputs are exactly the
  fold training tensor; purge arithmetic keeps train label horizons strictly
  before the first validation decision row; determinism itself is asserted.
- **Ingestion:** real CCXT clients (`ccxt.coinbase`, `ccxt.kraken`, v4.5.68)
  driven against exact-shape local simulators of both venues' public endpoints:
  multi-page pagination with strictly-advancing cursors, coverage, dedup on the
  UTC grid, bit-identical idempotent reruns, short pages, both venues' rate-limit
  dialects, 5xx, client timeouts, truncation, malformed rows, off-grid stamps,
  divergence, newest-candle mismatch, depth caps, in-progress-candle exclusion.
  Crash injection between renames plus journal roll-forward recovery; real
  cross-process lock contention via subprocesses.
- **ML lifecycle:** deterministic seeds (bitwise same-seed model identity, both
  families), early-stopping/NaN-guard/no-shuffle wiring, CPU-only execution,
  full TCN train→export→reload prediction parity (atol 1e-7), causal padding and
  dilation doubling verified structurally; bounded Optuna searches with
  adversarial objectives.
- **Bundles:** 21 hostile-bundle tests — every tamper class fails closed,
  including coherent tampering (metadata cross-checks), hard-link
  post-verification edits, and an instrumented verify-then-load race.
- **Promotion:** atomic under injected crashes and a genuine two-process race;
  manifest-anchored champion identity; corrupt incumbents block promotion
  without pointer movement.
- **Paper inference:** fail-closed battery incl. duplicated/reordered rows,
  in-progress candles in either feed, non-finite prices, stale/misaligned/
  missing feeds, mismatched features, invalid model outputs; no log entry on
  any failure; read-only output.
- **No live-order path:** full-tree scan — the only CCXT surface is
  `load_markets` / `fetch_ohlcv` / `milliseconds` / `has`; no order, balance,
  withdrawal, transfer, or credential-loading code exists.
- **Packaging:** sdist+wheel build, `twine check` PASSED, wheel-only install
  exercised from outside the checkout (import, version, all 14 commands,
  offline demo via packaged configs).
- **CI/Docker:** workflows YAML-validated, least-privilege checked, uv
  resolution proven (180 packages); Docker hardened to non-root. Building/
  running the image and executing GitHub Actions were not possible in this
  sandbox (no Docker daemon; no push allowed) — see residual risks.

## 3. Defects

17 findings (3 High, 6 Medium, 3 Low, 5 Informational), every functional one
reproduced or race-instrumented before repair, each fixed in a focused commit
with regression tests. Register: [`FABLE_5_FINDINGS.md`](FABLE_5_FINDINGS.md).
Highest severity: torn multi-venue commits on crash, a default configuration
that could never complete real long-range ingestion, and Optuna selection
accepting non-finite winners.

## 4. What the audit did NOT establish

- **Live Coinbase/Kraken compatibility.** Outbound requests to both venues are
  blocked by this sandbox's allowlist proxy (HTTP 403,
  `X-Proxy-Error: blocked-by-allowlist`, at CONNECT; DNS also unavailable;
  evidence in the command log §9). Exact-shape simulation is not live proof.
- **Docker image build/run** (no daemon available) and **GitHub Actions
  execution** (nothing was pushed).
- **Kraken's exact live depth-cap behavior** (asserted from public API
  documentation, modeled in the simulator).
- **Any statement about profitability.** None is made. The smoke lifecycle's
  stressed median return is negative and the README says so.

## 5. Verdict rationale

Every locally executable PASS criterion is met: exact source identity, clean
install, honest 3.11–3.13 matrix, static analysis, full test suite, ML smoke,
build + wheel-CLI outside checkout, adversarial ingestion preserving canonical
state, no demonstrated leakage, reference-exact backtest, reload parity,
fail-closed bundles/promotion/paper inference, no live-order path, and no
unresolved Critical/High defects. The remaining live-ingestion requirement is
environment-blocked and intentionally not simulated away:

> Live bounded public-endpoint ingestion was not executed in the audit sandbox
> because outbound requests to Coinbase and Kraken were blocked with HTTP 403.
> Exact-shape simulations and adversarial local-server tests passed, but they
> do not establish real endpoint compatibility. A portable verification command
> was created for execution in an unrestricted environment.

**Verdict: CONDITIONAL.** Upgrade path to PASS: run
`ares verify-public-ingestion` per
[`FABLE_5_EXTERNAL_INGESTION_HANDOFF.md`](FABLE_5_EXTERNAL_INGESTION_HANDOFF.md)
from an unrestricted machine, validate the artifacts with
`ares validate-ingestion-report`, and review them independently.

## 6. Companion documents

- [`FABLE_5_FINDINGS.md`](FABLE_5_FINDINGS.md) — full defect register
- [`FABLE_5_COMMAND_LOG.md`](FABLE_5_COMMAND_LOG.md) — commands, versions, counts, network evidence
- [`FABLE_5_REQUIREMENTS_MATRIX.md`](FABLE_5_REQUIREMENTS_MATRIX.md) — requirement-by-requirement status
- [`FABLE_5_RESIDUAL_RISKS.md`](FABLE_5_RESIDUAL_RISKS.md) — what still cannot be claimed
- [`FABLE_5_EXTERNAL_INGESTION_HANDOFF.md`](FABLE_5_EXTERNAL_INGESTION_HANDOFF.md) — exact external steps

## 7. Claim discipline

ARES Engine is an open-source machine-learning trading research and
paper-inference engine. This audit does not claim profitability, production
readiness for live capital, institutional-grade execution, autonomy, or live
deployment, and none of those claims appear in the repository.
