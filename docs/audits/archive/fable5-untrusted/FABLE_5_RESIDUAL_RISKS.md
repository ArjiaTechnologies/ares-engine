# Fable 5 audit — residual risks

Ordered by weight. Each entry states what is NOT established and why.

1. **Live public-endpoint compatibility is unverified.** The sandbox proxy
   blocks Coinbase and Kraken (403 `blocked-by-allowlist` at CONNECT; DNS
   unavailable). All venue behavior was validated against exact-shape local
   simulators driven through unmodified CCXT clients. Real endpoints can still
   differ (schema drift, pagination quirks, rate policies, TLS interception,
   symbol listings). Mitigation: run the portable
   `ares verify-public-ingestion` audit externally and review its artifacts.
   Until then the verdict is capped at CONDITIONAL.

2. **Kraken depth-cap policy asserted, not observed.** The ~720-candle OHLC
   history cap that motivated the validator-coverage fix comes from public API
   documentation and is modeled in the simulator; the live behavior (and any
   future change to it) is unobserved from this sandbox.

3. **Docker image not built or executed.** The Dockerfile was audited statically
   and hardened (non-root, writable dirs), but no daemon exists in the sandbox.
   First `docker build` + `docker compose run ares doctor` on a real host is
   still required evidence.

4. **GitHub Actions not executed.** Workflows are YAML-valid, least-privilege,
   and their exact command sequences pass locally with uv-resolvable
   dependencies (180 packages), but no push occurred, so no live runner
   execution exists. The ml-smoke job runs the x86_64 `tensorflow-cpu` path,
   which this ARM sandbox could not exercise.

5. **Model-bundle deserialization remains a trust boundary.**
   `scaler.joblib` (pickle) and `model.keras` execute code paths on load by
   design. Hash/manifest verification and the new snapshot re-hash close
   tamper and race windows, but they authenticate *content integrity*, not
   *authorship*. Only load bundles you produced. Signing is roadmap work.

6. **No locked final holdout; early stopping shares the fold window.** All
   reported scores are selection estimates. Any performance claim requires the
   protocol's frozen post-search evaluation plus a forward paper period. The
   engine's own documentation says this; nothing in this audit weakens it.

7. **Readers between a crash and the next locked ingestion can see torn
   multi-venue state.** The journal guarantees convergence at the next locked
   entry and the alignment gates block paper signals meanwhile, but a direct
   reader of one venue's Parquet during that window sees pre-recovery data.

8. **Equity semantics under catastrophic short moves.** A single-bar adverse
   move > 100% against a short position produces net_return < −1 and a
   sign-flipped compounding path (reference implementation agrees exactly, and
   the near-total-loss case is tested). This is a documented modeling limit of
   simple multiplicative equity, not an accounting bug; leverage is fixed at 1.

9. **Third-party version drift.** The audit ran against the newest resolutions
   inside the declared pins (TF 2.21.0, pandas 2.3.3, optuna 4.9.0, ccxt
   4.5.68, duckdb 1.5.5 on aarch64). Other platforms/resolutions within the
   same pins may surface different behavior; CI's matrix is the ongoing guard.

10. **Scheduler long-run behavior.** Quick/deep cycle logic, locking, failure
    visibility, and idempotency are tested synthetically; a multi-day live
    soak with real APScheduler timing was out of scope in this sandbox.
