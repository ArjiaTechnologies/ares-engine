# Sol Docker evidence

Execution: 2026-07-28 on Docker Desktop server 29.5.3 for Linux ARM64.

## Build

- Command: `docker build --target ml --tag ares-engine:sol .`
- Result: success from the repository context.
- Base resolved to Python 3.11 slim for ARM64.
- ML marker selected `tensorflow` 2.21.0 for Linux ARM64, not the x86-only `tensorflow-cpu` wheel.
- No exchange credentials or host-specific source path was supplied to the build.

## Runtime

The real built image passed:

- UID exactly 1000 (non-root).
- `/app/data` and `/app/artifacts` writable.
- Python 3.11.15.
- ARES 0.1.0.
- Keras 3.15.0 and TensorFlow 2.21.0; `ready_for_ml: true`.
- `ares doctor`.
- `ares demo --bars 500` with primary and cross-venue quality gates.
- `ares verify-offline --config configs/smoke.yaml --bars 2000` including quality, LSTM training, validation, backtest, bundle export, manifest verification, promotion, reload, and read-only paper inference.
- Seven expected bundle files present.
- Solvency gate and complete lifecycle passed. The permissive smoke result contained no profitability claim.

## Lifecycle and Compose

A detached non-root container installed a TERM trap. `docker stop` delivered `SIGTERM`; the container exited with status 0 and was removed. `docker compose config --quiet` passed.

Verdict: **PASS**.
