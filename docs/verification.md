# Current verification record

The authoritative verification record is the Sol independent audit dated 2026-07-28:

- [Audit verdict and scope](audits/sol/SOL_AUDIT.md)
- [Machine-readable test results](audits/sol/SOL_TEST_RESULTS.json)
- [Real Coinbase/Kraken evidence](audits/sol/SOL_LIVE_INGESTION_EVIDENCE.md)
- [Docker evidence](audits/sol/SOL_DOCKER_EVIDENCE.md)
- [GitHub Actions evidence](audits/sol/SOL_CI_EVIDENCE.md)

The deterministic lifecycle can be reproduced with:

```bash
uv run ares verify-offline --config configs/smoke.yaml --bars 2000
```

That command exercises synthetic quality gates, features, labels, walk-forward ML, cost-aware backtesting, bundle export/verification/reload, promotion, freshness and cross-venue inference gates, and one paper signal. It is software verification and contains no profitability claim.

The July 2026 Fable 5 records are preserved at `audits/archive/fable5-untrusted/`. They are historical AI-assisted notes, are superseded, and are not an independent certification.
