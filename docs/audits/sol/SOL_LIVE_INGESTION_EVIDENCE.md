# Sol live public-ingestion evidence

Execution: 2026-07-28 18:11:45 UTC on macOS, CPython 3.12.13, ARES 0.1.0. This was a real credential-free execution of the production CCXT provider and audit paths, not fixture mode.

## Request

| Field | Value |
|---|---|
| Primary | Coinbase |
| Validation | Kraken |
| Symbol / timeframe | ETH/USD / 1h |
| Requested interval | 2026-07-10 18:00 UTC inclusive to 2026-07-25 18:00 UTC exclusive |
| Current UTC at selection | 2026-07-28 18:00 UTC |
| Historical safety margin | 72 hours |
| Expected closed candles | 360 per venue |
| Controlled page limit | 60; forces six pages per venue |
| Runs | Two isolated requests with independent providers/telemetry |

## Result

- `mode`: `live-public-endpoints`
- `fixture_mode`: `false`
- `live_public_endpoints_reached`: `true`
- `credentials_used`: `false`
- `orders_possible`: `false`
- All coverage, quality, alignment, divergence, idempotency, and overall gates: `true`
- Coinbase: 12 actual HTTP requests, 6 `fetch_ohlcv` calls/pages, 360 raw and 360 normalized rows, 0 duplicate rows.
- Kraken: 8 actual HTTP requests, 6 `fetch_ohlcv` calls/pages, 360 raw and 360 normalized rows, 0 duplicate rows.
- Both venues: first candle 2026-07-10 18:00 UTC; last candle 2026-07-25 17:00 UTC; 360 aligned overlap rows.
- Cross-venue close divergence: latest 0.641224 bps, median 1.178302 bps, p95 3.555377 bps, maximum 7.846052 bps.
- Run-one and run-two canonical hashes are identical for each venue.
- Run-one and run-two physical Parquet hashes are identical for each venue.
- Independent `ares validate-ingestion-report` result: `valid: true`, `problems: []`.

## Hashes

| Artifact | SHA-256 |
|---|---|
| Coinbase canonical logical content | `9165bcee77057df23672c9b2cc07ffbde922309141ad8f00603fe5ae2e1adfde` |
| Kraken canonical logical content | `bc0a556c762dfbbc3008dc6f75291e8f44530f8284eb452dda08b65cdb9c6ba5` |
| `coinbase_normalized.parquet` | `b18a83f8e402d1a59ed94924cd206575ddf31f6f6f0777b2dfa67a213b35ae6d` |
| `kraken_normalized.parquet` | `bf1bfc3456c5200e134b8957b0fe693cba3a01012bc7c2bfdd93b3dd90bc9f68` |
| `manifest.json` | `5e69d0cd72b0abcd0bfe82e8860b3d3e7862b8c3cbe39221d3e545b45d8919b7` |
| `public_ingestion_report.json` | `45a7770120a9e2eb6f9ee6081d06c9d32c20962ec22a96fa5c99974aa2a9af56` |
| `quality_report.json` | `2bc80182b32216f5b0771b504f92f574bfef1c4993efa1269afc6782d408de4f` |
| `request_summary.json` | `192de585c5fef070e511b2c99cd95519ad62762461c4bb39a8e8556d1b0ad129` |
| `request_trace.json` | `37a7ed9e7d3d95d432b5c5dd4f166dd77d566114ac489c5379077a0b814e9a14` |
| `SHA256SUMS.txt` | `6b781679351ed92482153a2a8cd1d17ddc8706934f16e285ec2792d124b490a7` |

The generated directory is ignored from Git because it contains runtime market data. The report, trace structure, hashes, and result are retained here; the complete local directory was validated before documentation. Adversarial tests prove that fake Parquet, substituted/extra/missing files, modified reports, false live flags, fixture masquerading, empty/zero-request traces, inconsistent counts/ranges, non-finite values, bad grids/schema/OHLCV, and checksum-consistent invalid artifacts are rejected.
