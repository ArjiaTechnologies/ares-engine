# Offline paper coverage and outcome annotations

`ares paper-report` reads a versioned journal, an explicit plan and supplied
terminal-close prices. It prints a deterministic JSON report to stdout. It does
not collect data, load a model, rerun inference, select thresholds, modify the
journal or change a schedule. Shell redirection can save the printed report to a
separate file; never redirect over an input.

```bash
ares paper-report --journal data/paper/signals.jsonl \
  --plan paper-plan.json --prices outcome-prices.json \
  --as-of 2026-10-01T17:00:00+00:00
python scripts/demo_paper_report.py
```

The plan has exactly these fields. Replace the example hashes with the manifest
and effective-config hashes from the intended recorded model.

```json
{
  "schema_version": "ares-paper-plan-v1",
  "exchange": "coinbase",
  "symbol": "ETH/USD",
  "timeframe": "1h",
  "start": "2026-10-01T12:00:00+00:00",
  "end": "2026-10-01T16:00:00+00:00",
  "horizon_bars": 1,
  "manifest_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
  "config_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
  "max_recording_delay_seconds": 300
}
```

All times are canonical UTC strings and candle timestamps identify **openings**.
The aligned `[start, end)` grid is independent of available records. A slot becomes
due when its candle closes. The report lists each planned slot as `not_due`,
`missing_record`, `model_mismatch`, `late_record` or `recorded`.

`timely_matching_coverage` is `recorded / due_slots`; no due slots produces `null`,
while due slots without qualifying records produce zero. Model mismatches and
late records remain in the denominator. Both evaluation and recording time must
fall between candle close and the declared delay limit. A record written or
evaluated after report `as_of` cannot fill historical coverage. If a record has
both a model mismatch and a timing problem, its status is `model_mismatch`; both
flags remain visible. Signal counts and staleness extrema describe qualifying
recorded successes only, not failed attempts or the missing population.
In a due, in-plan receipt visible at the cutoff, an evaluation timestamp later
than its own recording timestamp is inconsistent and blocks the report.
`missing_record` means absent at the cutoff, including
while the allowed recording-delay window is still open; it is not proof that a
deadline was missed.

An absent receipt means **missing record, cause unknown**. The journal contains
successful signals only. It cannot prove missing/stale-feed downtime, an attempted
evaluation, or the reason an evaluation failed. A missing journal is an error;
an explicitly empty existing journal is a valid empty stream.

Outcome prices are a JSON list with exactly five fields per row:

```json
[
  {"timestamp": "2026-10-01T12:00:00+00:00", "exchange": "coinbase", "symbol": "ETH/USD", "timeframe": "1h", "close": 100.0},
  {"timestamp": "2026-10-01T13:00:00+00:00", "exchange": "coinbase", "symbol": "ETH/USD", "timeframe": "1h", "close": 110.0}
]
```

For a receipt at `t`, horizon `h` targets exactly `t + h × timeframe`. That
target's close becomes available at `t + (h + 1) × timeframe`. The report never
substitutes the next available row across a gap. A missing intermediate candle
does not prevent a terminal-close annotation; the exact terminal candle must
exist. Before maturity, the status is `pending`; after maturity it is
`missing_data`, `source_conflict` or `available`.

The anchor is the original receipt's close. A contradictory supplied anchor
produces `source_conflict`; an absent supplied anchor uses the receipt. An
available annotation is `terminal_close / recorded_close - 1`. This descriptive
price change is not signal-signed, cost-adjusted, compounded, executable P&L or
the model's training label. No aggregate performance or probability accuracy is
reported. Overlapping horizons are not treated as independent trades.

The report binds the validated journal bytes, normalized plan, cutoff, original
receipt identities, consumed price rows and protocol version. `outcome_id` binds
the receipt and horizon; revised price evidence keeps that identity but changes
`outcome_evidence_sha256` and `report_sha256`. Valid future prices cannot change
earlier-cutoff annotations or the consumed-price digest. Later valid journal
entries change its whole-snapshot digest, even when earlier-cutoff metrics are
unchanged. The entire journal and all supplied price rows are validated before
filtering, so malformed unused data still fails closed.

Limits: 10,000 planned slots; horizon 1–1,000 bars; delay up to seven days;
100,000 price rows; 16 KiB plan JSON and 16 MiB prices/journal. Inputs must be
strict JSON with unique keys. The journal reader creates no lock or directory;
it captures one open-file snapshot from the atomic writer. It assumes the same
trusted-local, cooperating-writer boundary as [paper recording](paper-recording.md).

The plan and receipt times do **not** prove authenticated preregistration or
independent forward observations. Backfilled records cannot become timely by
backdating only `as_of`, but both declared clocks remain unauthenticated. Every
report explicitly leaves independent forward performance unestablished and
statistical drift unavailable. Feature-distribution drift, failure coverage and
collection of independent forward outcomes remain separate work.
