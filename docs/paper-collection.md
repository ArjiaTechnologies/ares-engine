# Frozen paper plans and attempt accounting

`paper-collection-init`, `paper-collect` and `paper-collection-status` add a
bounded, opt-in local experiment around the existing paper recorder. Inputs are
**supplied local Parquet files**. These commands do not fetch market data, run a
scheduler, train/promote models or submit orders. Existing `paper` and scheduler
commands keep their behavior and do not populate a collection's attempt state.

## Register and pin a plan

Prepare a [paper-report plan](paper-report.md) with future aligned `[start, end)`
candles, the intended bundle's captured manifest/effective-config hashes and a
recording delay. The registration clock must be at or before the first candle's
opening. Registration embeds the complete validated plan and a separate
`registration_sha256`; later editing the original plan file cannot change it.
Keep that digest outside the collection file and pass it on every collect/status
call. It detects a changed registration even if the file's other hashes have
been recomputed. It is not a signature or authenticated timestamp.

```bash
ares paper-collection-init --collection experiment.json --plan future-plan.json

# Use the exact registration_sha256 returned above. One invocation, one candle.
ares paper-collect --collection experiment.json \
  --registration-sha256 REGISTRATION_SHA256 \
  --candle 2030-01-01T12:00:00+00:00 --bundle artifacts/pinned-bundle \
  --primary snapshots/coinbase.parquet --secondary kraken=snapshots/kraken.parquet

ares paper-collection-status --collection experiment.json \
  --registration-sha256 REGISTRATION_SHA256 --as-of 2030-01-01T17:00:00+00:00
```

Replace the example dates and digest with the registered values. `paper-collect`
uses the current local UTC clock; it has no backdated `--as-of` option. Admission
must occur between the selected candle's close and its declared delay deadline,
inclusive. The actual loaded model/config/market and latest supplied candle must
match the plan before a signal is written. All normal inference quality gates
still apply. The CLI accepts up to eight unique `--secondary EXCHANGE=LOCAL_PATH`
arguments; supply the validation venues required by the pinned model config.

## What the records establish

The collection file holds frozen registration and at most one attempt per
planned candle. Each attempt has a stable ID bound to registration and candle,
original start time and, when acknowledged, terminal time. A nonblocking lock
covers admission, input reads, inference and the final state update. State is
validated and replaced atomically using the recorder's persistence primitive.

`STARTED` is persisted **before** input reads and inference. It proves wrapper
admission, not that inference began or completed. Only a newly recorded,
persisted receipt with the exact event/model/config/evaluation time can produce
`SUCCEEDED`. This status means the invocation returned and its completion was
saved; it does not mean the receipt was timely or profitable. Completion may
occur after the delay deadline. The existing `paper-report` still determines
timely signal coverage separately.

Known pre-commit rejection classes produce `FAILED` with one bounded code:

| Code | Observed boundary |
|---|---|
| `LOCAL_INPUT_REJECTED` | A caught local-file read/parse error before inference |
| `DATA_QUALITY_REJECTED` | Existing inference data/feature/scaling quality gate |
| `MODEL_INTEGRITY_REJECTED` | Verified bundle loading rejected integrity |
| `PLAN_MISMATCH` | Loaded identity or candle differs from the pinned plan |

The codes do not parse exception text or invent detailed missing/stale-feed
causes. Raw exception messages and input paths are not copied into attempt state.
Other exceptions, termination and uncertain writes leave unresolved evidence.
Missing input files may become empty frames in the existing reader and then
fail a data gate; the code records the actual caught class, not an inferred cause.

## Retries, interruption and recovery

Every repeat invocation for an admitted slot returns `REPLAY_NO_DISPATCH` with
the saved attempt. It never reads new input files or invokes inference again,
even if the previous attempt explicitly failed or the caller changes paths.
The entire collection and successful receipt links are validated first. A
missing or altered backing success receipt blocks further dispatch/replay.

Signals live in the dedicated sibling `experiment.json.signals.jsonl`. Never
point the regular paper command or scheduler at that journal. A receipt already
present before admission blocks a new attempt; a receipt replay returned during
dispatch leaves completion unresolved. A crash after signal commit but before
terminal-state commit likewise remains `STARTED_UNKNOWN`, with any visible
receipt listed separately. Matching evidence alone does not prove which
invocation completed. This version provides no automatic reconciliation that
promotes unknown attempts to success or authorizes another attempt.

After an exception near a state replacement, inspect `paper-collection-status`
with the original registration digest. The terminal write may have committed;
the retry preserves the saved result. Preserve both files, the digest and any
interrupted temporary file. Do not delete/reset a slot, rewrite history, switch
to a fresh collection to hide uncertainty or treat lack of a receipt as proof
that no work happened. New independent experiments require explicit boundaries.

## Inspecting coverage

At the explicit report cutoff, every planned candle is exactly one of `NOT_DUE`,
`NOT_STARTED`, `STARTED_UNKNOWN`, `FAILED` or `SUCCEEDED`. A terminal timestamp
after the cutoff appears as `STARTED_UNKNOWN`; future transitions cannot improve
earlier metrics. `NOT_STARTED` means no observed admission by that cutoff,
including while its delay window is still open. The cause remains unknown.
Receipt hashes are reported independently of invocation status.

The status report embeds the frozen plan. Use that exact `registration.plan`
with the existing `paper-report` and the dedicated signal journal to annotate
prices. A missing signal journal is explicitly absent, never fabricated as an
empty successful stream. Whole-snapshot hashes change when later evidence is
added even when earlier-cutoff counts stay unchanged.

```bash
python scripts/demo_paper_collection.py
pytest tests/test_paper_collection.py -q
```

The demo uses a patched clock and synthetic evaluator to produce one success,
one explicit rejection, one interrupted attempt and one never-started slot.
It exercises the real state/receipt protocol and reuses `paper-report`; it is not
a model evaluation or independent market observation. Tests separately exercise
the existing inference path with a stub model and terminate actual processes.

Limits: 10,000 planned slots/attempts and 16 MiB each for state and signal
journal. All existing plan bounds apply. The trusted-local, cooperating-writer
and power-loss limitations of [paper recording](paper-recording.md) remain.
Input snapshots must stay stable while read; the files are not captured as a
transaction across venues. Failure records do not retain source-frame hashes;
successful receipts bind the values actually supplied to inference. This is
attempt accounting for offline supplied files, not authenticated preregistration,
independent forward collection, detailed failure telemetry, drift measurement
or investment evidence. Every status report leaves independent forward
performance unestablished.
