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
Without `--capture-inputs`, input snapshots must stay stable while read and failure
records do not retain source-frame hashes. Successful receipts bind the values
actually supplied to inference. Opt-in capture is described below. This is
attempt accounting for offline supplied files, not authenticated preregistration,
independent forward collection, detailed failure telemetry, drift measurement
or investment evidence. Every status report leaves independent forward
performance unestablished.

## Preserve the supplied input bytes

Add `--capture-inputs` to `paper-collect` to preserve local file evidence for that
new attempt. The existing frozen registration and stable attempt ID bind a
declared primary/secondary inventory before any input read. The choice is made
per attempt, not retroactively and not as a new guarantee in an older plan.
An already-admitted slot never captures again, even when this flag is newly added.
Legacy attempts remain readable without rewritten hashes or invented evidence;
their status rows expose `input_capture: null`.

Each source has these separately acknowledged stages:

| Status | Evidence |
|---|---|
| `PENDING` | Requested source, no observed read admission |
| `STARTED` | Admission saved before opening the supplied file |
| `CAPTURED` | Exact bounded bytes saved; parsing completion unacknowledged |
| `READY` | Bytes decoded and normalized-frame digest saved |
| `READ_REJECTED` | Known local read/path/size rejection; no captured byte claim |
| `PARSE_REJECTED` | Captured bytes preserved, decoding or normalized digest rejected |

`started_at` through `captured_at` bounds the observed local read and snapshot
publication. `finished_at` records the acknowledged parse/digest or rejection
observation. These are unauthenticated local-clock declarations. They are separate
from candle time and file modification time, and do not establish when a provider
retrieved, published or first made the data available. Sources are observed
sequentially; this is not a simultaneous or atomic cross-venue snapshot. A file
edited in place during its read may yield mixed bytes: the evidence preserves
exactly what was read, without certifying an upstream file generation.

Snapshots use generated names under
`experiment.json.inputs/<attempt_id>/<source_index>.parquet`. State records the
source role/exchange and exact raw SHA-256/byte count, without copying original
paths or exception text. Parsing uses the captured in-memory bytes, never a
second read of the original path. A separate frame hash covers the timestamp-
normalized, sorted OHLCV values used by inference; encoding changes, extra
columns and row order may change the raw hash without changing that frame hash.
Successful receipts must match the primary and every secondary frame hash, and
their recording time must follow all completed input observations.

A malformed secondary therefore preserves both an earlier valid primary and
the malformed secondary's bytes; later requested sources remain pending. The
attempt becomes `LOCAL_INPUT_REJECTED` only for known read/parse rejection.
Snapshot or state publication errors propagate as uncertain completion. A
crash can leave a saved raw file without acknowledged metadata. Preserve it;
the mere presence of bytes never advances the source or attempt state and never
authorizes recapture. There is no automatic repair or cleanup command.

Inspection, replay and new dispatch validate all referenced raw snapshots and
their declared normalized hashes. Missing, changed, linked or inconsistent
evidence blocks work without copying changed originals. Status at an earlier
cutoff hides later source transitions. Whole-state hashes can still change as
later evidence arrives. Inspection does not write source files or state.

Limits are nine sources per attempt, 16 MiB per file and 256 MiB of preserved
input files per collection, including interrupted/orphan files. Size exhaustion
requires explicit review, not deletion or a new collection to hide history.
The byte limits do not bound arbitrary Parquet decoder memory use; use trusted
local inputs. The existing cooperating-writer and power-loss limits apply.
This is supplied-file provenance, not original exchange-response capture or
independent forward acquisition. Run `python scripts/demo_paper_inputs.py` for
a synthetic partial-failure example with byte preservation and unchanged replay.
