# Paper recording and restart reconciliation

`ares paper` and the scheduler retain their existing `paper/signals.jsonl`
location. New records use the `ares-paper-record-v1` envelope. The API returns a
`PaperSignal` with a `recording` receipt when `log_path` is supplied. Its status is
`recorded` or `replayed`; `recording.record` is the canonical persisted record.
The outer signal describes the current evaluation, including current staleness.
When no log is requested, `recording` is `None`.

One journal is one stream. Within it, the event key contains the exact normalized
exchange, symbol, timeframe and UTC completed-candle timestamp. Model identity
does not create a second key: changing the model or supplied data for an existing
event produces a conflict. Independent model experiments need explicitly separate
journals. Aliased bundle paths and retry times do not change event identity.

Each event binds the captured verified bundle manifest, effective configuration,
normalized primary and every required secondary OHLCV frame, unscaled feature
window, scaled model input, prediction and cross-venue checks. Frame hashes cover
ordered timestamps, exchange/symbol/timeframe and OHLCV values supplied to the API;
they are not hashes of original Parquet files. A different history length can
therefore conflict even if the newest candle matches. Keep the same evidence for
retry reconciliation. Input frames are copied before evaluation; callers must not
mutate them during that initial copy.

The manifest digest comes from the snapshot used by `load_bundle`, never a later
read of the live manifest. This binds the loaded bytes; it does not authenticate
the publisher or prove the bundle is still the active champion after it loads.

The first record retains recording/evaluation time, bundle path and staleness.
An exact evidence replay returns that original record without changing journal
bytes, including when the recording clock has moved backward. Fresh inference
still has to pass current data-quality gates. New events must advance their
market's candle high-water mark and cannot regress the recording clock.

## Persistence and recovery boundary

A nonblocking file lock covers the complete read, validation, reconciliation and
commit. The writer validates all existing records, writes a sibling temporary
file, flushes and synchronizes it, and replaces the journal atomically. POSIX also
synchronizes the directory entry. There is no separate index to reconcile.

An exception near commit does not establish that the event was absent. Preserve
the journal and retry the same event/evidence: a committed event returns its
original receipt; an absent event can be committed once. Damaged/truncated history,
duplicate keys/events, invalid schema/digests, conflicting evidence and linked or
ambiguous target paths block recording without repairing history. A process killed
before cleanup can leave an unused sibling temporary file; it is never treated as
committed history. Inspect it before manual cleanup.

The journal is bounded to 16 MiB and scanned on each call. At the bound, stop and
review/export the stream explicitly. Starting a different journal starts a new
deduplication boundary; it is not automatic rotation or a bypass for an uncertain
record. This implementation assumes cooperating writers and a trusted local
filesystem. It is not a distributed database, authenticated audit log, or proof of
durability through arbitrary power loss. Keep independent backups.

## Existing logs

Legacy flat JSONL records have no sufficient event/evidence identity. They remain
unchanged and block new writes with an explicit legacy-log error. There is no
automatic migration or deletion. Stop scheduled writers, preserve and inspect the
old log, reconcile any uncertain last attempt, and deliberately archive it before
starting a new empty journal at the configured location. Retain the archive and
the chosen stream boundary; duplicates across that boundary are not prevented.
Consumers of new records read `event`, `evidence` and `observation` rather than the
old top-level `signal` and `timestamp` fields.

## Reproducible checks and limits

```bash
python scripts/demo_paper_recording.py
pytest tests/test_paper_recording.py tests/test_live.py -q
pytest tests/test_paper_inference_gates.py -q  # requires the ML extra
```

The demo uses constructed synthetic evidence to exercise the persistence
protocol. Tests separately cover owned-frame inference with a stub model, real
filesystem restart failures, multiple processes, and the ML bundle lifecycle.
This capability establishes recording integrity only. It does not yet collect
independent forward outcomes, quantify missing-feed coverage, measure drift,
establish profitability or provide any order-execution path.
