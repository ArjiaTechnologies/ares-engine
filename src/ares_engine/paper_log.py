"""Bounded, reconciled paper evidence for cooperating writers on a local filesystem.

One journal is one observation stream. Hashes detect accidental inconsistency;
they do not authenticate a writer or prove forward performance.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import DataConfig
from .exceptions import AresError
from .utils import timeframe_to_seconds

MAX_LOG_BYTES = 16 * 1024 * 1024
SCHEMA_VERSION = "ares-paper-record-v1"
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1, max_length=256)]
Nonnegative = Annotated[float, Field(ge=0)]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def evidence_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def utc_timestamp(value: Any) -> str:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Missing timestamp")
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.isoformat()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PaperEvent(_StrictModel):
    exchange: Name
    symbol: Name
    timeframe: Name
    timestamp: str

    @field_validator("timestamp")
    @classmethod
    def canonical_timestamp(cls, value: str) -> str:
        if utc_timestamp(value) != value:
            raise ValueError("Timestamp must use canonical UTC representation")
        return value

    @field_validator("timeframe")
    @classmethod
    def supported_timeframe(cls, value: str) -> str:
        if value != DataConfig.valid_timeframe(value):
            raise ValueError("Timeframe must use canonical lowercase form")
        timeframe_to_seconds(value)
        return value

    @field_validator("exchange")
    @classmethod
    def canonical_exchange(cls, value: str) -> str:
        if value != DataConfig.normalize_primary_exchange(value):
            raise ValueError("Exchange must use canonical lowercase form")
        return value

    @field_validator("symbol")
    @classmethod
    def canonical_symbol(cls, value: str) -> str:
        if value != DataConfig.normalize_symbol(value):
            raise ValueError("Symbol must use canonical market form")
        return value


class PaperEvidence(_StrictModel):
    manifest_sha256: Digest
    config_sha256: Digest
    primary_sha256: Digest
    secondary_sha256: dict[Name, Digest]
    feature_window_sha256: Digest
    scaled_window_sha256: Digest
    probability: Annotated[float, Field(ge=0, le=1)]
    signal: Annotated[int, Field(ge=-1, le=1)]
    close: Annotated[float, Field(gt=0)]
    data_quality_passed: Literal[True]
    cross_venue_p95_bps: dict[Name, Nonnegative]
    cross_venue_latest_bps: dict[Name, Nonnegative]

    @field_validator("data_quality_passed", mode="before")
    @classmethod
    def passed_boolean(cls, value: Any) -> bool:
        if value is not True:
            raise ValueError("Data-quality evidence must be the boolean true")
        return True


class PaperObservation(_StrictModel):
    recorded_at: str
    as_of: str
    bundle: Name
    staleness_bars: Nonnegative
    secondary_staleness_bars: dict[Name, Nonnegative]

    @field_validator("recorded_at", "as_of")
    @classmethod
    def canonical_timestamp(cls, value: str) -> str:
        return PaperEvent.canonical_timestamp(value)


class PaperRecord(_StrictModel):
    schema_version: Literal["ares-paper-record-v1"]
    event: PaperEvent
    event_id: Digest
    evidence: PaperEvidence
    evidence_sha256: Digest
    observation: PaperObservation
    record_sha256: Digest


@dataclass(frozen=True, slots=True)
class PaperRecordingReceipt:
    status: Literal["recorded", "replayed"]
    record: dict[str, Any]


def _validate_record(value: Any) -> dict[str, Any]:
    parsed = PaperRecord.model_validate(value)
    record = parsed.model_dump(mode="json")
    if record["event_id"] != evidence_digest(record["event"]):
        raise ValueError("Event identity mismatch")
    if record["evidence_sha256"] != evidence_digest(record["evidence"]):
        raise ValueError("Evidence digest mismatch")
    if record["record_sha256"] != evidence_digest(
        {key: item for key, item in record.items() if key != "record_sha256"}
    ):
        raise ValueError("Record digest mismatch")
    evidence = parsed.evidence
    venues = set(evidence.secondary_sha256)
    if any(
        set(values) != venues
        for values in (
            evidence.cross_venue_p95_bps,
            evidence.cross_venue_latest_bps,
            parsed.observation.secondary_staleness_bars,
        )
    ):
        raise ValueError("Secondary evidence is incomplete")
    candle = pd.Timestamp(parsed.event.timestamp)
    step = pd.Timedelta(seconds=timeframe_to_seconds(parsed.event.timeframe))
    if candle.value % step.value or pd.Timestamp(parsed.observation.as_of) < candle + step:
        raise ValueError("Paper event must identify a completed, aligned candle")
    return record


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _regular_file(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise AresError("Paper journal and lock must be regular, unlinked files")
    if info.st_size > MAX_LOG_BYTES:
        raise AresError("Paper journal exceeds the 16 MiB bound; review and archive explicitly")


def _read_records(path: Path, *, required: bool = False) -> tuple[bytes, list[dict[str, Any]]]:
    _regular_file(path)
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_LOG_BYTES + 1)
    except FileNotFoundError:
        if required:
            raise AresError(
                "Paper journal does not exist; absence is not an empty observation stream"
            ) from None
        return b"", []
    if len(raw) > MAX_LOG_BYTES or (raw and not raw.endswith(b"\n")):
        raise ValueError("Oversized or truncated paper journal")
    records = []
    seen: set[str] = set()
    highwater: dict[tuple[str, str, str], pd.Timestamp] = {}
    last_recorded: pd.Timestamp | None = None
    for line in raw.decode("utf-8").splitlines():
        value = json.loads(line, object_pairs_hook=_unique_object)
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Legacy or unsupported paper log; preserve and review before resuming")
        record = _validate_record(value)
        if record["event_id"] in seen:
            raise ValueError("Duplicate event in paper journal")
        event = record["event"]
        market = (event["exchange"], event["symbol"], event["timeframe"])
        candle = pd.Timestamp(event["timestamp"])
        recorded = pd.Timestamp(record["observation"]["recorded_at"])
        if market in highwater and candle <= highwater[market]:
            raise ValueError("Paper journal candle order regressed")
        if last_recorded is not None and recorded < last_recorded:
            raise ValueError("Paper journal recording clock regressed")
        highwater[market] = candle
        last_recorded = recorded
        seen.add(record["event_id"])
        records.append(record)
    return raw, records


def _journal_path(path: Path) -> Path:
    # Reject ambiguous Windows aliases on every platform so a journal does not
    # change identity when moved between hosts. Resolve parent/relative aliases
    # before deriving the one lock name, but never follow a linked target.
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    for part in path.parts:
        if part == path.anchor or part in (".", ".."):
            continue
        if part.endswith((".", " ")) or ":" in part or part.split(".")[0].upper() in reserved:
            raise AresError("Ambiguous paper journal path; use a regular canonical filename")
    _regular_file(path)
    return path.resolve()


def read_paper_journal(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Read one validated snapshot without creating a journal, lock or directory.

    Cooperating writers atomically replace whole journals. One open file handle
    therefore reads either the prior or the new snapshot. This is not protection
    against a writer that edits journal bytes in place.
    """
    try:
        raw, records = _read_records(_journal_path(path), required=True)
        return hashlib.sha256(raw).hexdigest(), records
    except (OSError, ValueError, UnicodeError) as exc:
        raise AresError(f"Invalid paper journal snapshot: {exc}") from exc


def _replace_journal(path: Path, payload: bytes) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # On POSIX persist the directory entry too. Windows lacks this API;
        # neither platform's tests establish arbitrary power-loss guarantees.
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def record_paper_signal(
    path: Path, *, event: dict[str, Any], evidence: dict[str, Any], observation: dict[str, Any]
) -> PaperRecordingReceipt:
    """Reconcile one event before committing. An exception never proves absence.

    Retry with the same journal and evidence after an uncertain write. Existing
    identical evidence returns its original record, even after a clock rollback.
    Legacy/damaged/conflicting history is never repaired or overwritten.
    """
    try:
        event = PaperEvent.model_validate(event).model_dump(mode="json")
        evidence = PaperEvidence.model_validate(evidence).model_dump(mode="json")
        observation = PaperObservation.model_validate(observation).model_dump(mode="json")
        record = {
            "schema_version": SCHEMA_VERSION,
            "event": event,
            "event_id": evidence_digest(event),
            "evidence": evidence,
            "evidence_sha256": evidence_digest(evidence),
            "observation": observation,
        }
        record["record_sha256"] = evidence_digest(record)
        record = _validate_record(record)
        path = _journal_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(str(path) + ".lock")
        _regular_file(lock_path)
        with FileLock(lock_path, timeout=0):
            raw, records = _read_records(path)
            for previous in records:
                if previous["event_id"] == record["event_id"]:
                    if previous["evidence"] != evidence:
                        raise AresError("Conflicting evidence for an already recorded paper event")
                    return PaperRecordingReceipt("replayed", previous)
            for previous in records:
                prior_event = previous["event"]
                same_market = all(
                    prior_event[key] == event[key] for key in ("exchange", "symbol", "timeframe")
                )
                if same_market and pd.Timestamp(prior_event["timestamp"]) >= pd.Timestamp(
                    event["timestamp"]
                ):
                    raise AresError("Unseen paper candle predates the journal high-water mark")
            if records and pd.Timestamp(observation["recorded_at"]) < pd.Timestamp(
                records[-1]["observation"]["recorded_at"]
            ):
                raise AresError("Paper recording clock regressed")
            payload = raw + canonical_json(record) + b"\n"
            if len(payload) > MAX_LOG_BYTES:
                raise AresError(
                    "Paper journal exceeds the 16 MiB bound; review and archive explicitly"
                )
            _replace_journal(path, payload)
            return PaperRecordingReceipt("recorded", record)
    except Timeout as exc:
        raise AresError("Another paper process is writing the signal log") from exc
    except (OSError, ValueError, UnicodeError, ValidationError) as exc:
        raise AresError(
            f"Paper journal blocked; preserve history and reconcile before retry: {exc}"
        ) from exc
