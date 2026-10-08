"""Frozen local plans and at-most-once dispatch of supplied-file paper attempts.

Local timestamps and hashes are declarations, not authenticated preregistration.
An admitted attempt with no acknowledged terminal result always remains unknown.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data.storage import read_market
from .exceptions import AresError, BundleIntegrityError, DataQualityError, PaperPlanMismatch
from .live import generate_paper_signal
from .paper_log import (
    MAX_LOG_BYTES,
    Digest,
    PaperEvent,
    PaperExpectation,
    _journal_path,
    _regular_file,
    _replace_journal,
    canonical_json,
    evidence_digest,
    read_paper_journal,
    utc_timestamp,
)
from .paper_report import PaperReportPlan, _read_json
from .utils import timeframe_to_seconds, utc_now

Failure = Literal[
    "LOCAL_INPUT_REJECTED", "DATA_QUALITY_REJECTED", "MODEL_INTEGRITY_REJECTED", "PLAN_MISMATCH"
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Registration(_Strict):
    schema_version: Literal["ares-paper-registration-v1"]
    source_mode: Literal["OFFLINE_SUPPLIED_FILES"]
    created_at: str
    plan: PaperReportPlan
    plan_sha256: Digest
    registration_sha256: Digest

    @field_validator("created_at")
    @classmethod
    def canonical_time(cls, value: str) -> str:
        return PaperEvent.canonical_timestamp(value)


class Attempt(_Strict):
    candle: str
    attempt_id: Digest
    started_at: str
    status: Literal["STARTED", "SUCCEEDED", "FAILED"]
    finished_at: str | None = None
    failure_code: Failure | None = None
    record_sha256: Digest | None = None

    @field_validator("candle", "started_at", "finished_at")
    @classmethod
    def canonical_time(cls, value: str | None) -> str | None:
        return PaperEvent.canonical_timestamp(value) if value is not None else None


class CollectionState(_Strict):
    schema_version: Literal["ares-paper-collection-v1"]
    registration: Registration
    attempts: list[Attempt] = Field(max_length=10000)
    state_sha256: Digest


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _attempt_id(registration: str, candle: str) -> str:
    return evidence_digest({"registration_sha256": registration, "candle": candle})


def _validate(value: Any, expected: str) -> dict[str, Any]:
    parsed = CollectionState.model_validate(value)
    state = parsed.model_dump(mode="json")
    reg = state["registration"]
    plan = parsed.registration.plan
    if (
        reg["plan_sha256"] != evidence_digest(reg["plan"])
        or reg["registration_sha256"] != evidence_digest(_without(reg, "registration_sha256"))
        or reg["registration_sha256"] != expected
        or state["state_sha256"] != evidence_digest(_without(state, "state_sha256"))
    ):
        raise AresError("Collection identity/digest mismatch; preserve the original registration")
    last_time = pd.Timestamp(reg["created_at"])
    if last_time > pd.Timestamp(plan.start):
        raise AresError("Plan registration follows its first planned candle")
    seen: set[str] = set()
    step = pd.Timedelta(seconds=timeframe_to_seconds(plan.timeframe))
    for attempt in parsed.attempts:
        candle, started = pd.Timestamp(attempt.candle), pd.Timestamp(attempt.started_at)
        if (
            attempt.candle in seen
            or attempt.attempt_id != _attempt_id(expected, attempt.candle)
            or not pd.Timestamp(plan.start) <= candle < pd.Timestamp(plan.end)
            or candle.value % step.value
            or not candle + step
            <= started
            <= candle + step + pd.Timedelta(seconds=plan.max_recording_delay_seconds)
            or started < last_time
        ):
            raise AresError("Invalid, repeated or out-of-window collection attempt")
        seen.add(attempt.candle)
        if attempt.status == "STARTED":
            valid = (
                attempt.finished_at is None
                and attempt.failure_code is None
                and attempt.record_sha256 is None
            )
        elif attempt.status == "FAILED":
            valid = (
                attempt.finished_at is not None
                and attempt.failure_code is not None
                and attempt.record_sha256 is None
            )
        else:
            valid = (
                attempt.finished_at is not None
                and attempt.failure_code is None
                and attempt.record_sha256 is not None
            )
        if not valid or (
            attempt.finished_at is not None and pd.Timestamp(attempt.finished_at) < started
        ):
            raise AresError("Inconsistent attempt transition")
        last_time = pd.Timestamp(attempt.finished_at or attempt.started_at)
    return state


def _read_state(path: Path, expected: str) -> dict[str, Any]:
    try:
        return _validate(_read_json(_journal_path(path), MAX_LOG_BYTES), expected)
    except (ValueError, OSError, OverflowError) as exc:
        raise AresError(f"Collection state blocked: {exc}") from exc


def _save(path: Path, state: dict[str, Any]) -> None:
    state["state_sha256"] = evidence_digest(_without(state, "state_sha256"))
    _validate(state, state["registration"]["registration_sha256"])
    payload = canonical_json(state) + b"\n"
    if len(payload) > MAX_LOG_BYTES:
        raise AresError("Collection state exceeds the 16 MiB bound")
    _replace_journal(path, payload)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    lock_path = Path(str(path) + ".lock")
    _regular_file(lock_path)
    try:
        with FileLock(lock_path, timeout=0):
            yield
    except Timeout as exc:
        raise AresError("Another process owns this paper collection") from exc


def collection_signals_path(path: Path) -> Path:
    """A dedicated signal stream; never share it with a scheduler/other writer."""
    return _journal_path(path).with_name(path.name + ".signals.jsonl")


def register_collection(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Freeze a plan before its start. An exact registration retry changes no bytes."""
    selected = PaperReportPlan.model_validate(plan).model_dump(mode="json")
    path = _journal_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        if path.exists():
            raw = _read_json(path, MAX_LOG_BYTES)
            expected = (
                raw.get("registration", {}).get("registration_sha256")
                if isinstance(raw, dict)
                else None
            )
            if not isinstance(expected, str):
                raise AresError("Missing frozen registration identity")
            state = _read_state(path, expected)
            if state["registration"]["plan"] != selected:
                raise AresError("Collection plan is already frozen; conflicting registration")
            return dict(state["registration"])
        _regular_file(collection_signals_path(path))
        if collection_signals_path(path).exists():
            raise AresError("Unowned signal history exists; preserve and reconcile it")
        reg = {
            "schema_version": "ares-paper-registration-v1",
            "source_mode": "OFFLINE_SUPPLIED_FILES",
            "created_at": utc_timestamp(utc_now()),
            "plan": selected,
            "plan_sha256": evidence_digest(selected),
        }
        reg["registration_sha256"] = evidence_digest(reg)
        state = {"schema_version": "ares-paper-collection-v1", "registration": reg, "attempts": []}
        _save(path, state)
        return reg


def _records(path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    signal_path = collection_signals_path(path)
    _regular_file(signal_path)
    # Absence remains explicit; it is not passed off as an empty paper journal.
    return read_paper_journal(signal_path) if signal_path.exists() else (None, [])


def _expected(plan: dict[str, Any], candle: str) -> PaperExpectation:
    return PaperExpectation(
        event=PaperEvent(
            **{key: plan[key] for key in ("exchange", "symbol", "timeframe")}, timestamp=candle
        ),
        manifest_sha256=plan["manifest_sha256"],
        config_sha256=plan["config_sha256"],
    )


def _validate_receipts(state: dict[str, Any], records: list[dict[str, Any]]) -> None:
    by_hash = {record["record_sha256"]: record for record in records}
    for attempt in state["attempts"]:
        if attempt["status"] != "SUCCEEDED":
            continue
        record = by_hash.get(attempt["record_sha256"])
        expected = _expected(state["registration"]["plan"], attempt["candle"])
        if (
            record is None
            or record["event"] != expected.event.model_dump()
            or any(
                record["evidence"][key] != getattr(expected, key)
                for key in ("manifest_sha256", "config_sha256")
            )
            or record["observation"]["as_of"] != attempt["started_at"]
            or not pd.Timestamp(attempt["started_at"])
            <= pd.Timestamp(record["observation"]["recorded_at"])
            <= pd.Timestamp(attempt["finished_at"])
        ):
            raise AresError("Successful attempt has no matching immutable receipt")


def collect_paper_slot(
    path: Path,
    *,
    registration_sha256: str,
    candle: str,
    bundle: Path,
    primary: Path,
    secondary: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Dispatch once from local files. Existing attempts NEVER invoke inference again.

    STARTED without an acknowledged terminal result is unresolved even if a
    matching signal is visible. Reread after any write error; do not repair it.
    """
    path = _journal_path(path)
    if not path.is_file():
        raise AresError("Register the collection before attempting a planned slot")
    with _locked(path):
        state = _read_state(path, registration_sha256)
        _, records = _records(path)
        _validate_receipts(state, records)
        for previous in state["attempts"]:
            if previous["candle"] == candle:
                return {"dispatch": "REPLAY_NO_DISPATCH", "attempt": previous}
        plan = state["registration"]["plan"]
        expected = _expected(plan, candle)
        if any(record["event"] == expected.event.model_dump() for record in records):
            raise AresError(
                "Pre-existing receipt cannot establish this attempt; reconcile externally"
            )
        attempt: dict[str, Any] = {
            "candle": candle,
            "attempt_id": _attempt_id(registration_sha256, candle),
            "started_at": utc_timestamp(utc_now()),
            "status": "STARTED",
            "finished_at": None,
            "failure_code": None,
            "record_sha256": None,
        }
        state["attempts"].append(attempt)
        _save(path, state)  # Durable admission precedes reading inputs or invoking inference.
        failure: Failure | None = None
        try:
            frame = read_market(primary)
            others = {name: read_market(source) for name, source in (secondary or {}).items()}
        except (OSError, ValueError, AresError):
            failure = "LOCAL_INPUT_REJECTED"
        if failure is None:
            try:
                result = generate_paper_signal(
                    bundle,
                    frame,
                    secondary_ohlcv=others,
                    log_path=collection_signals_path(path),
                    as_of=pd.Timestamp(attempt["started_at"]),
                    expected=expected,
                )
            except DataQualityError:
                failure = "DATA_QUALITY_REJECTED"
            except BundleIntegrityError:
                failure = "MODEL_INTEGRITY_REJECTED"
            except PaperPlanMismatch:
                failure = "PLAN_MISMATCH"
            else:
                receipt = result.recording
                if receipt is None or receipt.status != "recorded":
                    raise AresError("No newly acknowledged receipt; attempt remains unresolved")
                record = receipt.record
                _, persisted = _records(path)
                if record not in persisted or (
                    record["event"] != expected.event.model_dump()
                    or any(
                        record["evidence"][key] != getattr(expected, key)
                        for key in ("manifest_sha256", "config_sha256")
                    )
                    or record["observation"]["as_of"] != attempt["started_at"]
                    or pd.Timestamp(record["observation"]["recorded_at"])
                    < pd.Timestamp(attempt["started_at"])
                ):
                    raise AresError("Acknowledged receipt differs from the pinned attempt")
                attempt["record_sha256"] = record["record_sha256"]
        attempt["status"] = "FAILED" if failure is not None else "SUCCEEDED"
        attempt["failure_code"] = failure
        attempt["finished_at"] = utc_timestamp(utc_now())
        if attempt["record_sha256"] is not None and pd.Timestamp(
            record["observation"]["recorded_at"]
        ) > pd.Timestamp(attempt["finished_at"]):
            raise AresError("Completion clock precedes receipt; preserve unresolved attempt")
        _save(path, state)
        return {"dispatch": "ONE_ATTEMPT", "attempt": attempt}


def collection_status(path: Path, *, registration_sha256: str, as_of: str) -> dict[str, Any]:
    """Report declared attempt lifecycle separately from successful signal coverage."""
    PaperEvent.canonical_timestamp(as_of)
    path = _journal_path(path)
    state = _read_state(path, registration_sha256)
    reg = state["registration"]
    if pd.Timestamp(as_of) < pd.Timestamp(reg["created_at"]):
        raise AresError("Cutoff predates collection registration")
    journal_hash, records = _records(path)
    _validate_receipts(state, records)
    by_event = {evidence_digest(record["event"]): record for record in records}
    plan = reg["plan"]
    step = pd.Timedelta(seconds=timeframe_to_seconds(plan["timeframe"]))
    attempts = {item["candle"]: item for item in state["attempts"]}
    counts = dict.fromkeys(("NOT_DUE", "NOT_STARTED", "STARTED_UNKNOWN", "FAILED", "SUCCEEDED"), 0)
    rows = []
    for candle in pd.date_range(plan["start"], plan["end"], freq=step, inclusive="left"):
        stamp = utc_timestamp(candle)
        row: dict[str, Any] = {
            "candle": stamp,
            "status": "NOT_DUE" if candle + step > pd.Timestamp(as_of) else "NOT_STARTED",
        }
        attempt = attempts.get(stamp)
        if attempt is not None and pd.Timestamp(attempt["started_at"]) <= pd.Timestamp(as_of):
            row.update(
                attempt_id=attempt["attempt_id"],
                started_at=attempt["started_at"],
                status="STARTED_UNKNOWN",
            )
            if attempt["finished_at"] is not None and pd.Timestamp(
                attempt["finished_at"]
            ) <= pd.Timestamp(as_of):
                row.update(
                    status=attempt["status"],
                    finished_at=attempt["finished_at"],
                    failure_code=attempt["failure_code"],
                    record_sha256=attempt["record_sha256"],
                )
        event = _expected(plan, stamp).event.model_dump()
        observed = by_event.get(evidence_digest(event))
        row["observed_receipt_hashes"] = (
            [observed["record_sha256"]]
            if observed is not None
            and pd.Timestamp(observed["observation"]["recorded_at"]) <= pd.Timestamp(as_of)
            and pd.Timestamp(observed["observation"]["as_of"]) <= pd.Timestamp(as_of)
            else []
        )
        counts[row["status"]] += 1
        rows.append(row)
    report = {
        "schema_version": "ares-paper-attempt-report-v1",
        "registration": reg,
        "state_sha256": state["state_sha256"],
        "journal_sha256": journal_hash,
        "as_of": as_of,
        "planned_slots": len(rows),
        "counts": counts,
        "slots": rows,
        "independent_forward_performance_established": False,
        "scope": "Offline supplied files; STARTED is durable admission, not proof inference ran. Observed receipts do not resolve unknown invocation completion. Failure codes identify caught rejection classes, not detailed feed causes. Local hashes/clocks are unauthenticated.",
    }
    report["report_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report
