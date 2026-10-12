"""Bounded local supplied-file capture; never a provider acquisition receipt."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data.storage import read_market_bytes
from .exceptions import AresError
from .live import _frame_digest
from .paper_log import MAX_LOG_BYTES, Digest, PaperEvent, _regular_file, _replace_journal

MAX_CAPTURE_BYTES = 256 * 1024 * 1024


class SourceCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    role: Literal["PRIMARY", "SECONDARY"]
    exchange: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    status: Literal["PENDING", "STARTED", "CAPTURED", "READY", "READ_REJECTED", "PARSE_REJECTED"]
    started_at: str | None
    captured_at: str | None
    finished_at: str | None
    raw_sha256: Digest | None
    raw_bytes: int | None = Field(ge=0, le=MAX_LOG_BYTES)
    frame_sha256: Digest | None

    @field_validator("started_at", "captured_at", "finished_at")
    @classmethod
    def canonical_time(cls, value: str | None) -> str | None:
        return PaperEvent.canonical_timestamp(value) if value is not None else None


class InputCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["ares-local-input-capture-v1"]
    sources: list[SourceCapture] = Field(min_length=1, max_length=9)


def requested_capture(exchange: str, secondary: Mapping[str, Path]) -> dict[str, Any]:
    if exchange in secondary:
        raise AresError("Captured primary and secondary source names must differ")
    sources = []
    for role, name in [("PRIMARY", exchange), *[("SECONDARY", n) for n in sorted(secondary)]]:
        sources.append(
            dict(
                role=role,
                exchange=name,
                status="PENDING",
                started_at=None,
                captured_at=None,
                finished_at=None,
                raw_sha256=None,
                raw_bytes=None,
                frame_sha256=None,
            )
        )
    return InputCapture.model_validate(
        {"schema_version": "ares-local-input-capture-v1", "sources": sources}
    ).model_dump(mode="json")


def validate_capture(attempt: dict[str, Any], exchange: str) -> pd.Timestamp:
    """Validate causal source transitions, including acknowledged partial captures."""
    capture = attempt.get("input_capture")
    last = pd.Timestamp(attempt["started_at"])
    if capture is None:
        return last
    sources = capture["sources"]
    if (
        sources[0]["role"] != "PRIMARY"
        or sources[0]["exchange"] != exchange
        or any(s["role"] != "SECONDARY" for s in sources[1:])
        or len({s["exchange"] for s in sources}) != len(sources)
        or [s["exchange"] for s in sources[1:]] != sorted(s["exchange"] for s in sources[1:])
    ):
        raise AresError("Invalid captured source inventory")
    stopped = False
    for source in sources:
        status = source["status"]
        started, captured, finished = (
            source[k] for k in ("started_at", "captured_at", "finished_at")
        )
        has_raw = source["raw_sha256"] is not None and source["raw_bytes"] is not None
        any_raw = source["raw_sha256"] is not None or source["raw_bytes"] is not None
        has_frame = source["frame_sha256"] is not None
        valid = {
            "PENDING": started is None
            and captured is None
            and finished is None
            and not any_raw
            and not has_frame,
            "STARTED": started is not None
            and captured is None
            and finished is None
            and not any_raw
            and not has_frame,
            "CAPTURED": started is not None
            and captured is not None
            and finished is None
            and has_raw
            and not has_frame,
            "READY": started is not None
            and captured is not None
            and finished is not None
            and has_raw
            and has_frame,
            "READ_REJECTED": started is not None
            and captured is None
            and finished is not None
            and not any_raw
            and not has_frame,
            "PARSE_REJECTED": started is not None
            and captured is not None
            and finished is not None
            and has_raw
            and not has_frame,
        }[status]
        if not valid or (stopped and status != "PENDING"):
            raise AresError("Inconsistent captured source transition")
        for time in (started, captured, finished):
            if time is not None:
                current = pd.Timestamp(time)
                if current < last:
                    raise AresError("Local capture clock regressed")
                last = current
        stopped = status != "READY"
    statuses = {s["status"] for s in sources}
    if attempt["status"] == "SUCCEEDED" and statuses != {"READY"}:
        raise AresError("Successful capture attempt has incomplete inputs")
    if attempt["status"] == "FAILED":
        if statuses & {"STARTED", "CAPTURED"} or not (
            statuses & {"READ_REJECTED", "PARSE_REJECTED"} or statuses == {"READY"}
        ):
            raise AresError("Failed capture attempt has unresolved source completion")
        if (
            statuses & {"READ_REJECTED", "PARSE_REJECTED"}
            and attempt["failure_code"] != "LOCAL_INPUT_REJECTED"
        ):
            raise AresError("Source rejection differs from attempt rejection")
    if attempt["finished_at"] is not None and pd.Timestamp(attempt["finished_at"]) < last:
        raise AresError("Attempt completion precedes captured input evidence")
    return last


def snapshot_path(collection: Path, attempt_id: str, index: int) -> Path:
    if (
        len(attempt_id) != 64
        or any(c not in "0123456789abcdef" for c in attempt_id)
        or not 0 <= index < 9
    ):
        raise AresError("Invalid input snapshot identity")
    root = collection.with_name(collection.name + ".inputs")
    directory = root / attempt_id
    for part in (root, directory):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISDIR(info.st_mode)
            or part.is_symlink()
            or getattr(info, "st_file_attributes", 0) & 0x400
        ):
            raise AresError("Input snapshot directory must be unlinked")
    path = directory / f"{index}.parquet"
    _regular_file(path)
    return path


def _read_bytes(path: Path) -> bytes:
    _regular_file(path)
    with path.open("rb") as handle:
        raw = handle.read(MAX_LOG_BYTES + 1)
    if len(raw) > MAX_LOG_BYTES:
        raise AresError("Input exceeds the 16 MiB capture bound")
    return raw


def validate_snapshots(path: Path, state: dict[str, Any]) -> None:
    total = 0
    for attempt in state["attempts"]:
        capture = attempt.get("input_capture")
        if capture is None:
            continue
        for index, source in enumerate(capture["sources"]):
            if source["raw_sha256"] is None:
                continue
            raw = _read_bytes(snapshot_path(path, attempt["attempt_id"], index))
            if (
                len(raw) != source["raw_bytes"]
                or hashlib.sha256(raw).hexdigest() != source["raw_sha256"]
            ):
                raise AresError("Preserved input snapshot differs; do not recapture or repair")
            if (
                source["status"] == "READY"
                and _frame_digest(read_market_bytes(raw)) != source["frame_sha256"]
            ):
                raise AresError("Preserved input frame differs from its declared normalized hash")
            total += len(raw)
    if total > MAX_CAPTURE_BYTES:
        raise AresError("Collection input captures exceed the 256 MiB bound")
    if physical_capture_bytes(path, state) > MAX_CAPTURE_BYTES:
        raise AresError("Preserved input files exceed the 256 MiB capture bound")


def physical_capture_bytes(path: Path, state: dict[str, Any]) -> int:
    """Count acknowledged and orphan bytes without adopting unacknowledged evidence."""
    inventories = {
        a["attempt_id"]: a["input_capture"]
        for a in state["attempts"]
        if a.get("input_capture") is not None
    }
    root = path.with_name(path.name + ".inputs")
    # snapshot_path checks the root itself even when no attempt directory exists.
    snapshot_path(path, "0" * 64, 0)
    if not root.exists():
        return 0
    total = 0
    for directory in root.iterdir():
        if directory.name not in inventories:
            raise AresError("Unowned input capture directory; preserve and reconcile")
        snapshot_path(path, directory.name, 0)
        for candidate in directory.iterdir():
            # Retained temporary files from process death count toward the budget too.
            _regular_file(candidate)
            total += candidate.stat().st_size
            if total > MAX_CAPTURE_BYTES:
                raise AresError("Preserved input files exceed the 256 MiB capture bound")
    return total


def captured_frames(
    path: Path,
    state: dict[str, Any],
    attempt: dict[str, Any],
    *,
    primary: Path,
    secondary: Mapping[str, Path],
    clock: Callable[[], str],
    save: Callable[[], None],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]] | None:
    """Persistence failures propagate; only local reads/decoding are classified."""
    total = physical_capture_bytes(path, state)
    frames: list[pd.DataFrame] = []
    others: dict[str, pd.DataFrame] = {}
    for index, source in enumerate(attempt["input_capture"]["sources"]):
        source.update(status="STARTED", started_at=clock())
        save()
        rejected = False
        try:
            raw = _read_bytes(primary if index == 0 else secondary[source["exchange"]])
            if total + len(raw) > MAX_CAPTURE_BYTES:
                raise AresError("Collection input capture budget exhausted")
        except (OSError, ValueError, AresError):
            rejected = True
        if rejected:
            source.update(status="READ_REJECTED", finished_at=clock())
            save()
            return None
        # Never put publication/state writes inside the input-rejection catch.
        target = snapshot_path(path, attempt["attempt_id"], index)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise AresError("Unowned input snapshot exists; preserve unresolved attempt")
        _replace_journal(target, raw)
        source.update(
            status="CAPTURED",
            captured_at=clock(),
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            raw_bytes=len(raw),
        )
        total += len(raw)
        save()
        parse_rejected = False
        try:
            frame = read_market_bytes(raw)
            digest = _frame_digest(frame)
        except (OSError, ValueError, AresError, KeyError, TypeError, OverflowError):
            parse_rejected = True
        if parse_rejected:
            source.update(status="PARSE_REJECTED", finished_at=clock())
            save()
            return None
        source.update(status="READY", finished_at=clock(), frame_sha256=digest)
        save()
        frames.append(frame)
        if index:
            others[source["exchange"]] = frame
    return frames[0], others


def receipt_matches_capture(attempt: dict[str, Any], evidence: dict[str, Any]) -> bool:
    capture = attempt.get("input_capture")
    if capture is None:
        return True
    sources = capture["sources"]
    return (
        all(s["status"] == "READY" for s in sources)
        and evidence["primary_sha256"] == sources[0]["frame_sha256"]
        and evidence["secondary_sha256"] == {s["exchange"]: s["frame_sha256"] for s in sources[1:]}
    )


def receipt_not_before_inputs(attempt: dict[str, Any], recorded_at: str) -> bool:
    capture = attempt.get("input_capture")
    return capture is None or all(
        source["finished_at"] is not None
        and pd.Timestamp(recorded_at) >= pd.Timestamp(source["finished_at"])
        for source in capture["sources"]
    )


def capture_at_cutoff(attempt: dict[str, Any], cutoff: str) -> dict[str, Any] | None:
    capture = attempt.get("input_capture")
    if capture is None:
        return None
    rows = []
    for source in capture["sources"]:
        row = dict(source)
        if row["finished_at"] is not None and pd.Timestamp(row["finished_at"]) > pd.Timestamp(
            cutoff
        ):
            row.update(
                status="CAPTURED" if row["captured_at"] else "STARTED",
                finished_at=None,
                frame_sha256=None,
            )
        if row["captured_at"] is not None and pd.Timestamp(row["captured_at"]) > pd.Timestamp(
            cutoff
        ):
            row.update(status="STARTED", captured_at=None, raw_sha256=None, raw_bytes=None)
        if row["started_at"] is not None and pd.Timestamp(row["started_at"]) > pd.Timestamp(cutoff):
            row.update(status="PENDING", started_at=None)
        rows.append(row)
    return {"schema_version": capture["schema_version"], "sources": rows}
