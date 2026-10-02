"""Descriptive offline coverage and exact-horizon price annotations.

The plan is supplied, not authenticated preregistration. No forward-performance
claim, trading return, probability label, or failure-cause inference is made.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .exceptions import AresError
from .paper_log import (
    Digest,
    PaperEvent,
    _unique_object,
    evidence_digest,
    read_paper_journal,
    utc_timestamp,
)
from .utils import timeframe_to_seconds

MAX_PLAN_BYTES = 16384
MAX_PRICE_BYTES = 16 * 1024 * 1024
MAX_PRICE_ROWS = 100000
MAX_SLOTS = 10000
PROTOCOL = "ares-paper-report-v1"


class PaperReportPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    schema_version: Literal["ares-paper-plan-v1"]
    exchange: str
    symbol: str
    timeframe: str
    start: str
    end: str
    horizon_bars: int = Field(ge=1, le=1000)
    manifest_sha256: Digest
    config_sha256: Digest
    max_recording_delay_seconds: int = Field(ge=0, le=604800)

    @field_validator("start", "end")
    @classmethod
    def canonical_time(cls, value: str) -> str:
        return PaperEvent.canonical_timestamp(value)

    @model_validator(mode="after")
    def valid_window(self) -> PaperReportPlan:
        PaperEvent(
            exchange=self.exchange,
            symbol=self.symbol,
            timeframe=self.timeframe,
            timestamp=self.start,
        )
        step = pd.Timedelta(seconds=timeframe_to_seconds(self.timeframe))
        start, end = pd.Timestamp(self.start), pd.Timestamp(self.end)
        if start.value % step.value or end.value % step.value or end <= start:
            raise ValueError("Plan must define an aligned, nonempty [start, end) candle window")
        if (end - start) // step > MAX_SLOTS:
            raise ValueError("Plan exceeds the 10000-slot bound")
        return self


def _read_json(path: Path, limit: int) -> Any:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
        raise AresError("Report input must be a bounded regular JSON file")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise AresError("Report input exceeds its byte bound")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)


def _price_snapshot(
    prices: list[dict[str, Any]], plan: PaperReportPlan, cutoff: pd.Timestamp
) -> tuple[dict[str, float], str]:
    if not isinstance(prices, list) or len(prices) > MAX_PRICE_ROWS:
        raise ValueError("Outcome prices must be a list of at most 100000 rows")
    step = pd.Timedelta(seconds=timeframe_to_seconds(plan.timeframe))
    start = pd.Timestamp(plan.start)
    final = pd.Timestamp(plan.end) + (plan.horizon_bars - 1) * step
    seen: set[str] = set()
    consumed: list[dict[str, Any]] = []
    for row in prices:
        if not isinstance(row, dict) or set(row) != {
            "timestamp",
            "exchange",
            "symbol",
            "timeframe",
            "close",
        }:
            raise ValueError("Each outcome row needs exactly timestamp, market identity and close")
        event = PaperEvent.model_validate(
            {key: value for key, value in row.items() if key != "close"}
        )
        if any(
            getattr(event, key) != getattr(plan, key) for key in ("exchange", "symbol", "timeframe")
        ):
            raise ValueError("Outcome market does not match plan")
        stamp = pd.Timestamp(event.timestamp)
        if stamp.value % step.value or event.timestamp in seen:
            raise ValueError("Outcome timestamps must be aligned and unique")
        seen.add(event.timestamp)
        close = row["close"]
        if type(close) not in (float, int) or not math.isfinite(close) or close <= 0:
            raise ValueError("Outcome close must be finite and positive")
        # Only already closed candles in the needed window can affect annotation
        # values or the consumed-data digest. Unused valid future rows are ignored.
        if start <= stamp <= final and stamp + step <= cutoff:
            consumed.append({**event.model_dump(), "close": float(close)})
    consumed.sort(key=lambda row: row["timestamp"])
    return {row["timestamp"]: row["close"] for row in consumed}, evidence_digest(consumed)


def paper_report(
    journal_path: Path, *, plan: dict[str, Any], prices: list[dict[str, Any]], as_of: str
) -> dict[str, Any]:
    """Inspect a supplied schedule and immutable receipts; never evaluate a model.

    Coverage is timely, matching-model successes / due planned candle slots.
    Missing, late and model-mismatched slots stay in that denominator. Receipt
    timestamps are untrusted declarations and never establish forward provenance.
    """
    try:
        selected = PaperReportPlan.model_validate(plan)
        if utc_timestamp(as_of) != as_of:
            raise ValueError("Report as_of must use canonical UTC representation")
        cutoff = pd.Timestamp(as_of)
        step = pd.Timedelta(seconds=timeframe_to_seconds(selected.timeframe))
        journal_hash, records = read_paper_journal(journal_path)
        closes, price_hash = _price_snapshot(prices, selected, cutoff)
        by_candle = {}
        for entry in records:
            event, observed = entry["event"], entry["observation"]
            if any(
                event[key] != getattr(selected, key) for key in ("exchange", "symbol", "timeframe")
            ):
                continue
            if (
                pd.Timestamp(observed["recorded_at"]) > cutoff
                or pd.Timestamp(observed["as_of"]) > cutoff
            ):
                continue
            by_candle[event["timestamp"]] = entry
        rows: list[dict[str, Any]] = []
        counts = {
            key: 0
            for key in ("not_due", "missing_record", "late_record", "model_mismatch", "recorded")
        }
        outcomes = {key: 0 for key in ("pending", "missing_data", "source_conflict", "available")}
        staleness: list[float] = []
        signal_counts = {"long": 0, "flat": 0, "short": 0}
        for candle in pd.date_range(selected.start, selected.end, freq=step, inclusive="left"):
            stamp = utc_timestamp(candle)
            row: dict[str, Any] = {"timestamp": stamp, "status": "not_due"}
            if candle + step <= cutoff:
                row["status"] = "missing_record"
                record = by_candle.get(stamp)
                if record is not None:
                    evidence, observed = record["evidence"], record["observation"]
                    if pd.Timestamp(observed["as_of"]) > pd.Timestamp(observed["recorded_at"]):
                        raise ValueError("Receipt evaluation time follows its recording time")
                    row.update(
                        event_id=record["event_id"],
                        record_sha256=record["record_sha256"],
                        evidence_sha256=record["evidence_sha256"],
                    )
                    deadline = (
                        candle + step + pd.Timedelta(seconds=selected.max_recording_delay_seconds)
                    )
                    late = any(
                        pd.Timestamp(observed[key]) > deadline
                        or pd.Timestamp(observed[key]) < candle + step
                        for key in ("recorded_at", "as_of")
                    )
                    mismatch = any(
                        evidence[key] != getattr(selected, key)
                        for key in ("manifest_sha256", "config_sha256")
                    )
                    row.update(late_record=late, model_mismatch=mismatch)
                    row["status"] = (
                        "model_mismatch" if mismatch else "late_record" if late else "recorded"
                    )
                    if row["status"] == "recorded":
                        staleness.append(observed["staleness_bars"])
                        signal_counts[{1: "long", 0: "flat", -1: "short"}[evidence["signal"]]] += 1
                        target = candle + selected.horizon_bars * step
                        outcome: dict[str, Any] = {
                            "status": "pending",
                            "target_timestamp": utc_timestamp(target),
                            "matures_at": utc_timestamp(target + step),
                            "terminal_close_change": None,
                            "anchor_close": evidence["close"],
                        }
                        if target + step <= cutoff:
                            outcome["status"] = "missing_data"
                            anchor, terminal = closes.get(stamp), closes.get(utc_timestamp(target))
                            if anchor is not None and anchor != evidence["close"]:
                                outcome["status"] = "source_conflict"
                            elif terminal is not None:
                                change = terminal / evidence["close"] - 1.0
                                if not math.isfinite(change):
                                    raise ValueError("Non-finite terminal close change")
                                outcome.update(
                                    status="available",
                                    terminal_close=terminal,
                                    terminal_close_change=change,
                                )
                        outcome["outcome_id"] = evidence_digest(
                            {
                                "protocol": PROTOCOL,
                                "record_sha256": record["record_sha256"],
                                "horizon_bars": selected.horizon_bars,
                            }
                        )
                        outcome["outcome_evidence_sha256"] = evidence_digest(
                            {
                                "outcome": outcome,
                                "consumed_prices_sha256": price_hash,
                                "as_of": as_of,
                            }
                        )
                        outcomes[str(outcome["status"])] += 1
                        row["outcome"] = outcome
            counts[row["status"]] += 1
            rows.append(row)
        due = len(rows) - counts["not_due"]
        report = {
            "schema_version": PROTOCOL,
            "plan": selected.model_dump(),
            "plan_sha256": evidence_digest(selected.model_dump()),
            "as_of": as_of,
            "journal_sha256": journal_hash,
            "consumed_outcome_prices_sha256": price_hash,
            "coverage": {
                "planned_slots": len(rows),
                "due_slots": due,
                **counts,
                "timely_matching_coverage": counts["recorded"] / due if due else None,
            },
            "outcomes": outcomes,
            "recorded_signal_counts": signal_counts,
            "recorded_staleness_bars": {
                "count": len(staleness),
                "min": min(staleness) if staleness else None,
                "max": max(staleness) if staleness else None,
            },
            "observations": rows,
            "independent_forward_performance_established": False,
            "missing_record_causes": "UNKNOWN: successful receipts do not record attempted or failed evaluations",
            "statistical_drift": "UNAVAILABLE: no declared reference distribution or feature values",
            "metric_scope": "Descriptive terminal close change, not execution P&L, probability-target accuracy or compounded return; timestamps do not prove preregistration",
        }
        report["report_sha256"] = evidence_digest(report)
        return report
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        raise AresError(f"Paper report blocked: {exc}") from exc


def paper_report_from_files(
    journal_path: Path, plan_path: Path, prices_path: Path, as_of: str
) -> dict[str, Any]:
    try:
        plan = _read_json(plan_path, MAX_PLAN_BYTES)
        prices = _read_json(prices_path, MAX_PRICE_BYTES)
        return paper_report(journal_path, plan=plan, prices=prices, as_of=as_of)
    except (ValueError, OSError) as exc:
        raise AresError(f"Paper report input blocked: {exc}") from exc
