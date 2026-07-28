"""Credential-free, independently revalidated public-ingestion evidence."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from . import __version__
from .config import AresConfig
from .data.ingest import IngestionResult, ingest_market_data
from .data.providers import CCXTOHLCVProvider, FetchTelemetry, OHLCVProvider
from .data.quality import validate_cross_venue, validate_ohlcv
from .data.storage import current_generation, market_path, read_market
from .exceptions import AresError, DataQualityError
from .utils import atomic_write_json, sha256_file, timeframe_to_seconds

REPORT_NAME = "public_ingestion_report.json"
REPORT_MD_NAME = "public_ingestion_report.md"
REQUEST_TRACE_NAME = "request_trace.json"
REQUEST_SUMMARY_NAME = "request_summary.json"
QUALITY_NAME = "quality_report.json"
MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "SHA256SUMS.txt"
EVIDENCE_SCHEMA = "ares-public-ingestion-evidence-v2"
BASE_ARTIFACTS = {
    REPORT_NAME,
    REPORT_MD_NAME,
    REQUEST_TRACE_NAME,
    REQUEST_SUMMARY_NAME,
    QUALITY_NAME,
    MANIFEST_NAME,
    CHECKSUMS_NAME,
}
PARQUET_ARTIFACTS = {"coinbase_normalized.parquet", "kraken_normalized.parquet"}
PARQUET_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "exchange",
    "symbol",
    "timeframe",
]
TRACE_SUMMARY_FIELDS = {
    "http_request_count",
    "fetch_ohlcv_call_count",
    "page_count",
    "raw_rows_received",
    "normalized_rows",
    "duplicate_rows_removed",
    "overlapping_rows_removed",
    "out_of_range_rows_removed",
    "empty_pages",
    "short_pages",
    "first_request_cursor_ms",
    "last_request_cursor_ms",
    "cursor_progression_ms",
    "retry_count",
    "rate_limit_responses",
    "response_statuses",
    "failures",
}

REQUIRED_REPORT_FIELDS: dict[str, type | tuple[type, ...]] = {
    "schema": str,
    "live_public_endpoints_reached": bool,
    "credentials_used": bool,
    "orders_possible": bool,
    "fixture_mode": bool,
    "primary_exchange": str,
    "validation_exchange": str,
    "symbol": str,
    "timeframe": str,
    "requested_start": str,
    "requested_end": str,
    "expected_rows": int,
    "primary_http_requests": int,
    "validation_http_requests": int,
    "primary_fetch_ohlcv_calls": int,
    "validation_fetch_ohlcv_calls": int,
    "primary_pages": int,
    "validation_pages": int,
    "primary_rows": int,
    "validation_rows": int,
    "primary_raw_rows": int,
    "validation_raw_rows": int,
    "primary_duplicate_rows": int,
    "validation_duplicate_rows": int,
    "coverage_passed": bool,
    "quality_passed": bool,
    "alignment_passed": bool,
    "divergence_passed": bool,
    "idempotency_passed": bool,
    "overall_passed": bool,
    "failure": (str, type(None)),
    "failure_classification": (str, type(None)),
    "canonical_hashes_run1": dict,
    "canonical_hashes_run2": dict,
    "parquet_file_hashes_run1": dict,
    "parquet_file_hashes_run2": dict,
}


@dataclass
class InstrumentedProvider:
    """Capture measured provider telemetry without estimating pagination."""

    inner: OHLCVProvider
    exchange_id: str = ""
    pages: int = 0
    raw_rows: int = 0
    duplicate_rows: int = 0
    failures: list[str] = field(default_factory=list)
    telemetry: FetchTelemetry | None = None

    def __post_init__(self) -> None:
        self.exchange_id = self.inner.exchange_id

    def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime | None,
        *,
        limit: int,
        max_pages: int,
        retries: int,
    ) -> pd.DataFrame:
        try:
            frame = self.inner.fetch_range(
                symbol,
                timeframe,
                since,
                until,
                limit=limit,
                max_pages=max_pages,
                retries=retries,
            )
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            measured = getattr(self.inner, "last_trace", None)
            if isinstance(measured, FetchTelemetry):
                self.telemetry = measured
            raise
        measured = getattr(self.inner, "last_trace", None)
        if isinstance(measured, FetchTelemetry):
            self.telemetry = measured
            self.pages = measured.page_count
            self.raw_rows = measured.raw_rows_received
            self.duplicate_rows = measured.duplicate_rows_removed
            return frame
        # Fixtures that do not expose transport telemetry are truthfully labeled.
        self.pages = 1
        self.raw_rows = int(len(frame))
        if "timestamp" in frame.columns:
            self.duplicate_rows = int(frame["timestamp"].duplicated().sum())
        return frame

    def trace_dict(self) -> dict[str, Any]:
        if self.telemetry is not None:
            payload = self.telemetry.to_dict()
        else:
            payload = {
                "exchange": self.exchange_id,
                "symbol": "",
                "timeframe": "",
                "http_request_count": 0,
                "fetch_ohlcv_call_count": self.pages,
                "page_count": self.pages,
                "raw_rows_received": self.raw_rows,
                "normalized_rows": self.raw_rows - self.duplicate_rows,
                "duplicate_rows_removed": self.duplicate_rows,
                "overlapping_rows_removed": 0,
                "out_of_range_rows_removed": 0,
                "empty_pages": 0,
                "short_pages": 0,
                "first_request_cursor_ms": None,
                "last_request_cursor_ms": None,
                "cursor_progression_ms": [],
                "retry_count": 0,
                "rate_limit_responses": 0,
                "response_statuses": [],
                "requests": [],
            }
        payload["failures"] = list(self.failures)
        return payload


def _audit_config(
    *,
    primary: str,
    validation: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    storage_dir: Path,
    page_limit: int,
    retries: int,
) -> AresConfig:
    raw = {
        "project": {"name": "ARES public-ingestion audit", "seed": 1},
        "storage": {
            "root": str(storage_dir / "data"),
            "artifacts": str(storage_dir / "artifacts"),
            "duckdb_path": str(storage_dir / "data" / "ares.duckdb"),
        },
        "data": {
            "primary_exchange": primary,
            "validation_exchanges": [validation],
            "symbol": symbol,
            "timeframe": timeframe,
            "since": start.isoformat(),
            "until": end.isoformat(),
            "page_limit": page_limit,
            "max_pages": 10_000,
            "retries": retries,
            "drop_open_candle": True,
            "resume": False,
            "fail_on_quality": False,
            "max_gap_count": 0,
            "max_cross_venue_p95_bps": 75.0,
            "max_cross_venue_latest_bps": 150.0,
            "min_cross_venue_overlap": 100,
            "max_staleness_bars": 2,
        },
    }
    return AresConfig.model_validate(raw)


def _classify_failure(exc: Exception) -> str:
    description = f"{type(exc).__name__}: {exc}"
    lowered = description.lower()
    if "403" in lowered or "forbidden" in lowered or "blocked" in lowered or "tunnel" in lowered:
        return "environment_or_endpoint_block"
    if "timeout" in lowered or "timed out" in lowered:
        return "network_timeout"
    if "dns" in lowered or "name resolution" in lowered or "getaddrinfo" in lowered:
        return "dns_failure"
    return "ingestion_error"


def _logical_frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "absent"
    normalized = frame[PARQUET_COLUMNS].copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True).map(
        lambda value: value.isoformat()
    )
    payload = normalized.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_once(
    *,
    run_number: int,
    root: Path,
    primary: str,
    validation: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    page_limit: int,
    retries: int,
    providers: dict[str, OHLCVProvider] | None,
) -> dict[str, Any]:
    storage_dir = root / f"run-{run_number}"
    config = _audit_config(
        primary=primary,
        validation=validation,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        storage_dir=storage_dir,
        page_limit=page_limit,
        retries=retries,
    )
    instrumented = {
        exchange: InstrumentedProvider(
            (providers or {}).get(exchange) or CCXTOHLCVProvider(exchange)
        )
        for exchange in [primary, validation]
    }
    result: IngestionResult | None = None
    failure: str | None = None
    failure_classification: str | None = None
    try:
        result = ingest_market_data(config, providers=dict(instrumented))
    except (AresError, DataQualityError, OSError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        failure_classification = _classify_failure(exc)

    generation = current_generation(config.storage.root)
    frames: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    for exchange in [primary, validation]:
        path = market_path(
            config.storage.root,
            exchange,
            symbol,
            timeframe,
            generation=generation,
        )
        paths[exchange] = path
        frames[exchange] = read_market(path)
    return {
        "config": config,
        "result": result,
        "failure": failure,
        "failure_classification": failure_classification,
        "generation": generation,
        "frames": frames,
        "paths": paths,
        "logical_hashes": {name: _logical_frame_hash(frame) for name, frame in frames.items()},
        "file_hashes": {
            name: sha256_file(path) if path.is_file() else "absent" for name, path in paths.items()
        },
        "traces": {name: wrapper.trace_dict() for name, wrapper in instrumented.items()},
    }


def _write_manifest_and_checksums(output_dir: Path) -> None:
    manifest_targets = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in {MANIFEST_NAME, CHECKSUMS_NAME}
    )
    atomic_write_json(
        output_dir / MANIFEST_NAME,
        {
            "schema": EVIDENCE_SCHEMA,
            "files": {
                path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in manifest_targets
            },
        },
    )
    checksum_targets = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != CHECKSUMS_NAME
    )
    (output_dir / CHECKSUMS_NAME).write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_targets),
        encoding="utf-8",
    )


def run_public_ingestion_audit(
    *,
    primary: str = "coinbase",
    validation: str = "kraken",
    symbol: str = "ETH/USD",
    timeframe: str = "1h",
    start: datetime,
    end: datetime,
    output_dir: Path,
    page_limit: int = 300,
    retries: int = 3,
    providers: dict[str, OHLCVProvider] | None = None,
    fixture_note: str | None = None,
) -> dict[str, Any]:
    """Run the production path twice in isolated storage and emit strict evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_mode = providers is not None
    start = start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
    end = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
    if end <= start:
        raise ValueError("Public audit end must be after start")
    step_seconds = timeframe_to_seconds(timeframe)
    if not fixture_mode:
        if primary != "coinbase" or validation != "kraken":
            raise ValueError("Live public audit requires Coinbase primary and Kraken validation")
        if symbol != "ETH/USD" or timeframe != "1h":
            raise ValueError("Live public audit requires ETH/USD at 1h")
        if end > datetime.now(tz=UTC) - timedelta(hours=48):
            raise ValueError("Live public audit must end at least 48 hours before current UTC")
        if end - start < timedelta(days=14):
            raise ValueError("Live public audit must cover at least 14 days")
        if int(start.timestamp()) % step_seconds or int(end.timestamp()) % step_seconds:
            raise ValueError("Live public audit boundaries must be on the UTC timeframe grid")
        required_rows = int((end - start).total_seconds() // step_seconds)
        if math.ceil(required_rows / page_limit) < 4:
            raise ValueError("Live public audit must force at least four pages per venue")
    for name in BASE_ARTIFACTS | PARQUET_ARTIFACTS:
        (output_dir / name).unlink(missing_ok=True)
    expected_rows = int((end - start).total_seconds() // step_seconds)

    with tempfile.TemporaryDirectory(prefix=".ares-public-audit-", dir=output_dir.parent) as temp:
        root = Path(temp)
        runs = [
            _run_once(
                run_number=number,
                root=root,
                primary=primary,
                validation=validation,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                page_limit=page_limit,
                retries=retries,
                providers=providers,
            )
            for number in [1, 2]
        ]

        first, second = runs
        first_result = first["result"]
        primary_frame = first["frames"][primary]
        validation_frame = first["frames"][validation]
        coverage_reports = {
            exchange: validate_ohlcv(
                frame,
                timeframe,
                source=f"{exchange}-audit-coverage",
                max_gap_count=0,
                expected_exchange=exchange,
                expected_symbol=symbol,
                expected_timeframe=timeframe,
                expected_start=start,
                expected_end=end,
            )
            for exchange, frame in [
                (primary, primary_frame),
                (validation, validation_frame),
            ]
        }
        coverage_passed = all(report.passed for report in coverage_reports.values()) and all(
            len(frame) == expected_rows for frame in [primary_frame, validation_frame]
        )
        quality_passed = bool(first_result and first_result.passed)
        cross = (
            first_result.cross_venue_reports[0]
            if first_result and first_result.cross_venue_reports
            else None
        )
        cross_stats = (
            {
                key: str(value) if isinstance(value, pd.Timestamp) else value
                for key, value in cross.stats.items()
            }
            if cross
            else {}
        )
        cross_codes = {issue.code for issue in cross.issues} if cross else set()
        alignment_passed = bool(cross) and not (
            {"latest_timestamp_misalignment", "insufficient_overlap"} & cross_codes
        )
        divergence_passed = bool(cross) and not (
            {"cross_venue_divergence", "latest_cross_venue_divergence"} & cross_codes
        )
        idempotency_passed = bool(
            first_result
            and first_result.committed
            and second["result"]
            and second["result"].committed
            and first["logical_hashes"] == second["logical_hashes"]
            and all(value != "absent" for value in first["logical_hashes"].values())
        )
        no_failures = all(run["failure"] is None for run in runs)
        live_reached = bool(
            not fixture_mode
            and no_failures
            and all(
                int(run["traces"][exchange]["http_request_count"]) > 0
                and int(run["traces"][exchange]["fetch_ohlcv_call_count"]) > 0
                and int(run["traces"][exchange]["raw_rows_received"]) > 0
                for run in runs
                for exchange in [primary, validation]
            )
        )
        rows_present = len(primary_frame) > 0 and len(validation_frame) > 0
        overall_passed = bool(
            no_failures
            and quality_passed
            and coverage_passed
            and alignment_passed
            and divergence_passed
            and idempotency_passed
            and rows_present
            and (fixture_mode or live_reached)
        )
        trace_runs = {
            f"run_{number}": {"venues": run["traces"]} for number, run in enumerate(runs, start=1)
        }
        first_primary_trace = first["traces"][primary]
        first_validation_trace = first["traces"][validation]
        failures = [run["failure"] for run in runs if run["failure"]]
        classifications = [
            run["failure_classification"] for run in runs if run["failure_classification"]
        ]
        generated_at = datetime.now(tz=UTC).isoformat()
        mode_label = (
            "fixture-simulation (NOT live endpoints)" if fixture_mode else "live-public-endpoints"
        )
        report: dict[str, Any] = {
            "schema": EVIDENCE_SCHEMA,
            "generated_at": generated_at,
            "ares_version": __version__,
            "python": platform.python_version(),
            "mode": mode_label,
            "fixture_mode": fixture_mode,
            "fixture_note": fixture_note,
            "live_public_endpoints_reached": live_reached,
            "credentials_used": False,
            "orders_possible": False,
            "primary_exchange": primary,
            "validation_exchange": validation,
            "symbol": symbol,
            "timeframe": timeframe,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "expected_rows": expected_rows,
            "primary_http_requests": first_primary_trace["http_request_count"],
            "validation_http_requests": first_validation_trace["http_request_count"],
            "primary_fetch_ohlcv_calls": first_primary_trace["fetch_ohlcv_call_count"],
            "validation_fetch_ohlcv_calls": first_validation_trace["fetch_ohlcv_call_count"],
            "primary_pages": first_primary_trace["page_count"],
            "validation_pages": first_validation_trace["page_count"],
            "primary_raw_rows": first_primary_trace["raw_rows_received"],
            "validation_raw_rows": first_validation_trace["raw_rows_received"],
            "primary_rows": int(len(primary_frame)),
            "validation_rows": int(len(validation_frame)),
            "primary_duplicate_rows": first_primary_trace["duplicate_rows_removed"],
            "validation_duplicate_rows": first_validation_trace["duplicate_rows_removed"],
            "primary_first_timestamp": str(primary_frame["timestamp"].min())
            if len(primary_frame)
            else None,
            "primary_last_timestamp": str(primary_frame["timestamp"].max())
            if len(primary_frame)
            else None,
            "validation_first_timestamp": str(validation_frame["timestamp"].min())
            if len(validation_frame)
            else None,
            "validation_last_timestamp": str(validation_frame["timestamp"].max())
            if len(validation_frame)
            else None,
            "cross_venue": cross_stats,
            "coverage_passed": coverage_passed,
            "quality_passed": quality_passed,
            "alignment_passed": alignment_passed,
            "divergence_passed": divergence_passed,
            "idempotency_passed": idempotency_passed,
            "canonical_hashes_run1": first["logical_hashes"],
            "canonical_hashes_run2": second["logical_hashes"],
            "parquet_file_hashes_run1": first["file_hashes"],
            "parquet_file_hashes_run2": second["file_hashes"],
            "failure": "; ".join(failures) if failures else None,
            "failure_classification": classifications[0] if classifications else None,
            "overall_passed": overall_passed,
        }
        atomic_write_json(output_dir / REPORT_NAME, report)
        atomic_write_json(
            output_dir / REQUEST_TRACE_NAME,
            {
                "schema": EVIDENCE_SCHEMA,
                "mode": mode_label,
                "fixture_mode": fixture_mode,
                "generated_at": generated_at,
                "runs": trace_runs,
            },
        )
        atomic_write_json(
            output_dir / REQUEST_SUMMARY_NAME,
            {
                "schema": EVIDENCE_SCHEMA,
                "mode": mode_label,
                "fixture_mode": fixture_mode,
                "generated_at": generated_at,
                "runs": {
                    run_name: {
                        "venues": {
                            exchange: {key: venue[key] for key in sorted(TRACE_SUMMARY_FIELDS)}
                            for exchange, venue in run["venues"].items()
                        }
                    }
                    for run_name, run in trace_runs.items()
                },
            },
        )
        quality_payload: dict[str, Any] = {
            "schema": EVIDENCE_SCHEMA,
            "mode": mode_label,
            "fixture_mode": fixture_mode,
            "quality": first_result.to_dict() if first_result is not None else None,
            "independent_coverage": {
                exchange: report.to_dict() for exchange, report in coverage_reports.items()
            },
        }
        atomic_write_json(output_dir / QUALITY_NAME, quality_payload)

        if overall_passed:
            shutil.copyfile(first["paths"][primary], output_dir / "coinbase_normalized.parquet")
            shutil.copyfile(first["paths"][validation], output_dir / "kraken_normalized.parquet")

    lines = [
        f"# ARES public-ingestion audit ({mode_label})",
        "",
        f"- Generated: {generated_at}",
        f"- Live public endpoints reached: **{live_reached}**",
        "- Credentials used: **False** | Orders possible: **False**",
        f"- Venues: {primary} (primary) vs {validation} (validation)",
        f"- Market: {symbol} @ {timeframe}",
        f"- Window: {start.isoformat()} -> {end.isoformat()} ({expected_rows} expected candles)",
        f"- Measured run-one requests: {primary} HTTP={report['primary_http_requests']} "
        f"fetch_ohlcv={report['primary_fetch_ohlcv_calls']} pages={report['primary_pages']}; "
        f"{validation} HTTP={report['validation_http_requests']} "
        f"fetch_ohlcv={report['validation_fetch_ohlcv_calls']} pages={report['validation_pages']}",
        f"- Gates: coverage={coverage_passed} quality={quality_passed} "
        f"alignment={alignment_passed} divergence={divergence_passed} "
        f"idempotency={idempotency_passed}",
        f"- Failure: {report['failure'] or 'none'} ({report['failure_classification'] or 'n/a'})",
        f"- **OVERALL: {'PASSED' if overall_passed else 'FAILED'}**",
    ]
    if fixture_mode:
        lines.insert(
            1,
            "\n> SIMULATION NOTICE: injected providers were used. This does not prove live endpoint compatibility.\n",
        )
    (output_dir / REPORT_MD_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_manifest_and_checksums(output_dir)
    return report


def _append_nonfinite_problems(value: Any, path: str, problems: list[str]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        problems.append(f"non-finite numeric value at {path}")
    elif isinstance(value, dict):
        for key, nested in value.items():
            _append_nonfinite_problems(nested, f"{path}.{key}", problems)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _append_nonfinite_problems(nested, f"{path}[{index}]", problems)


def _parse_json(path: Path, label: str, problems: list[str]) -> Any | None:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key: {key}")
            payload[key] = value
        return payload

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, ValueError) as exc:
        problems.append(f"{label} is not valid JSON: {exc}")
        return None


def _validate_file_integrity(output_dir: Path, problems: list[str]) -> None:
    entries = list(output_dir.iterdir())
    unsafe_entries = [
        path.name
        for path in entries
        if path.is_symlink() or not path.is_file() or path.stat(follow_symlinks=False).st_nlink != 1
    ]
    if unsafe_entries:
        problems.append(f"nested, linked, or special artifact entries: {sorted(unsafe_entries)}")
    actual = {
        path.name
        for path in entries
        if path.is_file()
        and not path.is_symlink()
        and path.stat(follow_symlinks=False).st_nlink == 1
    }

    report_payload = _parse_json(output_dir / REPORT_NAME, REPORT_NAME, problems)
    passed = isinstance(report_payload, dict) and report_payload.get("overall_passed") is True
    expected_artifacts = BASE_ARTIFACTS | (PARQUET_ARTIFACTS if passed else set())
    if actual != expected_artifacts:
        problems.append(
            "artifact file set is not exact: "
            f"expected {sorted(expected_artifacts)}, got {sorted(actual)}"
        )

    manifest_path = output_dir / MANIFEST_NAME
    manifest = (
        _parse_json(manifest_path, MANIFEST_NAME, problems) if manifest_path.is_file() else None
    )
    if manifest is None:
        if not manifest_path.exists():
            problems.append(f"missing {MANIFEST_NAME}")
    else:
        if manifest.get("schema") != EVIDENCE_SCHEMA or not isinstance(manifest.get("files"), dict):
            problems.append("manifest schema or file table is invalid")
        else:
            expected_files = actual.difference({MANIFEST_NAME, CHECKSUMS_NAME})
            if set(manifest["files"]) != expected_files:
                problems.append("manifest file set differs from actual artifacts")
            for name, expected in manifest["files"].items():
                pure = PurePosixPath(name)
                path = output_dir / name
                if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
                    problems.append(f"unsafe manifest path: {name}")
                elif (
                    not path.is_file()
                    or path.is_symlink()
                    or path.stat(follow_symlinks=False).st_nlink != 1
                ):
                    problems.append(f"manifest file missing or non-regular: {name}")
                elif not isinstance(expected, dict):
                    problems.append(f"malformed manifest entry: {name}")
                else:
                    expected_bytes = expected.get("bytes")
                    expected_hash = expected.get("sha256")
                    if (
                        isinstance(expected_bytes, bool)
                        or not isinstance(expected_bytes, int)
                        or expected_bytes < 0
                    ):
                        problems.append(f"malformed manifest size: {name}")
                    elif path.stat().st_size != expected_bytes:
                        problems.append(f"manifest size mismatch: {name}")
                    if (
                        not isinstance(expected_hash, str)
                        or len(expected_hash) != 64
                        or any(character not in "0123456789abcdef" for character in expected_hash)
                    ):
                        problems.append(f"malformed manifest checksum: {name}")
                    elif sha256_file(path) != expected_hash:
                        problems.append(f"manifest checksum mismatch: {name}")

    checksums_path = output_dir / CHECKSUMS_NAME
    if not checksums_path.is_file():
        problems.append(f"missing {CHECKSUMS_NAME}")
        return
    parsed: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            problems.append("malformed checksum line")
            continue
        if name in parsed:
            problems.append(f"duplicate checksum entry: {name}")
            continue
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
            problems.append(f"unsafe checksum path: {name}")
            continue
        parsed[name] = digest
    expected_checksum_names = actual.difference({CHECKSUMS_NAME})
    if set(parsed) != expected_checksum_names:
        problems.append("checksum file set differs from actual artifacts")
    for name, digest in parsed.items():
        path = output_dir / name
        if path.is_file() and sha256_file(path) != digest:
            problems.append(f"checksum mismatch: {name}")


def _validate_trace(
    trace: Any,
    summary: Any,
    report: dict[str, Any],
    problems: list[str],
) -> None:
    if not isinstance(trace, dict) or not isinstance(summary, dict):
        return
    for label, payload in [("request trace", trace), ("request summary", summary)]:
        if payload.get("schema") != EVIDENCE_SCHEMA:
            problems.append(f"{label} schema is invalid")
        if payload.get("fixture_mode") is not report["fixture_mode"]:
            problems.append(f"{label} fixture mode differs from report")
    runs = trace.get("runs")
    summary_runs = summary.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"run_1", "run_2"}:
        problems.append("request trace must contain independent run_1 and run_2")
        return
    if not isinstance(summary_runs, dict) or set(summary_runs) != set(runs):
        problems.append("request summary run set differs from request trace")
        return
    venues_expected = {report["primary_exchange"], report["validation_exchange"]}
    for run_name, run in runs.items():
        venues = run.get("venues") if isinstance(run, dict) else None
        if not isinstance(venues, dict) or set(venues) != venues_expected:
            problems.append(f"{run_name} venue set is invalid")
            continue
        summary_run = summary_runs[run_name]
        if not isinstance(summary_run, dict):
            problems.append(f"{run_name} request summary is invalid")
            continue
        summary_venues = summary_run.get("venues", {})
        if not isinstance(summary_venues, dict) or set(summary_venues) != venues_expected:
            problems.append(f"{run_name} summary venue set is invalid")
            continue
        for exchange, venue in venues.items():
            if not isinstance(venue, dict):
                problems.append(f"{run_name}.{exchange} trace is invalid")
                continue
            for count in [
                "http_request_count",
                "fetch_ohlcv_call_count",
                "page_count",
                "raw_rows_received",
                "normalized_rows",
                "duplicate_rows_removed",
                "overlapping_rows_removed",
                "out_of_range_rows_removed",
                "empty_pages",
                "short_pages",
                "retry_count",
                "rate_limit_responses",
            ]:
                value = venue.get(count)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    problems.append(f"{run_name}.{exchange}.{count} is invalid")
            requests = venue.get("requests")
            if not isinstance(requests, list) or len(requests) != venue.get("http_request_count"):
                problems.append(f"{run_name}.{exchange} request log count mismatch")
            summary_venue = summary_venues.get(exchange, {})
            if not isinstance(summary_venue, dict) or set(summary_venue) != TRACE_SUMMARY_FIELDS:
                problems.append(f"{run_name}.{exchange} request summary fields are not exact")
                summary_venue = {}
            for key, value in summary_venue.items():
                if venue.get(key) != value:
                    problems.append(f"{run_name}.{exchange} summary mismatch for {key}")

            cursors = venue.get("cursor_progression_ms")
            fetch_calls = venue.get("fetch_ohlcv_call_count")
            page_count = venue.get("page_count")
            retry_count = venue.get("retry_count")
            if not isinstance(cursors, list) or any(
                isinstance(cursor, bool) or not isinstance(cursor, int) for cursor in cursors
            ):
                problems.append(f"{run_name}.{exchange} cursor progression is invalid")
            else:
                if len(cursors) != fetch_calls:
                    problems.append(f"{run_name}.{exchange} cursor call count mismatch")
                if cursors and (
                    venue.get("first_request_cursor_ms") != cursors[0]
                    or venue.get("last_request_cursor_ms") != cursors[-1]
                ):
                    problems.append(f"{run_name}.{exchange} cursor endpoints mismatch")
                if any(right < left for left, right in zip(cursors, cursors[1:], strict=False)):
                    problems.append(f"{run_name}.{exchange} cursor progression goes backwards")
            if (
                isinstance(fetch_calls, int)
                and isinstance(page_count, int)
                and isinstance(retry_count, int)
                and fetch_calls != page_count + retry_count
            ):
                problems.append(f"{run_name}.{exchange} page/retry accounting is inconsistent")
            raw = venue.get("raw_rows_received")
            normalized = venue.get("normalized_rows")
            removed = sum(
                venue.get(key, 0)
                for key in [
                    "duplicate_rows_removed",
                    "overlapping_rows_removed",
                    "out_of_range_rows_removed",
                ]
                if isinstance(venue.get(key), int)
            )
            if isinstance(raw, int) and isinstance(normalized, int) and normalized != raw - removed:
                problems.append(
                    f"{run_name}.{exchange} row normalization accounting is inconsistent"
                )
            statuses = venue.get("response_statuses")
            if not isinstance(statuses, list) or len(statuses) != venue.get("http_request_count"):
                problems.append(f"{run_name}.{exchange} response status count mismatch")
            failures = venue.get("failures")
            if not isinstance(failures, list) or any(
                not isinstance(item, str) for item in failures
            ):
                problems.append(f"{run_name}.{exchange} failure list is invalid")
            if report["overall_passed"] and failures:
                problems.append(f"{run_name}.{exchange} successful evidence records failures")

            if isinstance(requests, list):
                for sequence, request in enumerate(requests, start=1):
                    if not isinstance(request, dict):
                        problems.append(f"{run_name}.{exchange} request entry is invalid")
                        continue
                    if request.get("sequence") != sequence or request.get("method") != "GET":
                        problems.append(f"{run_name}.{exchange} request sequence/method is invalid")
            if report["overall_passed"] and not report["fixture_mode"]:
                if (
                    venue.get("http_request_count", 0) <= 0
                    or venue.get("fetch_ohlcv_call_count", 0) <= 0
                ):
                    problems.append(f"{run_name}.{exchange} has zero actual venue requests")
                if venue.get("page_count", 0) < 4:
                    problems.append(f"{run_name}.{exchange} has fewer than four measured pages")
                if venue.get("raw_rows_received", 0) <= 0:
                    problems.append(f"{run_name}.{exchange} has zero raw rows")
                for request in requests if isinstance(requests, list) else []:
                    url = str(request.get("url", ""))
                    parsed = urlsplit(url)
                    expected_suffix = "coinbase.com" if exchange == "coinbase" else "kraken.com"
                    hostname = parsed.hostname or ""
                    approved_host = hostname == expected_suffix or hostname.endswith(
                        "." + expected_suffix
                    )
                    if parsed.scheme != "https" or not approved_host:
                        problems.append(
                            f"{run_name}.{exchange} request endpoint is not approved: {url}"
                        )

    run_one_payload = runs.get("run_1", {})
    run_one = run_one_payload.get("venues", {}) if isinstance(run_one_payload, dict) else {}
    if not isinstance(run_one, dict):
        run_one = {}
    mappings = {
        "primary_http_requests": (report["primary_exchange"], "http_request_count"),
        "validation_http_requests": (report["validation_exchange"], "http_request_count"),
        "primary_fetch_ohlcv_calls": (report["primary_exchange"], "fetch_ohlcv_call_count"),
        "validation_fetch_ohlcv_calls": (report["validation_exchange"], "fetch_ohlcv_call_count"),
        "primary_pages": (report["primary_exchange"], "page_count"),
        "validation_pages": (report["validation_exchange"], "page_count"),
        "primary_raw_rows": (report["primary_exchange"], "raw_rows_received"),
        "validation_raw_rows": (report["validation_exchange"], "raw_rows_received"),
        "primary_duplicate_rows": (report["primary_exchange"], "duplicate_rows_removed"),
        "validation_duplicate_rows": (report["validation_exchange"], "duplicate_rows_removed"),
    }
    for report_key, (exchange, trace_key) in mappings.items():
        measured = run_one.get(exchange, {})
        measured_value = measured.get(trace_key) if isinstance(measured, dict) else None
        if measured_value != report[report_key]:
            problems.append(f"report {report_key} differs from measured request trace")
    row_mappings = {
        report["primary_exchange"]: report["primary_rows"],
        report["validation_exchange"]: report["validation_rows"],
    }
    if report["overall_passed"]:
        for exchange, rows in row_mappings.items():
            measured = run_one.get(exchange, {})
            normalized_rows = (
                measured.get("normalized_rows") if isinstance(measured, dict) else None
            )
            if normalized_rows != rows:
                problems.append(f"report {exchange} rows differ from normalized request trace")


def _validate_parquet(
    path: Path,
    *,
    exchange: str,
    expected_rows: int,
    report: dict[str, Any],
    first_field: str,
    last_field: str,
    problems: list[str],
) -> pd.DataFrame | None:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        problems.append(f"invalid parquet {path.name}: {type(exc).__name__}")
        return None
    if list(frame.columns) != PARQUET_COLUMNS:
        problems.append(f"{path.name} schema columns are invalid")
        return None
    if len(frame) != expected_rows:
        problems.append(f"{path.name} row count differs from report")
    report_quality = validate_ohlcv(
        frame,
        report["timeframe"],
        source=exchange,
        max_gap_count=0,
        expected_exchange=exchange,
        expected_symbol=report["symbol"],
        expected_timeframe=report["timeframe"],
        expected_start=datetime.fromisoformat(report["requested_start"]),
        expected_end=datetime.fromisoformat(report["requested_end"]),
    )
    for issue in report_quality.issues:
        problems.append(f"{path.name} quality failure: {issue.code}")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    if str(timestamps.min()) != report.get(first_field):
        problems.append(f"{path.name} first timestamp differs from report")
    if str(timestamps.max()) != report.get(last_field):
        problems.append(f"{path.name} last timestamp differs from report")
    logical_hash = _logical_frame_hash(frame)
    if logical_hash != report.get("canonical_hashes_run1", {}).get(exchange):
        problems.append(f"{path.name} logical hash differs from run-one report")
    expected_file_hash = report.get("parquet_file_hashes_run1", {}).get(exchange)
    if sha256_file(path) != expected_file_hash:
        problems.append(f"{path.name} physical hash differs from run-one report")
    return frame


def validate_public_ingestion_report(output_dir: Path) -> list[str]:
    """Recompute every locally provable gate; never trust the input verdict."""
    output_dir = Path(output_dir)
    problems: list[str] = []
    report_path = output_dir / REPORT_NAME
    if not report_path.is_file():
        return [f"missing {REPORT_NAME}"]
    report = _parse_json(report_path, REPORT_NAME, problems)
    if not isinstance(report, dict):
        return problems or ["report root must be an object"]
    _append_nonfinite_problems(report, "report", problems)
    for field_name, expected_type in REQUIRED_REPORT_FIELDS.items():
        if field_name not in report:
            problems.append(f"missing field: {field_name}")
            continue
        value = report[field_name]
        if expected_type is int and isinstance(value, bool):
            problems.append(f"field {field_name} must be an integer, got boolean")
        elif not isinstance(value, expected_type):
            problems.append(f"field {field_name} has wrong type {type(value).__name__}")
    if (
        report.get("fixture_mode") is False
        and report.get("overall_passed") is True
        and report.get("live_public_endpoints_reached") is not True
    ):
        problems.append("non-fixture success requires live_public_endpoints_reached exactly true")
    if problems:
        for name in PARQUET_ARTIFACTS:
            path = output_dir / name
            if path.is_file():
                try:
                    pd.read_parquet(path)
                except Exception as exc:
                    problems.append(f"invalid parquet {name}: {type(exc).__name__}")
        _validate_file_integrity(output_dir, problems)
        return problems

    if report.get("schema") != EVIDENCE_SCHEMA:
        problems.append("report schema is invalid")
    if report["credentials_used"] is not False:
        problems.append("credentials_used must be false")
    if report["orders_possible"] is not False:
        problems.append("orders_possible must be false")
    for field_name in [
        name
        for name in REQUIRED_REPORT_FIELDS
        if name.endswith(("requests", "calls", "pages", "rows"))
    ]:
        if report[field_name] < 0:
            problems.append(f"{field_name} is negative")

    range_valid = True
    try:
        requested_start = datetime.fromisoformat(report["requested_start"])
        requested_end = datetime.fromisoformat(report["requested_end"])
    except ValueError:
        range_valid = False
        problems.append("requested date range is not valid ISO-8601")
        requested_start = requested_end = datetime.now(tz=UTC)
    if requested_start.tzinfo is None or requested_end.tzinfo is None:
        problems.append("requested date range must be timezone-aware")
        requested_start = requested_start.replace(tzinfo=UTC)
        requested_end = requested_end.replace(tzinfo=UTC)
    timeframe_valid = True
    try:
        step_seconds = timeframe_to_seconds(report["timeframe"])
    except ValueError:
        timeframe_valid = False
        problems.append("report timeframe is invalid")
        step_seconds = 3600
    expected_rows = int((requested_end - requested_start).total_seconds() // step_seconds)
    if report["expected_rows"] != expected_rows or expected_rows <= 0:
        problems.append("expected_rows differs from the requested date range")
    if not report["fixture_mode"]:
        if (
            report["primary_exchange"] != "coinbase"
            or report["validation_exchange"] != "kraken"
            or report["symbol"] != "ETH/USD"
            or report["timeframe"] != "1h"
        ):
            problems.append("live evidence has the wrong venue, symbol, or timeframe")
        if requested_end - requested_start < timedelta(days=14):
            problems.append("live evidence covers fewer than 14 days")
        if requested_end > datetime.now(tz=UTC) - timedelta(hours=48):
            problems.append("live evidence ends less than 48 hours before current UTC")

    trace = _parse_json(output_dir / REQUEST_TRACE_NAME, REQUEST_TRACE_NAME, problems)
    summary = _parse_json(output_dir / REQUEST_SUMMARY_NAME, REQUEST_SUMMARY_NAME, problems)
    quality = _parse_json(output_dir / QUALITY_NAME, QUALITY_NAME, problems)
    _append_nonfinite_problems(trace, "trace", problems)
    _append_nonfinite_problems(summary, "summary", problems)
    _append_nonfinite_problems(quality, "quality", problems)
    _validate_trace(trace, summary, report, problems)
    if isinstance(quality, dict):
        if quality.get("schema") != EVIDENCE_SCHEMA:
            problems.append("quality evidence schema is invalid")
        if quality.get("fixture_mode") is not report["fixture_mode"]:
            problems.append("quality evidence fixture mode differs from report")
        quality_result = quality.get("quality")
        if report["overall_passed"] and (
            not isinstance(quality_result, dict)
            or quality_result.get("passed") is not True
            or quality_result.get("committed") is not True
        ):
            problems.append("quality evidence does not show a passed committed ingestion")

    hashes_one = report.get("canonical_hashes_run1")
    hashes_two = report.get("canonical_hashes_run2")
    computed_idempotency = bool(
        isinstance(hashes_one, dict)
        and isinstance(hashes_two, dict)
        and hashes_one == hashes_two
        and set(hashes_one) == {report["primary_exchange"], report["validation_exchange"]}
        and all(value != "absent" for value in hashes_one.values())
    )
    for label, hashes in [("run one", hashes_one), ("run two", hashes_two)]:
        if isinstance(hashes, dict) and any(
            value != "absent"
            and (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            )
            for value in hashes.values()
        ):
            problems.append(f"canonical {label} hashes are malformed")
    if report["idempotency_passed"] is not computed_idempotency:
        problems.append("idempotency_passed differs from recomputed run hashes")

    if report["fixture_mode"] and report["live_public_endpoints_reached"]:
        problems.append("fixture_mode runs cannot claim live_public_endpoints_reached")
    if (
        not report["fixture_mode"]
        and report["overall_passed"]
        and not report["live_public_endpoints_reached"]
    ):
        problems.append("non-fixture success requires live_public_endpoints_reached exactly true")
    if report["failure"] is not None and report["overall_passed"]:
        problems.append("overall_passed with a recorded failure")

    primary_frame: pd.DataFrame | None = None
    validation_frame: pd.DataFrame | None = None
    parquet_missing = PARQUET_ARTIFACTS.difference(
        path.name for path in output_dir.iterdir() if path.is_file()
    )
    if report["overall_passed"]:
        for name in sorted(parquet_missing):
            problems.append(f"missing artifact: {name}")
        if not parquet_missing and range_valid and timeframe_valid:
            primary_frame = _validate_parquet(
                output_dir / "coinbase_normalized.parquet",
                exchange=report["primary_exchange"],
                expected_rows=report["primary_rows"],
                report=report,
                first_field="primary_first_timestamp",
                last_field="primary_last_timestamp",
                problems=problems,
            )
            validation_frame = _validate_parquet(
                output_dir / "kraken_normalized.parquet",
                exchange=report["validation_exchange"],
                expected_rows=report["validation_rows"],
                report=report,
                first_field="validation_first_timestamp",
                last_field="validation_last_timestamp",
                problems=problems,
            )

    recomputed_coverage = False
    recomputed_quality = False
    recomputed_alignment = False
    recomputed_divergence = False
    if primary_frame is not None and validation_frame is not None:
        venue_reports = [
            validate_ohlcv(
                frame,
                report["timeframe"],
                source=exchange,
                max_gap_count=0,
                expected_exchange=exchange,
                expected_symbol=report["symbol"],
                expected_timeframe=report["timeframe"],
                expected_start=requested_start,
                expected_end=requested_end,
            )
            for exchange, frame in [
                (report["primary_exchange"], primary_frame),
                (report["validation_exchange"], validation_frame),
            ]
        ]
        cross = validate_cross_venue(
            primary_frame,
            validation_frame,
            primary_name=report["primary_exchange"],
            secondary_name=report["validation_exchange"],
            max_p95_bps=75.0,
            min_overlap=100,
            max_latest_bps=150.0,
        )
        cross_codes = {issue.code for issue in cross.issues}
        recomputed_coverage = all(item.passed for item in venue_reports) and all(
            len(frame) == report["expected_rows"] for frame in [primary_frame, validation_frame]
        )
        recomputed_alignment = not bool(
            {"latest_timestamp_misalignment", "insufficient_overlap"} & cross_codes
        )
        recomputed_divergence = not bool(
            {"cross_venue_divergence", "latest_cross_venue_divergence"} & cross_codes
        )
        recomputed_quality = all(item.passed for item in venue_reports) and cross.passed
    if report["overall_passed"]:
        for gate_field, recomputed in [
            ("coverage_passed", recomputed_coverage),
            ("quality_passed", recomputed_quality),
            ("alignment_passed", recomputed_alignment),
            ("divergence_passed", recomputed_divergence),
        ]:
            if report[gate_field] is not recomputed:
                problems.append(f"{gate_field} differs from independently recomputed artifacts")

    effective_coverage = (
        recomputed_coverage if report["overall_passed"] else report["coverage_passed"]
    )
    effective_quality = recomputed_quality if report["overall_passed"] else report["quality_passed"]
    effective_alignment = (
        recomputed_alignment if report["overall_passed"] else report["alignment_passed"]
    )
    effective_divergence = (
        recomputed_divergence if report["overall_passed"] else report["divergence_passed"]
    )

    gates = [
        effective_coverage,
        effective_quality,
        effective_alignment,
        effective_divergence,
        computed_idempotency,
        report["primary_rows"] > 0,
        report["validation_rows"] > 0,
        report["failure"] is None,
        report["fixture_mode"] or report["live_public_endpoints_reached"],
    ]
    recomputed_overall = all(gates)
    if report["overall_passed"] is not recomputed_overall:
        problems.append("overall_passed differs from independently recomputed gates")

    for artifact in [REPORT_MD_NAME, REQUEST_TRACE_NAME, REQUEST_SUMMARY_NAME, QUALITY_NAME]:
        path = output_dir / artifact
        if not path.is_file():
            problems.append(f"missing artifact: {artifact}")
        elif (
            report["fixture_mode"]
            and "fixture" not in path.read_text(encoding="utf-8", errors="replace").lower()
        ):
            problems.append(f"fixture run not labeled inside {artifact}")

    _validate_file_integrity(output_dir, problems)
    return problems
