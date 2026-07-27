"""Portable, credential-free public-ingestion audit harness.

Runs the real production ingestion path (`ingest_market_data` with real
`CCXTOHLCVProvider` instances) against public Coinbase/Kraken endpoints, twice,
inside an isolated output directory, and emits machine- and human-readable
evidence artifacts. It never uses credentials and has no order capability.

Fixture/simulation runs (injected providers) are labeled as such in every
artifact and can never claim that live public endpoints were reached.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .config import AresConfig
from .data.ingest import IngestionResult, ingest_market_data
from .data.providers import CCXTOHLCVProvider, OHLCVProvider
from .data.storage import market_path, read_market
from .exceptions import AresError, DataQualityError
from .utils import atomic_write_json, sha256_file, timeframe_to_seconds

REPORT_NAME = "public_ingestion_report.json"
REPORT_MD_NAME = "public_ingestion_report.md"
REQUEST_SUMMARY_NAME = "request_summary.json"
QUALITY_NAME = "quality_report.json"
CHECKSUMS_NAME = "SHA256SUMS.txt"

REQUIRED_REPORT_FIELDS: dict[str, type | tuple[type, ...]] = {
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
}


@dataclass
class InstrumentedProvider:
    """Counting wrapper that also enforces the no-credential guarantee."""

    inner: OHLCVProvider
    exchange_id: str = ""
    pages: int = 0
    raw_rows: int = 0
    duplicate_rows: int = 0
    failures: list[str] = field(default_factory=list)

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
        inner_exchange_factory = getattr(self.inner, "_exchange", None)
        if inner_exchange_factory is not None and isinstance(self.inner, CCXTOHLCVProvider):
            exchange = inner_exchange_factory()
            api_key = getattr(exchange, "apiKey", None)
            secret = getattr(exchange, "secret", None)
            if api_key or secret:
                raise AresError(
                    "Public-ingestion audit refuses to run with exchange credentials configured"
                )
        frame = self.inner.fetch_range(
            symbol, timeframe, since, until, limit=limit, max_pages=max_pages, retries=retries
        )
        self.raw_rows += int(len(frame))
        if "timestamp" in getattr(frame, "columns", []):
            self.duplicate_rows += int(frame["timestamp"].duplicated().sum())
        step_ms = timeframe_to_seconds(timeframe) * 1000
        window_ms = (
            int((until - since).total_seconds() * 1000) if until is not None else limit * step_ms
        )
        estimated_pages = max(1, -(-window_ms // (limit * step_ms)))
        self.pages += int(estimated_pages)
        return frame


def _audit_config(
    *,
    primary: str,
    validation: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    output_dir: Path,
    page_limit: int,
    retries: int,
) -> AresConfig:
    raw = {
        "project": {"name": "ARES public-ingestion audit", "seed": 1},
        "storage": {
            "root": str(output_dir / "data"),
            "artifacts": str(output_dir / "artifacts"),
            "duckdb_path": str(output_dir / "data" / "audit.duckdb"),
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
            "resume": True,
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
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "403" in lowered or "forbidden" in lowered or "blocked" in lowered or "tunnel" in lowered:
        return "environment_or_endpoint_block"
    if "timeout" in lowered or "timed out" in lowered:
        return "network_timeout"
    if "dns" in lowered or "name resolution" in lowered or "getaddrinfo" in lowered:
        return "dns_failure"
    return "ingestion_error"


def _hash_directory_parquet(config: AresConfig, exchanges: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for exchange in exchanges:
        path = market_path(config.storage.root, exchange, config.data.symbol, config.data.timeframe)
        hashes[exchange] = sha256_file(path) if path.exists() else "absent"
    return hashes


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
    """Execute the bounded two-run public-ingestion audit and write all artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_mode = providers is not None
    config = _audit_config(
        primary=primary,
        validation=validation,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        output_dir=output_dir,
        page_limit=page_limit,
        retries=retries,
    )
    step_seconds = timeframe_to_seconds(timeframe)
    expected_rows = int((end - start).total_seconds() // step_seconds)

    instrumented: dict[str, InstrumentedProvider] = {}
    for exchange in [primary, validation]:
        inner = (providers or {}).get(exchange) or CCXTOHLCVProvider(exchange)
        instrumented[exchange] = InstrumentedProvider(inner=inner)

    failure: str | None = None
    failure_classification: str | None = None
    first_result: IngestionResult | None = None
    second_result: IngestionResult | None = None
    try:
        provider_map: dict[str, OHLCVProvider] = dict(instrumented)
        first_result = ingest_market_data(config, providers=provider_map)
        first_hashes = _hash_directory_parquet(config, [primary, validation])
        second_result = ingest_market_data(config, providers=provider_map)
        second_hashes = _hash_directory_parquet(config, [primary, validation])
    except (AresError, DataQualityError, OSError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        failure_classification = _classify_failure(exc)
        first_hashes = _hash_directory_parquet(config, [primary, validation])
        second_hashes = first_hashes

    def _venue_report(result: IngestionResult | None, exchange: str) -> dict[str, Any]:
        if result is None:
            return {}
        for item in result.exchanges:
            if item.exchange == exchange:
                return item.report.to_dict()
        return {}

    primary_frame = read_market(market_path(config.storage.root, primary, symbol, timeframe))
    validation_frame = read_market(market_path(config.storage.root, validation, symbol, timeframe))

    cross_stats: dict[str, Any] = {}
    cross_passed = False
    alignment_passed = False
    if first_result is not None and first_result.cross_venue_reports:
        cross = first_result.cross_venue_reports[0]
        cross_stats = {
            key: (str(value) if isinstance(value, pd.Timestamp) else value)
            for key, value in cross.stats.items()
        }
        codes = {issue.code for issue in cross.issues}
        cross_passed = not ({"cross_venue_divergence", "latest_cross_venue_divergence"} & codes)
        alignment_passed = "latest_timestamp_misalignment" not in codes and bool(
            cross.stats.get("overlap_rows", 0)
        )

    primary_report = _venue_report(first_result, primary)
    validation_report = _venue_report(first_result, validation)
    primary_codes = {issue["code"] for issue in primary_report.get("issues", [])}
    validation_codes = {issue["code"] for issue in validation_report.get("issues", [])}
    coverage_passed = (
        bool(primary_report)
        and not (
            {"incomplete_start_coverage", "incomplete_end_coverage", "empty_dataset"}
            & primary_codes
        )
        and not ({"incomplete_end_coverage", "empty_dataset"} & validation_codes)
    )
    quality_passed = bool(first_result.passed) if first_result is not None else False
    idempotency_passed = (
        second_result is not None
        and second_result.committed
        and first_hashes == second_hashes
        and all(value != "absent" for value in first_hashes.values())
    )
    rows_present = len(primary_frame) > 0 and len(validation_frame) > 0
    overall_passed = bool(
        failure is None
        and quality_passed
        and coverage_passed
        and alignment_passed
        and cross_passed
        and idempotency_passed
        and rows_present
    )
    live_reached = bool(
        not fixture_mode
        and failure_classification not in {"environment_or_endpoint_block", "dns_failure"}
        and (instrumented[primary].raw_rows > 0 or instrumented[validation].raw_rows > 0)
    )

    generated_at = datetime.now(tz=UTC).isoformat()
    mode_label = (
        "fixture-simulation (NOT live endpoints)" if fixture_mode else "live-public-endpoints"
    )
    report: dict[str, Any] = {
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
        "primary_pages": instrumented[primary].pages,
        "validation_pages": instrumented[validation].pages,
        "primary_raw_rows": instrumented[primary].raw_rows,
        "validation_raw_rows": instrumented[validation].raw_rows,
        "primary_rows": int(len(primary_frame)),
        "validation_rows": int(len(validation_frame)),
        "primary_duplicate_rows": instrumented[primary].duplicate_rows,
        "validation_duplicate_rows": instrumented[validation].duplicate_rows,
        "primary_missing_candles": primary_report.get("stats", {}).get("missing_candle_slots"),
        "validation_missing_candles": validation_report.get("stats", {}).get(
            "missing_candle_slots"
        ),
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
        "divergence_passed": cross_passed,
        "idempotency_passed": bool(idempotency_passed),
        "canonical_hashes_run1": first_hashes,
        "canonical_hashes_run2": second_hashes,
        "failure": failure,
        "failure_classification": failure_classification,
        "overall_passed": overall_passed,
    }

    atomic_write_json(output_dir / REPORT_NAME, report)
    atomic_write_json(
        output_dir / REQUEST_SUMMARY_NAME,
        {
            "mode": mode_label,
            "fixture_mode": fixture_mode,
            "generated_at": generated_at,
            "venues": {
                exchange: {
                    "pages": item.pages,
                    "raw_rows": item.raw_rows,
                    "duplicate_rows": item.duplicate_rows,
                }
                for exchange, item in instrumented.items()
            },
        },
    )
    quality_source = config.storage.root / "quality" / "latest.json"
    quality_payload: dict[str, Any] = {"mode": mode_label, "fixture_mode": fixture_mode}
    if quality_source.exists():
        quality_payload["quality"] = json.loads(quality_source.read_text(encoding="utf-8"))
    else:
        quality_payload["missing"] = True
    atomic_write_json(output_dir / QUALITY_NAME, quality_payload)
    for exchange, name in [(primary, "coinbase"), (validation, "kraken")]:
        source = market_path(config.storage.root, exchange, symbol, timeframe)
        target = output_dir / f"{name}_normalized.parquet"
        if source.exists():
            target.write_bytes(source.read_bytes())

    lines = [
        f"# ARES public-ingestion audit ({mode_label})",
        "",
        f"- Generated: {generated_at}",
        f"- ARES version: {__version__}",
        f"- Mode: **{mode_label}**",
        f"- Live public endpoints reached: **{live_reached}**",
        "- Credentials used: **False**  |  Orders possible: **False**",
        f"- Venues: {primary} (primary) vs {validation} (validation)",
        f"- Market: {symbol} @ {timeframe}",
        f"- Window: {start.isoformat()} -> {end.isoformat()} ({expected_rows} expected candles)",
        f"- Pages: primary={instrumented[primary].pages}, validation={instrumented[validation].pages}",
        f"- Rows: primary raw={instrumented[primary].raw_rows} canonical={len(primary_frame)}; "
        f"validation raw={instrumented[validation].raw_rows} canonical={len(validation_frame)}",
        f"- Gates: coverage={coverage_passed} quality={quality_passed} alignment={alignment_passed} "
        f"divergence={cross_passed} idempotency={idempotency_passed}",
        f"- Failure: {failure or 'none'} ({failure_classification or 'n/a'})",
        f"- **OVERALL: {'PASSED' if overall_passed else 'FAILED'}**",
    ]
    if fixture_mode:
        lines.insert(
            1,
            "\n> SIMULATION NOTICE: this run used injected fixture providers. It proves "
            "pipeline behavior, not live endpoint compatibility.\n",
        )
    (output_dir / REPORT_MD_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    checksum_targets = sorted(
        item for item in output_dir.iterdir() if item.is_file() and item.name != CHECKSUMS_NAME
    )
    checksum_lines = [f"{sha256_file(item)}  {item.name}" for item in checksum_targets]
    (output_dir / CHECKSUMS_NAME).write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return report


def validate_public_ingestion_report(output_dir: Path) -> list[str]:
    """Validate report artifacts; return a list of problems (empty when valid)."""
    output_dir = Path(output_dir)
    problems: list[str] = []
    report_path = output_dir / REPORT_NAME
    if not report_path.exists():
        return [f"missing {REPORT_NAME}"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"report is not valid JSON: {exc}"]

    for fieldname, expected_type in REQUIRED_REPORT_FIELDS.items():
        if fieldname not in report:
            problems.append(f"missing field: {fieldname}")
            continue
        value = report[fieldname]
        if expected_type is int and isinstance(value, bool):
            problems.append(f"field {fieldname} must be an integer, got boolean")
        elif not isinstance(value, expected_type):
            problems.append(f"field {fieldname} has wrong type {type(value).__name__}")
    if problems:
        return problems

    if report["credentials_used"] is not False:
        problems.append("credentials_used must be false")
    if report["orders_possible"] is not False:
        problems.append("orders_possible must be false")
    if report["fixture_mode"] and report["live_public_endpoints_reached"]:
        problems.append("fixture_mode runs cannot claim live_public_endpoints_reached")
    if "fixture" in str(report.get("mode", "")).lower() and report["live_public_endpoints_reached"]:
        problems.append("mode labeled fixture but live_public_endpoints_reached is true")
    for count_field in [
        "primary_pages",
        "validation_pages",
        "primary_rows",
        "validation_rows",
        "primary_raw_rows",
        "validation_raw_rows",
    ]:
        if report[count_field] < 0:
            problems.append(f"{count_field} is negative")
    if report["overall_passed"]:
        for gate in [
            "coverage_passed",
            "quality_passed",
            "alignment_passed",
            "divergence_passed",
            "idempotency_passed",
        ]:
            if report[gate] is not True:
                problems.append(f"overall_passed requires {gate}")
        if report["primary_rows"] <= 0 or report["validation_rows"] <= 0:
            problems.append("overall_passed requires nonzero canonical rows")
        if report.get("failure"):
            problems.append("overall_passed with a recorded failure")
    if report["live_public_endpoints_reached"] and (
        report["primary_raw_rows"] <= 0 and report["validation_raw_rows"] <= 0
    ):
        problems.append("live_public_endpoints_reached without any fetched rows")

    checksums_path = output_dir / CHECKSUMS_NAME
    if not checksums_path.exists():
        problems.append(f"missing {CHECKSUMS_NAME}")
    else:
        for line in checksums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition("  ")
            target = output_dir / name
            if not target.exists():
                problems.append(f"checksummed file missing: {name}")
            elif sha256_file(target) != digest:
                problems.append(f"checksum mismatch: {name}")
    for artifact in [REPORT_MD_NAME, REQUEST_SUMMARY_NAME, QUALITY_NAME]:
        artifact_path = output_dir / artifact
        if not artifact_path.exists():
            problems.append(f"missing artifact: {artifact}")
        elif (
            report["fixture_mode"]
            and "fixture" not in artifact_path.read_text(encoding="utf-8", errors="replace").lower()
        ):
            problems.append(f"fixture run not labeled inside {artifact}")
    if report["overall_passed"] and not report["fixture_mode"]:
        for name in ["coinbase_normalized.parquet", "kraken_normalized.parquet"]:
            if not (output_dir / name).exists():
                problems.append(f"missing artifact: {name}")
    return problems
