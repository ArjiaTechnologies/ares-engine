"""Regression tests for the portable public-ingestion audit harness."""

import json
import shutil
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from venue_wire_sim import BASE_TS, HOUR, WireSimulator, make_real_ccxt_exchange

from ares_engine.data.providers import CCXTOHLCVProvider
from ares_engine.public_audit import (
    CHECKSUMS_NAME,
    EVIDENCE_SCHEMA,
    MANIFEST_NAME,
    PARQUET_ARTIFACTS,
    QUALITY_NAME,
    REPORT_NAME,
    REQUEST_SUMMARY_NAME,
    REQUEST_TRACE_NAME,
    _classify_failure,
    _validate_file_integrity,
    _validate_parquet,
    _validate_trace,
    run_public_ingestion_audit,
    validate_public_ingestion_report,
)
from ares_engine.utils import sha256_file

START = datetime.fromtimestamp(BASE_TS, tz=UTC)


def _end(hours: int) -> datetime:
    return datetime.fromtimestamp(BASE_TS + hours * HOUR, tz=UTC)


@pytest.fixture()
def sim(monkeypatch):
    simulator = WireSimulator().start()
    import ares_engine.data.providers as providers_module

    def patched_exchange(self):
        return make_real_ccxt_exchange(self.exchange_id, simulator.base_url, 5_000)

    monkeypatch.setattr(CCXTOHLCVProvider, "_exchange", patched_exchange)
    monkeypatch.setattr(providers_module.time, "sleep", lambda seconds: None)
    yield simulator
    simulator.stop()


def _fixture_providers():
    return {
        "coinbase": CCXTOHLCVProvider("coinbase"),
        "kraken": CCXTOHLCVProvider("kraken"),
    }


def _run(sim, tmp_path, hours=120, **kwargs):
    for venue in ("coinbase", "kraken"):
        sim.scenario(venue).history_end = BASE_TS + hours * HOUR
    return run_public_ingestion_audit(
        start=START,
        end=_end(hours),
        output_dir=tmp_path / "audit",
        page_limit=30,
        providers=_fixture_providers(),
        fixture_note="wire-simulator regression run",
        **kwargs,
    )


def _reseal(audit) -> None:
    files = sorted(
        path
        for path in audit.iterdir()
        if path.is_file() and path.name not in {MANIFEST_NAME, CHECKSUMS_NAME}
    )
    (audit / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema": EVIDENCE_SCHEMA,
                "files": {
                    path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                    for path in files
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checksum_files = sorted(
        path for path in audit.iterdir() if path.is_file() and path.name != CHECKSUMS_NAME
    )
    (audit / CHECKSUMS_NAME).write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_files),
        encoding="utf-8",
    )


def test_fixture_run_passes_gates_but_cannot_claim_live(sim, tmp_path) -> None:
    report = _run(sim, tmp_path)
    assert report["overall_passed"] is True
    assert report["fixture_mode"] is True
    assert report["live_public_endpoints_reached"] is False
    assert report["credentials_used"] is False and report["orders_possible"] is False
    assert report["primary_rows"] == 120 and report["validation_rows"] == 120
    assert report["idempotency_passed"] is True
    assert validate_public_ingestion_report(tmp_path / "audit") == []


def test_fixture_label_is_present_in_every_artifact(sim, tmp_path) -> None:
    _run(sim, tmp_path)
    audit = tmp_path / "audit"
    for name in [
        "public_ingestion_report.json",
        "public_ingestion_report.md",
        "request_summary.json",
        "quality_report.json",
    ]:
        content = (audit / name).read_text(encoding="utf-8", errors="replace").lower()
        assert "fixture" in content, f"{name} must carry the simulation label"


def test_audit_mode_cannot_touch_normal_canonical_storage(sim, tmp_path, monkeypatch) -> None:
    workdir = tmp_path / "cwd"
    sentinel_dir = workdir / "data" / "raw" / "coinbase" / "eth-usd"
    sentinel_dir.mkdir(parents=True)
    sentinel = sentinel_dir / "1h.parquet"
    sentinel.write_bytes(b"CANONICAL-SENTINEL")
    monkeypatch.chdir(workdir)
    _run(sim, tmp_path)
    assert sentinel.read_bytes() == b"CANONICAL-SENTINEL"
    assert sorted(p.name for p in (workdir / "data").rglob("*")) == [
        "1h.parquet",
        "coinbase",
        "eth-usd",
        "raw",
    ]


def test_no_order_method_is_ever_invoked(sim, tmp_path, monkeypatch) -> None:
    import ccxt

    forbidden = [
        "create_order",
        "createOrder",
        "create_market_buy_order",
        "create_market_sell_order",
        "create_limit_buy_order",
        "create_limit_sell_order",
        "cancel_order",
        "withdraw",
        "transfer",
        "fetch_balance",
    ]

    def explode(name):
        def _explode(*args, **kwargs):
            raise AssertionError(f"forbidden exchange method invoked: {name}")

        return _explode

    for cls in (ccxt.coinbase, ccxt.kraken):
        for name in forbidden:
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, explode(name), raising=False)
    report = _run(sim, tmp_path)
    assert report["overall_passed"] is True


def test_short_pages_do_not_stop_the_audit_early(sim, tmp_path) -> None:
    for venue in ("coinbase", "kraken"):
        sim.scenario(venue).short_page_size = 7
    report = _run(sim, tmp_path, hours=120)
    assert report["overall_passed"] is True
    assert report["primary_rows"] == 120 and report["validation_rows"] == 120


def test_repeated_cursor_fails_safely(sim, tmp_path) -> None:
    """Through the real CCXT client a repeating page is since-filtered into an
    empty batch: pagination stops early and the coverage gate blocks the run.
    Either failure mode (explicit non-advancing-cursor error, or truncation
    caught by coverage) must yield overall_passed=False with no commit."""
    sim.scenario("coinbase").duplicate_page = True
    report = _run(sim, tmp_path, hours=120)
    assert report["overall_passed"] is False
    assert report["coverage_passed"] is False or report["failure"] is not None
    assert report["canonical_hashes_run1"]["coinbase"] == "absent"
    assert validate_public_ingestion_report(tmp_path / "audit") == []


def test_partial_venue_success_cannot_yield_overall_pass(sim, tmp_path) -> None:
    sim.scenario("kraken").truncate_end = BASE_TS + 110 * HOUR
    report = _run(sim, tmp_path, hours=120)
    assert report["overall_passed"] is False
    assert validate_public_ingestion_report(tmp_path / "audit") == []


def test_blocked_endpoint_is_classified_as_environment_failure(tmp_path, monkeypatch) -> None:
    class BlockedProvider:
        def __init__(self, exchange_id: str) -> None:
            self.exchange_id = exchange_id

        def fetch_range(self, *args, **kwargs):
            raise OSError("Tunnel connection failed: 403 Forbidden (blocked-by-allowlist)")

    report = run_public_ingestion_audit(
        start=START,
        end=_end(48),
        output_dir=tmp_path / "audit",
        providers={"coinbase": BlockedProvider("coinbase"), "kraken": BlockedProvider("kraken")},
        fixture_note="simulated proxy block",
    )
    assert report["overall_passed"] is False
    assert report["failure_classification"] == "environment_or_endpoint_block"
    assert report["live_public_endpoints_reached"] is False
    assert validate_public_ingestion_report(tmp_path / "audit") == []


def test_report_tampering_fails_validation(sim, tmp_path) -> None:
    _run(sim, tmp_path)
    audit = tmp_path / "audit"
    report_path = audit / "public_ingestion_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    lying = dict(report)
    lying["live_public_endpoints_reached"] = True
    report_path.write_text(json.dumps(lying), encoding="utf-8")
    problems = validate_public_ingestion_report(audit)
    assert any("fixture_mode runs cannot claim" in p for p in problems)
    assert any("checksum mismatch" in p for p in problems)

    del lying["primary_pages"]
    lying["live_public_endpoints_reached"] = False
    report_path.write_text(json.dumps(lying), encoding="utf-8")
    assert any("missing field: primary_pages" in p for p in validate_public_ingestion_report(audit))

    lying["primary_pages"] = -3
    report_path.write_text(json.dumps(lying), encoding="utf-8")
    assert any("negative" in p for p in validate_public_ingestion_report(audit))

    inconsistent = dict(report)
    inconsistent["overall_passed"] = True
    inconsistent["quality_passed"] = False
    report_path.write_text(json.dumps(inconsistent), encoding="utf-8")
    assert any(
        "quality_passed differs from independently recomputed artifacts" in p
        for p in validate_public_ingestion_report(audit)
    )


def test_second_run_is_logically_identical(sim, tmp_path) -> None:
    report = _run(sim, tmp_path)
    assert report["canonical_hashes_run1"] == report["canonical_hashes_run2"]
    assert all(value != "absent" for value in report["canonical_hashes_run1"].values())


def test_coherently_resealed_evidence_attacks_still_fail(sim, tmp_path) -> None:
    _run(sim, tmp_path)
    baseline = tmp_path / "audit"
    attacks = {}

    missing = tmp_path / "attack-missing"
    shutil.copytree(baseline, missing)
    (missing / QUALITY_NAME).unlink()
    _reseal(missing)
    attacks["missing"] = validate_public_ingestion_report(missing)

    extra = tmp_path / "attack-extra"
    shutil.copytree(baseline, extra)
    (extra / "substituted.bin").write_bytes(b"substituted")
    _reseal(extra)
    attacks["extra"] = validate_public_ingestion_report(extra)

    counts = tmp_path / "attack-counts"
    shutil.copytree(baseline, counts)
    report = json.loads((counts / REPORT_NAME).read_text(encoding="utf-8"))
    report["primary_rows"] += 1
    (counts / REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
    _reseal(counts)
    attacks["counts"] = validate_public_ingestion_report(counts)

    nonfinite = tmp_path / "attack-nonfinite"
    shutil.copytree(baseline, nonfinite)
    report = json.loads((nonfinite / REPORT_NAME).read_text(encoding="utf-8"))
    report["attack_metric"] = float("nan")
    (nonfinite / REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
    _reseal(nonfinite)
    attacks["nonfinite"] = validate_public_ingestion_report(nonfinite)

    empty_trace = tmp_path / "attack-empty-trace"
    shutil.copytree(baseline, empty_trace)
    trace = json.loads((empty_trace / REQUEST_TRACE_NAME).read_text(encoding="utf-8"))
    trace["runs"]["run_1"]["venues"]["coinbase"]["requests"] = []
    (empty_trace / REQUEST_TRACE_NAME).write_text(json.dumps(trace), encoding="utf-8")
    _reseal(empty_trace)
    attacks["empty_trace"] = validate_public_ingestion_report(empty_trace)

    relabeled = tmp_path / "attack-relabeled"
    shutil.copytree(baseline, relabeled)
    for name in [REPORT_NAME, REQUEST_TRACE_NAME, REQUEST_SUMMARY_NAME, QUALITY_NAME]:
        payload = json.loads((relabeled / name).read_text(encoding="utf-8"))
        payload["fixture_mode"] = False
        payload["mode"] = "live-public-endpoints"
        if name == REPORT_NAME:
            payload["live_public_endpoints_reached"] = True
        (relabeled / name).write_text(json.dumps(payload), encoding="utf-8")
    _reseal(relabeled)
    attacks["relabeled"] = validate_public_ingestion_report(relabeled)

    fake_parquet = tmp_path / "attack-fake-parquet"
    shutil.copytree(baseline, fake_parquet)
    (fake_parquet / "coinbase_normalized.parquet").write_bytes(b"not parquet")
    _reseal(fake_parquet)
    attacks["fake_parquet"] = validate_public_ingestion_report(fake_parquet)

    expected_fragments = {
        "missing": "missing",
        "extra": "file set",
        "counts": "differs",
        "nonfinite": "non-finite",
        "empty_trace": "request log count mismatch",
        "relabeled": "live evidence covers fewer than 14 days",
        "fake_parquet": "invalid parquet",
    }
    for attack, fragment in expected_fragments.items():
        assert attacks[attack], attack
        assert any(fragment in problem for problem in attacks[attack]), (attack, attacks[attack])


def test_failure_classification_is_specific() -> None:
    assert _classify_failure(OSError("request timed out")) == "network_timeout"
    assert _classify_failure(OSError("DNS name resolution failed")) == "dns_failure"
    assert _classify_failure(OSError("ordinary failure")) == "ingestion_error"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"end": datetime(2025, 1, 1, tzinfo=UTC)}, "end must be after start"),
        ({"primary": "kraken", "validation": "coinbase"}, "Coinbase primary"),
        ({"symbol": "BTC/USD"}, "ETH/USD"),
        ({"end": datetime.now(tz=UTC) - timedelta(hours=1)}, "48 hours"),
        (
            {"start": datetime(2025, 1, 1, tzinfo=UTC), "end": datetime(2025, 1, 2, tzinfo=UTC)},
            "14 days",
        ),
        (
            {
                "start": datetime(2025, 1, 1, 0, 30, tzinfo=UTC),
                "end": datetime(2025, 1, 16, 0, 30, tzinfo=UTC),
            },
            "timeframe grid",
        ),
        (
            {
                "start": datetime(2025, 1, 1, tzinfo=UTC),
                "end": datetime(2025, 1, 15, tzinfo=UTC),
                "page_limit": 10_000,
            },
            "four pages",
        ),
    ],
)
def test_live_audit_rejects_invalid_evidence_windows_before_network(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    values = {
        "start": datetime(2025, 1, 1, tzinfo=UTC),
        "end": datetime(2025, 1, 16, tzinfo=UTC),
        "output_dir": tmp_path / "never-written",
        "page_limit": 30,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        run_public_ingestion_audit(**values)


def test_trace_validator_rejects_malformed_accounting_and_endpoints(sim, tmp_path) -> None:
    _run(sim, tmp_path)
    audit = tmp_path / "audit"
    report = json.loads((audit / REPORT_NAME).read_text(encoding="utf-8"))
    trace = json.loads((audit / REQUEST_TRACE_NAME).read_text(encoding="utf-8"))
    summary = json.loads((audit / REQUEST_SUMMARY_NAME).read_text(encoding="utf-8"))

    problems: list[str] = []
    _validate_trace([], summary, report, problems)
    _validate_trace({"runs": []}, summary, report, problems)
    _validate_trace(trace, {"runs": []}, report, problems)

    malformed = deepcopy(trace)
    malformed_summary = deepcopy(summary)
    malformed["schema"] = "wrong"
    malformed["fixture_mode"] = False
    malformed_summary["schema"] = "wrong"
    malformed_summary["fixture_mode"] = False
    malformed["runs"]["run_1"] = "bad"
    _validate_trace(malformed, malformed_summary, report, problems)

    malformed = deepcopy(trace)
    malformed_summary = deepcopy(summary)
    malformed_summary["runs"]["run_1"] = "bad"
    _validate_trace(malformed, malformed_summary, report, problems)

    malformed = deepcopy(trace)
    malformed_summary = deepcopy(summary)
    malformed_summary["runs"]["run_1"]["venues"] = []
    _validate_trace(malformed, malformed_summary, report, problems)

    malformed = deepcopy(trace)
    malformed_summary = deepcopy(summary)
    venue = malformed["runs"]["run_1"]["venues"]["coinbase"]
    summary_venue = malformed_summary["runs"]["run_1"]["venues"]["coinbase"]
    venue["http_request_count"] = True
    venue["requests"] = ["bad-request"]
    venue["response_statuses"] = []
    venue["cursor_progression_ms"] = [2, 1]
    venue["first_request_cursor_ms"] = 999
    venue["fetch_ohlcv_call_count"] = 9
    venue["page_count"] = 1
    venue["retry_count"] = 1
    venue["raw_rows_received"] = 10
    venue["normalized_rows"] = 20
    venue["failures"] = [1]
    summary_venue.pop(next(iter(summary_venue)))
    _validate_trace(malformed, malformed_summary, report, problems)

    malformed = deepcopy(trace)
    malformed_summary = deepcopy(summary)
    malformed["runs"]["run_1"]["venues"]["kraken"] = "bad"
    _validate_trace(malformed, malformed_summary, report, problems)

    malformed = deepcopy(trace)
    malformed_summary = deepcopy(summary)
    venue = malformed["runs"]["run_1"]["venues"]["coinbase"]
    venue["failures"] = ["claimed failure"]
    venue["requests"][0]["sequence"] = 999
    malformed_summary["runs"]["run_1"]["venues"]["coinbase"]["page_count"] = 999
    _validate_trace(malformed, malformed_summary, report, problems)

    live_report = deepcopy(report)
    live_report["fixture_mode"] = False
    live_trace = deepcopy(trace)
    live_summary = deepcopy(summary)
    for run in live_trace["runs"].values():
        for venue in run["venues"].values():
            venue["http_request_count"] = 0
            venue["fetch_ohlcv_call_count"] = 0
            venue["page_count"] = 0
            venue["raw_rows_received"] = 0
            venue["normalized_rows"] = 0
            venue["cursor_progression_ms"] = []
            venue["first_request_cursor_ms"] = None
            venue["last_request_cursor_ms"] = None
            venue["requests"] = []
            venue["response_statuses"] = []
            venue["failures"] = []
    for run in live_summary["runs"].values():
        for venue in run["venues"].values():
            venue.update(
                {
                    "http_request_count": 0,
                    "fetch_ohlcv_call_count": 0,
                    "page_count": 0,
                    "raw_rows_received": 0,
                    "normalized_rows": 0,
                    "first_request_cursor_ms": None,
                    "last_request_cursor_ms": None,
                    "cursor_progression_ms": [],
                    "response_statuses": [],
                    "requests": [],
                }
            )
    _validate_trace(live_trace, live_summary, live_report, problems)

    endpoint_trace = deepcopy(trace)
    endpoint_summary = deepcopy(summary)
    endpoint_report = deepcopy(report)
    endpoint_report["fixture_mode"] = False
    for run_name, run in endpoint_trace["runs"].items():
        for exchange, venue in run["venues"].items():
            venue["requests"][0]["url"] = "http://attacker.invalid/ohlcv"
            endpoint_summary["runs"][run_name]["venues"][exchange]["requests"] = venue["requests"]
    _validate_trace(endpoint_trace, endpoint_summary, endpoint_report, problems)

    assert any("schema is invalid" in problem for problem in problems)
    assert any("cursor progression goes backwards" in problem for problem in problems)
    assert any("page/retry accounting" in problem for problem in problems)
    assert any("row normalization accounting" in problem for problem in problems)
    assert any("endpoint is not approved" in problem for problem in problems)
    assert any("zero actual venue requests" in problem for problem in problems)


def test_file_integrity_validator_rejects_hostile_manifest_and_checksum_forms(
    sim, tmp_path
) -> None:
    _run(sim, tmp_path)
    baseline = tmp_path / "audit"

    missing = tmp_path / "integrity-missing"
    shutil.copytree(baseline, missing)
    (missing / MANIFEST_NAME).unlink()
    (missing / CHECKSUMS_NAME).unlink()
    problems: list[str] = []
    _validate_file_integrity(missing, problems)
    assert any("missing" in problem for problem in problems)

    bad_schema = tmp_path / "integrity-schema"
    shutil.copytree(baseline, bad_schema)
    manifest_path = bad_schema / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    problems = []
    _validate_file_integrity(bad_schema, problems)
    assert any("manifest schema" in problem for problem in problems)

    hostile = tmp_path / "integrity-hostile"
    shutil.copytree(baseline, hostile)
    (hostile / "nested").mkdir()
    manifest_path = hostile / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = list(manifest["files"])
    manifest["files"]["../escape"] = {"sha256": "0" * 64, "bytes": 0}
    manifest["files"][names[0]] = "malformed"
    manifest["files"][names[1]]["bytes"] = True
    manifest["files"][names[2]]["bytes"] += 1
    manifest["files"][names[3]]["sha256"] = "bad"
    manifest["files"][names[4]]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = "0" * 64
    (hostile / CHECKSUMS_NAME).write_text(
        f"malformed\n{digest}  {names[0]}\n{digest}  {names[0]}\n{digest}  ../escape\n",
        encoding="utf-8",
    )
    problems = []
    _validate_file_integrity(hostile, problems)
    expected = [
        "nested, linked, or special",
        "unsafe manifest path",
        "malformed manifest entry",
        "malformed manifest size",
        "manifest size mismatch",
        "malformed manifest checksum",
        "manifest checksum mismatch",
        "malformed checksum line",
        "duplicate checksum entry",
        "unsafe checksum path",
        "checksum file set differs",
    ]
    for fragment in expected:
        assert any(fragment in problem for problem in problems), (fragment, problems)


def test_parquet_validator_recomputes_schema_quality_time_and_hashes(sim, tmp_path) -> None:
    _run(sim, tmp_path)
    audit = tmp_path / "audit"
    report = json.loads((audit / REPORT_NAME).read_text(encoding="utf-8"))
    parquet = audit / "coinbase_normalized.parquet"
    hostile_report = deepcopy(report)
    hostile_report["primary_first_timestamp"] = "wrong"
    hostile_report["primary_last_timestamp"] = "wrong"
    hostile_report["canonical_hashes_run1"]["coinbase"] = "0" * 64
    hostile_report["parquet_file_hashes_run1"]["coinbase"] = "0" * 64
    problems: list[str] = []
    frame = _validate_parquet(
        parquet,
        exchange="coinbase",
        expected_rows=report["primary_rows"] + 1,
        report=hostile_report,
        first_field="primary_first_timestamp",
        last_field="primary_last_timestamp",
        problems=problems,
    )
    assert frame is not None
    for fragment in [
        "row count",
        "first timestamp",
        "last timestamp",
        "logical hash",
        "physical hash",
    ]:
        assert any(fragment in problem for problem in problems)

    bad_schema = tmp_path / "bad-schema.parquet"
    frame.drop(columns="volume").to_parquet(bad_schema, index=False)
    assert (
        _validate_parquet(
            bad_schema,
            exchange="coinbase",
            expected_rows=len(frame),
            report=report,
            first_field="primary_first_timestamp",
            last_field="primary_last_timestamp",
            problems=problems,
        )
        is None
    )

    bad_quality = tmp_path / "bad-quality.parquet"
    duplicate = pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)
    duplicate.to_parquet(bad_quality, index=False)
    _validate_parquet(
        bad_quality,
        exchange="coinbase",
        expected_rows=len(duplicate),
        report=report,
        first_field="primary_first_timestamp",
        last_field="primary_last_timestamp",
        problems=problems,
    )
    assert any("quality failure" in problem for problem in problems)


def test_report_validator_exercises_semantic_fail_closed_checks(sim, tmp_path) -> None:
    _run(sim, tmp_path)
    baseline = tmp_path / "audit"

    assert validate_public_ingestion_report(tmp_path / "absent") == [f"missing {REPORT_NAME}"]

    root_list = tmp_path / "report-list"
    shutil.copytree(baseline, root_list)
    (root_list / REPORT_NAME).write_text("[]", encoding="utf-8")
    assert validate_public_ingestion_report(root_list)

    semantic = tmp_path / "report-semantic"
    shutil.copytree(baseline, semantic)
    report_path = semantic / REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "schema": "wrong",
            "credentials_used": True,
            "orders_possible": True,
            "fixture_mode": False,
            "live_public_endpoints_reached": True,
            "primary_exchange": "wrong",
            "validation_exchange": "wrong",
            "symbol": "wrong",
            "timeframe": "invalid",
            "requested_start": "not-a-date",
            "requested_end": "not-a-date",
            "expected_rows": 999,
            "primary_pages": -1,
            "failure": "recorded failure",
            "idempotency_passed": False,
            "canonical_hashes_run1": {"wrong": "xyz"},
        }
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    quality_path = semantic / QUALITY_NAME
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["schema"] = "wrong"
    quality["fixture_mode"] = True
    quality["quality"] = None
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    problems = validate_public_ingestion_report(semantic)
    for fragment in [
        "report schema",
        "credentials_used",
        "orders_possible",
        "negative",
        "date range is not valid",
        "timeframe is invalid",
        "wrong venue",
        "quality evidence schema",
        "canonical run one hashes are malformed",
        "recorded failure",
    ]:
        assert any(fragment in problem for problem in problems), (fragment, problems)

    naive = tmp_path / "report-naive"
    shutil.copytree(baseline, naive)
    report_path = naive / REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["requested_start"] = "2025-01-01T00:00:00"
    report["requested_end"] = "2025-01-02T00:00:00"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert any("timezone-aware" in problem for problem in validate_public_ingestion_report(naive))

    missing_parquet = tmp_path / "report-missing-parquet"
    shutil.copytree(baseline, missing_parquet)
    for name in PARQUET_ARTIFACTS:
        (missing_parquet / name).unlink()
    problems = validate_public_ingestion_report(missing_parquet)
    assert any("missing artifact" in problem for problem in problems)

    unlabeled = tmp_path / "report-unlabeled"
    shutil.copytree(baseline, unlabeled)
    (unlabeled / "public_ingestion_report.md").write_text(
        "simulation label intentionally removed", encoding="utf-8"
    )
    assert any("not labeled" in problem for problem in validate_public_ingestion_report(unlabeled))
