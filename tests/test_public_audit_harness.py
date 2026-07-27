"""Regression tests for the portable public-ingestion audit harness."""

import json
from datetime import UTC, datetime

import pytest
from venue_wire_sim import BASE_TS, HOUR, WireSimulator, make_real_ccxt_exchange

from ares_engine.data.providers import CCXTOHLCVProvider
from ares_engine.public_audit import (
    run_public_ingestion_audit,
    validate_public_ingestion_report,
)

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
        "overall_passed requires quality_passed" in p
        for p in validate_public_ingestion_report(audit)
    )


def test_second_run_is_logically_identical(sim, tmp_path) -> None:
    report = _run(sim, tmp_path)
    assert report["canonical_hashes_run1"] == report["canonical_hashes_run2"]
    assert all(value != "absent" for value in report["canonical_hashes_run1"].values())
