"""Independent hand-calculated coverage and candle-time outcome oracles."""

import json

import pandas as pd
import pytest
from test_paper_recording import record_input
from typer.testing import CliRunner

from ares_engine.cli import app
from ares_engine.exceptions import AresError
from ares_engine.paper_log import record_paper_signal, utc_timestamp
from ares_engine.paper_report import paper_report, paper_report_from_files


def plan(**changes):
    return {
        "schema_version": "ares-paper-plan-v1",
        "exchange": "coinbase",
        "symbol": "ETH/USD",
        "timeframe": "1h",
        "start": "2026-10-01T12:00:00+00:00",
        "end": "2026-10-01T16:00:00+00:00",
        "horizon_bars": 1,
        "manifest_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "max_recording_delay_seconds": 300,
        **changes,
    }


def prices(*pairs):
    return [
        {
            "exchange": "coinbase",
            "symbol": "ETH/USD",
            "timeframe": "1h",
            "timestamp": f"2026-10-01T{hour:02d}:00:00+00:00",
            "close": close,
        }
        for hour, close in pairs
    ]


def add(path, hour=12, *, signal=1, delay=0, manifest=None, config=None, close=100.0):
    data = record_input()
    data["event"]["timestamp"] = f"2026-10-01T{hour:02d}:00:00+00:00"
    at = pd.Timestamp(data["event"]["timestamp"]) + pd.Timedelta(hours=1)
    data["observation"].update(
        recorded_at=utc_timestamp(at + pd.Timedelta(seconds=delay)), as_of=utc_timestamp(at)
    )
    data["evidence"].update(signal=signal, close=close)
    if manifest:
        data["evidence"]["manifest_sha256"] = manifest
    if config:
        data["evidence"]["config_sha256"] = config
    return record_paper_signal(path, **data)


@pytest.fixture
def journal(tmp_path):
    path = tmp_path / "signals.jsonl"
    path.write_bytes(b"")
    return path


def report(path, *, spec=None, data=None, cutoff="2026-10-01T17:00:00+00:00"):
    return paper_report(
        path,
        plan=plan() if spec is None else spec,
        prices=[] if data is None else data,
        as_of=cutoff,
    )


def test_declared_denominator_exact_outcomes_and_no_writes(journal):
    receipt = add(journal, 12)
    add(journal, 14, signal=-1)
    before = {p.name: p.read_bytes() for p in journal.parent.iterdir()}
    actual = report(journal, data=prices((12, 100), (13, 110), (14, 100), (15, 90)))
    assert actual["coverage"] == {
        "planned_slots": 4,
        "due_slots": 4,
        "not_due": 0,
        "missing_record": 2,
        "late_record": 0,
        "model_mismatch": 0,
        "recorded": 2,
        "timely_matching_coverage": 0.5,
    }
    assert actual["outcomes"] == {
        "pending": 0,
        "missing_data": 0,
        "source_conflict": 0,
        "available": 2,
    }
    first, last = actual["observations"][0], actual["observations"][2]
    assert first["record_sha256"] == receipt.record["record_sha256"]
    assert first["outcome"]["terminal_close_change"] == pytest.approx(0.1)
    assert last["outcome"]["terminal_close_change"] == pytest.approx(-0.1)
    assert actual["recorded_signal_counts"] == {"long": 1, "short": 1, "flat": 0}
    assert actual["independent_forward_performance_established"] is False
    assert actual["missing_record_causes"].startswith("UNKNOWN")
    assert actual == report(journal, data=prices((12, 100), (13, 110), (14, 100), (15, 90)))
    assert before == {p.name: p.read_bytes() for p in journal.parent.iterdir()}


@pytest.mark.parametrize(
    "cutoff,expected",
    [("2026-10-01T13:59:59+00:00", "pending"), ("2026-10-01T14:00:00+00:00", "available")],
)
def test_candle_open_timestamp_matures_only_at_target_close(journal, cutoff, expected):
    add(journal)
    actual = report(journal, data=prices((13, 110)), cutoff=cutoff)
    outcome = actual["observations"][0]["outcome"]
    assert outcome["status"] == expected
    assert outcome["target_timestamp"] == "2026-10-01T13:00:00+00:00"
    assert outcome["matures_at"] == "2026-10-01T14:00:00+00:00"


def test_exact_timestamp_join_never_skips_missing_terminal(journal):
    add(journal)
    actual = report(journal, data=prices((12, 100), (14, 500)))
    assert actual["observations"][0]["outcome"]["status"] == "missing_data"
    # h=2 requires 14:00 exactly; an absent intermediate candle is not substituted.
    h2 = report(journal, spec=plan(horizon_bars=2), data=prices((14, 110)))
    assert h2["observations"][0]["outcome"]["terminal_close_change"] == pytest.approx(0.1)


def test_later_receipts_and_future_prices_cannot_fill_earlier_cutoff(journal):
    add(journal)
    cutoff = "2026-10-01T14:00:00+00:00"
    earlier = report(journal, data=prices((13, 110)), cutoff=cutoff)
    add(journal, 13, delay=60)
    later = report(journal, data=prices((13, 110), (16, 999999)), cutoff=cutoff)
    for field in (
        "coverage",
        "outcomes",
        "observations",
        "recorded_signal_counts",
        "consumed_outcome_prices_sha256",
    ):
        assert later[field] == earlier[field]
    assert later["journal_sha256"] != earlier["journal_sha256"]


def test_receipt_evaluated_after_cutoff_does_not_count(journal):
    data = record_input()
    data["observation"]["as_of"] = "2026-10-01T14:05:00+00:00"
    record_paper_signal(journal, **data)
    actual = report(journal, cutoff="2026-10-01T14:00:00+00:00")
    assert actual["coverage"]["recorded"] == 0


def test_backdated_evaluation_does_not_hide_late_recording(journal):
    add(journal, delay=3600)
    actual = report(journal, data=prices((13, 110)))
    assert actual["coverage"]["late_record"] == 1
    assert actual["coverage"]["timely_matching_coverage"] == 0
    assert "outcome" not in actual["observations"][0]


def test_evaluation_cannot_follow_receipt_recording(journal):
    data = record_input()
    data["observation"]["as_of"] = "2026-10-01T13:04:00+00:00"
    record_paper_signal(journal, **data)
    before = journal.read_bytes()
    with pytest.raises(AresError, match="evaluation time follows"):
        report(journal)
    assert journal.read_bytes() == before


@pytest.mark.parametrize("extra_ns,expected", [(0, "recorded"), (1, "late_record")])
@pytest.mark.parametrize("evaluation_at_deadline", [False, True])
def test_recording_deadline_is_inclusive_at_nanosecond_precision(
    journal, extra_ns, expected, evaluation_at_deadline
):
    data = record_input()
    stamp = pd.Timestamp("2026-10-01T13:05:00+00:00") + pd.Timedelta(extra_ns, unit="ns")
    data["observation"]["recorded_at"] = utc_timestamp(stamp)
    if evaluation_at_deadline:
        data["observation"]["as_of"] = utc_timestamp(stamp)
    record_paper_signal(journal, **data)
    assert report(journal)["observations"][0]["status"] == expected


def test_flat_signal_does_not_zero_a_descriptive_price_change(journal):
    add(journal, signal=0)
    result = report(journal, data=prices((13, 110)))
    assert result["observations"][0]["outcome"]["terminal_close_change"] == pytest.approx(0.1)


def test_finite_prices_with_overflowing_ratio_fail_without_writes(journal):
    add(journal, close=1e-308)
    before = journal.read_bytes()
    with pytest.raises(AresError, match="Non-finite"):
        report(journal, data=prices((13, 1e308)))
    assert journal.read_bytes() == before


def test_missing_at_cutoff_can_still_be_within_recording_allowance(journal):
    result = report(journal, cutoff="2026-10-01T13:01:00+00:00")
    assert result["coverage"]["due_slots"] == 1
    assert result["coverage"]["missing_record"] == 1
    assert result["observations"][0]["status"] == "missing_record"


@pytest.mark.parametrize("field", ["manifest", "config"])
def test_model_identity_mismatch_stays_in_denominator(journal, field):
    add(journal, **{field: "a" * 64})
    actual = report(journal)
    assert actual["coverage"]["due_slots"] == 4
    assert actual["coverage"]["model_mismatch"] == 1
    assert actual["coverage"]["recorded"] == 0
    assert "outcome" not in actual["observations"][0]


def test_empty_explicit_stream_zero_due_and_zero_coverage(journal):
    assert report(journal)["coverage"]["timely_matching_coverage"] == 0
    actual = report(journal, cutoff="2026-10-01T12:30:00+00:00")
    assert actual["coverage"]["due_slots"] == 0
    assert actual["coverage"]["not_due"] == 4
    assert actual["coverage"]["timely_matching_coverage"] is None
    assert not (journal.parent / "signals.jsonl.lock").exists()


def test_missing_file_is_not_empty_stream(tmp_path):
    path = tmp_path / "new" / "missing.jsonl"
    with pytest.raises(AresError, match="does not exist"):
        report(path)
    assert not path.parent.exists()


def test_source_anchor_conflict_is_visible_without_mutation(journal):
    add(journal)
    before = journal.read_bytes()
    actual = report(journal, data=prices((12, 99), (13, 110)))
    assert actual["observations"][0]["outcome"]["status"] == "source_conflict"
    assert actual["observations"][0]["outcome"]["terminal_close_change"] is None
    assert journal.read_bytes() == before


def test_revised_prices_change_evidence_not_outcome_identity(journal):
    add(journal)
    a = report(journal, data=prices((13, 110)))
    b = report(journal, data=prices((13, 120)))
    first = a["observations"][0]["outcome"]
    second = b["observations"][0]["outcome"]
    assert first["outcome_id"] == second["outcome_id"]
    assert first["outcome_evidence_sha256"] != second["outcome_evidence_sha256"]
    assert a["report_sha256"] != b["report_sha256"]


@pytest.mark.parametrize("signal", [-1, 0, 1])
def test_price_annotation_is_not_signal_signed_trading_return(journal, signal):
    add(journal, signal=signal)
    actual = report(journal, data=prices((13, 100)))
    assert actual["observations"][0]["outcome"]["terminal_close_change"] == 0
    assert sum(actual["recorded_signal_counts"].values()) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"horizon_bars": 0},
        {"horizon_bars": True},
        {"horizon_bars": 1001},
        {"start": "2026-10-01T12:30:00+00:00"},
        {"end": "2030-01-01T00:00:00+00:00"},
        {"manifest_sha256": "bad"},
        {"timeframe": "1H"},
        {"max_recording_delay_seconds": -1},
        {"extra": 1},
    ],
)
def test_invalid_or_unbounded_plan_fails_closed(journal, change):
    with pytest.raises(AresError):
        report(journal, spec=plan(**change))


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "wrong-market", "nonfinite", "negative", "misaligned", "unknown-field", "bool"],
)
def test_bad_outcome_rows_fail_closed(journal, mutation):
    data = prices((13, 110))
    if mutation == "duplicate":
        data *= 2
    elif mutation == "wrong-market":
        data[0]["symbol"] = "BTC/USD"
    elif mutation == "nonfinite":
        data[0]["close"] = float("nan")
    elif mutation == "negative":
        data[0]["close"] = -1
    elif mutation == "misaligned":
        data[0]["timestamp"] = "2026-10-01T13:15:00+00:00"
    elif mutation == "unknown-field":
        data[0]["order"] = 1
    else:
        data[0]["close"] = True
    with pytest.raises(AresError):
        report(journal, data=data)


@pytest.mark.parametrize("raw", [b'{"signal":1}\n', b"{", b"\xff\n"])
def test_malformed_or_legacy_journal_is_not_filtered_away(journal, raw):
    journal.write_bytes(raw)
    with pytest.raises(AresError):
        report(journal)
    assert journal.read_bytes() == raw


def test_cli_emits_json_and_does_not_write_inputs(journal):
    add(journal)
    plan_path = journal.parent / "plan.json"
    price_path = journal.parent / "prices.json"
    plan_path.write_text(json.dumps(plan()), encoding="utf-8")
    price_path.write_text(json.dumps(prices((13, 110))), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in journal.parent.iterdir()}
    result = CliRunner().invoke(
        app,
        [
            "paper-report",
            "--journal",
            str(journal),
            "--plan",
            str(plan_path),
            "--prices",
            str(price_path),
            "--as-of",
            "2026-10-01T17:00:00+00:00",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["outcomes"]["available"] == 1
    assert before == {p.name: p.read_bytes() for p in journal.parent.iterdir()}
    plan_path.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
    with pytest.raises(AresError, match="Duplicate"):
        paper_report_from_files(journal, plan_path, price_path, "2026-10-01T17:00:00+00:00")
