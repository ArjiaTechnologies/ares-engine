"""Planned-attempt lifecycle, real inference gates, and interrupted-write oracles."""

import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from filelock import FileLock
from test_paper_recording import inference_env as inference_env
from test_paper_recording import record_input
from test_paper_report import plan
from typer.testing import CliRunner

import ares_engine.live as live
import ares_engine.paper_collection as collection
from ares_engine.bundles import load_bundle
from ares_engine.cli import app
from ares_engine.exceptions import (
    AresError,
    BundleIntegrityError,
    DataQualityError,
    PaperPlanMismatch,
)
from ares_engine.paper_log import evidence_digest, record_paper_signal, utc_timestamp
from ares_engine.paper_report import paper_report


@pytest.fixture
def registered(tmp_path, monkeypatch):
    clock = [pd.Timestamp("2026-10-01T12:00:00+00:00")]
    monkeypatch.setattr(collection, "utc_now", lambda: clock[0])
    path = tmp_path / "collection.json"
    reg = collection.register_collection(path, plan())
    clock[0] = pd.Timestamp("2026-10-01T13:00:00+00:00")
    monkeypatch.setattr(collection, "read_market", lambda path: pd.DataFrame())
    return path, reg["registration_sha256"], clock


def call(registered, hour=12, **changes):
    path, digest, _ = registered
    return collection.collect_paper_slot(
        path,
        registration_sha256=digest,
        candle=f"2026-10-01T{hour:02d}:00:00+00:00",
        bundle=Path("unused"),
        primary=Path("unused"),
        **changes,
    )


def status(registered, cutoff="2026-10-01T17:00:00+00:00"):
    return collection.collection_status(
        registered[0], registration_sha256=registered[1], as_of=cutoff
    )


def fake_success(bundle, frame, *, expected, log_path, as_of, **kwargs):
    data = record_input()
    data["event"] = expected.event.model_dump()
    data["evidence"].update(
        manifest_sha256=expected.manifest_sha256, config_sha256=expected.config_sha256
    )
    data["observation"].update(as_of=utc_timestamp(as_of), recorded_at=utc_timestamp(as_of))
    return SimpleNamespace(recording=record_paper_signal(log_path, **data))


def boom(exc):
    def fail(*args, **kwargs):
        raise exc

    return fail


def test_frozen_registration_exact_replay_does_not_touch_bytes(registered):
    path, digest, clock = registered
    before = path.read_bytes()
    clock[0] += pd.Timedelta(days=1)
    assert collection.register_collection(path, plan())["registration_sha256"] == digest
    assert path.read_bytes() == before
    with pytest.raises(AresError, match="frozen"):
        collection.register_collection(path, plan(manifest_sha256="a" * 64))
    assert path.read_bytes() == before


def test_registration_after_first_candle_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        collection, "utc_now", lambda: pd.Timestamp("2026-10-01T12:00:00.000000001+00:00")
    )
    path = tmp_path / "late.json"
    with pytest.raises(AresError, match="follows"):
        collection.register_collection(path, plan())
    assert not path.exists()


def test_hand_count_planned_success_failure_unknown_and_unattempted(registered, monkeypatch):
    path, digest, clock = registered
    monkeypatch.setattr(collection, "generate_paper_signal", fake_success)
    assert call(registered)["attempt"]["status"] == "SUCCEEDED"
    clock[0] += pd.Timedelta(hours=1)
    monkeypatch.setattr(
        collection, "generate_paper_signal", boom(DataQualityError("private details"))
    )
    assert call(registered, 13)["attempt"]["failure_code"] == "DATA_QUALITY_REJECTED"
    clock[0] += pd.Timedelta(hours=1)
    monkeypatch.setattr(collection, "generate_paper_signal", boom(RuntimeError("interrupted")))
    with pytest.raises(RuntimeError):
        call(registered, 14)
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    report = status(registered)
    assert report["planned_slots"] == 4
    assert report["counts"] == {
        "NOT_DUE": 0,
        "NOT_STARTED": 1,
        "STARTED_UNKNOWN": 1,
        "FAILED": 1,
        "SUCCEEDED": 1,
    }
    assert report["independent_forward_performance_established"] is False
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}
    assert b"private details" not in path.read_bytes()
    assert status(registered, "2026-10-01T12:59:59+00:00")["counts"]["NOT_DUE"] == 4
    # Reuse the original planned-denominator report without rebuilding outcomes.
    prices = [
        {
            "timestamp": "2026-10-01T13:00:00+00:00",
            "exchange": "coinbase",
            "symbol": "ETH/USD",
            "timeframe": "1h",
            "close": 110.0,
        }
    ]
    outcome = paper_report(
        collection.collection_signals_path(path),
        plan=report["registration"]["plan"],
        prices=prices,
        as_of=report["as_of"],
    )
    assert outcome["coverage"]["timely_matching_coverage"] == 0.25
    assert outcome["outcomes"]["available"] == 1
    assert outcome["observations"][0]["outcome"]["terminal_close_change"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "exception,code",
    [
        (DataQualityError("secret"), "DATA_QUALITY_REJECTED"),
        (BundleIntegrityError("secret"), "MODEL_INTEGRITY_REJECTED"),
        (PaperPlanMismatch("secret"), "PLAN_MISMATCH"),
    ],
)
def test_only_known_precommit_rejections_are_explicit_failures(
    registered, monkeypatch, exception, code
):
    monkeypatch.setattr(collection, "generate_paper_signal", boom(exception))
    assert call(registered)["attempt"]["failure_code"] == code
    assert not collection.collection_signals_path(registered[0]).exists()
    assert "secret" not in registered[0].read_text()


def test_local_input_error_is_a_recorded_rejection_after_admission(registered, monkeypatch):
    def reject(path):
        assert json.loads(registered[0].read_text())["attempts"][0]["status"] == "STARTED"
        raise AresError("bad local file")

    monkeypatch.setattr(collection, "read_market", reject)
    monkeypatch.setattr(collection, "generate_paper_signal", boom(AssertionError("must not run")))
    assert call(registered)["attempt"]["failure_code"] == "LOCAL_INPUT_REJECTED"


@pytest.mark.parametrize("mode", ["success", "failure", "unknown"])
def test_same_slot_never_dispatches_again_even_with_changed_inputs(registered, monkeypatch, mode):
    monkeypatch.setattr(
        collection,
        "generate_paper_signal",
        fake_success
        if mode == "success"
        else boom(DataQualityError() if mode == "failure" else RuntimeError()),
    )
    if mode == "unknown":
        with pytest.raises(RuntimeError):
            call(registered)
    else:
        call(registered)
    before = registered[0].read_bytes()
    monkeypatch.setattr(collection, "read_market", boom(AssertionError("must not reread")))
    actual = call(registered, secondary={"changed": Path("new")})
    assert actual["dispatch"] == "REPLAY_NO_DISPATCH"
    assert registered[0].read_bytes() == before


@pytest.mark.parametrize("delta", [-1, 300_000_000_001])
def test_admission_checks_exact_close_and_delay_before_inference(registered, monkeypatch, delta):
    registered[2][0] += pd.Timedelta(nanoseconds=delta)
    before = registered[0].read_bytes()
    monkeypatch.setattr(collection, "generate_paper_signal", boom(AssertionError("must not run")))
    with pytest.raises(AresError, match="out-of-window"):
        call(registered)
    assert registered[0].read_bytes() == before


def test_exact_delay_boundary_is_allowed(registered, monkeypatch):
    registered[2][0] += pd.Timedelta(seconds=300)
    monkeypatch.setattr(collection, "generate_paper_signal", fake_success)
    assert call(registered)["attempt"]["status"] == "SUCCEEDED"


def test_terminal_after_historical_cutoff_is_still_unknown(registered, monkeypatch):
    def succeed(*args, **kwargs):
        result = fake_success(*args, **kwargs)
        registered[2][0] += pd.Timedelta(seconds=2)
        return result

    monkeypatch.setattr(collection, "generate_paper_signal", succeed)
    call(registered)
    before = registered[0].read_bytes()
    report = status(registered, "2026-10-01T13:00:01+00:00")
    assert report["counts"]["STARTED_UNKNOWN"] == 1
    assert len(report["slots"][0]["observed_receipt_hashes"]) == 1
    assert status(registered, "2026-10-01T13:00:02+00:00")["counts"]["SUCCEEDED"] == 1
    assert registered[0].read_bytes() == before


def test_receipt_after_interruption_never_infers_invocation_completion(registered, monkeypatch):
    def crash(*args, **kwargs):
        fake_success(*args, **kwargs)
        raise RuntimeError("died after paper commit")

    monkeypatch.setattr(collection, "generate_paper_signal", crash)
    with pytest.raises(RuntimeError):
        call(registered)
    report = status(registered)
    assert report["counts"]["STARTED_UNKNOWN"] == 1
    assert len(report["slots"][0]["observed_receipt_hashes"]) == 1
    assert call(registered)["attempt"]["status"] == "STARTED"


@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize("terminal", ["SUCCEEDED", "FAILED"])
def test_uncertain_terminal_replace_reconciles_disk_without_redispatch(
    registered, monkeypatch, after_commit, terminal
):
    monkeypatch.setattr(
        collection,
        "generate_paper_signal",
        fake_success if terminal == "SUCCEEDED" else boom(DataQualityError()),
    )
    original = collection._replace_journal

    def uncertain(path, payload):
        if json.loads(payload)["attempts"][-1]["status"] == "STARTED":
            return original(path, payload)
        if after_commit:
            original(path, payload)
        raise OSError("uncertain replace")

    monkeypatch.setattr(collection, "_replace_journal", uncertain)
    with pytest.raises(OSError):
        call(registered)
    before = registered[0].read_bytes()
    monkeypatch.setattr(collection, "generate_paper_signal", boom(AssertionError("never retry")))
    assert call(registered)["attempt"]["status"] == (terminal if after_commit else "STARTED")
    assert registered[0].read_bytes() == before


def test_preexisting_unlinked_receipt_blocks_dispatch(registered, monkeypatch):
    record_paper_signal(collection.collection_signals_path(registered[0]), **record_input())
    before = registered[0].read_bytes()
    monkeypatch.setattr(collection, "generate_paper_signal", boom(AssertionError("never run")))
    with pytest.raises(AresError, match="Pre-existing"):
        call(registered)
    assert registered[0].read_bytes() == before
    assert status(registered)["slots"][0]["status"] == "NOT_STARTED"


def test_independent_writer_replay_cannot_be_attributed_to_collection(registered, monkeypatch):
    def replay(*args, **kwargs):
        fake_success(*args, **kwargs)
        return fake_success(*args, **kwargs)

    monkeypatch.setattr(collection, "generate_paper_signal", replay)
    with pytest.raises(AresError, match="newly acknowledged"):
        call(registered)
    assert status(registered)["counts"]["STARTED_UNKNOWN"] == 1


def test_clock_rollback_after_receipt_preserves_unknown(registered, monkeypatch):
    def rollback(*args, **kwargs):
        result = fake_success(*args, **kwargs)
        registered[2][0] -= pd.Timedelta(seconds=1)
        return result

    monkeypatch.setattr(collection, "generate_paper_signal", rollback)
    with pytest.raises(AresError, match="clock precedes"):
        call(registered)
    assert status(registered)["counts"]["STARTED_UNKNOWN"] == 1


def test_pin_and_tamper_rejection_preserve_state(registered):
    path, digest, _ = registered
    before = path.read_bytes()
    with pytest.raises(AresError, match="identity"):
        collection.collection_status(
            path, registration_sha256="a" * 64, as_of="2026-10-01T17:00:00+00:00"
        )
    data = json.loads(before)
    data["registration"]["plan"]["end"] = "2026-10-01T17:00:00+00:00"
    path.write_text(json.dumps(data))
    bad = path.read_bytes()
    with pytest.raises(AresError, match="digest"):
        status(registered)
    assert path.read_bytes() == bad


def test_deleted_success_receipt_is_not_silently_trusted(registered, monkeypatch):
    monkeypatch.setattr(collection, "generate_paper_signal", fake_success)
    call(registered)
    collection.collection_signals_path(registered[0]).unlink()
    with pytest.raises(AresError, match="no matching"):
        status(registered)
    before = registered[0].read_bytes()
    monkeypatch.setattr(collection, "generate_paper_signal", boom(AssertionError("must not run")))
    for hour in (12, 13):
        with pytest.raises(AresError, match="no matching"):
            call(registered, hour)
    assert registered[0].read_bytes() == before


@pytest.mark.parametrize("malformed", ["primary", "secondary"])
def test_real_malformed_parquet_has_explicit_local_failure(registered, monkeypatch, malformed):
    from ares_engine.data.storage import read_market

    path, digest, _ = registered
    primary, secondary = path.parent / "primary.parquet", path.parent / "secondary.parquet"
    good = pd.DataFrame({"timestamp": [pd.Timestamp("2026-10-01T12:00:00+00:00")]})
    bad = pd.DataFrame({"unrelated": [1]})
    (bad if malformed == "primary" else good).to_parquet(primary)
    (bad if malformed == "secondary" else good).to_parquet(secondary)
    monkeypatch.setattr(collection, "read_market", read_market)
    monkeypatch.setattr(collection, "generate_paper_signal", boom(AssertionError("must not run")))
    result = collection.collect_paper_slot(
        path,
        registration_sha256=digest,
        candle="2026-10-01T12:00:00+00:00",
        bundle=Path("unused"),
        primary=primary,
        secondary={"kraken": secondary},
    )
    assert result["attempt"]["failure_code"] == "LOCAL_INPUT_REJECTED"


@pytest.mark.parametrize(
    "change", ["duplicate", "bad-id", "bad-time", "false-terminal", "unknown-field"]
)
def test_rehashed_invalid_state_is_rejected(registered, monkeypatch, change):
    monkeypatch.setattr(collection, "generate_paper_signal", boom(DataQualityError()))
    call(registered)
    state = json.loads(registered[0].read_text())
    attempt = state["attempts"][0]
    if change == "duplicate":
        state["attempts"].append(dict(attempt))
    elif change == "bad-id":
        attempt["attempt_id"] = "a" * 64
    elif change == "bad-time":
        attempt["finished_at"] = "2026-10-01T12:59:59+00:00"
    elif change == "false-terminal":
        attempt["status"] = "SUCCEEDED"
    else:
        state["approved"] = True
    state["state_sha256"] = evidence_digest({k: v for k, v in state.items() if k != "state_sha256"})
    registered[0].write_text(json.dumps(state))
    before = registered[0].read_bytes()
    with pytest.raises(AresError):
        status(registered)
    assert registered[0].read_bytes() == before


def test_rehashed_plan_cannot_evade_external_registration_pin(registered):
    state = json.loads(registered[0].read_text())
    reg = state["registration"]
    reg["plan"]["end"] = "2026-10-01T17:00:00+00:00"
    reg["plan_sha256"] = evidence_digest(reg["plan"])
    reg["registration_sha256"] = evidence_digest(
        {k: v for k, v in reg.items() if k != "registration_sha256"}
    )
    state["state_sha256"] = evidence_digest({k: v for k, v in state.items() if k != "state_sha256"})
    registered[0].write_text(json.dumps(state))
    with pytest.raises(AresError, match="identity"):
        call(registered)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b"{", b"[]", b"null"])
def test_malformed_state_blocks_without_repair(registered, raw):
    registered[0].write_bytes(raw)
    with pytest.raises(AresError):
        status(registered)
    assert registered[0].read_bytes() == raw


def test_state_byte_limit_and_missing_collection(registered, monkeypatch):
    before = registered[0].read_bytes()
    monkeypatch.setattr(collection, "MAX_LOG_BYTES", len(before) - 1)
    with pytest.raises(AresError, match="bounded"):
        call(registered)
    assert registered[0].read_bytes() == before
    path = registered[0].parent / "absent" / "state.json"
    with pytest.raises(AresError, match="Register"):
        collection.collect_paper_slot(
            path,
            registration_sha256=registered[1],
            candle="2026-10-01T12:00:00+00:00",
            bundle=Path("unused"),
            primary=Path("unused"),
        )
    assert not path.parent.exists()


def test_collection_lock_contention_does_not_admit(registered):
    before = registered[0].read_bytes()
    with FileLock(str(registered[0]) + ".lock"), pytest.raises(AresError, match="owns"):
        call(registered)
    assert registered[0].read_bytes() == before


@pytest.mark.parametrize("target", ["state", "lock", "signals"])
def test_linked_control_paths_are_rejected(registered, target):
    path = registered[0]
    original = path.parent / "original"
    original.write_bytes(b"keep")
    linked = (
        path
        if target == "state"
        else Path(str(path) + (".lock" if target == "lock" else ".signals.jsonl"))
    )
    linked.unlink(missing_ok=True)
    os.link(original, linked)
    with pytest.raises(AresError, match="unlinked"):
        call(registered)
    assert original.read_bytes() == b"keep"


def _crash_process(path, digest, after_signal):
    collection.utc_now = lambda: pd.Timestamp("2026-10-01T13:00:00+00:00")
    collection.read_market = lambda path: pd.DataFrame()

    def crash(*args, **kwargs):
        if after_signal:
            fake_success(*args, **kwargs)
        os._exit(79)

    collection.generate_paper_signal = crash
    collection.collect_paper_slot(
        Path(path),
        registration_sha256=digest,
        candle="2026-10-01T12:00:00+00:00",
        bundle=Path("unused"),
        primary=Path("unused"),
    )


@pytest.mark.parametrize("after_signal", [False, True])
def test_real_process_death_after_admission_or_signal_never_reexecutes(
    registered, monkeypatch, after_signal
):
    proc = multiprocessing.get_context("spawn").Process(
        target=_crash_process, args=(str(registered[0]), registered[1], after_signal)
    )
    proc.start()
    proc.join(30)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail("Crash process did not stop")
    assert proc.exitcode == 79
    monkeypatch.setattr(collection, "generate_paper_signal", boom(AssertionError("never rerun")))
    assert call(registered)["attempt"]["status"] == "STARTED"
    report = status(registered)
    assert report["counts"]["STARTED_UNKNOWN"] == 1
    assert len(report["slots"][0]["observed_receipt_hashes"]) == int(after_signal)


@pytest.mark.parametrize("mismatch", [None, "manifest", "config", "candle"])
def test_real_inference_enforces_loaded_plan_before_recording(
    inference_env, tmp_path, monkeypatch, mismatch
):
    bundle, primary, secondary, as_of = inference_env
    loaded = load_bundle(bundle)
    candle = as_of - pd.Timedelta(hours=1)
    spec = plan(
        start=utc_timestamp(candle),
        end=utc_timestamp(candle + pd.Timedelta(hours=1)),
        manifest_sha256=loaded.manifest_sha256,
        config_sha256=evidence_digest(loaded.config.model_dump(mode="json")),
    )
    if mismatch in ("manifest", "config"):
        spec[mismatch + "_sha256"] = "a" * 64
    if mismatch == "candle":
        spec.update(
            start=utc_timestamp(candle + pd.Timedelta(hours=1)),
            end=utc_timestamp(candle + pd.Timedelta(hours=2)),
        )
    monkeypatch.setattr(collection, "utc_now", lambda: candle)
    path = tmp_path / "actual.json"
    reg = collection.register_collection(path, spec)
    now = as_of + (pd.Timedelta(hours=1) if mismatch == "candle" else pd.Timedelta(0))
    monkeypatch.setattr(collection, "utc_now", lambda: now)
    monkeypatch.setattr(live, "utc_now", lambda: now)
    monkeypatch.setattr(
        collection, "read_market", lambda p: primary if p.name == "primary" else secondary
    )
    if mismatch:
        monkeypatch.setattr(
            live, "predict_probabilities", boom(AssertionError("must reject before prediction"))
        )
    result = collection.collect_paper_slot(
        path,
        registration_sha256=reg["registration_sha256"],
        candle=spec["start"],
        bundle=bundle,
        primary=Path("primary"),
        secondary={"kraken": Path("secondary")},
    )
    if mismatch:
        assert result["attempt"]["failure_code"] == "PLAN_MISMATCH"
        assert not collection.collection_signals_path(path).exists()
    else:
        assert result["attempt"]["status"] == "SUCCEEDED"
        assert (
            collection.collection_status(
                path, registration_sha256=reg["registration_sha256"], as_of=utc_timestamp(now)
            )["counts"]["SUCCEEDED"]
            == 1
        )


def test_cli_rejection_and_read_only_status(registered, monkeypatch):
    monkeypatch.setattr(collection, "generate_paper_signal", boom(DataQualityError()))
    path, digest, _ = registered
    args = ["--collection", str(path), "--registration-sha256", digest]
    result = CliRunner().invoke(
        app,
        [
            "paper-collect",
            *args,
            "--candle",
            "2026-10-01T12:00:00+00:00",
            "--bundle",
            "unused",
            "--primary",
            "unused",
        ],
    )
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["attempt"]["status"] == "FAILED"
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    result = CliRunner().invoke(
        app, ["paper-collection-status", *args, "--as-of", "2026-10-01T17:00:00+00:00"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["counts"]["FAILED"] == 1
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}
