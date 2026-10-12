"""Preserved raw inputs, partial acquisition evidence, and non-retryable crashes."""

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_paper_recording import inference_env as inference_env
from test_paper_recording import record_input
from test_paper_report import plan
from typer.testing import CliRunner

import ares_engine.live as live
import ares_engine.paper_collection as collection
import ares_engine.paper_inputs as inputs
from ares_engine.bundles import load_bundle
from ares_engine.cli import app
from ares_engine.data.storage import read_market
from ares_engine.exceptions import AresError
from ares_engine.paper_log import (
    evidence_digest,
    read_paper_journal,
    record_paper_signal,
    utc_timestamp,
)

SOURCE_FIELDS = {
    "role",
    "exchange",
    "status",
    "started_at",
    "captured_at",
    "finished_at",
    "raw_sha256",
    "raw_bytes",
    "frame_sha256",
}


def fail_if_called(*args, **kwargs):
    raise AssertionError("An admitted attempt must not dispatch again")


def canonical_digest(value):
    """Independent old-envelope digest, without current Pydantic defaults."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_bytes(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def capture_env(inference_env, tmp_path, monkeypatch):
    bundle, primary_frame, secondary_frame, as_of = inference_env
    loaded = load_bundle(bundle)
    candle = as_of - pd.Timedelta(hours=1)
    clock = [candle]
    monkeypatch.setattr(collection, "utc_now", lambda: clock[0])
    monkeypatch.setattr(live, "utc_now", lambda: clock[0])
    path = tmp_path / "capture.json"
    spec = plan(
        start=utc_timestamp(candle),
        end=utc_timestamp(candle + pd.Timedelta(hours=2)),
        manifest_sha256=loaded.manifest_sha256,
        config_sha256=evidence_digest(loaded.config.model_dump(mode="json")),
    )
    registration = collection.register_collection(path, spec)
    primary = tmp_path / "supplied-primary.parquet"
    secondary = tmp_path / "supplied-secondary.parquet"
    # read_market sorts and normalizes; a rewritten frame is not the original file.
    raw_primary = primary_frame.iloc[::-1].copy()
    raw_primary["audit_only"] = "preserve this extra column in raw bytes"
    raw_primary.to_parquet(primary, index=False, compression="gzip")
    secondary_frame.to_parquet(secondary, index=False, compression="snappy")
    clock[0] = as_of
    return SimpleNamespace(
        path=path,
        registration=registration["registration_sha256"],
        candle=utc_timestamp(candle),
        next_candle=utc_timestamp(candle + pd.Timedelta(hours=1)),
        bundle=bundle,
        primary=primary,
        secondary=secondary,
        primary_frame=primary_frame,
        secondary_frame=secondary_frame,
        clock=clock,
        as_of=as_of,
    )


def collect(env, **changes):
    arguments = {
        "registration_sha256": env.registration,
        "candle": env.candle,
        "bundle": env.bundle,
        "primary": env.primary,
        "secondary": {"kraken": env.secondary},
        "capture_inputs": True,
        **changes,
    }
    return collection.collect_paper_slot(env.path, **arguments)


def status(env, cutoff=None):
    return collection.collection_status(
        env.path,
        registration_sha256=env.registration,
        as_of=utc_timestamp(env.clock[0]) if cutoff is None else cutoff,
    )


def saved_attempt(env):
    return json.loads(env.path.read_bytes())["attempts"][0]


def snapshot(env, index):
    return inputs.snapshot_path(env.path, saved_attempt(env)["attempt_id"], index)


def test_real_inference_binds_exact_raw_bytes_and_normalized_frame_identity(capture_env):
    env = capture_env
    original = [env.primary.read_bytes(), env.secondary.read_bytes()]
    os.utime(env.primary, (0, 0))
    result = collect(env)
    attempt = result["attempt"]
    assert attempt["status"] == "SUCCEEDED"
    capture = attempt["input_capture"]
    assert capture["schema_version"] == "ares-local-input-capture-v1"
    sources = capture["sources"]
    assert [(source["role"], source["exchange"]) for source in sources] == [
        ("PRIMARY", "coinbase"),
        ("SECONDARY", "kraken"),
    ]
    _, records = read_paper_journal(collection.collection_signals_path(env.path))
    receipt = records[0]
    expected_frames = [env.primary_frame, env.secondary_frame]
    receipt_hashes = [
        receipt["evidence"]["primary_sha256"],
        receipt["evidence"]["secondary_sha256"]["kraken"],
    ]
    for index, source in enumerate(sources):
        assert set(source) == SOURCE_FIELDS
        assert source["status"] == "READY"
        assert snapshot(env, index).read_bytes() == original[index]
        assert source["raw_sha256"] == hashlib.sha256(original[index]).hexdigest()
        assert source["raw_bytes"] == len(original[index])
        assert source["frame_sha256"] == live._frame_digest(expected_frames[index])
        assert source["frame_sha256"] == receipt_hashes[index]
        assert source["frame_sha256"] == live._frame_digest(read_market(snapshot(env, index)))
        assert source["raw_sha256"] != source["frame_sha256"]
        assert (
            pd.Timestamp(attempt["started_at"])
            <= pd.Timestamp(source["started_at"])
            <= pd.Timestamp(source["captured_at"])
            <= pd.Timestamp(source["finished_at"])
            <= pd.Timestamp(receipt["observation"]["recorded_at"])
            <= pd.Timestamp(attempt["finished_at"])
        )
    raw_frame = pd.read_parquet(snapshot(env, 0))
    assert "audit_only" in raw_frame
    assert raw_frame.timestamp.is_monotonic_decreasing
    assert sources[0]["started_at"] == utc_timestamp(env.as_of)
    assert status(env)["slots"][0]["input_capture"] == capture


def test_replaced_originals_cannot_change_inference_or_replay(capture_env, monkeypatch):
    env = capture_env
    original = [env.primary.read_bytes(), env.secondary.read_bytes()]
    real_generate = collection.generate_paper_signal

    def replace_originals_then_infer(bundle, frame, **kwargs):
        attempt = saved_attempt(env)
        assert [item["status"] for item in attempt["input_capture"]["sources"]] == [
            "READY",
            "READY",
        ]
        assert snapshot(env, 0).read_bytes() == original[0]
        assert snapshot(env, 1).read_bytes() == original[1]
        env.primary.write_bytes(b"source replaced after capture")
        env.secondary.unlink()
        assert live._frame_digest(frame) == live._frame_digest(env.primary_frame)
        assert live._frame_digest(kwargs["secondary_ohlcv"]["kraken"]) == live._frame_digest(
            env.secondary_frame
        )
        return real_generate(bundle, frame, **kwargs)

    monkeypatch.setattr(collection, "generate_paper_signal", replace_originals_then_infer)
    assert collect(env)["attempt"]["status"] == "SUCCEEDED"
    before = file_bytes(env.path.parent)
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    assert status(env)["counts"]["SUCCEEDED"] == 1
    assert collect(env)["dispatch"] == "REPLAY_NO_DISPATCH"
    assert file_bytes(env.path.parent) == before


def test_partial_malformed_secondary_preserves_raw_and_completed_primary(capture_env, monkeypatch):
    env = capture_env
    pd.DataFrame({"not_timestamp": [1, 2]}).to_parquet(env.secondary, index=False)
    raw_primary, raw_bad = env.primary.read_bytes(), env.secondary.read_bytes()
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    result = collect(env, secondary={"kraken": env.secondary, "zeta": env.primary})
    attempt = result["attempt"]
    assert attempt["status"] == "FAILED"
    assert attempt["failure_code"] == "LOCAL_INPUT_REJECTED"
    sources = attempt["input_capture"]["sources"]
    assert [item["status"] for item in sources] == ["READY", "PARSE_REJECTED", "PENDING"]
    assert snapshot(env, 0).read_bytes() == raw_primary
    assert snapshot(env, 1).read_bytes() == raw_bad
    assert not snapshot(env, 2).exists()
    assert sources[0]["frame_sha256"] == live._frame_digest(env.primary_frame)
    assert sources[1]["raw_sha256"] == hashlib.sha256(raw_bad).hexdigest()
    assert sources[1]["raw_bytes"] == len(raw_bad)
    assert sources[1]["captured_at"] is not None
    assert sources[1]["finished_at"] is not None
    assert sources[1]["frame_sha256"] is None
    for key in SOURCE_FIELDS - {"role", "exchange", "status"}:
        assert sources[2][key] is None
    before = file_bytes(env.path.parent)
    assert status(env)["counts"]["FAILED"] == 1
    assert collect(env)["dispatch"] == "REPLAY_NO_DISPATCH"
    assert file_bytes(env.path.parent) == before


def test_missing_requested_file_is_explicit_without_fabricated_snapshot(capture_env, monkeypatch):
    env = capture_env
    env.primary.unlink()
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    attempt = collect(env)["attempt"]
    assert attempt["status"] == "FAILED"
    assert attempt["failure_code"] == "LOCAL_INPUT_REJECTED"
    sources = attempt["input_capture"]["sources"]
    assert [item["status"] for item in sources] == ["READ_REJECTED", "PENDING"]
    assert sources[0]["started_at"] is not None
    assert sources[0]["finished_at"] is not None
    for key in ("captured_at", "raw_sha256", "raw_bytes", "frame_sha256"):
        assert sources[0][key] is None
    assert not snapshot(env, 0).exists()
    assert not snapshot(env, 1).exists()


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_damaged_capture_blocks_status_replay_and_new_dispatch(
    capture_env, monkeypatch, index, damage
):
    env = capture_env
    assert collect(env)["attempt"]["status"] == "SUCCEEDED"
    target = snapshot(env, index)
    if damage == "missing":
        target.unlink()
    else:
        data = target.read_bytes()
        target.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    before = file_bytes(env.path.parent)
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    with pytest.raises(AresError):
        status(env)
    with pytest.raises(AresError):
        collect(env)
    env.clock[0] += pd.Timedelta(hours=1)
    with pytest.raises(AresError):
        collect(env, candle=env.next_candle)
    assert file_bytes(env.path.parent) == before


def test_failed_attempt_also_requires_its_acknowledged_raw_snapshot(capture_env, monkeypatch):
    env = capture_env
    pd.DataFrame({"bad": [1]}).to_parquet(env.secondary, index=False)
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    assert collect(env)["attempt"]["status"] == "FAILED"
    snapshot(env, 1).unlink()
    before = file_bytes(env.path.parent)
    with pytest.raises(AresError):
        status(env)
    with pytest.raises(AresError):
        collect(env)
    assert file_bytes(env.path.parent) == before


def test_success_receipt_must_match_captured_normalized_frames(capture_env, monkeypatch):
    env = capture_env

    def mismatched_receipt(bundle, frame, *, expected, log_path, as_of, secondary_ohlcv, **kwargs):
        data = record_input()
        data["event"] = expected.event.model_dump()
        data["evidence"].update(
            manifest_sha256=expected.manifest_sha256,
            config_sha256=expected.config_sha256,
            primary_sha256="f" * 64,
            secondary_sha256={"kraken": live._frame_digest(secondary_ohlcv["kraken"])},
        )
        data["observation"].update(
            as_of=utc_timestamp(as_of), recorded_at=utc_timestamp(env.clock[0])
        )
        return SimpleNamespace(recording=record_paper_signal(log_path, **data))

    monkeypatch.setattr(collection, "generate_paper_signal", mismatched_receipt)
    with pytest.raises(AresError):
        collect(env)
    assert saved_attempt(env)["status"] == "STARTED"
    assert saved_attempt(env)["record_sha256"] is None
    before = file_bytes(env.path.parent)
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    assert collect(env)["dispatch"] == "REPLAY_NO_DISPATCH"
    assert file_bytes(env.path.parent) == before


def test_initial_receipt_cannot_predate_ready_input_capture(capture_env, monkeypatch):
    env = capture_env

    def advancing_clock():
        value = env.clock[0]
        env.clock[0] += pd.Timedelta(seconds=1)
        return value

    def early_receipt(bundle, frame, *, expected, log_path, as_of, secondary_ohlcv, **kwargs):
        attempt = saved_attempt(env)
        assert all(
            pd.Timestamp(source["finished_at"]) > pd.Timestamp(attempt["started_at"])
            for source in attempt["input_capture"]["sources"]
        )
        data = record_input()
        data["event"] = expected.event.model_dump()
        data["evidence"].update(
            manifest_sha256=expected.manifest_sha256,
            config_sha256=expected.config_sha256,
            primary_sha256=live._frame_digest(frame),
            secondary_sha256={
                exchange: live._frame_digest(other) for exchange, other in secondary_ohlcv.items()
            },
        )
        data["observation"].update(as_of=utc_timestamp(as_of), recorded_at=attempt["started_at"])
        return SimpleNamespace(recording=record_paper_signal(log_path, **data))

    monkeypatch.setattr(collection, "utc_now", advancing_clock)
    monkeypatch.setattr(collection, "generate_paper_signal", early_receipt)
    with pytest.raises(AresError):
        collect(env)
    attempt = saved_attempt(env)
    assert attempt["status"] == "STARTED"
    assert attempt["finished_at"] is None
    assert attempt["failure_code"] is None
    assert attempt["record_sha256"] is None
    _, records = read_paper_journal(collection.collection_signals_path(env.path))
    assert len(records) == 1
    before = file_bytes(env.path.parent)
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    assert collect(env)["dispatch"] == "REPLAY_NO_DISPATCH"
    assert status(env)["counts"]["STARTED_UNKNOWN"] == 1
    assert file_bytes(env.path.parent) == before


def test_saved_receipt_link_rechecks_capture_causal_order(capture_env, monkeypatch):
    env = capture_env

    def advancing_clock():
        value = env.clock[0]
        env.clock[0] += pd.Timedelta(seconds=1)
        return value

    monkeypatch.setattr(collection, "utc_now", advancing_clock)
    assert collect(env)["attempt"]["status"] == "SUCCEEDED"
    state = json.loads(env.path.read_bytes())
    attempt = state["attempts"][0]
    journal = collection.collection_signals_path(env.path)
    _, records = read_paper_journal(journal)
    record = records[0]
    # Keep every digest and the captured frame identities valid while violating
    # only the receipt's causal relationship to the completed input acquisition.
    record["observation"]["recorded_at"] = attempt["started_at"]
    record["record_sha256"] = canonical_digest(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    journal.write_bytes(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    assert read_paper_journal(journal)[1][0] == record
    attempt["record_sha256"] = record["record_sha256"]
    state["state_sha256"] = canonical_digest(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )
    env.path.write_bytes(json.dumps(state, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    before = file_bytes(env.path.parent)
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    with pytest.raises(AresError):
        status(env)
    with pytest.raises(AresError):
        collect(env)
    env.clock[0] += pd.Timedelta(hours=1)
    with pytest.raises(AresError):
        collect(env, candle=env.next_candle)
    assert file_bytes(env.path.parent) == before


@pytest.mark.parametrize("orphan_name", ["orphan.parquet", ".0.parquet.interrupted"])
def test_orphan_snapshot_and_atomic_temporary_bytes_count_toward_budget(
    capture_env, monkeypatch, orphan_name
):
    env = capture_env
    assert collect(env)["attempt"]["status"] == "SUCCEEDED"
    declared_bytes = sum(
        source["raw_bytes"] for source in saved_attempt(env)["input_capture"]["sources"]
    )
    orphan = snapshot(env, 0).parent / orphan_name
    orphan.write_bytes(b"published bytes whose completion was never acknowledged")
    # A bound equal to the acknowledged total would pass a metadata-only sum.
    monkeypatch.setattr(inputs, "MAX_CAPTURE_BYTES", declared_bytes)
    before = file_bytes(env.path.parent)
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    with pytest.raises(AresError, match="bound"):
        status(env)
    with pytest.raises(AresError, match="bound"):
        collect(env)
    env.clock[0] += pd.Timedelta(hours=1)
    with pytest.raises(AresError, match="bound"):
        collect(env, candle=env.next_candle)
    assert file_bytes(env.path.parent) == before


@pytest.mark.parametrize("after_commit", [False, True])
def test_uncertain_raw_publication_never_becomes_explicit_input_failure(
    capture_env, monkeypatch, after_commit
):
    env = capture_env
    real_replace = os.replace

    def interrupted(source, destination):
        if Path(destination).suffix == ".parquet":
            assert saved_attempt(env)["status"] == "STARTED"
            if after_commit:
                real_replace(source, destination)
            raise OSError("snapshot replacement acknowledgement lost")
        return real_replace(source, destination)

    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupted)
        with pytest.raises((OSError, AresError)):
            collect(env)
    attempt = saved_attempt(env)
    assert attempt["status"] == "STARTED"
    assert attempt["failure_code"] is None
    assert attempt["finished_at"] is None
    assert attempt["input_capture"]["sources"][0]["status"] == "STARTED"
    assert snapshot(env, 0).exists() is after_commit
    env.primary.unlink()
    before = file_bytes(env.path.parent)
    assert collect(env)["dispatch"] == "REPLAY_NO_DISPATCH"
    assert status(env)["counts"]["STARTED_UNKNOWN"] == 1
    assert file_bytes(env.path.parent) == before


@pytest.mark.parametrize("phase", ["CAPTURED", "READY"])
@pytest.mark.parametrize("after_commit", [False, True])
def test_uncertain_capture_checkpoint_never_resumes_or_infers(
    capture_env, monkeypatch, phase, after_commit
):
    env = capture_env
    real_replace = collection._replace_journal

    def interrupted(path, payload):
        state = json.loads(payload)
        attempt = state["attempts"][-1]
        if attempt["input_capture"]["sources"][0]["status"] == phase:
            if after_commit:
                real_replace(path, payload)
            raise OSError("capture checkpoint acknowledgement lost")
        return real_replace(path, payload)

    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    with monkeypatch.context() as patch:
        patch.setattr(collection, "_replace_journal", interrupted)
        with pytest.raises((OSError, AresError)):
            collect(env)
    attempt = saved_attempt(env)
    assert attempt["status"] == "STARTED"
    assert attempt["failure_code"] is None
    predecessor = "STARTED" if phase == "CAPTURED" else "CAPTURED"
    assert attempt["input_capture"]["sources"][0]["status"] == (
        phase if after_commit else predecessor
    )
    assert snapshot(env, 0).exists()
    before = file_bytes(env.path.parent)
    assert collect(env)["dispatch"] == "REPLAY_NO_DISPATCH"
    assert status(env)["counts"]["STARTED_UNKNOWN"] == 1
    assert file_bytes(env.path.parent) == before


def _crash_after_capture(path, digest, candle, bundle, primary, secondary, now):
    collection.utc_now = lambda: pd.Timestamp(now)

    def crash(*args, **kwargs):
        os._exit(83)

    collection.generate_paper_signal = crash
    collection.collect_paper_slot(
        Path(path),
        registration_sha256=digest,
        candle=candle,
        bundle=Path(bundle),
        primary=Path(primary),
        secondary={"kraken": Path(secondary)},
        capture_inputs=True,
    )


def test_real_process_death_with_ready_snapshots_does_not_authorize_inference(
    capture_env, monkeypatch
):
    env = capture_env
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_after_capture,
        args=(
            str(env.path),
            env.registration,
            env.candle,
            str(env.bundle),
            str(env.primary),
            str(env.secondary),
            utc_timestamp(env.clock[0]),
        ),
    )
    process.start()
    process.join(30)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("Capture crash process did not stop")
    assert process.exitcode == 83
    attempt = saved_attempt(env)
    assert attempt["status"] == "STARTED"
    assert [source["status"] for source in attempt["input_capture"]["sources"]] == [
        "READY",
        "READY",
    ]
    assert not collection.collection_signals_path(env.path).exists()
    before = file_bytes(env.path.parent)
    # Process death leaves an ephemeral OS-lock file. Recovery may remove it;
    # every durable collection, input, model and signal byte must stay unchanged.
    before.pop(env.path.name + ".lock", None)
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    assert collect(env)["dispatch"] == "REPLAY_NO_DISPATCH"
    assert status(env)["counts"]["STARTED_UNKNOWN"] == 1
    assert file_bytes(env.path.parent) == before


def test_oversized_source_is_rejected_before_raw_publication(capture_env, monkeypatch):
    env = capture_env
    env.primary.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    attempt = collect(env)["attempt"]
    assert attempt["status"] == "FAILED"
    assert attempt["failure_code"] == "LOCAL_INPUT_REJECTED"
    source = attempt["input_capture"]["sources"][0]
    assert source["status"] == "READ_REJECTED"
    assert source["raw_sha256"] is None
    assert not snapshot(env, 0).exists()


def test_hardlinked_source_is_rejected_without_capturing_it(capture_env, monkeypatch):
    env = capture_env
    alias = env.primary.with_name("primary-alias.parquet")
    os.link(env.primary, alias)
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    attempt = collect(env)["attempt"]
    assert attempt["status"] == "FAILED"
    assert attempt["input_capture"]["sources"][0]["status"] == "READ_REJECTED"
    assert not snapshot(env, 0).exists()


def test_hardlinked_snapshot_blocks_readback_without_repair(capture_env, monkeypatch):
    env = capture_env
    assert collect(env)["attempt"]["status"] == "SUCCEEDED"
    os.link(snapshot(env, 0), env.primary.with_name("snapshot-alias.parquet"))
    before = file_bytes(env.path.parent)
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    with pytest.raises(AresError):
        status(env)
    with pytest.raises(AresError):
        collect(env)
    assert file_bytes(env.path.parent) == before


def test_capture_clock_rollback_leaves_admitted_attempt_unknown(capture_env, monkeypatch):
    env = capture_env
    ticks = iter([env.clock[0], env.clock[0] - pd.Timedelta(seconds=1)])
    monkeypatch.setattr(collection, "utc_now", lambda: next(ticks))
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    with pytest.raises(AresError, match="clock regressed"):
        collect(env)
    attempt = saved_attempt(env)
    assert attempt["status"] == "STARTED"
    assert all(s["status"] == "PENDING" for s in attempt["input_capture"]["sources"])
    assert not snapshot(env, 0).exists()
    before = file_bytes(env.path.parent)
    monkeypatch.setattr(collection, "utc_now", lambda: env.clock[0])
    assert collect(env)["dispatch"] == "REPLAY_NO_DISPATCH"
    assert status(env)["counts"]["STARTED_UNKNOWN"] == 1
    assert file_bytes(env.path.parent) == before


def test_historical_cutoff_hides_future_capture_completion(capture_env, monkeypatch):
    env = capture_env

    def advancing_clock():
        value = env.clock[0]
        env.clock[0] += pd.Timedelta(seconds=1)
        return value

    monkeypatch.setattr(collection, "utc_now", advancing_clock)
    attempt = collect(env)["attempt"]
    cutoff = attempt["started_at"]
    first = attempt["input_capture"]["sources"][0]
    assert pd.Timestamp(first["captured_at"]) > pd.Timestamp(cutoff)
    before = file_bytes(env.path.parent)
    earlier = status(env, cutoff)
    assert earlier["counts"]["STARTED_UNKNOWN"] == 1
    projected = earlier["slots"][0]["input_capture"]
    for source in projected["sources"]:
        assert source["status"] in {"PENDING", "STARTED"}
        for key in ("captured_at", "finished_at", "raw_sha256", "raw_bytes", "frame_sha256"):
            assert source[key] is None
    assert file_bytes(env.path.parent) == before


def literal_legacy_state(terminal):
    """The pre-capture V1 shape, independent of all new model serialization."""
    registration = json.loads(
        """{
          "schema_version":"ares-paper-registration-v1",
          "source_mode":"OFFLINE_SUPPLIED_FILES",
          "created_at":"2026-10-01T12:00:00+00:00",
          "plan":{
            "schema_version":"ares-paper-plan-v1",
            "exchange":"coinbase", "symbol":"ETH/USD", "timeframe":"1h",
            "start":"2026-10-01T12:00:00+00:00",
            "end":"2026-10-01T14:00:00+00:00",
            "horizon_bars":1,
            "manifest_sha256":"1111111111111111111111111111111111111111111111111111111111111111",
            "config_sha256":"2222222222222222222222222222222222222222222222222222222222222222",
            "max_recording_delay_seconds":300
          }
        }"""
    )
    registration["plan_sha256"] = canonical_digest(registration["plan"])
    registration["registration_sha256"] = canonical_digest(registration)
    attempt = {
        "candle": "2026-10-01T12:00:00+00:00",
        "attempt_id": canonical_digest(
            {
                "registration_sha256": registration["registration_sha256"],
                "candle": "2026-10-01T12:00:00+00:00",
            }
        ),
        "started_at": "2026-10-01T13:00:00+00:00",
        "status": terminal,
        "finished_at": "2026-10-01T13:00:00+00:00" if terminal == "FAILED" else None,
        "failure_code": "DATA_QUALITY_REJECTED" if terminal == "FAILED" else None,
        "record_sha256": None,
    }
    state = {
        "schema_version": "ares-paper-collection-v1",
        "registration": registration,
        "attempts": [attempt],
    }
    state["state_sha256"] = canonical_digest(state)
    return state


@pytest.mark.parametrize("terminal", ["STARTED", "FAILED"])
def test_literal_legacy_attempt_retains_hash_and_unobserved_capture(
    tmp_path, monkeypatch, terminal
):
    state = literal_legacy_state(terminal)
    path = tmp_path / "legacy.json"
    path.write_bytes(json.dumps(state, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    original = path.read_bytes()
    digest = state["registration"]["registration_sha256"]
    monkeypatch.setattr(collection, "generate_paper_signal", fail_if_called)
    report = collection.collection_status(
        path, registration_sha256=digest, as_of="2026-10-01T13:00:00+00:00"
    )
    assert report["state_sha256"] == state["state_sha256"]
    assert report["slots"][0]["input_capture"] is None
    result = collection.collect_paper_slot(
        path,
        registration_sha256=digest,
        candle="2026-10-01T12:00:00+00:00",
        bundle=Path("never-read"),
        primary=Path("never-read"),
        capture_inputs=True,
    )
    assert result["dispatch"] == "REPLAY_NO_DISPATCH"
    assert result["attempt"]["status"] == terminal
    assert "input_capture" not in result["attempt"]
    assert path.read_bytes() == original
    assert not path.with_name(path.name + ".inputs").exists()


def test_default_attempt_does_not_silently_enable_capture(capture_env):
    env = capture_env
    attempt = collect(env, capture_inputs=False)["attempt"]
    assert attempt["status"] == "SUCCEEDED"
    assert "input_capture" not in attempt
    assert "input_capture" not in saved_attempt(env)
    assert status(env)["slots"][0]["input_capture"] is None
    assert not env.path.with_name(env.path.name + ".inputs").exists()


def test_cli_capture_inputs_flag_uses_preserved_sources(capture_env):
    env = capture_env
    result = CliRunner().invoke(
        app,
        [
            "paper-collect",
            "--collection",
            str(env.path),
            "--registration-sha256",
            env.registration,
            "--candle",
            env.candle,
            "--bundle",
            str(env.bundle),
            "--primary",
            str(env.primary),
            "--secondary",
            f"kraken={env.secondary}",
            "--capture-inputs",
        ],
    )
    assert result.exit_code == 0, result.output
    attempt = json.loads(result.stdout)["attempt"]
    assert attempt["status"] == "SUCCEEDED"
    assert [source["status"] for source in attempt["input_capture"]["sources"]] == [
        "READY",
        "READY",
    ]
    assert snapshot(env, 0).read_bytes() == env.primary.read_bytes()
