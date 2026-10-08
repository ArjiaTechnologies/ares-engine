"""Restart reconciliation with real filesystem writes and synthetic inference."""

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from filelock import FileLock

import ares_engine.bundles as bundles
import ares_engine.live as live
import ares_engine.paper_log as journal
from ares_engine.config import load_config
from ares_engine.exceptions import AresError, BundleIntegrityError, DataQualityError
from ares_engine.features import build_features
from ares_engine.synthetic import make_synthetic_ohlcv
from ares_engine.utils import sha256_file


def record_input():
    return {
        "event": {
            "exchange": "coinbase",
            "symbol": "ETH/USD",
            "timeframe": "1h",
            "timestamp": "2026-10-01T12:00:00+00:00",
        },
        "evidence": {
            "manifest_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "primary_sha256": "3" * 64,
            "secondary_sha256": {"kraken": "4" * 64},
            "feature_window_sha256": "5" * 64,
            "scaled_window_sha256": "6" * 64,
            "probability": 0.7,
            "signal": 1,
            "close": 100.0,
            "data_quality_passed": True,
            "cross_venue_p95_bps": {"kraken": 0.0},
            "cross_venue_latest_bps": {"kraken": 0.0},
        },
        "observation": {
            "recorded_at": "2026-10-01T13:00:00+00:00",
            "as_of": "2026-10-01T13:00:00+00:00",
            "bundle": "bundle",
            "staleness_bars": 1.0,
            "secondary_staleness_bars": {"kraken": 1.0},
        },
    }


def test_exact_retry_preserves_original_receipt_and_bytes(tmp_path):
    path = tmp_path / "signals.jsonl"
    original = journal.record_paper_signal(path, **record_input())
    before = path.read_bytes()
    retry = record_input()
    retry["observation"].update(
        recorded_at="2026-10-01T12:59:00+00:00", bundle="alias", staleness_bars=1.5
    )
    replay = journal.record_paper_signal(path, **retry)
    assert original.status == "recorded"
    assert replay.status == "replayed"
    assert replay.record == original.record
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("probability", 0.71),
        ("signal", 0),
        ("close", 101.0),
        ("manifest_sha256", "a" * 64),
        ("config_sha256", "b" * 64),
        ("primary_sha256", "c" * 64),
        ("secondary_sha256", {"kraken": "d" * 64}),
        ("feature_window_sha256", "e" * 64),
        ("scaled_window_sha256", "f" * 64),
    ],
)
def test_conflicting_replay_preserves_history(tmp_path, field, value):
    path = tmp_path / "signals.jsonl"
    journal.record_paper_signal(path, **record_input())
    before = path.read_bytes()
    retry = record_input()
    retry["evidence"][field] = value
    with pytest.raises(AresError, match="Conflicting evidence"):
        journal.record_paper_signal(path, **retry)
    assert path.read_bytes() == before


@pytest.mark.parametrize("after_commit", [False, True])
def test_interrupted_replace_is_reconciled_after_restart(tmp_path, monkeypatch, after_commit):
    path = tmp_path / "signals.jsonl"
    real_replace = os.replace

    def interrupted(source, destination):
        if after_commit:
            real_replace(source, destination)
        raise OSError("simulated process failure at commit boundary")

    with monkeypatch.context() as patch:
        patch.setattr(journal.os, "replace", interrupted)
        with pytest.raises(AresError, match="reconcile"):
            journal.record_paper_signal(path, **record_input())
    committed_bytes = path.read_bytes() if path.exists() else None
    receipt = journal.record_paper_signal(path, **record_input())
    assert receipt.status == ("replayed" if after_commit else "recorded")
    assert len(path.read_bytes().splitlines()) == 1
    if after_commit:
        assert path.read_bytes() == committed_bytes


def test_failure_after_temp_file_fsync_leaves_previous_journal(tmp_path, monkeypatch):
    path = tmp_path / "signals.jsonl"
    journal.record_paper_signal(path, **record_input())
    before = path.read_bytes()
    next_input = record_input()
    next_input["event"]["timestamp"] = "2026-10-01T13:00:00+00:00"
    next_input["observation"].update(
        as_of="2026-10-01T14:00:00+00:00", recorded_at="2026-10-01T14:00:00+00:00"
    )
    real_fsync = os.fsync

    def synced_then_failed(fd):
        real_fsync(fd)
        raise OSError("simulated crash after flush")

    with monkeypatch.context() as patch:
        patch.setattr(journal.os, "fsync", synced_then_failed)
        with pytest.raises(AresError, match="reconcile"):
            journal.record_paper_signal(path, **next_input)
    assert path.read_bytes() == before
    assert journal.record_paper_signal(path, **next_input).status == "recorded"


@pytest.mark.parametrize(
    "damage",
    [
        "truncated",
        "no-newline",
        "utf8",
        "duplicate-key",
        "nan",
        "infinity",
        "legacy",
        "duplicate-event",
        "digest",
        "extra-field",
        "bool-signal",
    ],
)
def test_damaged_or_legacy_history_is_never_repaired(tmp_path, damage):
    path = tmp_path / "signals.jsonl"
    journal.record_paper_signal(path, **record_input())
    raw = path.read_bytes()
    variants = {
        "truncated": raw + b'{"schema_version":',
        "no-newline": raw[:-1],
        "utf8": raw + b"\xff\n",
        "duplicate-key": raw.replace(b'"signal":1', b'"signal":1,"signal":1'),
        "nan": raw.replace(b'"probability":0.7', b'"probability":NaN'),
        "infinity": raw.replace(b'"probability":0.7', b'"probability":Infinity'),
        "legacy": b'{"timestamp":"2026-10-01","signal":1}\n',
        "duplicate-event": raw + raw,
        "digest": raw.replace(b'"signal":1', b'"signal":0'),
        "extra-field": raw.replace(b'"signal":1', b'"signal":1,"quantity":2'),
        "bool-signal": raw.replace(b'"signal":1', b'"signal":true'),
    }
    path.write_bytes(variants[damage])
    with pytest.raises(AresError):
        journal.record_paper_signal(path, **record_input())
    assert path.read_bytes() == variants[damage]


def test_unseen_older_candle_and_regressed_clock_are_blocked(tmp_path):
    path = tmp_path / "signals.jsonl"
    journal.record_paper_signal(path, **record_input())
    before = path.read_bytes()
    older = record_input()
    older["event"]["timestamp"] = "2026-10-01T11:00:00+00:00"
    with pytest.raises(AresError, match="high-water"):
        journal.record_paper_signal(path, **older)
    newer = record_input()
    newer["event"]["timestamp"] = "2026-10-01T13:00:00+00:00"
    newer["observation"].update(
        as_of="2026-10-01T14:00:00+00:00", recorded_at="2026-10-01T12:59:00+00:00"
    )
    with pytest.raises(AresError, match="clock regressed"):
        journal.record_paper_signal(path, **newer)
    assert path.read_bytes() == before


def test_old_committed_event_replays_after_newer_candle(tmp_path):
    path = tmp_path / "signals.jsonl"
    first = journal.record_paper_signal(path, **record_input())
    newer = record_input()
    newer["event"]["timestamp"] = "2026-10-01T13:00:00+00:00"
    newer["observation"].update(
        as_of="2026-10-01T14:00:00+00:00", recorded_at="2026-10-01T14:00:00+00:00"
    )
    journal.record_paper_signal(path, **newer)
    before = path.read_bytes()
    replay = journal.record_paper_signal(path, **record_input())
    assert replay.status == "replayed" and replay.record == first.record
    assert path.read_bytes() == before
    assert len(before.splitlines()) == 2


def test_append_crossing_size_bound_preserves_prior_event(tmp_path, monkeypatch):
    path = tmp_path / "signals.jsonl"
    journal.record_paper_signal(path, **record_input())
    before = path.read_bytes()
    monkeypatch.setattr(journal, "MAX_LOG_BYTES", len(before) + 10)
    newer = record_input()
    newer["event"]["timestamp"] = "2026-10-01T13:00:00+00:00"
    newer["observation"].update(
        as_of="2026-10-01T14:00:00+00:00", recorded_at="2026-10-01T14:00:00+00:00"
    )
    with pytest.raises(AresError, match="bound"):
        journal.record_paper_signal(path, **newer)
    assert path.read_bytes() == before


@pytest.mark.skipif(os.name != "posix", reason="Directory fsync is a POSIX capability")
def test_directory_fsync_failure_after_replace_reconciles(tmp_path, monkeypatch):
    path = tmp_path / "signals.jsonl"
    real_fsync = os.fsync
    calls = 0

    def interrupted(fd):
        nonlocal calls
        calls += 1
        real_fsync(fd)
        if calls == 2:
            raise OSError("directory synchronized but acknowledgement lost")

    with monkeypatch.context() as patch:
        patch.setattr(journal.os, "fsync", interrupted)
        with pytest.raises(AresError, match="reconcile"):
            journal.record_paper_signal(path, **record_input())
    before = path.read_bytes()
    assert journal.record_paper_signal(path, **record_input()).status == "replayed"
    assert path.read_bytes() == before


def test_bound_and_lock_contention_preserve_history(tmp_path, monkeypatch):
    path = tmp_path / "signals.jsonl"
    journal.record_paper_signal(path, **record_input())
    before = path.read_bytes()
    with FileLock(str(path) + ".lock"), pytest.raises(AresError, match="Another paper process"):
        journal.record_paper_signal(path, **record_input())
    monkeypatch.setattr(journal, "MAX_LOG_BYTES", len(before) - 1)
    with pytest.raises(AresError, match="bound"):
        journal.record_paper_signal(path, **record_input())
    assert path.read_bytes() == before


@pytest.mark.parametrize("target", ["journal", "lock"])
def test_hardlinks_are_rejected_without_touching_target(tmp_path, target):
    original = tmp_path / "original"
    original.write_bytes(b"retained")
    path = tmp_path / "signals.jsonl"
    linked = path if target == "journal" else Path(str(path) + ".lock")
    os.link(original, linked)
    with pytest.raises(AresError, match="unlinked"):
        journal.record_paper_signal(path, **record_input())
    assert original.read_bytes() == b"retained"


@pytest.mark.parametrize(
    "name", ["signals.jsonl.", "signals.jsonl ", "signals.jsonl:stream", "NUL", "COM1.log"]
)
def test_windows_aliases_cannot_bypass_lock(tmp_path, name):
    path = tmp_path / "signals.jsonl"
    journal.record_paper_signal(path, **record_input())
    before = path.read_bytes()
    with FileLock(str(path) + ".lock"), pytest.raises(AresError, match="Ambiguous"):
        journal.record_paper_signal(tmp_path / name, **record_input())
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeframe", "1H"),
        ("timeframe", " 1h "),
        ("exchange", "Coinbase"),
        ("exchange", "coinbase "),
    ],
)
def test_market_aliases_cannot_create_distinct_events(tmp_path, field, value):
    path = tmp_path / "signals.jsonl"
    journal.record_paper_signal(path, **record_input())
    before = path.read_bytes()
    candidate = record_input()
    candidate["event"][field] = value
    with pytest.raises(AresError, match="canonical lowercase"):
        journal.record_paper_signal(path, **candidate)
    assert path.read_bytes() == before


def _process_record(path):
    try:
        return journal.record_paper_signal(Path(path), **record_input()).status
    except AresError as exc:
        if "Another paper process" not in str(exc):
            raise
        return "contended"


def _process_crash(path, after_commit):
    real_replace = os.replace

    def crash(source, destination):
        if after_commit:
            real_replace(source, destination)
        os._exit(79)

    journal.os.replace = crash
    journal.record_paper_signal(Path(path), **record_input())


@pytest.mark.parametrize("after_commit", [False, True])
def test_abrupt_writer_death_then_fresh_process_retry(tmp_path, after_commit):
    path = tmp_path / "signals.jsonl"
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=_process_crash, args=(str(path), after_commit))
    child.start()
    child.join(timeout=30)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail("Crash worker did not terminate")
    assert child.exitcode == 79
    before = path.read_bytes() if path.exists() else None
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        result = pool.submit(_process_record, str(path)).result(timeout=30)
    assert result == ("replayed" if after_commit else "recorded")
    assert len(path.read_bytes().splitlines()) == 1
    if after_commit:
        assert path.read_bytes() == before


def test_two_processes_never_duplicate_an_event(tmp_path):
    path = tmp_path / "signals.jsonl"
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        statuses = list(pool.map(_process_record, [str(path), str(path)]))
    assert statuses.count("recorded") == 1
    assert set(statuses) <= {"recorded", "replayed", "contended"}
    assert journal.record_paper_signal(path, **record_input()).status == "replayed"
    assert len(path.read_bytes().splitlines()) == 1


class IdentityScaler:
    def __init__(self, count):
        self.n_features_in_ = count

    def transform(self, values):
        return values


@pytest.fixture
def inference_env(tmp_path, monkeypatch):
    config = load_config("configs/smoke.yaml")
    primary = make_synthetic_ohlcv(300, seed=36, exchange="coinbase")
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    columns = build_features(primary, config.features).columns
    model = SimpleNamespace(
        input_shape=(None, config.model.lookback_bars, len(columns)),
        output_shape=(None, 1),
        save=lambda path: path.write_bytes(b"synthetic model placeholder"),
    )
    bundle_path = bundles.export_bundle(
        model=model,
        scaler=IdentityScaler(len(columns)),
        feature_columns=columns,
        config=config,
        metrics={},
        provenance={"synthetic": True},
        artifacts_root=tmp_path / "artifacts",
        name="fixture",
    )
    monkeypatch.setattr(
        "ares_engine.models.keras_api",
        lambda: SimpleNamespace(models=SimpleNamespace(load_model=lambda path: model)),
    )
    monkeypatch.setattr(live, "predict_probabilities", lambda model, values: np.asarray([0.7]))
    as_of = pd.Timestamp(primary.timestamp.max()) + pd.Timedelta(hours=1)
    return bundle_path, primary, secondary, as_of


def test_inference_retry_binds_loaded_manifest_and_owned_inputs(
    inference_env, tmp_path, monkeypatch
):
    bundle, primary, secondary, as_of = inference_env
    original_primary = primary.copy()
    original_secondary = secondary.copy()
    expected_manifest = sha256_file(bundle / "manifest.json")
    expected_primary = live._frame_digest(primary)
    expected_secondary = live._frame_digest(secondary)

    def mutate_during_prediction(model, values):
        (bundle / "manifest.json").write_text("{}", encoding="utf-8")
        primary.loc[0, "volume"] += 100
        secondary.loc[0, "volume"] += 100
        return np.asarray([0.7])

    captured_manifest = (bundle / "manifest.json").read_bytes()
    monkeypatch.setattr(live, "predict_probabilities", mutate_during_prediction)
    path = tmp_path / "signals.jsonl"
    first = live.generate_paper_signal(
        bundle, primary, secondary_ohlcv=secondary, as_of=as_of, log_path=path
    )
    evidence = first.recording.record["evidence"]
    assert evidence["manifest_sha256"] == expected_manifest
    assert evidence["primary_sha256"] == expected_primary
    assert evidence["secondary_sha256"]["kraken"] == expected_secondary
    before = path.read_bytes()
    (bundle / "manifest.json").write_bytes(captured_manifest)
    monkeypatch.setattr(live, "predict_probabilities", lambda model, values: np.asarray([0.7]))
    second = live.generate_paper_signal(
        bundle,
        original_primary,
        secondary_ohlcv=original_secondary,
        as_of=as_of + pd.Timedelta(minutes=15),
        log_path=path,
    )
    assert second.recording.status == "replayed"
    assert second.recording.record == first.recording.record
    assert path.read_bytes() == before
    assert not {"order", "quantity", "order_id"} & set(evidence)


def test_manifest_change_before_verification_fails_closed(inference_env, monkeypatch):
    bundle, _, _, _ = inference_env
    real_verify = bundles.verify_bundle

    def changed_manifest(path):
        manifest_path = path / "manifest.json"
        content = json.loads(manifest_path.read_text())
        content["created_at"] = "changed"
        manifest_path.write_text(json.dumps(content), encoding="utf-8")
        return real_verify(path)

    monkeypatch.setattr(bundles, "verify_bundle", changed_manifest)
    with pytest.raises(BundleIntegrityError, match="capture and verification"):
        bundles.load_bundle(bundle)


def test_manifest_capture_remains_bound_after_verification(inference_env, monkeypatch):
    bundle, _, _, _ = inference_env
    expected = sha256_file(bundle / "manifest.json")
    real_verify = bundles.verify_bundle

    def changed_after_verify(path):
        verified = real_verify(path)
        content = {**verified, "created_at": "changed after verification"}
        (path / "manifest.json").write_text(json.dumps(content), encoding="utf-8")
        return verified

    monkeypatch.setattr(bundles, "verify_bundle", changed_after_verify)
    loaded = bundles.load_bundle(bundle)
    assert loaded.manifest_sha256 == expected
    assert loaded.manifest_sha256 != sha256_file(bundle / "manifest.json")


@pytest.mark.parametrize("changed", ["primary", "secondary"])
def test_changed_valid_input_conflicts_even_with_identical_prediction(
    inference_env, tmp_path, changed
):
    bundle, primary, secondary, as_of = inference_env
    path = tmp_path / "signals.jsonl"
    live.generate_paper_signal(
        bundle, primary, secondary_ohlcv=secondary, as_of=as_of, log_path=path
    )
    before = path.read_bytes()
    frame = primary if changed == "primary" else secondary
    frame.loc[0, "volume"] += 1
    with pytest.raises(AresError, match="Conflicting evidence"):
        live.generate_paper_signal(
            bundle, primary, secondary_ohlcv=secondary, as_of=as_of, log_path=path
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "failure", ["missing", "stale", "future", "divergent", "invalid-probability"]
)
def test_quality_and_prediction_failures_never_change_history(
    inference_env, tmp_path, monkeypatch, failure
):
    bundle, primary, secondary, as_of = inference_env
    path = tmp_path / "signals.jsonl"
    live.generate_paper_signal(
        bundle, primary, secondary_ohlcv=secondary, as_of=as_of, log_path=path
    )
    before = path.read_bytes()
    if failure == "missing":
        secondary = None
    elif failure == "stale":
        as_of += pd.Timedelta(days=1)
    elif failure == "future":
        as_of -= pd.Timedelta(minutes=1)
    elif failure == "divergent":
        secondary = secondary.copy()
        secondary[["open", "high", "low", "close"]] *= 2
    else:
        monkeypatch.setattr(
            live, "predict_probabilities", lambda model, values: np.asarray([np.nan])
        )
    with pytest.raises((DataQualityError, AresError)):
        live.generate_paper_signal(
            bundle, primary, secondary_ohlcv=secondary, as_of=as_of, log_path=path
        )
    assert path.read_bytes() == before
