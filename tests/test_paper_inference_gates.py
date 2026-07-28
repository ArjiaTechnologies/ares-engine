"""Paper inference fail-closed battery with a real exported bundle."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from filelock import FileLock
from test_bundle_security import _training_config  # shared real-bundle recipe

import ares_engine.live as live_module
from ares_engine.bundles import load_bundle
from ares_engine.exceptions import AresError, DataQualityError
from ares_engine.live import generate_paper_signal
from ares_engine.synthetic import make_synthetic_ohlcv
from ares_engine.training import train_candidate
from ares_engine.utils import atomic_write_json, sha256_file

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def paper_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("paper-fixture")
    config = _training_config(root)
    frame = make_synthetic_ohlcv(430, seed=52, exchange="coinbase")
    bundle, _ = train_candidate(frame, config, bundle_name="paper_bundle")
    secondary = frame.copy()
    secondary["exchange"] = "kraken"
    as_of = pd.Timestamp(frame["timestamp"].max()) + pd.Timedelta(hours=1)
    return {"bundle": bundle, "frame": frame, "secondary": secondary, "as_of": as_of}


def _run(paper_env, primary=None, secondary=None, as_of=None, log_path=None):
    return generate_paper_signal(
        paper_env["bundle"],
        paper_env["frame"] if primary is None else primary,
        secondary_ohlcv=paper_env["secondary"] if secondary is None else secondary,
        as_of=paper_env["as_of"] if as_of is None else as_of,
        log_path=log_path,
    )


def test_happy_path_emits_read_only_signal_from_newest_completed_candle(
    paper_env, tmp_path
) -> None:
    log_path = tmp_path / "signals.jsonl"
    result = _run(paper_env, log_path=log_path)
    assert result.signal in (-1, 0, 1)
    assert 0.0 <= result.probability <= 1.0
    assert result.timestamp == pd.Timestamp(paper_env["frame"]["timestamp"].max())
    assert result.data_quality_passed is True
    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["signal"] == result.signal
    assert "manifest_sha256" in payload
    # Read-only: nothing in the payload resembles an order instruction.
    assert not {"order", "order_id", "size", "quantity", "venue_order"} & set(payload)


def test_duplicated_timestamps_are_refused(paper_env) -> None:
    primary = pd.concat([paper_env["frame"], paper_env["frame"].iloc[[-1]]], ignore_index=True)
    with pytest.raises(DataQualityError, match="Duplicate"):
        _run(paper_env, primary=primary)


def test_out_of_order_timestamps_are_refused(paper_env) -> None:
    primary = paper_env["frame"].iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataQualityError, match="monotonically|Duplicate|unsorted"):
        _run(paper_env, primary=primary)


def test_in_progress_candle_in_primary_feed_is_refused(paper_env) -> None:
    frame = paper_env["frame"]
    step = pd.Timedelta(hours=1)
    open_candle = frame.iloc[[-1]].copy()
    open_candle["timestamp"] = frame["timestamp"].max() + step
    primary = pd.concat([frame, open_candle], ignore_index=True)
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    as_of = pd.Timestamp(frame["timestamp"].max()) + step + pd.Timedelta(minutes=30)
    with pytest.raises(DataQualityError, match="in-progress"):
        _run(paper_env, primary=primary, secondary=secondary, as_of=as_of)


def test_in_progress_candle_in_secondary_feed_is_refused(paper_env) -> None:
    frame = paper_env["frame"]
    step = pd.Timedelta(hours=1)
    secondary = paper_env["secondary"]
    open_candle = secondary.iloc[[-1]].copy()
    open_candle["timestamp"] = secondary["timestamp"].max() + step
    secondary_open = pd.concat([secondary, open_candle], ignore_index=True)
    as_of = pd.Timestamp(frame["timestamp"].max()) + step + pd.Timedelta(minutes=30)
    # Primary lacking that candle also trips alignment; require the explicit
    # open-candle message so the guard itself is proven.
    with pytest.raises(DataQualityError, match="in-progress"):
        _run(paper_env, secondary=secondary_open, as_of=as_of)


def test_non_finite_prices_are_refused(paper_env) -> None:
    primary = paper_env["frame"].copy()
    primary.loc[primary.index[-3], "close"] = np.inf
    with pytest.raises(DataQualityError, match="finite|NaN"):
        _run(paper_env, primary=primary)


def test_failed_run_writes_no_signal_line(paper_env, tmp_path) -> None:
    log_path = tmp_path / "blocked.jsonl"
    primary = paper_env["frame"].copy()
    primary.loc[primary.index[-3], "close"] = np.nan
    with pytest.raises(DataQualityError):
        _run(paper_env, primary=primary, log_path=log_path)
    assert not log_path.exists()


def test_happy_path_without_log_returns_without_side_effect(paper_env) -> None:
    result = _run(paper_env)
    assert result.data_quality_passed is True


def test_completed_feed_path_works_when_open_candle_gate_is_disabled(paper_env, tmp_path) -> None:
    bundle = tmp_path / "open-candle-disabled-bundle"
    shutil.copytree(paper_env["bundle"], bundle)
    config_path = bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["data"]["drop_open_candle"] = False
    atomic_write_json(config_path, config)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["config.json"] = {
        "sha256": sha256_file(config_path),
        "bytes": config_path.stat().st_size,
    }
    atomic_write_json(manifest_path, manifest)
    result = generate_paper_signal(
        bundle,
        paper_env["frame"],
        secondary_ohlcv=paper_env["secondary"],
        as_of=paper_env["as_of"],
    )
    assert result.data_quality_passed is True


def test_too_short_latest_feature_window_is_refused(paper_env, tmp_path) -> None:
    bundle = tmp_path / "short-window-bundle"
    shutil.copytree(paper_env["bundle"], bundle)
    config_path = bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["data"]["min_cross_venue_overlap"] = 1
    atomic_write_json(config_path, config)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["config.json"] = {
        "sha256": sha256_file(config_path),
        "bytes": config_path.stat().st_size,
    }
    atomic_write_json(manifest_path, manifest)

    primary = paper_env["frame"].tail(5).reset_index(drop=True)
    secondary = paper_env["secondary"].tail(5).reset_index(drop=True)
    as_of = pd.Timestamp(primary["timestamp"].max()) + pd.Timedelta(hours=1)
    with pytest.raises(DataQualityError, match="latest inference window"):
        generate_paper_signal(
            bundle,
            primary,
            secondary_ohlcv=secondary,
            as_of=as_of,
        )


def test_nonfinite_scaler_output_is_refused(paper_env, monkeypatch) -> None:
    loaded = load_bundle(paper_env["bundle"])
    monkeypatch.setattr(
        loaded.scaler,
        "transform",
        lambda values: np.full(values.shape, np.inf, dtype="float64"),
    )
    monkeypatch.setattr(live_module, "load_bundle", lambda path: loaded)
    with pytest.raises(DataQualityError, match="Scaled inference window"):
        _run(paper_env)


def test_signal_log_lock_contention_fails_closed(paper_env, tmp_path) -> None:
    log_path = tmp_path / "locked-signals.jsonl"
    with FileLock(str(log_path) + ".lock"):
        with pytest.raises(AresError, match="Another paper process"):
            _run(paper_env, log_path=log_path)
    assert not log_path.exists()
