"""Paper inference fail-closed battery with a real exported bundle."""

import json

import numpy as np
import pandas as pd
import pytest
from test_bundle_security import _training_config  # shared real-bundle recipe

from ares_engine.exceptions import DataQualityError
from ares_engine.live import generate_paper_signal
from ares_engine.synthetic import make_synthetic_ohlcv
from ares_engine.training import train_candidate

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
