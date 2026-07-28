"""Small fail-closed tests for paper-input normalization helpers."""

from __future__ import annotations

import pandas as pd

from ares_engine.live import _normalize_secondary_frames, _open_candle_failures


def test_open_candle_guard_accepts_empty_or_timestamp_free_inputs() -> None:
    reference = pd.Timestamp("2025-01-01T12:30:00Z")
    assert (
        _open_candle_failures(pd.DataFrame(), source="empty", timeframe="1h", reference=reference)
        == []
    )
    assert (
        _open_candle_failures(
            pd.DataFrame({"close": [1.0]}),
            source="missing-timestamp",
            timeframe="1h",
            reference=reference,
        )
        == []
    )


def test_secondary_input_normalization_rejects_ambiguous_and_malformed_inputs() -> None:
    assert _normalize_secondary_frames(None, ["kraken"]) == ({}, [])
    frame = pd.DataFrame({"timestamp": []})
    frames, failures = _normalize_secondary_frames(frame, ["kraken", "bitstamp"])
    assert frames == {}
    assert any("exactly one" in failure for failure in failures)

    frames, failures = _normalize_secondary_frames(
        {"kraken": "not-a-frame", "unexpected": frame}, ["kraken"]
    )
    assert set(frames) == {"unexpected"}
    assert any("not a pandas DataFrame" in failure for failure in failures)
    assert any("unexpected secondary feeds" in failure for failure in failures)
