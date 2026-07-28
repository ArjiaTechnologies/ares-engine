"""Adversarial regressions for defects independently reproduced after Fable 5."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ares_engine.backtest import run_backtest
from ares_engine.config import AresConfig, BacktestConfig
from ares_engine.public_audit import (
    CHECKSUMS_NAME,
    QUALITY_NAME,
    REPORT_MD_NAME,
    REPORT_NAME,
    REQUEST_SUMMARY_NAME,
    InstrumentedProvider,
    validate_public_ingestion_report,
)
from ares_engine.search import _data_fingerprint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_forged_live_report_and_fake_parquet_are_rejected(tmp_path: Path) -> None:
    report = {
        "live_public_endpoints_reached": False,
        "credentials_used": False,
        "orders_possible": False,
        "fixture_mode": False,
        "mode": "live-public-endpoints",
        "primary_exchange": "coinbase",
        "validation_exchange": "kraken",
        "symbol": "ETH/USD",
        "timeframe": "1h",
        "requested_start": "2026-01-01T00:00:00+00:00",
        "requested_end": "2026-01-15T00:00:00+00:00",
        "primary_http_requests": 4,
        "validation_http_requests": 4,
        "primary_fetch_ohlcv_calls": 4,
        "validation_fetch_ohlcv_calls": 4,
        "primary_pages": 4,
        "validation_pages": 4,
        "primary_rows": 336,
        "validation_rows": 336,
        "primary_raw_rows": 336,
        "validation_raw_rows": 336,
        "primary_duplicate_rows": 0,
        "validation_duplicate_rows": 0,
        "coverage_passed": True,
        "quality_passed": True,
        "alignment_passed": True,
        "divergence_passed": True,
        "idempotency_passed": True,
        "failure": None,
        "overall_passed": True,
    }
    (tmp_path / REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / REPORT_MD_NAME).write_text("forged live report\n", encoding="utf-8")
    (tmp_path / REQUEST_SUMMARY_NAME).write_text("{}\n", encoding="utf-8")
    (tmp_path / QUALITY_NAME).write_text("{}\n", encoding="utf-8")
    (tmp_path / "coinbase_normalized.parquet").write_bytes(b"not parquet")
    (tmp_path / "kraken_normalized.parquet").write_bytes(b"also not parquet")
    targets = sorted(path for path in tmp_path.iterdir() if path.is_file())
    (tmp_path / CHECKSUMS_NAME).write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in targets), encoding="utf-8"
    )

    problems = validate_public_ingestion_report(tmp_path)

    assert problems, "forged non-live success with fake Parquet must fail closed"
    assert any("live_public_endpoints_reached" in problem for problem in problems)
    assert any("parquet" in problem.lower() for problem in problems)


class _OneCallProvider:
    exchange_id = "coinbase"

    def fetch_range(self, *args: object, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            [[pd.Timestamp("2026-01-01", tz="UTC"), 1.0, 1.0, 1.0, 1.0, 1.0]],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )


def test_request_pages_are_measured_not_estimated() -> None:
    wrapped = InstrumentedProvider(_OneCallProvider())
    wrapped.fetch_range(
        "ETH/USD",
        "1h",
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=10),
        limit=2,
        max_pages=100,
        retries=0,
    )

    assert wrapped.pages == 1, (
        "one provider invocation must not be reported as five estimated pages"
    )


def test_full_ohlcv_and_material_config_are_in_study_identity() -> None:
    config = AresConfig()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [10.5, 11.5, 12.5, 13.5],
            "volume": [1.0, 2.0, 3.0, 4.0],
            "exchange": ["coinbase"] * 4,
            "symbol": ["ETH/USD"] * 4,
            "timeframe": ["1h"] * 4,
        }
    )
    baseline = _data_fingerprint(frame, config)
    high_changed = frame.copy()
    high_changed.loc[1, "high"] += 0.25
    feature_changed = config.model_copy(deep=True)
    feature_changed.features.rsi_period += 1

    assert _data_fingerprint(high_changed, config) != baseline
    assert _data_fingerprint(frame, feature_changed) != baseline
    assert _data_fingerprint(frame.copy(), config.model_copy(deep=True)) == baseline


def test_catastrophic_short_return_causes_terminal_bankruptcy() -> None:
    result = run_backtest(
        pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
        np.asarray([0.0, 2.0, 2.0, -0.5]),
        np.asarray([0.0, 0.0, 0.0, 0.0]),
        BacktestConfig(fee_bps=0.0, slippage_bps=0.0),
        timeframe="1h",
    )

    assert result.metrics.bankrupt is True
    assert result.metrics.final_equity == 0.0
    assert result.metrics.total_return == -1.0
    assert result.ledger["equity"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert result.ledger["bankrupt"].tolist() == [False, True, True, True]
