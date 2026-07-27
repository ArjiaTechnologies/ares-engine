from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ares_engine.config import load_config
from ares_engine.exceptions import AresError, DataQualityError
from ares_engine.features import build_features
from ares_engine.live import generate_paper_signal
from ares_engine.synthetic import make_synthetic_ohlcv


class IdentityScaler:
    def transform(self, values):
        return values


def test_paper_inference_fails_closed_before_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    fake_bundle = SimpleNamespace(config=config, feature_spec={})
    monkeypatch.setattr("ares_engine.live.load_bundle", lambda _: fake_bundle)

    primary = make_synthetic_ohlcv(300, seed=30, exchange="coinbase")
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    primary.loc[10, "high"] = primary.loc[10, "low"] - 1.0
    log_path = tmp_path / "signals.jsonl"

    with pytest.raises(DataQualityError, match="Paper inference blocked"):
        generate_paper_signal(
            Path("fake-bundle"),
            primary,
            secondary_ohlcv=secondary,
            log_path=log_path,
            as_of=pd.Timestamp(primary["timestamp"].max()) + pd.Timedelta(hours=1),
        )
    assert not log_path.exists()


def test_paper_inference_rejects_stale_data(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config("configs/smoke.yaml")
    fake_bundle = SimpleNamespace(config=config, feature_spec={})
    monkeypatch.setattr("ares_engine.live.load_bundle", lambda _: fake_bundle)

    primary = make_synthetic_ohlcv(300, seed=31, exchange="coinbase")
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    as_of = pd.Timestamp(primary["timestamp"].max()) + pd.Timedelta(hours=10)

    with pytest.raises(DataQualityError, match="stale"):
        generate_paper_signal(
            Path("fake-bundle"),
            primary,
            secondary_ohlcv=secondary,
            as_of=as_of,
        )


def test_paper_inference_rejects_stale_secondary_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/smoke.yaml")
    fake_bundle = SimpleNamespace(config=config, feature_spec={})
    monkeypatch.setattr("ares_engine.live.load_bundle", lambda _: fake_bundle)

    primary = make_synthetic_ohlcv(300, seed=32, exchange="coinbase")
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    secondary["timestamp"] = secondary["timestamp"] - pd.Timedelta(hours=10)
    as_of = pd.Timestamp(primary["timestamp"].max()) + pd.Timedelta(hours=1)

    with pytest.raises(DataQualityError, match="kraken data is"):
        generate_paper_signal(
            Path("fake-bundle"),
            primary,
            secondary_ohlcv=secondary,
            as_of=as_of,
        )


def test_paper_inference_requires_every_configured_secondary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/smoke.yaml")
    config.data.validation_exchanges = ["kraken", "bitstamp"]
    fake_bundle = SimpleNamespace(config=config, feature_spec={})
    monkeypatch.setattr("ares_engine.live.load_bundle", lambda _: fake_bundle)

    primary = make_synthetic_ohlcv(300, seed=33, exchange="coinbase")
    kraken = primary.copy()
    kraken["exchange"] = "kraken"
    as_of = pd.Timestamp(primary["timestamp"].max()) + pd.Timedelta(hours=1)

    with pytest.raises(DataQualityError, match="bitstamp is missing"):
        generate_paper_signal(
            Path("fake-bundle"),
            primary,
            secondary_ohlcv={"kraken": kraken},
            as_of=as_of,
        )


def test_paper_inference_requires_market_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config("configs/smoke.yaml")
    fake_bundle = SimpleNamespace(config=config, feature_spec={})
    monkeypatch.setattr("ares_engine.live.load_bundle", lambda _: fake_bundle)

    primary = make_synthetic_ohlcv(300, seed=34, exchange="coinbase").drop(columns="exchange")
    secondary = make_synthetic_ohlcv(300, seed=34, exchange="kraken")
    as_of = pd.Timestamp(primary["timestamp"].max()) + pd.Timedelta(hours=1)

    with pytest.raises(DataQualityError, match="missing required exchange metadata"):
        generate_paper_signal(
            Path("fake-bundle"),
            primary,
            secondary_ohlcv=secondary,
            as_of=as_of,
        )


def test_paper_inference_refuses_to_fall_back_to_an_older_complete_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/smoke.yaml")
    primary = make_synthetic_ohlcv(300, seed=35, exchange="coinbase")
    flat_price = float(primary.loc[primary.index[-31], "close"])
    primary.loc[primary.index[-30:], ["open", "high", "low", "close"]] = flat_price
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    columns = build_features(primary, config.features).columns
    fake_bundle = SimpleNamespace(
        config=config,
        feature_spec={
            "columns": columns,
            "lookback_bars": config.model.lookback_bars,
            "thresholds": {
                "long": config.backtest.long_threshold,
                "short": config.backtest.short_threshold,
            },
        },
        scaler=IdentityScaler(),
        model=object(),
    )
    monkeypatch.setattr("ares_engine.live.load_bundle", lambda _: fake_bundle)
    as_of = pd.Timestamp(primary["timestamp"].max()) + pd.Timedelta(hours=1)

    with pytest.raises(DataQualityError, match="Latest inference window"):
        generate_paper_signal(
            Path("fake-bundle"),
            primary,
            secondary_ohlcv=secondary,
            as_of=as_of,
        )


def test_paper_inference_rejects_non_finite_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/smoke.yaml")
    primary = make_synthetic_ohlcv(300, seed=36, exchange="coinbase")
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    columns = build_features(primary, config.features).columns
    fake_bundle = SimpleNamespace(
        config=config,
        feature_spec={
            "columns": columns,
            "lookback_bars": config.model.lookback_bars,
            "thresholds": {
                "long": config.backtest.long_threshold,
                "short": config.backtest.short_threshold,
            },
        },
        scaler=IdentityScaler(),
        model=object(),
    )
    monkeypatch.setattr("ares_engine.live.load_bundle", lambda _: fake_bundle)
    monkeypatch.setattr(
        "ares_engine.live.predict_probabilities", lambda model, values: np.asarray([np.nan])
    )
    as_of = pd.Timestamp(primary["timestamp"].max()) + pd.Timedelta(hours=1)

    with pytest.raises(AresError, match="invalid probability"):
        generate_paper_signal(
            Path("fake-bundle"),
            primary,
            secondary_ohlcv=secondary,
            as_of=as_of,
        )


def test_paper_inference_rejects_unaligned_latest_secondary_candle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/smoke.yaml")
    fake_bundle = SimpleNamespace(config=config, feature_spec={})
    monkeypatch.setattr("ares_engine.live.load_bundle", lambda _: fake_bundle)

    primary = make_synthetic_ohlcv(300, seed=37, exchange="coinbase")
    secondary = primary.iloc[:-1].copy()
    secondary["exchange"] = "kraken"
    as_of = pd.Timestamp(primary["timestamp"].max()) + pd.Timedelta(hours=1)

    with pytest.raises(DataQualityError, match="Newest candle is not aligned"):
        generate_paper_signal(
            Path("fake-bundle"),
            primary,
            secondary_ohlcv=secondary,
            as_of=as_of,
        )
