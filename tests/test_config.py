from pathlib import Path

import pytest
from pydantic import ValidationError

from ares_engine.config import AresConfig, load_config


def test_default_config_loads() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.data.primary_exchange == "coinbase"
    assert config.data.validation_exchanges == ["kraken"]
    assert config.validation.purge_bars == config.labels.horizon_bars
    assert config.model.lookback_bars < config.validation.min_train_bars


def test_purge_must_cover_label_horizon() -> None:
    with pytest.raises(ValidationError, match="purge_bars"):
        AresConfig.model_validate(
            {
                "labels": {"horizon_bars": 12},
                "validation": {"purge_bars": 6},
            }
        )


def test_validation_exchanges_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        AresConfig.model_validate(
            {
                "data": {
                    "primary_exchange": "coinbase",
                    "validation_exchanges": ["kraken", "Kraken"],
                }
            }
        )


def test_at_least_one_validation_exchange_is_required() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        AresConfig.model_validate({"data": {"validation_exchanges": []}})


@pytest.mark.parametrize("exchange", ["../kraken", "kraken/../../tmp", "kraken\\tmp"])
def test_exchange_identifiers_cannot_escape_storage_paths(exchange: str) -> None:
    raw = AresConfig().model_dump(mode="python")
    raw["data"]["validation_exchanges"] = [exchange]
    with pytest.raises(ValidationError, match="safe CCXT exchange"):
        AresConfig.model_validate(raw)


def test_non_finite_configuration_is_rejected() -> None:
    raw = AresConfig().model_dump(mode="python")
    raw["gates"]["min_median_sharpe"] = float("nan")
    with pytest.raises(ValidationError, match="finite"):
        AresConfig.model_validate(raw)
