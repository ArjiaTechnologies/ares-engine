"""Typed YAML configuration for ARES Engine."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .utils import timeframe_to_seconds


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class ProjectConfig(StrictModel):
    name: str = "ARES Engine"
    seed: int = 42


class StorageConfig(StrictModel):
    root: Path = Path("data")
    artifacts: Path = Path("artifacts")
    duckdb_path: Path = Path("data/ares.duckdb")


class DataConfig(StrictModel):
    primary_exchange: str = "coinbase"
    validation_exchanges: list[str] = Field(default_factory=lambda: ["kraken"], min_length=1)
    symbol: str = "ETH/USD"
    timeframe: str = "1h"
    since: datetime = Field(default_factory=lambda: datetime(2023, 1, 1, tzinfo=UTC))
    until: datetime | None = None
    page_limit: int = Field(default=300, ge=10, le=1000)
    max_pages: int = Field(default=10_000, ge=1)
    retries: int = Field(default=4, ge=0, le=10)
    drop_open_candle: bool = True
    resume: bool = True
    fail_on_quality: bool = True
    max_gap_count: int = Field(default=0, ge=0)
    max_cross_venue_p95_bps: float = Field(default=75.0, gt=0)
    max_cross_venue_latest_bps: float = Field(default=150.0, gt=0)
    min_cross_venue_overlap: int = Field(default=100, ge=1)
    max_staleness_bars: int = Field(default=2, ge=1)

    @field_validator("primary_exchange")
    @classmethod
    def normalize_primary_exchange(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9_]+", normalized):
            raise ValueError("primary_exchange must be a safe CCXT exchange identifier")
        return normalized

    @field_validator("validation_exchanges")
    @classmethod
    def normalize_validation_exchanges(cls, values: list[str]) -> list[str]:
        normalized = [value.strip().lower() for value in values]
        if any(not re.fullmatch(r"[a-z0-9_]+", value) for value in normalized):
            raise ValueError("validation_exchanges must contain safe CCXT exchange identifiers")
        if len(normalized) != len(set(normalized)):
            raise ValueError("validation_exchanges must be unique")
        return normalized

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)?", normalized):
            raise ValueError("symbol must be a safe BASE/QUOTE market identifier")
        return normalized

    @field_validator("since", "until", mode="before")
    @classmethod
    def ensure_timezone(cls, value: object) -> object:
        if value is None:
            return None
        parsed = (
            datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if not isinstance(value, datetime)
            else value
        )
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @field_validator("timeframe")
    @classmethod
    def valid_timeframe(cls, value: str) -> str:
        normalized = value.strip().lower()
        timeframe_to_seconds(normalized)
        return normalized

    @model_validator(mode="after")
    def valid_range(self) -> DataConfig:
        if self.until is not None and self.until <= self.since:
            raise ValueError("data.until must be after data.since")
        if self.primary_exchange in self.validation_exchanges:
            raise ValueError("primary_exchange must not also appear in validation_exchanges")
        return self


class FeatureConfig(StrictModel):
    ema_periods: list[int] = Field(default_factory=lambda: [12, 26, 50])
    rsi_period: int = Field(default=14, ge=2)
    bollinger_period: int = Field(default=20, ge=2)
    bollinger_std: float = Field(default=2.0, gt=0)
    volatility_windows: list[int] = Field(default_factory=lambda: [12, 24, 72])
    volume_z_window: int = Field(default=48, ge=2)

    @field_validator("ema_periods", "volatility_windows")
    @classmethod
    def positive_unique(cls, values: list[int]) -> list[int]:
        cleaned = sorted(set(values))
        if not cleaned or any(value < 2 for value in cleaned):
            raise ValueError("period lists must contain unique integers >= 2")
        return cleaned


class LabelConfig(StrictModel):
    method: Literal["k_ahead", "triple_barrier"] = "k_ahead"
    horizon_bars: int = Field(default=12, ge=1)
    dead_zone_bps: float = Field(default=15.0, ge=0)
    take_profit_bps: float = Field(default=60.0, gt=0)
    stop_loss_bps: float = Field(default=45.0, gt=0)


class ModelConfig(StrictModel):
    family: Literal["lstm", "tcn"] = "lstm"
    lookback_bars: int = Field(default=72, ge=4)
    hidden_units: int = Field(default=64, ge=4, le=512)
    tcn_kernel_size: int = Field(default=3, ge=2, le=9)
    tcn_blocks: int = Field(default=3, ge=1, le=6)
    dropout: float = Field(default=0.2, ge=0, lt=0.9)
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=128, ge=8)
    epochs: int = Field(default=30, ge=1)
    patience: int = Field(default=5, ge=1)


class BacktestConfig(StrictModel):
    long_threshold: float = Field(default=0.58, gt=0.5, lt=1)
    short_threshold: float = Field(default=0.42, gt=0, lt=0.5)
    fee_bps: float = Field(default=6.0, ge=0)
    slippage_bps: float = Field(default=3.0, ge=0)
    execution_delay_bars: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def threshold_order(self) -> BacktestConfig:
        if self.short_threshold >= self.long_threshold:
            raise ValueError("short_threshold must be below long_threshold")
        return self


class ValidationConfig(StrictModel):
    min_train_bars: int = Field(default=4_000, ge=100)
    validation_bars: int = Field(default=720, ge=50)
    step_bars: int = Field(default=720, ge=1)
    max_folds: int = Field(default=5, ge=1)
    purge_bars: int | None = Field(default=None, ge=0)
    stress_cost_multiplier: float = Field(default=2.0, ge=1)


class GateConfig(StrictModel):
    min_median_sharpe: float = 0.0
    min_median_return: float = 0.0
    max_worst_drawdown: float = Field(default=0.30, gt=0, lt=1)
    max_median_turnover: float = Field(default=500.0, gt=0)
    min_total_trades: int = Field(default=20, ge=0)
    require_positive_cost_stress: bool = True
    min_promotion_score_improvement: float = Field(default=0.05, ge=0)


class SearchConfig(StrictModel):
    n_trials: int = Field(default=25, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    study_name: str = "ares-eth"
    storage_url: str = "sqlite:///artifacts/optuna.db"


class SchedulerConfig(StrictModel):
    quick_minutes: int = Field(default=60, ge=1)
    deep_hours: int = Field(default=168, ge=1)
    run_search_on_deep: bool = True


class AresConfig(StrictModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    gates: GateConfig = Field(default_factory=GateConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    @model_validator(mode="after")
    def model_horizon_consistency(self) -> AresConfig:
        if self.validation.purge_bars is None:
            self.validation.purge_bars = self.labels.horizon_bars
        if self.validation.purge_bars < self.labels.horizon_bars:
            raise ValueError("validation.purge_bars must be at least labels.horizon_bars")
        if self.validation.min_train_bars <= self.model.lookback_bars:
            raise ValueError("validation.min_train_bars must exceed model.lookback_bars")
        return self


def load_config(path: str | Path) -> AresConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AresConfig.model_validate(raw)


def dump_config(config: AresConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)
