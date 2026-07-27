import importlib.util

import pytest

from ares_engine.config import load_config
from ares_engine.synthetic import make_synthetic_ohlcv
from ares_engine.validation import run_walk_forward


@pytest.mark.ml
@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None and importlib.util.find_spec("keras") is None,
    reason="No Keras runtime is installed",
)
def test_tensorflow_walk_forward_smoke() -> None:
    config = load_config("configs/smoke.yaml")
    config.model.epochs = 1
    config.model.patience = 1
    config.validation.max_folds = 1
    config.validation.min_train_bars = 900
    config.validation.validation_bars = 120
    config.validation.step_bars = 120
    config.gates.require_positive_cost_stress = False
    frame = make_synthetic_ohlcv(1_500, seed=12)
    result = run_walk_forward(frame, config)
    assert result.sample_count > 1_000
    assert len(result.folds) == 1
    assert "median_sharpe" in result.aggregate
