from pathlib import Path
from types import SimpleNamespace

import pytest

from ares_engine.config import load_config
from ares_engine.search import run_search
from ares_engine.synthetic import make_synthetic_ohlcv


def test_search_refuses_to_select_a_gate_failing_trial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config("configs/smoke.yaml")
    config.search.n_trials = 1
    config.search.study_name = "no-passing-candidate"
    config.search.storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    summary = SimpleNamespace(
        passed=False,
        aggregate={"median_sharpe": -1.0},
        gates={"median_sharpe": False},
        score=-1.0,
    )
    monkeypatch.setattr("ares_engine.search.run_walk_forward", lambda *args, **kwargs: summary)

    with pytest.raises(ValueError, match="without a gate-passing candidate"):
        run_search(
            make_synthetic_ohlcv(1_500),
            config,
            output_dir=tmp_path / "artifacts",
        )
    assert not (tmp_path / "artifacts" / "best_config.yaml").exists()
