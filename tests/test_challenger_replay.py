import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_bundle_promotion import make_fake_bundle, rewrite_bundle_payload

import ares_engine.replay as replay_module
from ares_engine.config import GateConfig, load_config
from ares_engine.exceptions import BundleIntegrityError, PromotionRejected
from ares_engine.promotion import promote, resolve_champion
from ares_engine.replay import (
    run_challenger_replay,
    split_replay_window,
    validate_replay_report,
)
from ares_engine.synthetic import make_synthetic_ohlcv
from ares_engine.utils import atomic_write_json, sha256_file
from ares_engine.validation import prepare_dataset


def _gates() -> GateConfig:
    return GateConfig(
        min_promotion_score_improvement=0.0,
        require_recent_replay=True,
        recent_replay_bars=50,
        min_recent_replay_trades=0,
        max_recent_replay_drawdown=0.99,
        require_positive_recent_replay_stress=False,
    )


def _valid_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifacts = tmp_path / "artifacts"
    challenger = make_fake_bundle(artifacts, "candidate", score=1.0)
    source = tmp_path / "source.parquet"
    frame = make_synthetic_ohlcv(400, seed=86, exchange="coinbase")
    frame.to_parquet(source, index=False)
    config = load_config("configs/smoke.yaml")
    config.storage.artifacts = artifacts
    config.gates = _gates()
    _, dataset = prepare_dataset(frame, config)
    cutoff = frame["timestamp"].iloc[-51]
    rewrite_bundle_payload(
        challenger,
        "provenance.json",
        {
            "latest_feature_timestamp": cutoff,
            "research_dataset_identity_sha256": "a" * 64,
        },
    )
    loaded = SimpleNamespace(
        config=config,
        feature_spec={"columns": dataset.feature_columns},
        provenance={"latest_feature_timestamp": cutoff},
        model=object(),
        scaler=object(),
    )
    monkeypatch.setattr(replay_module, "load_bundle", lambda _: loaded)
    monkeypatch.setattr(replay_module, "transform_sequences", lambda _scaler, values: values)
    monkeypatch.setattr(
        replay_module,
        "predict_probabilities",
        lambda _model, values: np.resize(np.array([0.35, 0.65]), len(values)),
    )
    report_path = artifacts / "replays" / "candidate.json"
    report = run_challenger_replay(
        challenger,
        frame,
        source_path=source,
        report_path=report_path,
        gates=config.gates,
    )
    return artifacts, challenger, report_path, report, config.gates


def test_recent_window_is_reserved_before_research() -> None:
    config = load_config("configs/smoke.yaml")
    frame = make_synthetic_ohlcv(2_000, seed=87, exchange="coinbase")
    research, replay = split_replay_window(frame, config)
    assert len(replay) == config.gates.recent_replay_bars
    assert len(research) + len(replay) == len(frame)
    assert research["timestamp"].max() < replay["timestamp"].min()


def test_replay_report_binds_unseen_latest_window_and_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, challenger, report_path, report, gates = _valid_replay(tmp_path, monkeypatch)
    assert report["window"]["sample_count"] == gates.recent_replay_bars
    assert report["training"]["latest_feature_timestamp"] < report["window"]["start"]
    assert report["window"]["end"] == report["source"]["latest_timestamp"]
    assert report["passed"] is True

    decision = promote(challenger, artifacts, gates, replay_report=report_path)
    assert decision.approved is True
    assert resolve_champion(artifacts) == challenger.resolve()
    pointer = json.loads((artifacts / "champion.json").read_text(encoding="utf-8"))
    assert pointer["replay_report_sha256"] == sha256_file(report_path)


def test_required_replay_cannot_be_omitted(tmp_path: Path) -> None:
    challenger = make_fake_bundle(tmp_path, "candidate", score=1.0)
    with pytest.raises(PromotionRejected, match="replay report is required"):
        promote(challenger, tmp_path, _gates())


@pytest.mark.parametrize("mutation", ["cross_bundle", "overlap", "weakened", "stale"])
def test_replay_evidence_fails_closed_on_binding_and_gate_attacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    _, challenger, _, report, gates = _valid_replay(tmp_path, monkeypatch)
    attacked = copy.deepcopy(report)
    if mutation == "cross_bundle":
        attacked["bundle"]["manifest_sha256"] = "0" * 64
    elif mutation == "overlap":
        attacked["training"]["latest_feature_timestamp"] = attacked["window"]["start"]
    elif mutation == "weakened":
        attacked["backtest"]["bankrupt"] = True
    else:
        attacked["source"]["latest_timestamp"] = attacked["window"]["start"]
    with pytest.raises(PromotionRejected):
        validate_replay_report(attacked, challenger=challenger, gates=gates)


def test_tampered_promoted_replay_report_invalidates_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, challenger, report_path, _, gates = _valid_replay(tmp_path, monkeypatch)
    promote(challenger, artifacts, gates, replay_report=report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["passed"] = False
    atomic_write_json(report_path, payload)
    with pytest.raises(BundleIntegrityError, match="replay report no longer matches"):
        resolve_champion(artifacts)


def test_replay_report_must_stay_inside_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts, challenger, report_path, report, gates = _valid_replay(tmp_path, monkeypatch)
    external = tmp_path / "external-replay.json"
    atomic_write_json(external, report)
    report_path.unlink()
    with pytest.raises(BundleIntegrityError, match="contained"):
        promote(challenger, artifacts, gates, replay_report=external)
