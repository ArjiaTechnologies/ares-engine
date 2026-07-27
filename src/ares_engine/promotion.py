"""Champion/challenger promotion with immutable bundles and an atomic pointer."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from .bundles import verify_bundle
from .config import GateConfig
from .exceptions import BundleIntegrityError, PromotionRejected
from .utils import atomic_write_json, sha256_file, utc_now


@dataclass(slots=True)
class PromotionDecision:
    approved: bool
    reason: str
    challenger_score: float
    champion_score: float | None
    improvement: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metrics(bundle: Path) -> dict[str, Any]:
    with (bundle / "metrics.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BundleIntegrityError("Bundle metrics must be a JSON object")
    passed = payload.get("passed")
    if not isinstance(passed, bool):
        raise BundleIntegrityError("Bundle metrics `passed` field must be boolean")
    score_value = payload.get("score")
    if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
        raise BundleIntegrityError("Bundle metrics `score` field must be numeric")
    score = float(score_value)
    if not math.isfinite(score):
        raise BundleIntegrityError("Bundle metrics `score` field must be finite")
    return payload


def champion_pointer(artifacts_root: Path) -> Path:
    return artifacts_root / "champion.json"


def _contained_bundle(path: Path, artifacts_root: Path, *, label: str) -> Path:
    resolved_root = artifacts_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise BundleIntegrityError(
            f"{label} bundle must be contained inside the configured artifacts root"
        ) from exc
    return resolved_path


def resolve_champion(artifacts_root: Path) -> Path | None:
    resolved_root = artifacts_root.resolve()
    pointer = champion_pointer(resolved_root)
    if not pointer.exists():
        return None
    with pointer.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BundleIntegrityError("Champion pointer must be a JSON object")
    bundle_value = payload.get("bundle_path")
    if not isinstance(bundle_value, str) or not bundle_value:
        raise BundleIntegrityError("Champion pointer has an invalid bundle path")
    candidate = Path(bundle_value)
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    path = _contained_bundle(candidate, resolved_root, label="Champion")
    verify_bundle(path)
    _metrics(path)
    expected_manifest_hash = payload.get("manifest_sha256")
    if not isinstance(expected_manifest_hash, str):
        raise BundleIntegrityError("Champion pointer is missing its manifest hash")
    actual_manifest_hash = sha256_file(path / "manifest.json")
    if actual_manifest_hash != expected_manifest_hash:
        raise BundleIntegrityError("Champion manifest no longer matches the promoted pointer")
    return path


def decide_promotion(challenger: Path, champion: Path | None, gates: GateConfig) -> PromotionDecision:
    verify_bundle(challenger)
    challenger_metrics = _metrics(challenger)
    challenger_score = float(challenger_metrics["score"])
    if challenger_metrics["passed"] is not True:
        return PromotionDecision(
            approved=False,
            reason="challenger failed validation gates",
            challenger_score=challenger_score,
            champion_score=None,
            improvement=None,
        )
    if champion is None:
        return PromotionDecision(
            approved=True,
            reason="first gate-passing champion",
            challenger_score=challenger_score,
            champion_score=None,
            improvement=None,
        )

    verify_bundle(champion)
    champion_metrics = _metrics(champion)
    if champion_metrics["passed"] is not True:
        raise BundleIntegrityError("Incumbent champion metrics no longer show passed gates")
    champion_score = float(champion_metrics["score"])
    improvement = challenger_score - champion_score
    approved = improvement >= gates.min_promotion_score_improvement
    return PromotionDecision(
        approved=approved,
        reason=(
            f"score improvement {improvement:.6f} met required margin"
            if approved
            else f"score improvement {improvement:.6f} is below required "
            f"{gates.min_promotion_score_improvement:.6f}"
        ),
        challenger_score=challenger_score,
        champion_score=champion_score,
        improvement=improvement,
    )


def _promote_unlocked(
    challenger: Path,
    artifacts_root: Path,
    gates: GateConfig,
) -> PromotionDecision:
    champion = resolve_champion(artifacts_root)
    decision = decide_promotion(challenger, champion, gates)
    atomic_write_json(artifacts_root / "last_promotion_decision.json", decision.to_dict())
    if not decision.approved:
        raise PromotionRejected(decision.reason)

    bundle_path = str(challenger.relative_to(artifacts_root))
    atomic_write_json(
        champion_pointer(artifacts_root),
        {
            "bundle_path": bundle_path,
            "promoted_at": utc_now(),
            "manifest_sha256": sha256_file(challenger / "manifest.json"),
            "decision": decision.to_dict(),
        },
    )
    return decision


def promote(challenger: Path, artifacts_root: Path, gates: GateConfig) -> PromotionDecision:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    resolved_root = artifacts_root.resolve()
    resolved_challenger = _contained_bundle(challenger, resolved_root, label="Challenger")
    lock = FileLock(resolved_root / ".ares-promotion.lock", timeout=0)
    try:
        with lock:
            return _promote_unlocked(resolved_challenger, resolved_root, gates)
    except Timeout as exc:
        raise PromotionRejected("Another ARES promotion is already running") from exc
