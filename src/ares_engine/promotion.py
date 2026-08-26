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
from .replay import validate_replay_report
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
    try:
        with pointer.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise BundleIntegrityError("Champion pointer is not valid JSON") from exc
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
    replay_value = payload.get("replay_report_path")
    replay_hash = payload.get("replay_report_sha256")
    if replay_value is not None or replay_hash is not None:
        if not isinstance(replay_value, str) or not replay_value:
            raise BundleIntegrityError("Champion pointer has an invalid replay report path")
        replay_path = _contained_bundle(
            resolved_root / replay_value, resolved_root, label="Replay report"
        )
        if not replay_path.is_file() or replay_path.is_symlink():
            raise BundleIntegrityError("Champion replay report is not a regular file")
        if not isinstance(replay_hash, str) or sha256_file(replay_path) != replay_hash:
            raise BundleIntegrityError("Champion replay report no longer matches the pointer")
    return path


def _load_replay_report(path: Path, artifacts_root: Path) -> tuple[Path, dict[str, Any]]:
    report_path = _contained_bundle(path, artifacts_root, label="Replay report")
    try:
        stat = report_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PromotionRejected("Unable to inspect replay report") from exc
    if not report_path.is_file() or report_path.is_symlink() or stat.st_nlink != 1:
        raise PromotionRejected("Replay report must be a regular single-link file")
    if stat.st_size > 8_388_608:
        raise PromotionRejected("Replay report exceeds its allowed size")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key: {key}")
            payload[key] = value
        return payload

    try:
        with report_path.open(encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique_object)
    except (OSError, UnicodeError, ValueError) as exc:
        raise PromotionRejected("Replay report is not valid unique-key UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise PromotionRejected("Replay report must be a JSON object")
    return report_path, payload


def decide_promotion(
    challenger: Path, champion: Path | None, gates: GateConfig
) -> PromotionDecision:
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
    replay_report: Path | None,
) -> PromotionDecision:
    manifest_path = challenger / "manifest.json"
    verify_bundle(challenger)
    verified_manifest_hash = sha256_file(manifest_path)
    verified_replay_path: Path | None = None
    verified_replay_hash: str | None = None
    if replay_report is not None:
        verified_replay_path, replay_payload = _load_replay_report(replay_report, artifacts_root)
        validate_replay_report(replay_payload, challenger=challenger, gates=gates)
        verified_replay_hash = sha256_file(verified_replay_path)
    champion = resolve_champion(artifacts_root)
    decision = decide_promotion(challenger, champion, gates)
    atomic_write_json(artifacts_root / "last_promotion_decision.json", decision.to_dict())
    if not decision.approved:
        raise PromotionRejected(decision.reason)

    # Re-verify immediately before publishing the pointer. This anchors the
    # decision to the same immutable bytes that the pointer names.
    verify_bundle(challenger)
    if sha256_file(manifest_path) != verified_manifest_hash:
        raise BundleIntegrityError("Challenger changed during promotion")
    if (
        verified_replay_path is not None
        and sha256_file(verified_replay_path) != verified_replay_hash
    ):
        raise BundleIntegrityError("Replay report changed during promotion")

    bundle_path = str(challenger.relative_to(artifacts_root))
    pointer_payload: dict[str, Any] = {
        "bundle_path": bundle_path,
        "promoted_at": utc_now(),
        "manifest_sha256": verified_manifest_hash,
        "decision": decision.to_dict(),
    }
    if verified_replay_path is not None:
        pointer_payload["replay_report_path"] = str(
            verified_replay_path.relative_to(artifacts_root)
        )
        pointer_payload["replay_report_sha256"] = verified_replay_hash
    atomic_write_json(champion_pointer(artifacts_root), pointer_payload)
    return decision


def promote(
    challenger: Path,
    artifacts_root: Path,
    gates: GateConfig,
    *,
    replay_report: Path | None = None,
) -> PromotionDecision:
    if gates.require_recent_replay and replay_report is None:
        raise PromotionRejected("A verified recent-window replay report is required")
    artifacts_root.mkdir(parents=True, exist_ok=True)
    resolved_root = artifacts_root.resolve()
    resolved_challenger = _contained_bundle(challenger, resolved_root, label="Challenger")
    lock = FileLock(resolved_root / ".ares-promotion.lock", timeout=0)
    try:
        with lock:
            return _promote_unlocked(resolved_challenger, resolved_root, gates, replay_report)
    except Timeout as exc:
        raise PromotionRejected("Another ARES promotion is already running") from exc
