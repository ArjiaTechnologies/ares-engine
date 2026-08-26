"""Promotion atomicity, crash injection, and cross-process serialization."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from test_bundle_promotion import make_fake_bundle  # reuse the fixture builder

import ares_engine.promotion as promotion_module
from ares_engine.config import GateConfig
from ares_engine.exceptions import BundleIntegrityError, PromotionRejected
from ares_engine.promotion import promote, resolve_champion
from ares_engine.utils import sha256_file


def test_crash_between_decision_and_pointer_preserves_previous_champion(
    tmp_path: Path, monkeypatch
) -> None:
    gates = GateConfig(min_promotion_score_improvement=0.0, require_recent_replay=False)
    first = make_fake_bundle(tmp_path, "first", score=1.0)
    promote(first, tmp_path, gates)
    champion_before = resolve_champion(tmp_path)

    real_write = promotion_module.atomic_write_json

    def crashing_write(path, payload):
        if path.name == "champion.json":
            raise OSError("simulated crash during pointer update")
        real_write(path, payload)

    monkeypatch.setattr(promotion_module, "atomic_write_json", crashing_write)
    challenger = make_fake_bundle(tmp_path, "challenger", score=2.0)
    with pytest.raises(OSError, match="simulated crash"):
        promote(challenger, tmp_path, gates)
    monkeypatch.setattr(promotion_module, "atomic_write_json", real_write)
    assert resolve_champion(tmp_path) == champion_before
    decision = json.loads((tmp_path / "last_promotion_decision.json").read_text(encoding="utf-8"))
    assert decision["approved"] is True  # the recorded intent survives, pointer did not move


def test_garbage_champion_pointer_fails_closed(tmp_path: Path) -> None:
    make_fake_bundle(tmp_path, "anything", score=1.0)
    (tmp_path / "champion.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="not valid JSON"):
        resolve_champion(tmp_path)


def test_corrupt_incumbent_blocks_promotion_without_pointer_change(tmp_path: Path) -> None:
    gates = GateConfig(min_promotion_score_improvement=0.0, require_recent_replay=False)
    first = make_fake_bundle(tmp_path, "first", score=1.0)
    promote(first, tmp_path, gates)
    (first / "metrics.json").write_text("tampered", encoding="utf-8")
    pointer_before = (tmp_path / "champion.json").read_bytes()
    challenger = make_fake_bundle(tmp_path, "challenger", score=5.0)
    with pytest.raises(BundleIntegrityError):
        promote(challenger, tmp_path, gates)
    assert (tmp_path / "champion.json").read_bytes() == pointer_before


def test_gate_failed_challenger_never_promotes_even_without_incumbent(tmp_path: Path) -> None:
    gates = GateConfig(min_promotion_score_improvement=0.0, require_recent_replay=False)
    failed = make_fake_bundle(tmp_path, "failed", score=9.9, passed=False)
    with pytest.raises(PromotionRejected, match="failed validation gates"):
        promote(failed, tmp_path, gates)
    assert resolve_champion(tmp_path) is None


def test_equal_score_respects_margin_semantics(tmp_path: Path) -> None:
    strict = GateConfig(min_promotion_score_improvement=0.05, require_recent_replay=False)
    first = make_fake_bundle(tmp_path, "first", score=1.0)
    promote(first, tmp_path, strict)
    equal = make_fake_bundle(tmp_path, "equal", score=1.0)
    with pytest.raises(PromotionRejected, match="below required"):
        promote(equal, tmp_path, strict)
    permissive = GateConfig(min_promotion_score_improvement=0.0, require_recent_replay=False)
    promote(equal, tmp_path, permissive)  # explicit zero margin allows ties
    assert resolve_champion(tmp_path) == equal.resolve()


def test_infinite_score_fails_closed(tmp_path: Path) -> None:
    bundle = make_fake_bundle(tmp_path, "infinite", score=float("inf"))
    with pytest.raises(BundleIntegrityError, match="finite"):
        promote(
            bundle,
            tmp_path,
            GateConfig(min_promotion_score_improvement=0.0, require_recent_replay=False),
        )


def test_cross_process_promotion_lock_fails_closed(tmp_path: Path) -> None:
    bundle = make_fake_bundle(tmp_path, "locked-out", score=1.0)
    lock_file = tmp_path / ".ares-promotion.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import time
                from filelock import FileLock
                lock = FileLock({str(lock_file)!r})
                lock.acquire()
                print("locked", flush=True)
                time.sleep(30)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(PromotionRejected, match="already running"):
            promote(
                bundle,
                tmp_path,
                GateConfig(min_promotion_score_improvement=0.0, require_recent_replay=False),
            )
    finally:
        holder.kill()
        holder.wait()
    assert resolve_champion(tmp_path) is None


def test_concurrent_promotions_from_two_processes_leave_a_single_coherent_champion(
    tmp_path: Path,
) -> None:
    for name, score in [("bundle-a", 1.0), ("bundle-b", 2.0)]:
        make_fake_bundle(tmp_path, name, score=score)
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(Path.cwd() / "src")!r})
        from pathlib import Path
        from ares_engine.config import GateConfig
        from ares_engine.exceptions import PromotionRejected
        from ares_engine.promotion import promote
        root = Path({str(tmp_path)!r})
        try:
            decision = promote(
                root / sys.argv[1],
                root,
                GateConfig(min_promotion_score_improvement=0.0, require_recent_replay=False),
            )
            print("PROMOTED")
        except PromotionRejected as exc:
            print(f"REJECTED: {{exc}}")
        """
    )
    first = subprocess.Popen(
        [sys.executable, "-c", script, "bundle-a"], stdout=subprocess.PIPE, text=True
    )
    second = subprocess.Popen(
        [sys.executable, "-c", script, "bundle-b"], stdout=subprocess.PIPE, text=True
    )
    outputs = [process.communicate(timeout=30)[0].strip() for process in (first, second)]
    assert all(out.startswith(("PROMOTED", "REJECTED")) for out in outputs), outputs
    champion = resolve_champion(tmp_path)
    assert champion is not None and champion.name in {"bundle-a", "bundle-b"}
    pointer = json.loads((tmp_path / "champion.json").read_text(encoding="utf-8"))
    assert pointer["manifest_sha256"] == sha256_file(champion / "manifest.json")


def test_permission_failure_leaves_champion_intact(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("permission semantics require a non-root user")
    gates = GateConfig(min_promotion_score_improvement=0.0, require_recent_replay=False)
    first = make_fake_bundle(tmp_path, "first", score=1.0)
    promote(first, tmp_path, gates)
    challenger = make_fake_bundle(tmp_path, "challenger", score=2.0)
    pointer_before = (tmp_path / "champion.json").read_bytes()
    os.chmod(tmp_path, 0o500)
    try:
        with pytest.raises(OSError):
            promote(challenger, tmp_path, gates)
    finally:
        os.chmod(tmp_path, 0o700)
    assert (tmp_path / "champion.json").read_bytes() == pointer_before
    assert resolve_champion(tmp_path) == first.resolve()


def test_challenger_mutation_during_decision_never_updates_pointer(
    tmp_path: Path, monkeypatch
) -> None:
    gates = GateConfig(min_promotion_score_improvement=0.0, require_recent_replay=False)
    first = make_fake_bundle(tmp_path, "first", score=1.0)
    promote(first, tmp_path, gates)
    pointer_before = (tmp_path / "champion.json").read_bytes()
    challenger = make_fake_bundle(tmp_path, "challenger-race", score=2.0)
    real_decide = promotion_module.decide_promotion

    def mutate_after_decision(challenger_path, champion_path, gate_config):
        decision = real_decide(challenger_path, champion_path, gate_config)
        (challenger_path / "metrics.json").write_text("mutated", encoding="utf-8")
        return decision

    monkeypatch.setattr(promotion_module, "decide_promotion", mutate_after_decision)
    with pytest.raises(BundleIntegrityError):
        promote(challenger, tmp_path, gates)
    assert (tmp_path / "champion.json").read_bytes() == pointer_before


def test_manifest_identity_change_after_reverification_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    challenger = make_fake_bundle(tmp_path, "manifest-race", score=2.0)
    real_verify = promotion_module.verify_bundle
    calls = 0

    def mutate_manifest_on_final_verify(bundle_path):
        nonlocal calls
        calls += 1
        if calls == 3:
            manifest_path = bundle_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["race_nonce"] = "changed-after-decision"
            promotion_module.atomic_write_json(manifest_path, manifest)
        real_verify(bundle_path)

    monkeypatch.setattr(promotion_module, "verify_bundle", mutate_manifest_on_final_verify)
    with pytest.raises(BundleIntegrityError, match="changed during promotion"):
        promote(
            challenger,
            tmp_path,
            GateConfig(min_promotion_score_improvement=0.0, require_recent_replay=False),
        )
    assert not (tmp_path / "champion.json").exists()
