"""Adversarial branch coverage for immutable generation publication."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

import ares_engine.data.storage as storage
from ares_engine.data.storage import (
    GENERATION_MANIFEST,
    cleanup_abandoned_generations,
    create_generation,
    current_generation,
    generation_root,
    market_path,
    merge_market,
    publish_generation,
    read_market,
    verify_generation,
)
from ares_engine.exceptions import AresError
from ares_engine.synthetic import make_synthetic_ohlcv


def _frames() -> dict[str, pd.DataFrame]:
    coinbase = make_synthetic_ohlcv(24, seed=91, exchange="coinbase")
    kraken = coinbase.copy()
    kraken["exchange"] = "kraken"
    return {"coinbase": coinbase, "kraken": kraken}


def _create(root: Path, *, fault_hook=None) -> str:
    return create_generation(
        root,
        frames=_frames(),
        symbol="ETH/USD",
        timeframe="1h",
        quality_documents={"latest": {"passed": True}},
        metadata={"security_test": True},
        fault_hook=fault_hook,
    )


def _manifest(directory: Path) -> tuple[Path, dict]:
    path = directory / GENERATION_MANIFEST
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.chmod(0o644)
    return path, payload


def _write_manifest(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_path_and_pointer_guards_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(AresError, match="Invalid market generation"):
        generation_root(tmp_path, "../escape")
    with pytest.raises(AresError, match="Unsafe exchange"):
        market_path(tmp_path, "../coinbase", "ETH/USD", "1h", generation=None)

    root = tmp_path / "root"
    root.mkdir()
    assert current_generation(root) is None
    (root / "CURRENT").write_text("malformed\n", encoding="ascii")
    with pytest.raises(AresError, match="malformed"):
        current_generation(root)

    generation = _create(tmp_path / "valid")
    valid_root = tmp_path / "valid"
    (valid_root / "CURRENT").write_text(generation + "\n", encoding="ascii")
    assert current_generation(valid_root, verify=False) == generation

    pointer = valid_root / "CURRENT"
    pointer.unlink()
    target = tmp_path / "pointer-target"
    target.write_text(generation + "\n", encoding="ascii")
    pointer.symlink_to(target)
    with pytest.raises(AresError, match="regular file"):
        current_generation(valid_root)
    pointer.unlink()
    os.link(target, pointer)
    with pytest.raises(AresError, match="hard-linked"):
        current_generation(valid_root)


def test_create_and_publish_fault_hooks_cover_every_publication_stage(tmp_path: Path) -> None:
    events: list[str] = []
    root = tmp_path / "events"
    generation = _create(root, fault_hook=events.append)
    assert events == [
        "generation_created",
        "venue_written:coinbase",
        "venue_written:kraken",
        "quality_written",
        "duckdb_written",
        "manifest_written",
    ]
    publish_generation(root, generation, fault_hook=events.append)
    assert events[-2:] == ["before_pointer", "after_pointer"]

    def fail_after_quality(event: str) -> None:
        if event == "quality_written":
            raise OSError("injected generation failure")

    with pytest.raises(OSError, match="injected generation failure"):
        _create(tmp_path / "failed", fault_hook=fail_after_quality)
    assert list((tmp_path / "failed" / "snapshots").iterdir())


def test_generation_manifest_corruption_matrix_fails_closed(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    generation = _create(baseline)

    def case(name: str) -> tuple[Path, Path, dict]:
        root = tmp_path / name
        shutil.copytree(baseline, root)
        directory = generation_root(root, generation)
        directory.chmod(0o755)
        manifest_path, payload = _manifest(directory)
        return root, manifest_path, payload

    mutations = []

    root, path, payload = case("root-list")
    _write_manifest(path, [])
    mutations.append((root, "manifest root"))

    root, path, payload = case("identity")
    payload["schema"] = "wrong"
    _write_manifest(path, payload)
    mutations.append((root, "identity"))

    root, path, payload = case("no-files")
    payload["files"] = {}
    _write_manifest(path, payload)
    mutations.append((root, "no files"))

    root, path, payload = case("unsafe-path")
    payload["files"]["../escape"] = {"sha256": "0" * 64, "bytes": 0}
    _write_manifest(path, payload)
    mutations.append((root, "unsafe manifest path"))

    root, path, payload = case("bad-record")
    first_name = next(iter(payload["files"]))
    payload["files"][first_name] = "bad"
    _write_manifest(path, payload)
    mutations.append((root, "unsafe manifest path"))

    root, path, payload = case("extra-file")
    (path.parent / "extra.bin").write_bytes(b"extra")
    _write_manifest(path, payload)
    mutations.append((root, "file set differs"))

    for name, field, value, message in [
        ("bool-size", "bytes", True, "byte count"),
        ("negative-size", "bytes", -1, "byte count"),
        ("bad-hash", "sha256", "not-a-hash", "checksum is invalid"),
        ("size-mismatch", "bytes", 999999, "size mismatch"),
    ]:
        root, path, payload = case(name)
        first_name = next(iter(payload["files"]))
        payload["files"][first_name][field] = value
        _write_manifest(path, payload)
        mutations.append((root, message))

    root, path, payload = case("checksum-mismatch")
    first_name = next(iter(payload["files"]))
    artifact = path.parent / first_name
    artifact.chmod(0o644)
    original = artifact.read_bytes()
    artifact.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    _write_manifest(path, payload)
    mutations.append((root, "checksum mismatch"))

    root, path, payload = case("bad-market-identity")
    payload["exchanges"] = []
    _write_manifest(path, payload)
    mutations.append((root, "market identity"))

    root, path, payload = case("missing-required")
    missing = path.parent / "quality/latest.json"
    missing.parent.chmod(0o755)
    missing.chmod(0o644)
    missing.unlink()
    payload["files"].pop("quality/latest.json")
    _write_manifest(path, payload)
    mutations.append((root, "missing required canonical"))

    root, path, payload = case("linked-artifact")
    artifact = path.parent / "quality/latest.json"
    artifact.parent.chmod(0o755)
    artifact.chmod(0o644)
    outside = root / "outside.json"
    shutil.copyfile(artifact, outside)
    artifact.unlink()
    os.link(outside, artifact)
    _write_manifest(path, payload)
    mutations.append((root, "linked or special"))

    root, path, payload = case("special-entry")
    (path.parent / "linked").symlink_to(root / "outside-target")
    _write_manifest(path, payload)
    mutations.append((root, "symbolic or special"))

    root, path, payload = case("duplicate-key")
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    mutations.append((root, "unreadable"))

    root, path, payload = case("linked-manifest")
    outside_manifest = root / "manifest-copy.json"
    shutil.copyfile(path, outside_manifest)
    path.unlink()
    os.link(outside_manifest, path)
    mutations.append((root, "single regular file"))

    for root, message in mutations:
        with pytest.raises(AresError, match=message):
            verify_generation(root, generation)


def test_storage_miscellaneous_defensive_branches(tmp_path: Path, monkeypatch) -> None:
    assert cleanup_abandoned_generations(tmp_path / "absent") == []
    empty = pd.DataFrame()
    assert merge_market(empty, empty).empty
    no_timestamp = pd.DataFrame({"value": [1.0]})
    assert list(merge_market(no_timestamp, empty)) == ["value"]

    malformed = tmp_path / "malformed.parquet"
    no_timestamp.to_parquet(malformed, index=False)
    with pytest.raises(AresError, match="no timestamp"):
        read_market(malformed)

    monkeypatch.setattr(
        storage.os, "open", lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
    )
    storage._fsync_path(tmp_path / "unopenable")


def test_publish_failure_before_pointer_leaves_no_temporary_pointer(tmp_path: Path) -> None:
    root = tmp_path / "publish-failure"
    generation = _create(root)

    def fail(event: str) -> None:
        if event == "before_pointer":
            raise OSError("injected pointer failure")

    with pytest.raises(OSError, match="injected pointer failure"):
        publish_generation(root, generation, fault_hook=fail)
    assert current_generation(root) is None
    assert not list(root.glob(".CURRENT.*"))
