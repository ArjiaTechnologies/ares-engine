"""Immutable market-data generations with one atomic canonical pointer."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pandas as pd

from ..exceptions import AresError
from ..utils import atomic_write_json, sha256_file, slugify_symbol

CURRENT_NAME = "CURRENT"
SNAPSHOTS_NAME = "snapshots"
GENERATION_MANIFEST = "manifest.json"
GENERATION_SCHEMA = "ares-market-generation-v1"
_GENERATION_RE = re.compile(r"^[0-9]{8}T[0-9]{6}\.[0-9]{6}Z-[0-9a-f]{12}$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe_component(value: str, *, label: str) -> str:
    if not _SAFE_COMPONENT_RE.fullmatch(value) or value in {".", ".."}:
        raise AresError(f"Unsafe {label} path component: {value!r}")
    return value


def snapshots_root(root: Path) -> Path:
    return Path(root) / SNAPSHOTS_NAME


def generation_root(root: Path, generation: str) -> Path:
    if not _GENERATION_RE.fullmatch(generation):
        raise AresError(f"Invalid market generation id: {generation!r}")
    return snapshots_root(root) / generation


def current_generation(root: Path, *, verify: bool = True) -> str | None:
    """Capture the canonical generation ID once for a complete reader operation."""
    pointer = Path(root) / CURRENT_NAME
    if not pointer.exists():
        return None
    if pointer.is_symlink() or not pointer.is_file():
        raise AresError("Canonical market generation pointer must be a regular file")
    if pointer.stat(follow_symlinks=False).st_nlink != 1:
        raise AresError("Canonical market generation pointer must not be hard-linked")
    try:
        generation = pointer.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise AresError("Cannot read the canonical market generation pointer") from exc
    if not _GENERATION_RE.fullmatch(generation):
        raise AresError("Canonical market generation pointer is malformed")
    if verify:
        verify_generation(root, generation)
    return generation


def market_path(
    root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    *,
    generation: str | None = None,
) -> Path:
    """Resolve one market file, preferably through an already captured generation."""
    selected = generation if generation is not None else current_generation(root)
    base = generation_root(root, selected) if selected is not None else Path(root)
    safe_exchange = _safe_component(exchange, label="exchange")
    safe_symbol = _safe_component(slugify_symbol(symbol), label="symbol")
    safe_timeframe = _safe_component(timeframe, label="timeframe")
    return base / "raw" / safe_exchange / safe_symbol / f"{safe_timeframe}.parquet"


def quality_path(root: Path, name: str, *, generation: str | None = None) -> Path:
    safe_name = _safe_component(
        name.replace("/", "-").replace("~", "__").lower(), label="quality report"
    )
    selected = generation if generation is not None else current_generation(root)
    base = generation_root(root, selected) if selected is not None else Path(root)
    return base / "quality" / f"{safe_name}.json"


def generation_duckdb_path(root: Path, generation: str) -> Path:
    return generation_root(root, generation) / "ares.duckdb"


def read_market(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "timestamp" not in frame:
        raise AresError(f"Market data file has no timestamp column: {path}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def merge_market(current: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge a candidate batch in memory without mutating canonical storage."""
    if current.empty:
        merged = incoming.copy()
    elif incoming.empty:
        merged = current.copy()
    else:
        merged = pd.concat([current, incoming], ignore_index=True)
    if merged.empty:
        return merged
    if "timestamp" in merged:
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
        merged = (
            merged.sort_values("timestamp", na_position="last")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )
    return merged


def _fsync_path(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_duckdb(
    database_path: Path,
    parquet_root: Path,
    *,
    generation: str | None = None,
) -> None:
    """Bind a DuckDB view to exactly one captured immutable generation."""
    try:
        import duckdb
    except ImportError as exc:
        raise AresError("DuckDB is not installed. Install ARES base dependencies first.") from exc

    selected = generation if generation is not None else current_generation(parquet_root)
    base = generation_root(parquet_root, selected) if selected is not None else Path(parquet_root)
    raw_root = base / "raw"
    parquet_files = (
        sorted(
            path
            for path in raw_root.rglob("*.parquet")
            if path.is_file() and not path.name.startswith(".")
        )
        if raw_root.exists()
        else []
    )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database_path))
    try:
        if not parquet_files:
            connection.execute(
                """
                CREATE OR REPLACE VIEW ohlcv AS
                SELECT
                    CAST(NULL AS TIMESTAMPTZ) AS timestamp,
                    CAST(NULL AS DOUBLE) AS open,
                    CAST(NULL AS DOUBLE) AS high,
                    CAST(NULL AS DOUBLE) AS low,
                    CAST(NULL AS DOUBLE) AS close,
                    CAST(NULL AS DOUBLE) AS volume,
                    CAST(NULL AS VARCHAR) AS exchange,
                    CAST(NULL AS VARCHAR) AS symbol,
                    CAST(NULL AS VARCHAR) AS timeframe,
                    CAST(NULL AS VARCHAR) AS filename
                WHERE FALSE
                """
            )
            return
        quoted = ", ".join(
            "'" + path.resolve().as_posix().replace("'", "''") + "'" for path in parquet_files
        )
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW ohlcv AS
            SELECT *
            FROM read_parquet([{quoted}], union_by_name = true, filename = true)
            """
        )
    finally:
        connection.close()


def new_generation_id() -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:12]}"


def create_generation(
    root: Path,
    *,
    frames: dict[str, pd.DataFrame],
    symbol: str,
    timeframe: str,
    quality_documents: dict[str, Any],
    metadata: dict[str, Any],
    generation: str | None = None,
    fault_hook: Callable[[str], None] | None = None,
) -> str:
    """Write and fsync a complete unpublished immutable generation."""
    root = Path(root)
    generation = generation or new_generation_id()
    directory = generation_root(root, generation)
    directory.mkdir(parents=True, exist_ok=False)
    if fault_hook is not None:
        fault_hook("generation_created")
    try:
        for exchange, frame in sorted(frames.items()):
            path = market_path(root, exchange, symbol, timeframe, generation=generation)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            _fsync_path(path)
            _fsync_path(path.parent)
            if fault_hook is not None:
                fault_hook(f"venue_written:{exchange}")

        for name, payload in sorted(quality_documents.items()):
            atomic_write_json(quality_path(root, name, generation=generation), payload)
        if fault_hook is not None:
            fault_hook("quality_written")

        database = generation_duckdb_path(root, generation)
        sync_duckdb(database, root, generation=generation)
        _fsync_path(database)
        if fault_hook is not None:
            fault_hook("duckdb_written")

        files: dict[str, dict[str, Any]] = {}
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if path == directory / GENERATION_MANIFEST:
                continue
            relative = path.relative_to(directory).as_posix()
            files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        manifest = {
            "schema": GENERATION_SCHEMA,
            "generation": generation,
            "created_at": datetime.now(tz=UTC),
            "symbol": symbol,
            "timeframe": timeframe,
            "exchanges": sorted(frames),
            "metadata": metadata,
            "files": files,
        }
        atomic_write_json(directory / GENERATION_MANIFEST, manifest)
        _fsync_path(directory / GENERATION_MANIFEST)
        _fsync_path(directory)
        if fault_hook is not None:
            fault_hook("manifest_written")
        verify_generation(root, generation)
        for path in sorted(directory.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        directory.chmod(0o555)
        _fsync_path(snapshots_root(root))
        return generation
    except Exception:
        # Deliberately retain the unpublished directory as crash evidence. It is
        # unreachable through CURRENT and may be removed by explicit cleanup.
        raise


def verify_generation(root: Path, generation: str) -> dict[str, Any]:
    directory = generation_root(root, generation)
    manifest_path = directory / GENERATION_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AresError(f"Generation {generation} has no manifest")
    manifest_stat = manifest_path.stat(follow_symlinks=False)
    if not stat.S_ISREG(manifest_stat.st_mode) or manifest_stat.st_nlink != 1:
        raise AresError(f"Generation {generation} manifest must be a single regular file")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key: {key}")
            payload[key] = value
        return payload

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except (OSError, ValueError) as exc:
        raise AresError(f"Generation {generation} manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise AresError(f"Generation {generation} manifest root is invalid")
    if manifest.get("schema") != GENERATION_SCHEMA or manifest.get("generation") != generation:
        raise AresError(f"Generation {generation} manifest identity is invalid")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise AresError(f"Generation {generation} manifest has no files")
    # JSON object keys are always strings. Validate their lexical safety before
    # comparing them with the on-disk set so a traversal-shaped declaration is
    # diagnosed directly rather than hidden behind a generic set mismatch.
    for relative, expected in declared.items():
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or not isinstance(expected, dict)
        ):
            raise AresError(f"Generation {generation} contains an unsafe manifest path")
    all_entries = list(directory.rglob("*"))
    special = [
        path
        for path in all_entries
        if path.is_symlink() or (not path.is_file() and not path.is_dir())
    ]
    if special:
        raise AresError(f"Generation {generation} contains symbolic or special entries")
    actual = {
        path.relative_to(directory).as_posix()
        for path in all_entries
        if path.is_file() and path != manifest_path
    }
    if actual != set(declared):
        raise AresError(f"Generation {generation} file set differs from its manifest")
    for relative, expected in declared.items():
        pure = PurePosixPath(relative)
        path = directory.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file():
            raise AresError(f"Generation {generation} contains a non-regular file")
        file_stat = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise AresError(f"Generation {generation} contains a linked or special file")
        expected_bytes = expected.get("bytes")
        expected_hash = expected.get("sha256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise AresError(f"Generation {generation} byte count is invalid for {relative}")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise AresError(f"Generation {generation} checksum is invalid for {relative}")
        if file_stat.st_size != expected_bytes:
            raise AresError(f"Generation {generation} size mismatch for {relative}")
        if sha256_file(path) != expected_hash:
            raise AresError(f"Generation {generation} checksum mismatch for {relative}")

    exchanges = manifest.get("exchanges")
    symbol = manifest.get("symbol")
    timeframe = manifest.get("timeframe")
    if (
        not isinstance(exchanges, list)
        or not exchanges
        or not all(isinstance(exchange, str) for exchange in exchanges)
        or len(exchanges) != len(set(exchanges))
        or not isinstance(symbol, str)
        or not isinstance(timeframe, str)
    ):
        raise AresError(f"Generation {generation} market identity is invalid")
    required = {"ares.duckdb", "quality/latest.json"}
    required.update(
        market_path(root, exchange, symbol, timeframe, generation=generation)
        .relative_to(directory)
        .as_posix()
        for exchange in exchanges
    )
    if not required.issubset(actual):
        raise AresError(f"Generation {generation} is missing required canonical artifacts")
    return cast(dict[str, Any], manifest)


def publish_generation(
    root: Path,
    generation: str,
    *,
    fault_hook: Callable[[str], None] | None = None,
) -> None:
    """Atomically make a verified complete generation canonical."""
    root = Path(root)
    verify_generation(root, generation)
    root.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=root, prefix=".CURRENT.")
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(generation + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if fault_hook is not None:
            fault_hook("before_pointer")
        os.replace(temp_name, root / CURRENT_NAME)
        _fsync_path(root)
        if fault_hook is not None:
            fault_hook("after_pointer")
    finally:
        Path(temp_name).unlink(missing_ok=True)


def cleanup_abandoned_generations(root: Path) -> list[Path]:
    """Remove only unpublished generations that do not have a valid complete manifest."""
    root = Path(root)
    active = current_generation(root) if (root / CURRENT_NAME).exists() else None
    snapshots = snapshots_root(root)
    removed: list[Path] = []
    if not snapshots.exists():
        return removed
    for directory in sorted(item for item in snapshots.iterdir() if item.is_dir()):
        if directory.name == active:
            continue
        try:
            verify_generation(root, directory.name)
        except AresError as exc:
            if directory.parent.resolve() != snapshots.resolve():
                raise AresError(
                    "Refusing to clean a generation outside the snapshots directory"
                ) from exc
            shutil.rmtree(directory)
            removed.append(directory)
    return removed
