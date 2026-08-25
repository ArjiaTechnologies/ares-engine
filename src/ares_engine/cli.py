"""ARES Engine command-line interface."""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
import platform
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from . import __version__
from .config import AresConfig, load_config
from .data.ingest import ingest_market_data
from .data.quality import validate_cross_venue, validate_ohlcv
from .data.storage import current_generation, market_path, quality_path, read_market
from .holdout import run_locked_holdout
from .live import generate_paper_signal
from .models import backend_name
from .promotion import promote as promote_bundle
from .promotion import resolve_champion
from .public_audit import run_public_ingestion_audit, validate_public_ingestion_report
from .scheduler import deep_cycle, quick_cycle, run_scheduler
from .search import run_search
from .synthetic import make_synthetic_ohlcv
from .training import train_candidate
from .utils import atomic_write_json, timeframe_to_seconds, utc_now
from .validation import minimum_required_bars, run_walk_forward

app = typer.Typer(
    name="ares",
    help="ARES Engine - auditable ETH ML research and paper-trading pipeline.",
    no_args_is_help=True,
)


def _resolve_config(path: Path) -> Path:
    """Resolve a config path: filesystem first, then the packaged defaults.

    Installed wheels have no ./configs checkout, so `configs/default.yaml` and
    `configs/smoke.yaml` fall back to the copies shipped inside the package.
    """
    if path.exists():
        return path
    if path.parent == Path("configs"):
        packaged = importlib.resources.files("ares_engine") / "configs" / path.name
        if packaged.is_file():
            with importlib.resources.as_file(packaged) as concrete:
                return Path(concrete)
    raise typer.BadParameter(f"Configuration file not found: {path}")


def _json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _primary_frame(config_path: Path) -> tuple[AresConfig, Path, pd.DataFrame]:
    config = load_config(_resolve_config(config_path))
    generation = current_generation(config.storage.root)
    path = market_path(
        config.storage.root,
        config.data.primary_exchange,
        config.data.symbol,
        config.data.timeframe,
        generation=generation,
    )
    frame = read_market(path)
    if frame.empty:
        raise typer.BadParameter(f"No primary data at {path}; run `ares ingest` first")
    secondary_frames = _secondary_frames(config, generation=generation)
    reference = utc_now()
    freshness_limit = config.data.max_staleness_bars if config.data.until is None else None
    failures: list[str] = []
    primary_report = validate_ohlcv(
        frame,
        config.data.timeframe,
        source=config.data.primary_exchange,
        max_gap_count=config.data.max_gap_count,
        expected_exchange=config.data.primary_exchange,
        expected_symbol=config.data.symbol,
        expected_timeframe=config.data.timeframe,
        as_of=reference if freshness_limit is not None else None,
        max_staleness_bars=freshness_limit,
        expected_start=config.data.since,
        expected_end=config.data.until,
    )
    failures.extend(f"{primary_report.source}: {issue.message}" for issue in primary_report.issues)
    for exchange in config.data.validation_exchanges:
        secondary = secondary_frames[exchange]
        secondary_report = validate_ohlcv(
            secondary,
            config.data.timeframe,
            source=exchange,
            max_gap_count=config.data.max_gap_count,
            expected_exchange=exchange,
            expected_symbol=config.data.symbol,
            expected_timeframe=config.data.timeframe,
            as_of=reference if freshness_limit is not None else None,
            max_staleness_bars=freshness_limit,
            expected_start=None,
            expected_end=config.data.until,
        )
        failures.extend(
            f"{secondary_report.source}: {issue.message}" for issue in secondary_report.issues
        )
        required_cross_columns = {"timestamp", "close"}
        if required_cross_columns.issubset(frame.columns) and required_cross_columns.issubset(
            secondary.columns
        ):
            cross = validate_cross_venue(
                frame,
                secondary,
                primary_name=config.data.primary_exchange,
                secondary_name=exchange,
                max_p95_bps=config.data.max_cross_venue_p95_bps,
                min_overlap=config.data.min_cross_venue_overlap,
                max_latest_bps=config.data.max_cross_venue_latest_bps,
            )
            failures.extend(f"{cross.source}: {issue.message}" for issue in cross.issues)
    if failures:
        raise typer.BadParameter(
            "Stored market data failed fresh validation; rerun `ares ingest`: "
            + "; ".join(failures)
        )
    return config, path, frame


def _secondary_frames(
    config: AresConfig, *, generation: str | None = None
) -> dict[str, pd.DataFrame]:
    return {
        exchange: read_market(
            market_path(
                config.storage.root,
                exchange,
                config.data.symbol,
                config.data.timeframe,
                generation=generation,
            )
        )
        for exchange in config.data.validation_exchanges
    }


@app.command()
def version() -> None:
    """Print the installed ARES Engine version."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Check runtime compatibility and dependency availability."""
    packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "pyarrow",
        "duckdb",
        "ccxt",
        "optuna",
        "tensorflow-cpu",
        "tensorflow",
        "keras",
        "torch",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 14)
    _json(
        {
            "ares": __version__,
            "python": platform.python_version(),
            "supported_python": supported_python,
            "platform": platform.platform(),
            "packages": versions,
            "ready_for_ml": bool(
                versions["tensorflow-cpu"]
                or versions["tensorflow"]
                or (versions["keras"] and versions["torch"])
            ),
            "reference_backend": "tensorflow",
            "compatibility_backend": "keras+torch",
        }
    )
    if not supported_python:
        raise typer.Exit(code=1)


@app.command()
def ingest(
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
) -> None:
    """Fetch Coinbase and validation venues, then enforce data quality gates."""
    config = load_config(_resolve_config(config_path))
    result = ingest_market_data(config)
    _json(result.to_dict())


@app.command("quality")
def quality_report(
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
) -> None:
    """Print the most recent persisted data-quality report."""
    config = load_config(_resolve_config(config_path))
    generation = current_generation(config.storage.root)
    path = quality_path(config.storage.root, "latest", generation=generation)
    if not path.exists():
        raise typer.BadParameter("No quality report exists; run `ares ingest` first")
    typer.echo(path.read_text(encoding="utf-8"), nl=False)


@app.command()
def validate(
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    verbose: int = typer.Option(0, min=0, max=2),
) -> None:
    """Run leak-aware walk-forward validation without exporting a bundle."""
    config, _, frame = _primary_frame(config_path)
    summary = run_walk_forward(frame, config, verbose=verbose)
    _json(summary.to_dict())
    if not summary.passed:
        raise typer.Exit(code=2)


@app.command()
def search(
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    verbose: int = typer.Option(0, min=0, max=2),
) -> None:
    """Run Optuna and persist best_params.json, best_meta.json, and best_config.yaml."""
    config, _, frame = _primary_frame(config_path)
    study, best_config = run_search(frame, config, verbose=verbose)
    selected_number = int(study.user_attrs["ares_selected_trial_number"])
    selected_trial = next(trial for trial in study.trials if trial.number == selected_number)
    _json(
        {
            "study": study.study_name,
            "trials": len(study.trials),
            "selected_trial": selected_trial.number,
            "selected_value": selected_trial.value,
            "selected_params": selected_trial.params,
            "best_config": best_config.model_dump(mode="json"),
        }
    )


@app.command("locked-holdout")
def locked_holdout(
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    holdout_bars: int = typer.Option(..., "--holdout-bars", min=50),
    embargo_bars: int | None = typer.Option(None, "--embargo-bars", min=1),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    verbose: int = typer.Option(0, min=0, max=2),
) -> None:
    """Quarantine final history, search earlier bars, and evaluate it exactly once."""
    config, _, frame = _primary_frame(config_path)
    report = run_locked_holdout(
        frame,
        config,
        holdout_bars=holdout_bars,
        embargo_bars=embargo_bars,
        output_dir=output_dir,
        verbose=verbose,
    )
    _json(report)


@app.command()
def train(
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    bundle_name: str | None = typer.Option(None, "--name"),
    verbose: int = typer.Option(0, min=0, max=2),
) -> None:
    """Validate, train on all directional samples, and export an immutable challenger bundle."""
    config, source, frame = _primary_frame(config_path)
    bundle, metrics = train_candidate(
        frame,
        config,
        source_path=source,
        bundle_name=bundle_name,
        repository_root=Path.cwd(),
        verbose=verbose,
    )
    _json({"bundle": bundle, "score": metrics["score"], "passed": metrics["passed"]})


@app.command()
def promote(
    challenger: Path = typer.Argument(..., exists=True, file_okay=False),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
) -> None:
    """Promote a challenger only when it passes gates and beats the incumbent."""
    config = load_config(_resolve_config(config_path))
    decision = promote_bundle(challenger, config.storage.artifacts, config.gates)
    _json(decision.to_dict())


@app.command()
def paper(
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    bundle: Path | None = typer.Option(None, "--bundle", file_okay=False),
) -> None:
    """Load a verified bundle and emit one read-only paper signal."""
    config = load_config(_resolve_config(config_path))
    selected = bundle or resolve_champion(config.storage.artifacts)
    if selected is None:
        raise typer.BadParameter("No champion exists and no --bundle was supplied")
    generation = current_generation(config.storage.root)
    primary = read_market(
        market_path(
            config.storage.root,
            config.data.primary_exchange,
            config.data.symbol,
            config.data.timeframe,
            generation=generation,
        )
    )
    result = generate_paper_signal(
        selected,
        primary,
        secondary_ohlcv=_secondary_frames(config, generation=generation),
        log_path=config.storage.root / "paper" / "signals.jsonl",
    )
    _json(result.to_dict())
    if not result.data_quality_passed:
        raise typer.Exit(code=3)


@app.command()
def demo(
    config_path: Path = typer.Option(Path("configs/smoke.yaml"), "--config"),
    bars: int = typer.Option(2_500, min=500),
    ml: bool = typer.Option(False, "--ml", help="Also run Keras walk-forward validation."),
) -> None:
    """Run deterministic offline data-quality and optional ML smoke tests."""
    config = load_config(_resolve_config(config_path))
    primary = make_synthetic_ohlcv(
        bars,
        timeframe=config.data.timeframe,
        seed=config.project.seed,
        exchange="coinbase-synthetic",
    )
    secondary = make_synthetic_ohlcv(
        bars,
        timeframe=config.data.timeframe,
        seed=config.project.seed,
        exchange="kraken-synthetic",
        price_noise_bps=2.0,
    )
    primary_report = validate_ohlcv(
        primary,
        config.data.timeframe,
        source="coinbase-synthetic",
        max_gap_count=0,
    )
    cross_report = validate_cross_venue(
        primary,
        secondary,
        primary_name="coinbase-synthetic",
        secondary_name="kraken-synthetic",
        max_p95_bps=20.0,
        min_overlap=100,
        max_latest_bps=40.0,
    )
    output: dict[str, Any] = {
        "primary_quality": primary_report.to_dict(),
        "cross_venue_quality": cross_report.to_dict(),
    }
    if ml:
        required_bars = minimum_required_bars(config)
        if bars < required_bars:
            raise typer.BadParameter(
                f"--bars {bars} cannot satisfy the configured folds; "
                f"at least {required_bars} rows are required"
            )
        summary = run_walk_forward(primary, config, verbose=0)
        output["validation"] = summary.to_dict()
    _json(output)


@app.command("verify-offline")
def verify_offline(
    config_path: Path = typer.Option(Path("configs/smoke.yaml"), "--config"),
    output_dir: Path = typer.Option(Path("artifacts/verification"), "--output-dir"),
    bars: int = typer.Option(2_500, min=1_500),
) -> None:
    """Exercise quality, ML, export, promotion, reload, and paper inference on synthetic data."""
    config = load_config(_resolve_config(config_path))
    required_bars = minimum_required_bars(config)
    if bars < required_bars:
        raise typer.BadParameter(
            f"--bars {bars} cannot satisfy the configured folds: feature warm-up, lookback, "
            f"min_train_bars, purge, and validation_bars need at least {required_bars} rows"
        )
    run_name = datetime.now(tz=UTC).strftime("run-%Y%m%dT%H%M%SZ")
    run_root = output_dir / run_name
    config.storage.artifacts = run_root

    primary = make_synthetic_ohlcv(
        bars,
        timeframe=config.data.timeframe,
        seed=config.project.seed,
        exchange="coinbase",
    )
    secondary = make_synthetic_ohlcv(
        bars,
        timeframe=config.data.timeframe,
        seed=config.project.seed,
        exchange="kraken",
        price_noise_bps=2.0,
    )
    primary_report = validate_ohlcv(
        primary, config.data.timeframe, source="coinbase", max_gap_count=0
    )
    secondary_report = validate_ohlcv(
        secondary, config.data.timeframe, source="kraken", max_gap_count=0
    )
    cross_report = validate_cross_venue(
        primary,
        secondary,
        primary_name="coinbase",
        secondary_name="kraken",
        max_p95_bps=20.0,
        min_overlap=100,
        max_latest_bps=40.0,
    )
    if not (primary_report.passed and secondary_report.passed and cross_report.passed):
        raise typer.BadParameter("Synthetic verification data unexpectedly failed quality gates")

    bundle, metrics = train_candidate(
        primary,
        config,
        bundle_name="ares_smoke_verified",
        repository_root=Path.cwd(),
        verbose=0,
    )
    decision = promote_bundle(bundle, run_root, config.gates)
    as_of = primary["timestamp"].max() + timedelta(
        seconds=timeframe_to_seconds(config.data.timeframe)
    )
    signal = generate_paper_signal(
        bundle,
        primary,
        secondary_ohlcv=secondary,
        log_path=run_root / "paper_signals.jsonl",
        as_of=as_of,
    )
    report = {
        "verification_type": "deterministic synthetic smoke lifecycle",
        "profitability_claim": False,
        "gate_profile": "smoke/permissive",
        "cost_stress_gate_required": config.gates.require_positive_cost_stress,
        "backend": backend_name(),
        "bundle": str(bundle),
        "bundle_files": sorted(path.name for path in bundle.iterdir() if path.is_file()),
        "quality": {
            "primary": primary_report.to_dict(),
            "secondary": secondary_report.to_dict(),
            "cross_venue": cross_report.to_dict(),
        },
        "validation": metrics,
        "promotion": decision.to_dict(),
        "paper_signal": signal.to_dict(),
    }
    report_path = run_root / "verification.json"
    atomic_write_json(report_path, report)
    _json({"report": report_path, **report})


@app.command("verify-public-ingestion")
def verify_public_ingestion(
    primary: str = typer.Option("coinbase", "--primary"),
    validation: str = typer.Option("kraken", "--validation"),
    symbol: str = typer.Option("ETH/USD", "--symbol"),
    timeframe: str = typer.Option("1h", "--timeframe"),
    start: datetime = typer.Option(..., "--start", formats=["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]),
    end: datetime = typer.Option(..., "--end", formats=["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]),
    output: Path = typer.Option(Path("artifacts/public-ingestion-audit"), "--output"),
    page_limit: int = typer.Option(300, min=10, max=1000),
    retries: int = typer.Option(3, min=0, max=10),
) -> None:
    """Run the credential-free bounded public-ingestion audit against live endpoints.

    Uses public market-data endpoints only, through the production ingestion
    path, in an isolated output directory. Runs ingestion twice to prove
    idempotency and exits nonzero when any required gate fails.
    """
    start_utc = start.replace(tzinfo=UTC) if start.tzinfo is None else start
    end_utc = end.replace(tzinfo=UTC) if end.tzinfo is None else end
    report = run_public_ingestion_audit(
        primary=primary,
        validation=validation,
        symbol=symbol,
        timeframe=timeframe,
        start=start_utc,
        end=end_utc,
        output_dir=output,
        page_limit=page_limit,
        retries=retries,
    )
    _json(report)
    problems = validate_public_ingestion_report(output)
    if problems:
        typer.echo("Report validation problems: " + "; ".join(problems), err=True)
        raise typer.Exit(code=4)
    if not report["overall_passed"]:
        raise typer.Exit(code=1)


@app.command("validate-ingestion-report")
def validate_ingestion_report(
    output: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Validate a public-ingestion audit directory; nonzero exit on any problem."""
    problems = validate_public_ingestion_report(output)
    _json({"output": output, "problems": problems, "valid": not problems})
    if problems:
        raise typer.Exit(code=4)


@app.command("cycle")
def cycle(
    mode: str = typer.Argument(..., help="quick or deep"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
) -> None:
    """Run one scheduler cycle immediately."""
    if mode == "quick":
        quick_cycle(config_path)
    elif mode == "deep":
        deep_cycle(config_path)
    else:
        raise typer.BadParameter("mode must be quick or deep")


@app.command("scheduler")
def scheduler_command(
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
) -> None:
    """Run recurring APScheduler quick and deep cycles."""
    run_scheduler(config_path)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
