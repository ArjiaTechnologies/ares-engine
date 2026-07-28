#!/usr/bin/env python3
"""Render the ARES v0.1.0 launch demo from authentic JSON captures."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1440
HEIGHT = 900
BACKGROUND = "#08111f"
PANEL = "#101b2d"
TERMINAL = "#07101a"
WHITE = "#f6f8fb"
MUTED = "#aab6c8"
TEAL = "#42d3b5"
BLUE = "#61a8ff"
PURPLE = "#a98bff"
ORANGE = "#ff9b54"
RED = "#ff6684"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Capture is not a JSON object: {path}")
    return payload


def font(size: int, *, mono: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
        if mono
        else [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/SFNSRounded.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def base_frame(
    kicker: str, title: str, subtitle: str = ""
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((44, 34, 242, 72), radius=19, fill="#14352f", outline=TEAL, width=2)
    draw.text((143, 53), kicker.upper(), font=font(17, bold=True), fill=TEAL, anchor="mm")
    draw.text((56, 104), title, font=font(42, bold=True), fill=WHITE)
    if subtitle:
        draw.text((58, 158), subtitle, font=font(23), fill=MUTED)
    return image, draw


def footer(draw: ImageDraw.ImageDraw, scene: int, text: str) -> None:
    draw.line((56, 845, 1384, 845), fill="#27364d", width=2)
    draw.text((58, 867), text, font=font(18), fill=MUTED, anchor="lm")
    draw.text((1382, 867), f"{scene}/6", font=font(18, mono=True), fill="#6f8099", anchor="rm")


def terminal(
    draw: ImageDraw.ImageDraw,
    command_lines: list[str],
    output_lines: list[tuple[str, str]],
    *,
    caption: str,
) -> None:
    draw.rounded_rectangle(
        (56, 204, 1384, 812), radius=20, fill=TERMINAL, outline="#2b3b52", width=2
    )
    for index, color in enumerate((RED, "#ffc857", TEAL)):
        x = 86 + index * 28
        draw.ellipse((x, 228, x + 14, 242), fill=color)
    draw.text(
        (184, 236),
        "sanitized shell · authentic output",
        font=font(17, mono=True),
        fill="#74849b",
        anchor="lm",
    )
    y = 278
    mono = font(23, mono=True)
    for index, line in enumerate(command_lines):
        prompt = "$ " if index == 0 else "  "
        rendered = line if line.startswith("$ ") else prompt + line
        draw.text((86, y), rendered, font=mono, fill=WHITE)
        y += 34
    y += 16
    draw.line((86, y, 1352, y), fill="#233249", width=2)
    y += 24
    for line, color in output_lines:
        draw.text((86, y), line, font=mono, fill=color)
        y += 34
    draw.rounded_rectangle((84, 748, 1354, 790), radius=10, fill="#11283a")
    draw.text((104, 769), caption, font=font(18, bold=True), fill=TEAL, anchor="lm")


def scene_identity(architecture_png: Path) -> Image.Image:
    image, draw = base_frame(
        "OPEN SOURCE",
        "ARES Engine v0.1.0",
        "ML trading research and paper inference — no live orders",
    )
    diagram = Image.open(architecture_png).convert("RGB")
    diagram.thumbnail((1260, 580), Image.Resampling.LANCZOS)
    x = (WIDTH - diagram.width) // 2
    y = 196
    draw.rounded_rectangle(
        (x - 3, y - 3, x + diagram.width + 3, y + diagram.height + 3), radius=15, fill="#32445f"
    )
    image.paste(diagram, (x, y))
    draw.text(
        (720, 812),
        "github.com/ArjiaTechnologies/ares-engine",
        font=font(25, mono=True),
        fill=BLUE,
        anchor="mm",
    )
    footer(draw, 1, "Architecture shown from docs/assets/ares-architecture.svg")
    return image


def scene_doctor(doctor: dict[str, Any]) -> Image.Image:
    image, draw = base_frame(
        "ENVIRONMENT", "Check the runtime", "The installed CLI reports its own dependency state"
    )
    packages = doctor["packages"]
    terminal(
        draw,
        ["ares doctor"],
        [
            (f'"ares": "{doctor["ares"]}"', BLUE),
            (f'"python": "{doctor["python"]}"', WHITE),
            (f'"supported_python": {str(doctor["supported_python"]).lower()}', TEAL),
            (f'"tensorflow": "{packages["tensorflow"]}"', WHITE),
            (f'"ready_for_ml": {str(doctor["ready_for_ml"]).lower()}', TEAL),
            (f'"reference_backend": "{doctor["reference_backend"]}"', PURPLE),
        ],
        caption="Real ares doctor output · successful environment and ML dependency checks",
    )
    footer(draw, 2, "Capture: ares doctor · exit 0")
    return image


def scene_demo(demo: dict[str, Any]) -> Image.Image:
    image, draw = base_frame(
        "OFFLINE DATA", "Run the deterministic demo", "No network, credentials, or orders"
    )
    primary = demo["primary_quality"]
    cross = demo["cross_venue_quality"]
    terminal(
        draw,
        ["ares demo --bars 500"],
        [
            (f'"primary_quality.passed": {str(primary["passed"]).lower()}', TEAL),
            (f'"primary_quality.stats.rows": {primary["stats"]["rows"]}', WHITE),
            (
                f'"primary_quality.stats.missing_candle_slots": {primary["stats"]["missing_candle_slots"]}',
                WHITE,
            ),
            (f'"cross_venue_quality.passed": {str(cross["passed"]).lower()}', TEAL),
            (f'"cross_venue_quality.stats.overlap_rows": {cross["stats"]["overlap_rows"]}', WHITE),
            (
                f'"cross_venue_quality.stats.p95_divergence_bps": {cross["stats"]["p95_divergence_bps"]:.6f}',
                PURPLE,
            ),
        ],
        caption="Authentic deterministic output · 500 aligned bars pass quality checks",
    )
    footer(draw, 3, "Capture: ares demo --bars 500 · exit 0")
    return image


def scene_lifecycle(offline: dict[str, Any]) -> Image.Image:
    image, draw = base_frame(
        "MODEL LIFECYCLE", "Verify, promote, reload, infer", "Idle training time condensed"
    )
    promotion = offline["promotion"]
    signal = offline["paper_signal"]
    validation = offline["validation"]
    bundle_files = offline["bundle_files"]
    terminal(
        draw,
        [
            "ares verify-offline --config configs/smoke.yaml --bars 2000 \\",
            "  --output-dir /tmp/ares-demo-verification",
        ],
        [
            (f'"verification_type": "{offline["verification_type"]}"', BLUE),
            (f'"backend": "{offline["backend"]}"', WHITE),
            (f'"bundle_files": {json.dumps(bundle_files[:3])[:-1]},', PURPLE),
            (f"  {json.dumps(bundle_files[3:])[1:]}", PURPLE),
            (f'"validation.passed": {str(validation["passed"]).lower()}', TEAL),
            (
                f'"validation.aggregate.bankrupt_folds": {validation["aggregate"]["bankrupt_folds"]}',
                WHITE,
            ),
            (f'"promotion.approved": {str(promotion["approved"]).lower()}', TEAL),
            (
                f'"paper_signal.data_quality_passed": {str(signal["data_quality_passed"]).lower()}',
                TEAL,
            ),
            (f'"paper_signal.signal": {signal["signal"]}  // read-only', ORANGE),
            (f'"profitability_claim": {str(offline["profitability_claim"]).lower()}', MUTED),
        ],
        caption="Real lifecycle output · bundle verified, promoted, reloaded, paper signal emitted",
    )
    footer(draw, 4, "Capture: ares verify-offline · exit 0 · no order execution")
    return image


def scene_public(public: dict[str, Any], validated: dict[str, Any]) -> Image.Image:
    image, draw = base_frame(
        "PUBLIC ENDPOINTS",
        "Verify Coinbase + Kraken",
        "Real public verification — idle time condensed",
    )
    terminal(
        draw,
        [
            "ares verify-public-ingestion --primary coinbase --validation kraken \\",
            "  --symbol ETH/USD --timeframe 1h \\",
            f"  --start {public['requested_start'][:19]}Z --end {public['requested_end'][:19]}Z \\",
            "  --page-limit 60 --retries 3 --output /tmp/ares-public-ingestion-launch",
            "$ ares validate-ingestion-report /tmp/ares-public-ingestion-launch",
        ],
        [
            (
                f'"coinbase": requests={public["primary_http_requests"]}, pages={public["primary_pages"]}, rows={public["primary_rows"]}',
                BLUE,
            ),
            (
                f'"kraken": requests={public["validation_http_requests"]}, pages={public["validation_pages"]}, rows={public["validation_rows"]}',
                PURPLE,
            ),
            (
                f'"quality_passed": {str(public["quality_passed"]).lower()}, '
                f'"alignment_passed": {str(public["alignment_passed"]).lower()}',
                TEAL,
            ),
            (f'"overall_passed": {str(public["overall_passed"]).lower()}', TEAL),
            (
                f'"credentials_used": {str(public["credentials_used"]).lower()}, '
                f'"orders_possible": {str(public["orders_possible"]).lower()}',
                ORANGE,
            ),
            (f'"validate-ingestion-report.valid": {str(validated["valid"]).lower()}', TEAL),
        ],
        caption="Real public Coinbase + Kraken verification — idle time condensed",
    )
    footer(draw, 5, "Captures: verify-public-ingestion + validate-ingestion-report · exit 0")
    return image


def scene_call_to_action() -> Image.Image:
    image, draw = base_frame("ARES ENGINE v0.1.0", "Stress-test the boundary conditions")
    draw.rounded_rectangle((130, 236, 1310, 706), radius=30, fill=PANEL, outline="#2a3b54", width=2)
    lines = [
        ("Clone it.", WHITE),
        ("Run the verifier.", TEAL),
        ("Open an issue if you break it.", WHITE),
    ]
    y = 320
    for line, color in lines:
        draw.text((720, y), line, font=font(47, bold=True), fill=color, anchor="mm")
        y += 90
    draw.text(
        (720, 640),
        "github.com/ArjiaTechnologies/ares-engine",
        font=font(29, mono=True),
        fill=BLUE,
        anchor="mm",
    )
    draw.rounded_rectangle((485, 722, 955, 775), radius=26, fill="#3a1721", outline=RED, width=2)
    draw.text((720, 748), "NO LIVE-ORDER PATH", font=font(20, bold=True), fill=RED, anchor="mm")
    footer(draw, 6, "Public data only · no credentials required · sanitize reports before posting")
    return image


def validate_captures(
    doctor: dict[str, Any],
    demo: dict[str, Any],
    offline: dict[str, Any],
    public: dict[str, Any],
    validated: dict[str, Any],
) -> None:
    checks = {
        "doctor supported Python": doctor.get("supported_python") is True,
        "doctor ML readiness": doctor.get("ready_for_ml") is True,
        "demo primary quality": demo.get("primary_quality", {}).get("passed") is True,
        "demo cross-venue quality": demo.get("cross_venue_quality", {}).get("passed") is True,
        "offline validation": offline.get("validation", {}).get("passed") is True,
        "offline promotion": offline.get("promotion", {}).get("approved") is True,
        "offline paper signal": offline.get("paper_signal", {}).get("data_quality_passed") is True,
        "offline no profitability claim": offline.get("profitability_claim") is False,
        "public overall": public.get("overall_passed") is True,
        "public no credentials": public.get("credentials_used") is False,
        "public no orders": public.get("orders_possible") is False,
        "report validation": validated.get("valid") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Refusing to render from captures that did not pass: {failed}")


def render_video(scenes: list[Image.Image], output: Path, durations: list[int]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render the H.264 MP4")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ares-demo-render.") as directory_name:
        directory = Path(directory_name)
        concat_lines: list[str] = []
        for index, (scene, duration) in enumerate(zip(scenes, durations, strict=True)):
            image_path = directory / f"scene-{index:02d}.png"
            video_path = directory / f"scene-{index:02d}.mp4"
            scene.save(image_path, format="PNG", optimize=True)
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-loop",
                    "1",
                    "-framerate",
                    "30",
                    "-i",
                    str(image_path),
                    "-t",
                    str(duration),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "slow",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    str(video_path),
                ],
                check=True,
            )
            concat_lines.append(f"file '{video_path}'")
        concat_file = directory / "scenes.txt"
        concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doctor", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--validated", type=Path, required=True)
    parser.add_argument("--architecture-png", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    doctor = load_json(args.doctor)
    demo = load_json(args.demo)
    offline = load_json(args.offline)
    public = load_json(args.public)
    validated = load_json(args.validated)
    validate_captures(doctor, demo, offline, public, validated)

    scenes = [
        scene_identity(args.architecture_png),
        scene_doctor(doctor),
        scene_demo(demo),
        scene_lifecycle(offline),
        scene_public(public, validated),
        scene_call_to_action(),
    ]
    render_video(scenes, args.output, [9, 9, 9, 14, 14, 8])


if __name__ == "__main__":
    main()
