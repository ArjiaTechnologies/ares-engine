# ARES v0.1.0 demo provenance

The public demo is assembled from real ARES CLI JSON, not invented terminal output. Captions select fields from the complete outputs so they remain readable at video speed. Training and network idle time are condensed.

## Capture commands

Run from a clean checkout after `uv sync --extra dev --extra ml`:

```bash
uv run ares doctor > /tmp/ares-launch-doctor.json
uv run ares demo --bars 500 > /tmp/ares-launch-demo.json
uv run ares verify-offline \
  --config configs/smoke.yaml --bars 2000 \
  --output-dir /tmp/ares-demo-verification \
  > /tmp/ares-launch-offline.json

eval "$(uv run python - <<'PY'
from datetime import UTC, datetime, timedelta
end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=72)
start = end - timedelta(days=15)
print(f'ARES_START={start:%Y-%m-%dT%H:%M:%SZ}')
print(f'ARES_END={end:%Y-%m-%dT%H:%M:%SZ}')
PY
)"
uv run ares verify-public-ingestion \
  --primary coinbase --validation kraken \
  --symbol ETH/USD --timeframe 1h \
  --start "$ARES_START" --end "$ARES_END" \
  --page-limit 60 --retries 3 \
  --output /tmp/ares-public-ingestion-launch \
  > /tmp/ares-launch-public.json
uv run ares validate-ingestion-report \
  /tmp/ares-public-ingestion-launch \
  > /tmp/ares-launch-public-validation.json
```

The launch capture used the interval `2026-07-10T22:00:00Z` through `2026-07-25T22:00:00Z`, selected at runtime on 2026-07-28 with a 72-hour safety margin. Both isolated runs returned 360 rows per venue across six measured pages. The independent validator returned `valid: true` with no problems.

Capture SHA-256 values:

```text
doctor                  c9ececa5abade4914d7232d7c394bd6471af56e8aab4cce8952decdcd70fe69a
demo                    8d0d4b227900ba0403f0cb4cac092fee73f204db1defeec58aa902af7c3cfe1a
offline lifecycle       c776a00d99b9c3ebac5c79c7ee22b19124717c885801ed947cb178338424d106
public ingestion        1f27763eb76856df113016f043e9860c22957c9429a2e8ef359eedf99c040674
public report validator ced0a725c2e4f60f8733a7106361887f61dc42c48cc0a5b38859e0e71aed4503
```

`render_demo.py` refuses to render unless the captured safety and success fields pass. It displays only controlled fields, so no username, checkout path, hostname, credential, token, account data, or unrelated application appears.

## Render

Rasterize the SVG with an SVG renderer, then run:

```bash
uv run --with pillow python docs/launch/demo/render_demo.py \
  --doctor /tmp/ares-launch-doctor.json \
  --demo /tmp/ares-launch-demo.json \
  --offline /tmp/ares-launch-offline.json \
  --public /tmp/ares-launch-public.json \
  --validated /tmp/ares-launch-public-validation.json \
  --architecture-png /tmp/ares-architecture.png \
  --output docs/assets/ares-v0.1.0-demo.mp4
```

The render is captioned and silent. `ffmpeg` writes H.264 video with `yuv420p` compatibility and fast-start metadata.
