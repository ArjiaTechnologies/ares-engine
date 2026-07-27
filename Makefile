.PHONY: install install-ml test test-ml lint format doctor demo ingest validate

install:
	uv sync --extra dev

install-ml:
	uv sync --extra dev --extra ml

test:
	uv run pytest -m "not ml"

test-ml:
	uv run pytest -m ml

lint:
	uv run ruff check .

format:
	uv run ruff format .

doctor:
	uv run ares doctor

demo:
	uv run ares demo

ingest:
	uv run ares ingest --config configs/default.yaml

validate:
	uv run ares validate --config configs/default.yaml
