# Contributing

1. Fork the repository and create a focused branch.
2. Install Python 3.11 or 3.12 and run `uv sync --extra dev`.
3. Run `uv run ruff check .`, `uv run pytest -m "not ml"`, and relevant ML smoke tests.
4. Add tests for every behavior change, especially timestamp alignment, scaling, split boundaries,
   cost accounting, bundle integrity, and promotion gates.
5. Keep pull requests small. Trading research becomes unauditable when unrelated changes are mixed.

Do not submit strategies that report only headline return. Include assumptions, chronology, costs,
drawdown, turnover, trade count, and failure cases.
