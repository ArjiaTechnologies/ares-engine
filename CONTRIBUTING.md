# Contributing

1. Create a focused branch and keep unrelated changes separate.
2. Use Python 3.11, 3.12, or 3.13 and install `uv sync --extra dev --extra ml`.
3. Run compileall, Ruff lint and format checks, strict mypy, the complete branch-aware pytest suite, `pip-audit`, package build, and `twine check`.
4. Test an installed wheel outside the source checkout when packaging or CLI behavior changes.
5. Add independent or adversarial tests for timestamp alignment, leakage, scaling, fold boundaries, cost accounting, bankruptcy, storage publication, evidence validation, bundle integrity, promotion, and inference gates.
6. Do not weaken a data, risk, integrity, or freshness gate to make a result pass.

Research pull requests must state chronology, data sources, fees, slippage, drawdown, turnover, trade count, rejected candidates, and limitations. A headline return is not sufficient evidence.
