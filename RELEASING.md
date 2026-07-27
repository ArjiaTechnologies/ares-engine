# Releasing ARES Engine

1. Run the complete checks in a clean environment:

   ```bash
   uv sync --extra dev --extra ml
   uv run ruff check .
   uv run pytest -m "not ml" --cov
   uv run pytest -m ml
   uv run ares verify-offline --config configs/smoke.yaml
   uv build
   ```

2. Update `CHANGELOG.md`, `CITATION.cff`, and the version in `pyproject.toml` and `src/ares_engine/__init__.py`.
3. Commit the release, create an annotated `vX.Y.Z` tag, and push the tag.
4. `.github/workflows/release.yml` builds the source/wheel artifacts and creates the GitHub release from the verified tag.
5. Do not publish model bundles or market datasets as source releases unless their licenses, provenance, hashes, and risk statement have been reviewed separately.
