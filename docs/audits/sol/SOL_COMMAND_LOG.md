# Sol command log

Commands are summarized with material outcomes; secrets and noisy package/build output are intentionally omitted. All timestamps are 2026-07-28 UTC unless noted.

| Area | Command or operation | Result |
|---|---|---|
| Input hashes | `shasum -a 256 -c ARES-Engine-Fable5-Audit-SHA256SUMS.txt` | Every supplied entry matched |
| Git provenance | `git bundle verify`, `git fsck --full`, `git log --graph --all`, full baseline-to-Fable diff | Bundle valid and complete; expected commits/trees verified; every hunk read |
| Archive equivalence | Extract tar and zip, initialize Git, compare tree | Both reconstructed exact Fable final tree `cd2790c…` |
| GitHub preflight | `gh auth status`; repository metadata; `git ls-remote` | Authenticated as Dalekware; existing repository empty; visibility changed to and verified PRIVATE before source push |
| Static | `python -m compileall -q src tests` | Pass |
| Static | `ruff check .` | Pass |
| Static | `ruff format --check .` | 75 files formatted |
| Static | `mypy --strict src` | Pass, 24 source files |
| Tests | `pytest --cov=ares_engine --cov-branch --cov-report=term-missing` | 245 passed; 0 skipped; 32 warnings; 90% total |
| Dependencies | `pip-audit` | No known vulnerabilities |
| Build | `python -m build`; `twine check dist/*` | sdist and wheel built; both passed |
| Installed wheel | Clean temporary venv outside checkout; all CLI `--help`; `version`; `doctor`; `demo --bars 500` | Pass; `0.1.0`; Python 3.12.13; deterministic quality demo passed |
| Live ingestion | `ares verify-public-ingestion --primary coinbase --validation kraken --symbol ETH/USD --timeframe 1h --start 2026-07-10T18:00:00Z --end 2026-07-25T18:00:00Z --page-limit 60 --retries 3` | Pass; live endpoints; two runs; 360 rows per venue |
| Live validation | `ares validate-ingestion-report artifacts/public-ingestion-audit` | `valid: true`, zero problems |
| Docker daemon | `docker info` after launching Docker Desktop | Server 29.5.3 |
| Docker build | `docker build --target ml --tag ares-engine:sol .` | Pass, Linux ARM64 image |
| Docker runtime | UID/writability checks, `ares doctor`, `ares demo --bars 500`, `ares verify-offline ... --bars 2000` | Pass; UID 1000; Python 3.11.15; TensorFlow 2.21.0; complete lifecycle passed |
| Docker stop | Start trap loop, `docker stop`, inspect exit, remove test container | Exit code 0; Compose config valid |
| Git commits | Focused `git commit` operations | `00fe9cc`, `8dbac5d`, `34472f5`, `1cd3fed`, `da0df74`, `ad19c4a`, `bf92447`, `8b52bb3` before final evidence sealing |
| First dependency audit run | Private CI run `30388440422` | Found PyArrow 21.0.0 / `PYSEC-2026-113`; clean-environment-only defect reproduced |
| Patched CI | Private CI run `30388958288` | Seven of seven jobs passed: Python 3.11–3.13, ML lifecycle, package/wheel, dependency audit, Docker |
| Python matrix | Jobs `90375288426`, `90375288317`, `90375288280` | 195 non-ML passed and 50 ML deselected on each interpreter; compile/Ruff/mypy green |
| ML | Job `90375288206` | 50 ML passed; 195 deselected; 2,000-bar offline lifecycle passed |
| Dependency Review | Workflow `30388958215` | Workflow passed; review action skipped because private GHAS is unavailable; explicit status recorded |
| CodeQL | Workflow `30388958147` | Workflow passed; analysis skipped because private Code Security is unavailable; pinned v4.36.0 job retained |

Draft PR: `https://github.com/ArjiaTechnologies/ares-engine/pull/1`. Merge commit, final privacy proof, and no-release proof are recorded in the completion response because a commit cannot contain its own future merge identity.
