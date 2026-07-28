# Sol GitHub Actions evidence

Repository: `ArjiaTechnologies/ares-engine` (private)

Draft PR: `https://github.com/ArjiaTechnologies/ares-engine/pull/1`

Audited code commit: `8b52bb3041c15fb939300ee6d1db165dfab1c4ca`

## Supported private checks

CI run `30388958288` passed all seven jobs:

| Job | Result | Evidence |
|---|---|---|
| Python 3.11 non-ML | PASS | 195 tests; 50 ML deselected; compile, Ruff lint/format, strict mypy; job `90375288426` |
| Python 3.12 non-ML | PASS | 195 tests; 50 ML deselected; compile, Ruff lint/format, strict mypy; job `90375288317` |
| Python 3.13 non-ML | PASS | 195 tests; 50 ML deselected; compile, Ruff lint/format, strict mypy; job `90375288280` |
| TensorFlow CPU lifecycle | PASS | 50 ML tests; 2,000-bar offline verification; job `90375288206` |
| Build and installed-wheel | PASS | sdist/wheel, Twine, clean wheel install, CLI/version/doctor/demo; job `90375288258` |
| Python dependency audit | PASS | No known vulnerabilities after PyArrow floor repair; job `90375288275` |
| Docker | PASS | Build, non-root/runtime/offline, graceful shutdown; job `90375288349` |

Workflow URL: `https://github.com/ArjiaTechnologies/ares-engine/actions/runs/30388958288`.

## Security feature availability

The first Dependency Review execution failed with GitHub's explicit message that the feature is unsupported without GitHub Advanced Security on this private repository. The workflow now runs the pinned action when the repository is public or private GHAS is enabled and otherwise records the limitation. Workflow `30388958215` is green; its unsupported-private step passed and review action was skipped.

The first CodeQL execution initialized and analyzed for 2m50s but GitHub rejected access/upload because code scanning is not enabled for this private repository. Official GitHub terms/documentation require private GitHub Code Security for CodeQL. The repaired workflow pins CodeQL v4.36.0 and runs it automatically when that capability exists; today it records an explicit unsupported-private result. Workflow `30388958147` is green; the status job passed and analysis job was skipped.

These are non-critical external product/licensing limitations, not source alerts. Ruff, strict mypy, dependency audit, adversarial tests, and the complete Python/ML/Docker matrix remain green. The Sol verdict is therefore `CONDITIONAL`, not `PASS`.

## Workflow controls

- Read-only default permissions; write scope only inside a supported CodeQL job.
- Actions pinned to full commits.
- No untrusted pull-request job receives a write token.
- Public ingestion is manual `workflow_dispatch` only.
- Release workflow is a deliberate no-op failure with no write/package/release operation.
- Branch push CI runs only on `main`; pull-request CI avoids duplicate Sol-branch executions.

Final evidence-only CI is run before merge. The merge identity and final privacy/no-release proof are reported at task completion.
