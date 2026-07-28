# ARES Engine v0.1.0 public release verification

## Verdict

CONDITIONAL

This record was added to trigger and capture the first real public security
analyses. It becomes `PASS` only after public CI, CodeQL, Dependency Review,
alert review, branch protection, merge, tag-build, artifact verification, and
anonymous-clone checks are complete.

## Provenance and privacy gate

- Original pre-rewrite main: `90c6f3646ca32a4a0ca0e68a887e0853418cc5cd`
- Original pre-rewrite tree: `181a15cae0c44fbaacd6101e6e83ef17a43d2156`
- Rewritten clean main: `3de1360cbe95d004c04a05345a89ba3eab29f240`
- Release commit: `870f09555e6a01d1914f76c57893b9151509053c`
- Old repository ID: `R_kgDOTlVbxQ`
- Replacement repository ID: `R_kgDOTmei-A`
- Original recovery bundle SHA-256: `b8a5601ca07ffb475f1a60ddee09c7d8d86799fc4de1362549e013888c2380b1`
- Clean pre-replacement bundle SHA-256: `e6e39e57909b1bc116b96eb324b755def649a9b89a44ddd0b695a98788f2fea0`
- GitHub Support ticket: `4610730` (no purge response before replacement)
- Clean release PR: `https://github.com/ArjiaTechnologies/ares-engine/pull/1`

The original GitHub repository object was deleted only after both bundles,
the replacement history, its managed PR ref, and fresh private CI were
verified. The replacement kept the repository private until GitHub's commit
API, a clean mirror, and the commit page all proved that offending commit
`226c17e528e8881d72f7c79fcd8e902715708afd` was absent. Its current PR ref
points to the sanitized release commit instead.

## Secret and metadata review

- Gitleaks `8.30.1`: `gitleaks git . --log-opts="--all"`; 29 commits; no leaks.
- TruffleHog `3.96.0`: `trufflehog git file://$(pwd) --only-verified`; 524 chunks; zero verified and zero unverified secrets.
- Complete author/committer review: only GitHub noreply identities and the
  intentional Fable audit noreply identity remain; no `.local`, hostname,
  workspace path, or unintended private address remains.
- `git fsck --full`: clean in the replacement mirror.

## Visibility and verification

- Public visibility timestamp: `2026-07-28T20:52:58Z`
- Anonymous repository request: HTTP 200
- Anonymous old-commit request: HTTP 404
- Anonymous refs: `main`, `release/public-v0.1.0`, and clean PR #1 refs only
- README rendered and GitHub detected the MIT license.

## Test state

- Fresh replacement private CI: run `30397558077`, seven required jobs passed.
- Local suite: 245 passed; 90% branch-aware coverage.
- Python 3.11, 3.12, and 3.13 non-ML jobs: passed.
- TensorFlow lifecycle and offline verification: passed.
- Dependency audit: passed with no known vulnerabilities.
- Package, Twine, isolated installed-wheel, Docker non-root, and offline smoke: passed.

## Pending public gates

The first real public CodeQL, Dependency Review, public CI, alert review,
protected-branch configuration, tag build, release artifact checks, and fresh
anonymous clone are pending at this commit.

