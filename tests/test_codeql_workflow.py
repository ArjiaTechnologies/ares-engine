"""Regression coverage for the CodeQL action-family version contract."""

from __future__ import annotations

import re
from pathlib import Path


def test_codeql_init_and_analyze_use_the_same_immutable_revision() -> None:
    workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "codeql.yml").read_text(
        encoding="utf-8"
    )
    revisions = re.findall(
        r"uses:\s+github/codeql-action/(?:init|analyze)@([0-9a-f]{40})", workflow
    )

    expected_revision = "cdf488f595d80d6e07e03d4674febd5ab45fa938"  # CodeQL v4.37.9
    assert revisions == [expected_revision, expected_revision]
