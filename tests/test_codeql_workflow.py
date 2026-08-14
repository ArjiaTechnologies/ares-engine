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

    expected_revision = "5595ccaf912efad79be6eef63a5619ff05969be3"  # CodeQL v4.37.6
    assert revisions == [expected_revision, expected_revision]
