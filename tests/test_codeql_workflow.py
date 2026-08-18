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

    expected_revision = "ff2f1c621b7f889edc0d3c761ac2e6a3f8cdb0dd"  # CodeQL v4.37.7
    assert revisions == [expected_revision, expected_revision]
