"""Repo-integrity rules (REP). Local repo targets only."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from .registry import finding

if TYPE_CHECKING:
    from ..report import Finding
    from ..repo import LinkRecord

MAX_REP001_FINDINGS = 20


def analyze_repo_links(links: Iterable["LinkRecord"]) -> list["Finding"]:
    """One REP001 per escaping allowlisted file, capped at 20."""
    findings: list["Finding"] = []
    escapes = sorted(
        (link for link in links if link.kind == "escape"),
        key=lambda link: (link.name, link.target),
    )
    for link in escapes[:MAX_REP001_FINDINGS]:
        findings.append(finding(
            "REP001",
            f"Repo file {link.name!r} is a symlink whose resolved target "
            f"{link.target!r} is outside the repository root. The target was not "
            f"read, so this surface was not audited; manual review is required.",
            location=f"{link.name} -> {link.target}",
            artifact=link.name,
        ))
    return findings
