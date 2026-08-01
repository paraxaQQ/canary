"""Finding data model and deterministic human / JSON renderers.

Every check in c4nary produces :class:`Finding` objects. Reports sort findings
by ``(severity rank, rule_id, location)`` so the same input always yields the
same byte-for-byte output (invariant §7.4). No timestamps or other
nondeterministic fields ever appear in machine output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coverage import Coverage

# Severity levels. Ordered FAIL < WARN < INFO for sorting (most severe first).
FAIL = "FAIL"
WARN = "WARN"
INFO = "INFO"

_SEVERITY_RANK = {FAIL: 0, WARN: 1, INFO: 2}


@dataclass(frozen=True)
class Finding:
    """A single explainable result, tied to a registered rule id."""

    rule_id: str          # e.g. "TPL001"
    severity: str         # FAIL | WARN | INFO
    title: str
    detail: str           # plain-language explanation
    location: str | None  # where (node path / line, metadata key) or None
    artifact: str | None = None
    line: int | None = None
    suppressed: bool = False
    suppression_justification: str | None = None
    # Discriminates repeat occurrences of one rule at one location so a baseline
    # entry suppresses a single finding rather than the whole (rule, location)
    # group. Fingerprint input only; never rendered.
    subject: str | None = None

    def sort_key(self) -> tuple[int, str, str]:
        return (_SEVERITY_RANK.get(self.severity, 99), self.rule_id, self.location or "")


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Deterministic stable ordering of findings."""

    return sorted(findings, key=Finding.sort_key)


def summarize(findings: list[Finding]) -> dict[str, int]:
    counts = {FAIL: 0, WARN: 0, INFO: 0}
    for f in findings:
        if not f.suppressed and f.severity in counts:
            counts[f.severity] += 1
    return counts


def verdict_line(findings: list[Finding]) -> str:
    """Honest, non-alarmist verdict wording (spec §1)."""

    counts = summarize(findings)
    if counts[FAIL]:
        return (
            "POTENTIALLY DANGEROUS CONSTRUCTS DETECTED - manual review required. "
            "This flags risk indicators; it is not proof the model is malicious."
        )
    if counts[WARN]:
        return (
            "Risk indicators found - review recommended. "
            "These are heuristic flags, not proof of malicious behavior."
        )
    if any(f.suppressed for f in findings):
        return (
            "Detected risk indicators are suppressed by the supplied baseline. "
            "They remain in this report with their review justifications."
        )
    return (
        "No risk indicators detected. "
        "This does not prove the model is safe - only that no known-dangerous "
        "construct was found by these rules."
    )


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #

_SEVERITY_ORDER = (FAIL, WARN, INFO)


def render_human(
    *,
    file: str,
    sha256: str,
    template_sha256: str | None,
    findings: list[Finding],
) -> str:
    findings = sort_findings(findings)
    counts = summarize(findings)
    lines: list[str] = []
    lines.append(f"c4nary scan: {file}")
    lines.append(f"  sha256          {sha256 or '(remote scan - whole-file hash unavailable)'}")
    lines.append(f"  template_sha256 {template_sha256 or '(none)'}")
    lines.append("")
    lines.append(verdict_line(findings))
    lines.append(
        f"  {counts[FAIL]} fail, {counts[WARN]} warn, {counts[INFO]} info"
    )
    suppressed = sum(f.suppressed for f in findings)
    if suppressed:
        lines.append(f"  {suppressed} suppressed by baseline")

    for severity in _SEVERITY_ORDER:
        group = [f for f in findings if f.severity == severity]
        if not group:
            continue
        lines.append("")
        lines.append(f"[{severity}]")
        for f in group:
            loc = f" ({f.location})" if f.location else ""
            suffix = " [suppressed by baseline]" if f.suppressed else ""
            lines.append(f"  {f.rule_id} {f.title}{loc}{suffix}")
            lines.append(f"      {f.detail}")
            if f.suppression_justification is not None:
                lines.append(
                    f"      baseline justification: {f.suppression_justification}"
                )
    lines.append("")
    return "\n".join(lines)


def findings_to_dicts(findings: list[Finding]) -> list[dict]:
    from .policy import finding_fingerprint

    return [
        {
            "rule_id": f.rule_id,
            "severity": f.severity,
            "title": f.title,
            "detail": f.detail,
            "location": f.location,
            "artifact": f.artifact,
            "line": f.line,
            "suppressed": f.suppressed,
            "suppression_justification": f.suppression_justification,
            "fingerprint": finding_fingerprint(f),
        }
        for f in sort_findings(findings)
    ]


def render_json(
    *,
    file: str,
    sha256: str,
    template_sha256: str | None,
    findings: list[Finding],
    coverage: "Coverage | None" = None,
) -> str:
    from . import __version__
    from .provenance import rules_bundle_sha256

    payload = {
        "file": file,
        "sha256": sha256,
        "template_sha256": template_sha256,
        "findings": findings_to_dicts(findings),
        "summary": _summary_lower(findings),
        "artifacts": artifact_rows(file=file, sha256=sha256, findings=findings),
        "coverage": coverage.to_dict() if coverage is not None else [],
        # Paired with the ruleset digest: the digest says which rules ran, this says
        # which build ran them. The MCP scan tool already returned both.
        "tool_version": __version__,
        "rules_bundle_sha256": rules_bundle_sha256(),
    }
    # Fixed separators + no sort_keys: field order is the literal order above,
    # which is stable -> deterministic bytes.
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _summary_lower(findings: list[Finding]) -> dict[str, int]:
    counts = summarize(findings)
    return {
        "fail": counts[FAIL],
        "warn": counts[WARN],
        "info": counts[INFO],
        "suppressed": sum(f.suppressed for f in findings),
    }


def artifact_rows(
    *,
    file: str,
    sha256: str,
    findings: list[Finding],
) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = [{"uri": file, "sha256": sha256 or None}]
    rows.extend(
        {"uri": artifact, "sha256": None}
        for artifact in sorted({
            finding.artifact
            for finding in findings
            if finding.artifact is not None and finding.artifact != file
        })
    )
    return rows
