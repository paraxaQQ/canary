"""Finding data model and deterministic human / JSON renderers.

Every check in c4nary produces :class:`Finding` objects. Reports sort findings
by ``(severity rank, rule_id, location)`` so the same input always yields the
same byte-for-byte output (invariant §7.4). No timestamps or other
nondeterministic fields ever appear in machine output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

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

    def sort_key(self) -> tuple[int, str, str]:
        return (_SEVERITY_RANK.get(self.severity, 99), self.rule_id, self.location or "")


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Deterministic stable ordering of findings."""

    return sorted(findings, key=Finding.sort_key)


def summarize(findings: list[Finding]) -> dict[str, int]:
    counts = {FAIL: 0, WARN: 0, INFO: 0}
    for f in findings:
        if f.severity in counts:
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

    for severity in _SEVERITY_ORDER:
        group = [f for f in findings if f.severity == severity]
        if not group:
            continue
        lines.append("")
        lines.append(f"[{severity}]")
        for f in group:
            loc = f" ({f.location})" if f.location else ""
            lines.append(f"  {f.rule_id} {f.title}{loc}")
            lines.append(f"      {f.detail}")
    lines.append("")
    return "\n".join(lines)


def findings_to_dicts(findings: list[Finding]) -> list[dict]:
    return [
        {
            "rule_id": f.rule_id,
            "severity": f.severity,
            "title": f.title,
            "detail": f.detail,
            "location": f.location,
        }
        for f in sort_findings(findings)
    ]


def render_json(
    *,
    file: str,
    sha256: str,
    template_sha256: str | None,
    findings: list[Finding],
) -> str:
    payload = {
        "file": file,
        "sha256": sha256,
        "template_sha256": template_sha256,
        "findings": findings_to_dicts(findings),
        "summary": _summary_lower(findings),
    }
    # Fixed separators + no sort_keys: field order is the literal order above,
    # which is stable -> deterministic bytes.
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _summary_lower(findings: list[Finding]) -> dict[str, int]:
    counts = summarize(findings)
    return {"fail": counts[FAIL], "warn": counts[WARN], "info": counts[INFO]}
