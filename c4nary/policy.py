"""Per-report severity policy and justified finding baselines."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from .report import FAIL, INFO, WARN, Finding
from .rules.registry import get_rule


FailOn = Literal["fail", "warn", "none"]
_FAIL_ON = frozenset({"fail", "warn", "none"})
_SEVERITIES = frozenset({FAIL, WARN, INFO})
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class Policy:
    severity_overrides: dict[str, str]
    fail_on: FailOn = "fail"

    def __post_init__(self) -> None:
        if not isinstance(self.fail_on, str) or self.fail_on not in _FAIL_ON:
            raise PolicyError("policy fail_on must be 'fail', 'warn', or 'none'")
        overrides: dict[str, str] = {}
        for rule_id, severity in self.severity_overrides.items():
            if not isinstance(rule_id, str):
                raise PolicyError("policy rule ids must be strings")
            try:
                get_rule(rule_id)
            except KeyError as exc:
                raise PolicyError(str(exc)) from None
            if not isinstance(severity, str) or severity not in _SEVERITIES:
                raise PolicyError(
                    f"policy severity for {rule_id!r} must be FAIL, WARN, or INFO"
                )
            overrides[rule_id] = severity
        object.__setattr__(
            self,
            "severity_overrides",
            dict(sorted(overrides.items())),
        )


@dataclass(frozen=True)
class Baseline:
    suppressions: dict[str, str]

    def __post_init__(self) -> None:
        normalized: dict[str, str] = {}
        for fingerprint, justification in self.suppressions.items():
            if (
                not isinstance(fingerprint, str)
                or not _FINGERPRINT.fullmatch(fingerprint)
            ):
                raise PolicyError(
                    "baseline fingerprint must be 64 lowercase hex characters"
                )
            if not isinstance(justification, str) or not justification.strip():
                raise PolicyError("baseline suppression requires a justification")
            normalized[fingerprint] = justification.strip()
        object.__setattr__(
            self,
            "suppressions",
            dict(sorted(normalized.items())),
        )


def _read_object(path: str | os.PathLike[str], label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise PolicyError(f"{label} must be a JSON object")
    return value


def load_policy(path: str | os.PathLike[str] | None) -> Policy:
    if path is None:
        return Policy({})
    value = _read_object(path, "policy")
    unknown = sorted(set(value) - {"severity_overrides", "fail_on"})
    if unknown:
        raise PolicyError(f"unknown policy keys: {unknown}")

    raw_overrides = value.get("severity_overrides", {})
    if not isinstance(raw_overrides, dict):
        raise PolicyError("policy severity_overrides must be an object")
    overrides: dict[str, str] = {}
    for rule_id, severity in raw_overrides.items():
        if not isinstance(rule_id, str):
            raise PolicyError("policy rule ids must be strings")
        try:
            get_rule(rule_id)
        except KeyError as exc:
            raise PolicyError(str(exc)) from None
        if not isinstance(severity, str) or severity not in _SEVERITIES:
            raise PolicyError(
                f"policy severity for {rule_id!r} must be FAIL, WARN, or INFO"
            )
        overrides[rule_id] = severity

    fail_on = value.get("fail_on", "fail")
    if not isinstance(fail_on, str) or fail_on not in _FAIL_ON:
        raise PolicyError("policy fail_on must be 'fail', 'warn', or 'none'")
    return Policy(dict(sorted(overrides.items())), cast(FailOn, fail_on))


def load_baseline(path: str | os.PathLike[str] | None) -> Baseline:
    if path is None:
        return Baseline({})
    value = _read_object(path, "baseline")
    unknown = sorted(set(value) - {"suppressions"})
    if unknown:
        raise PolicyError(f"unknown baseline keys: {unknown}")
    entries = value.get("suppressions")
    if not isinstance(entries, list):
        raise PolicyError("baseline suppressions must be a list")

    suppressions: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PolicyError(f"baseline suppression {index} must be an object")
        entry_unknown = sorted(set(entry) - {"fingerprint", "justification"})
        if entry_unknown:
            raise PolicyError(
                f"unknown keys in baseline suppression {index}: {entry_unknown}"
            )
        fingerprint = entry.get("fingerprint")
        justification = entry.get("justification")
        if not isinstance(fingerprint, str) or not _FINGERPRINT.fullmatch(fingerprint):
            raise PolicyError(
                f"baseline suppression {index} fingerprint must be 64 lowercase hex characters"
            )
        if not isinstance(justification, str) or not justification.strip():
            raise PolicyError(
                f"baseline suppression {index} requires a justification"
            )
        if fingerprint in suppressions:
            raise PolicyError(f"duplicate baseline fingerprint: {fingerprint}")
        suppressions[fingerprint] = justification.strip()
    return Baseline(dict(sorted(suppressions.items())))


def finding_fingerprint(finding: Finding) -> str:
    # NUL-separated: without it, (artifact="modeling_", location="a.py:L1") and
    # (artifact="modeling_a.", location="py:L1") hash identically, and a fingerprint
    # collision silently suppresses an unreviewed finding.
    material = "\x00".join((
        finding.rule_id,
        finding.artifact or "",
        finding.location or "",
        finding.subject or "",
    ))
    # surrogatepass: a lone surrogate from a hostile \udXXX JSON escape would raise
    # UnicodeEncodeError here, and this runs on every finding of every scan.
    return hashlib.sha256(material.encode("utf-8", "surrogatepass")).hexdigest()


def apply_report_policy(
    findings: list[Finding],
    policy: Policy,
    baseline: Baseline,
) -> list[Finding]:
    processed: list[Finding] = []
    for item in findings:
        severity = policy.severity_overrides.get(item.rule_id, item.severity)
        justification = baseline.suppressions.get(finding_fingerprint(item))
        processed.append(replace(
            item,
            severity=severity,
            suppressed=justification is not None,
            suppression_justification=justification,
        ))
    return processed
