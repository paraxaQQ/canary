"""Deterministic SARIF 2.1.0 output."""

from __future__ import annotations

import json

from . import __version__
from .policy import finding_fingerprint
from .provenance import rules_bundle_sha256
from .report import FAIL, INFO, WARN, Finding, artifact_rows, sort_findings
from .rules.registry import all_rules


SARIF_SCHEMA = (
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/"
    "sarif-schema-2.1.0.json"
)
_SARIF_LEVEL = {FAIL: "error", WARN: "warning", INFO: "note"}


def _uri(value: str) -> str:
    return value.replace("\\", "/")


def render_sarif(
    *,
    file: str,
    sha256: str | None,
    findings: list[Finding],
) -> str:
    stamp = rules_bundle_sha256()
    rules = all_rules()
    rule_indexes = {rule.rule_id: index for index, rule in enumerate(rules)}
    artifact_table = artifact_rows(
        file=file,
        sha256=sha256,
        findings=findings,
    )
    artifacts = []
    artifact_indexes: dict[str, int] = {}
    for index, row in enumerate(artifact_table):
        uri = _uri(row["uri"])
        artifact_indexes[row["uri"]] = index
        artifact = {"location": {"uri": uri}}
        if row["sha256"] is not None:
            artifact["hashes"] = {"sha-256": row["sha256"]}
        artifacts.append(artifact)

    results = []
    for finding in sort_findings(findings):
        artifact = finding.artifact or file
        logical_location = {
            "fullyQualifiedName": finding.location or finding.rule_id,
            "kind": "review",
        }
        location = {"logicalLocations": [logical_location]}
        if finding.line is not None:
            location = {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": _uri(artifact),
                        "index": artifact_indexes[artifact],
                    },
                    "region": {"startLine": finding.line},
                },
                "logicalLocations": [logical_location],
            }
        result = {
            "ruleId": finding.rule_id,
            "ruleIndex": rule_indexes[finding.rule_id],
            "level": _SARIF_LEVEL[finding.severity],
            "message": {"text": finding.detail},
            "locations": [location],
            "properties": {
                "legacyLocation": finding.location,
                "artifact": finding.artifact,
                "rulesBundleSha256": stamp,
                "baselineFingerprint": finding_fingerprint(finding),
            },
        }
        if finding.suppressed:
            result["suppressions"] = [{
                "kind": "external",
                "status": "accepted",
                "justification": finding.suppression_justification,
            }]
        results.append(result)

    payload = {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "c4nary",
                    "version": __version__,
                    "informationUri": "https://github.com/paraxaQQ/canary",
                    "rules": [
                        {
                            "id": rule.rule_id,
                            "shortDescription": {"text": rule.title},
                            "fullDescription": {"text": rule.description},
                            "defaultConfiguration": {
                                "level": _SARIF_LEVEL[rule.severity],
                            },
                        }
                        for rule in rules
                    ],
                },
            },
            "artifacts": artifacts,
            "results": results,
            "properties": {"rulesBundleSha256": stamp},
        }],
    }
    return json.dumps(payload, indent=2, ensure_ascii=True)
