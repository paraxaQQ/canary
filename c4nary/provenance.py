"""Deterministic provenance stamp for rule and known-template behavior."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from .rules.registry import Rule, all_rules


KNOWN_TEMPLATES = Path(__file__).with_name("known_templates")


def rules_bundle_sha256(
    *,
    rules: Iterable[Rule] | None = None,
    known_templates: Path = KNOWN_TEMPLATES,
) -> str:
    rule_rows = [
        {
            "rule_id": rule.rule_id,
            "severity": rule.severity,
            "title": rule.title,
            "description": rule.description,
        }
        for rule in (all_rules() if rules is None else rules)
    ]
    template_rows = [
        {
            "name": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(known_templates.glob("*.jinja"))
    ]
    payload = json.dumps(
        {"rules": rule_rows, "known_templates": template_rows},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
