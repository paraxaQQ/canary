"""Registry consistency: every finding maps to a registered rule (invariant §7.5)."""

from pathlib import Path

import pytest

from c4nary.rules.registry import all_rules, finding, get_rule
from c4nary.rules.template import analyze_template

FIX = Path(__file__).parent / "fixtures"


def test_rule_ids_unique():
    ids = [r.rule_id for r in all_rules()]
    assert len(ids) == len(set(ids))


def test_speculative_tokenizer_rules_are_not_shipped():
    ids = {r.rule_id for r in all_rules()}
    assert {"MET022", "MET023"}.isdisjoint(ids)


def test_finding_rejects_unregistered_id():
    with pytest.raises(KeyError):
        finding("NOPE999", "detail")


def test_emitted_findings_are_all_registered():
    # Scan every fixture template; every emitted rule_id must resolve.
    for jinja in FIX.glob("*.jinja"):
        for f in analyze_template(jinja.read_text(encoding="utf-8")):
            rule = get_rule(f.rule_id)          # raises if unregistered
            assert f.severity == rule.severity  # severity cannot drift
