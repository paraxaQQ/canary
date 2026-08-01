from __future__ import annotations

import dataclasses
import json

from c4nary import __version__, cli
from c4nary.provenance import KNOWN_TEMPLATES, rules_bundle_sha256
from c4nary.rules.registry import all_rules


def test_rules_bundle_stamp_is_stable_and_behavior_complete(tmp_path) -> None:
    baseline = rules_bundle_sha256()
    assert rules_bundle_sha256() == baseline

    rules = list(all_rules())
    rules[0] = dataclasses.replace(rules[0], severity="CHANGED")
    assert rules_bundle_sha256(rules=rules) != baseline

    copied = tmp_path / "known_templates"
    copied.mkdir()
    for source in KNOWN_TEMPLATES.glob("*.jinja"):
        (copied / source.name).write_bytes(source.read_bytes())
    assert rules_bundle_sha256(known_templates=copied) == baseline

    (copied / "unrelated.txt").write_text("ignored", encoding="utf-8")
    assert rules_bundle_sha256(known_templates=copied) == baseline

    first_template = next(iter(sorted(copied.glob("*.jinja"))))
    first_template.write_bytes(first_template.read_bytes() + b"\n")
    assert rules_bundle_sha256(known_templates=copied) != baseline


def test_rules_json_is_a_stamped_envelope(capsys) -> None:
    assert cli.main(["rules", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert list(payload) == ["tool_version", "rules_bundle_sha256", "rules"]
    assert payload["tool_version"] == __version__
    assert payload["rules_bundle_sha256"] == rules_bundle_sha256()
    assert len(payload["rules"]) == len(all_rules())
