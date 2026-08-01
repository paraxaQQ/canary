from __future__ import annotations

import hashlib
import json
from pathlib import Path

from c4nary.report import FAIL
from c4nary.rules.template import analyze_template


FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = FIXTURES / "corpus-fail"
EXPECTED = FIXTURES / "corpus-fail-expected.json"


def test_frozen_v022_template_fail_baseline() -> None:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    rows = expected["findings"]
    files = sorted(CORPUS.glob("*.jinja"))
    referenced = {row["fixture"] for row in rows}

    assert expected["fixtures"] == len(files) == len(referenced) == 28
    assert len(rows) == 140
    assert {path.name for path in files} == referenced

    actual: list[dict[str, str]] = []
    for path in files:
        payload = path.read_bytes()
        template_sha256 = hashlib.sha256(payload).hexdigest()
        fixture_rows = [row for row in rows if row["fixture"] == path.name]
        assert fixture_rows
        assert {row["template_sha256"] for row in fixture_rows} == {template_sha256}

        for result in analyze_template(payload.decode("utf-8")):
            if result.severity != FAIL:
                continue
            actual.append({
                "fixture": path.name,
                "repo": fixture_rows[0]["repo"],
                "template_sha256": template_sha256,
                "rule": result.rule_id,
                "severity": result.severity,
                "detail": result.detail,
                "location": result.location,
            })

    actual.sort(key=lambda item: (
        item["fixture"],
        item["rule"],
        item["severity"],
        item["detail"],
        item["location"],
    ))
    assert actual == rows
