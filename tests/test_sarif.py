from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from _ggufgen import write_gguf

from c4nary import cli
from c4nary.rules.registry import finding
from c4nary.sarif import render_sarif


SCHEMA = Path(__file__).parent / "fixtures" / "sarif-schema-2.1.0.json"


def _payload() -> dict:
    return json.loads(render_sarif(
        file="model.gguf",
        sha256="a" * 64,
        findings=[
            finding(
                "TPL003",
                "dangerous call",
                location="template:L7",
                artifact="modeling_x.py",
            ),
            finding(
                "INT004",
                "tensor map changed",
                location="tensor map",
            ),
        ],
    ))


def test_sarif_validates_against_official_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(_payload(), schema)


def test_sarif_is_byte_stable_and_has_github_physical_location() -> None:
    first = render_sarif(
        file="model.gguf",
        sha256="a" * 64,
        findings=[
            finding(
                "TPL003",
                "dangerous call",
                location="template:L7",
                artifact="modeling_x.py",
            ),
        ],
    )
    second = render_sarif(
        file="model.gguf",
        sha256="a" * 64,
        findings=[
            finding(
                "TPL003",
                "dangerous call",
                location="template:L7",
                artifact="modeling_x.py",
            ),
        ],
    )
    result = json.loads(first)["runs"][0]["results"][0]
    location = result["locations"][0]

    assert first == second
    assert location["physicalLocation"] == {
        "artifactLocation": {"uri": "modeling_x.py", "index": 1},
        "region": {"startLine": 7},
    }
    assert location["logicalLocations"]
    assert "startTimeUtc" not in first


def test_sarif_no_line_is_logical_only() -> None:
    result = _payload()["runs"][0]["results"][1]
    location = result["locations"][0]

    assert result["ruleId"] == "INT004"
    assert "physicalLocation" not in location
    assert location["logicalLocations"][0]["fullyQualifiedName"] == "tensor map"


def test_cli_sarif_output(tmp_path, capsys) -> None:
    model = write_gguf(tmp_path / "model.gguf", {
        "tokenizer.chat_template": "{{ lipsum.__globals__['os'] }}",
    })
    assert cli.main(["scan", str(model), "--sarif"]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["version"] == "2.1.0"
    assert payload["runs"][0]["results"]


def test_sarif_emits_external_suppression_and_still_validates() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    suppressed = replace(
        finding("TPL001", "reviewed", location="template:L2"),
        suppressed=True,
        suppression_justification="accepted fixture",
    )
    payload = json.loads(render_sarif(
        file="model.gguf",
        sha256="a" * 64,
        findings=[suppressed],
    ))
    result = payload["runs"][0]["results"][0]

    jsonschema.validate(payload, schema)
    assert result["suppressions"] == [{
        "kind": "external",
        "status": "accepted",
        "justification": "accepted fixture",
    }]
