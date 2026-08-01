from __future__ import annotations

import json

from c4nary.bundle import bundle_findings
from c4nary.coverage import Coverage, Surface, build_scan_coverage
from c4nary.parser import GGUFModel, MetaArray
from c4nary.rules.template import analyze_templates


def _model(metadata: dict[str, object] | None = None) -> GGUFModel:
    return GGUFModel(
        path="model.gguf",
        version=3,
        tensor_count=0,
        metadata=metadata or {},
        metadata_types={},
        tensors=(),
    )


def test_coverage_dict_is_sorted_and_byte_stable() -> None:
    first = Coverage.from_surfaces([
        Surface("z", "skipped", "z reason"),
        Surface("a", "examined", "a reason"),
    ])
    second = Coverage.from_surfaces(reversed(first.surfaces))

    assert first.to_dict() == [
        {"id": "a", "state": "examined", "reason": "a reason"},
        {"id": "z", "state": "skipped", "reason": "z reason"},
    ]
    assert json.dumps(first.to_dict()) == json.dumps(second.to_dict())


def test_shard_and_truncated_deep_seam_each_get_one_partial_row() -> None:
    model = _model({
        "split.count": 2,
        "tokenizer.ggml.tokens": MetaArray("string", 2, ("a",), True),
        "tokenizer.ggml.token_type": MetaArray("uint32", 2, (1,), True),
    })
    coverage = build_scan_coverage(
        model,
        remote=False,
        deep_tokenizer=True,
        bundle=False,
        manifest_requested=False,
        template_findings=analyze_templates(model),
    )
    rows = {row["id"]: row for row in coverage.to_dict()}

    assert rows["tensor_map.consistency"]["state"] == "partial"
    assert all(
        rule in rows["tensor_map.consistency"]["reason"]
        for rule in ("MET010", "MET011", "MET013", "MET014", "TOK002")
    )
    assert rows["tokenizer.deep"]["state"] == "partial"
    assert sum(row["id"] == "tensor_map.consistency" for row in coverage.to_dict()) == 1
    assert sum(row["id"] == "tokenizer.deep" for row in coverage.to_dict()) == 1


def test_bundle_absent_and_unparseable_surfaces_are_explicit() -> None:
    files = {
        "config.json": "{",
        "tokenizer.json": "{",
        "generation_config.json": json.dumps({"system_prompt": "{{ unfinished_value"}),
    }
    _findings, surfaces = bundle_findings(
        _model(),
        lambda name, max_bytes=1 << 20: files.get(name),
    )
    rows = {surface.id: surface for surface in surfaces}

    assert len(rows) == 10
    assert rows["bundle.config.json"].state == "unparseable"
    assert rows["bundle.tokenizer.json"].state == "unparseable"
    assert rows["bundle.generation_config.json"].state == "partial"
    assert rows["bundle.README.md"].state == "absent"
