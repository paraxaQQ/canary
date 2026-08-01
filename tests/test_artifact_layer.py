from __future__ import annotations

import json

from c4nary.bundle import bundle_findings
from c4nary.parser import GGUFModel
from c4nary.rules.config import analyze_config
from c4nary.rules.metadata import analyze_metadata
from c4nary.rules.template import analyze_repo_templates, analyze_templates


CLEAN = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
PAYLOAD = "{{ lipsum.__globals__['os'] }}"


def _model(metadata: dict[str, object] | None = None) -> GGUFModel:
    return GGUFModel(
        path="model.gguf",
        version=3,
        tensor_count=0,
        metadata=metadata or {},
        metadata_types={},
        tensors=(),
    )


def _tpl001(findings):
    return next(finding for finding in findings if finding.rule_id == "TPL001")


def test_all_legacy_retags_also_set_artifact() -> None:
    multi = _tpl001(analyze_templates(_model({
        "tokenizer.chat_template": CLEAN,
        "tokenizer.chat_template.tool_use": PAYLOAD,
    })))
    assert multi.artifact == "chat_template[tool_use]"
    assert multi.location == "chat_template[tool_use] template:L1"
    assert multi.line == 1

    repo = _tpl001(analyze_repo_templates(
        _model({"tokenizer.chat_template": CLEAN}),
        {"chat_template": PAYLOAD},
        None,
    ))
    assert repo.artifact == "tokenizer_config.json"
    assert repo.location == "tokenizer_config.json:template:L1"

    metadata = _tpl001(analyze_metadata(_model({"general.description": PAYLOAD})))
    assert metadata.artifact == "general.description"
    assert metadata.location == "general.description:template:L1"

    config = _tpl001(analyze_config(_model(), {"system_prompt": PAYLOAD}))
    assert config.artifact == "system_prompt"
    assert config.location == "system_prompt:template:L1"

    files = {"config.json": json.dumps({"system_prompt": PAYLOAD})}
    bundle_results, _surfaces = bundle_findings(
        _model(),
        lambda name, max_bytes=1 << 20: files.get(name),
    )
    bundled = _tpl001(bundle_results)
    assert bundled.artifact == "config.json"
    assert bundled.location == "config.json:system_prompt:template:L1"
