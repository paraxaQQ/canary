"""Shared repo-bundle audit (c4nary.bundle.bundle_findings) used by both the CLI --bundle
path and the MCP scan tool, so they run the identical CFG / NRM / DOC / TPL030 audit."""

import json

from c4nary.bundle import bundle_findings
from c4nary.parser import GGUFModel


def _model(meta=None):
    return GGUFModel(path="t", version=3, tensor_count=0, metadata=meta or {},
                     metadata_types={}, tensors=())


def test_bundle_routes_to_cfg_nrm_doc():
    files = {
        "config.json": json.dumps({"suppress_tokens": [0], "eos_token_id": [0]}),
        "tokenizer.json": json.dumps({"normalizer": {"type": "Replace",
                                       "pattern": {"String": "cannot"}, "content": "can"}}),
        "README.md": "a model card with ‮ a bidi override hidden in it",
    }
    ids = {f.rule_id for f in bundle_findings(_model(), lambda n, max_bytes=1 << 20: files.get(n))}
    assert "CFG001" in ids   # a suppressed stop token
    assert "NRM001" in ids   # a content-rewriting normalizer
    assert "DOC001" in ids   # bidi override concealed in the card


def test_bundle_empty_reader_is_clean():
    assert bundle_findings(_model(), lambda n, max_bytes=1 << 20: None) == []
