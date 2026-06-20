"""Integrity, manifest, and structural-diff tests."""

from pathlib import Path

from _ggufgen import write_gguf

from c4nary.integrity import (
    build_manifest,
    compare_manifest,
    diff_is_empty,
    diff_models,
    sha256_file,
)
from c4nary.parser import parse_gguf

CHATML = (
    Path(__file__).parents[1] / "c4nary" / "known_templates" / "chatml.jinja"
).read_text(encoding="utf-8")


def _model(tmp_path, name, template, *, tail=b"weightdata"):
    return write_gguf(tmp_path / name, {
        "general.architecture": "llama",
        "llama.context_length": 4096,
        "tokenizer.chat_template": template,
    }, tensors=[("token_embd.weight", (4, 8), 0)], tail=tail)


def test_file_hash_is_stable(tmp_path):
    p = _model(tmp_path, "m.gguf", CHATML)
    assert sha256_file(p) == sha256_file(p)


def test_manifest_detects_tamper(tmp_path):
    good = _model(tmp_path, "good.gguf", CHATML)
    manifest = build_manifest(parse_gguf(good), sha256_file(good))

    bad = _model(tmp_path, "bad.gguf", CHATML + "{{ lipsum.__globals__ }}")
    findings = compare_manifest(parse_gguf(bad), sha256_file(bad), manifest)
    ids = {f.rule_id for f in findings}
    assert "INT001" in ids  # file hash drift
    assert "INT002" in ids  # template hash drift


def test_manifest_matches_self(tmp_path):
    p = _model(tmp_path, "m.gguf", CHATML)
    model = parse_gguf(p)
    manifest = build_manifest(model, sha256_file(p))
    assert compare_manifest(model, sha256_file(p), manifest) == []


def test_diff_surfaces_only_template_change(tmp_path):
    a = _model(tmp_path, "a.gguf", CHATML, tail=b"X" * 32)
    b = _model(tmp_path, "b.gguf", CHATML.replace("im_start", "im_START"),
               tail=b"X" * 32)
    diff = diff_models(parse_gguf(a), parse_gguf(b))
    assert diff["template_changed"] is True
    assert diff["template_diff"]  # has a unified diff body
    assert diff["metadata"]["changed"] == {}
    assert diff["metadata"]["added"] == {}
    assert diff["metadata"]["removed"] == {}
    assert diff["tensors"]["changed"] == []
    assert not diff_is_empty(diff)


def test_diff_identical_is_empty(tmp_path):
    a = _model(tmp_path, "a.gguf", CHATML, tail=b"Z" * 8)
    b = _model(tmp_path, "b.gguf", CHATML, tail=b"different-tail")
    # tail (weight bytes) differs but structure is identical -> empty diff.
    diff = diff_models(parse_gguf(a), parse_gguf(b))
    assert diff_is_empty(diff)
