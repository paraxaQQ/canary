"""Metadata sanity-rule tests."""

from pathlib import Path

from _ggufgen import write_gguf

from c4nary.parser import parse_gguf
from c4nary.rules.metadata import analyze_metadata


def _ids(model):
    return {f.rule_id for f in analyze_metadata(model)}


def test_url_and_nonstandard_key(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "llama.context_length": 4096,
        "general.name": "test-model",
        "weird.custom_key": "http://evil.example.com/payload.bin",
    }, tensors=[("token_embd.weight", (8, 32), 0)])
    ids = _ids(parse_gguf(p))
    assert "MET001" in ids  # embedded URL in a non-provenance key
    assert "MET003" in ids  # non-standard namespace
    assert "MET004" in ids  # string listing


def test_oversized_field(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "llama.context_length": 4096,
        "general.blob": "A" * 9000,
    })
    ids = _ids(parse_gguf(p))
    assert "MET002" in ids


def test_architecture_without_arch_keys_warns(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "general.name": "stripped",
    }, tensors=[("a", (2, 2), 0)])
    ids = _ids(parse_gguf(p))
    assert "MET005" in ids


def test_provenance_url_not_flagged(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "general.source.url": "https://huggingface.co/org/model",
        "general.repo_url": "https://github.com/org/model",
    })
    assert "MET001" not in _ids(parse_gguf(p))  # provenance keys hold URLs


def test_clean_metadata_no_warn(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "llama.context_length": 4096,
        "general.name": "clean",
    }, tensors=[("a", (2, 2), 0)])
    findings = analyze_metadata(parse_gguf(p))
    severities = {f.severity for f in findings}
    assert "WARN" not in severities
    assert "FAIL" not in severities
