"""Determinism + exit-code acceptance tests (spec §8)."""

import json
from pathlib import Path

from _ggufgen import write_gguf

from c4nary import cli
from c4nary.provenance import rules_bundle_sha256
from c4nary.report import render_json
from c4nary.rules.registry import finding

CHATML = (
    Path(__file__).parents[1] / "c4nary" / "known_templates" / "chatml.jinja"
).read_text(encoding="utf-8")
CVE = (
    Path(__file__).parent / "fixtures" / "cve_llama_drama.jinja"
).read_text(encoding="utf-8")


def _clean(tmp_path):
    return write_gguf(tmp_path / "clean.gguf", {
        "general.architecture": "llama",
        "llama.context_length": 4096,
        "general.name": "clean",
        "tokenizer.chat_template": CHATML,
    }, tensors=[("a", (2, 2), 0)], tail=b"data")


def _malicious(tmp_path):
    return write_gguf(tmp_path / "evil.gguf", {
        "general.architecture": "llama",
        "llama.context_length": 4096,
        "tokenizer.chat_template": CVE,
    }, tensors=[("a", (2, 2), 0)], tail=b"data")


def test_scan_json_is_byte_identical(tmp_path, capsys):
    p = _clean(tmp_path)
    cli.main(["scan", str(p), "--json"])
    out1 = capsys.readouterr().out
    cli.main(["scan", str(p), "--json"])
    out2 = capsys.readouterr().out
    assert out1 == out2
    assert '"summary"' in out1  # sanity: it really produced the report


def test_scan_json_keeps_legacy_key_prefixes() -> None:
    payload = json.loads(render_json(
        file="model.gguf",
        sha256="abc",
        template_sha256="def",
        findings=[
            finding(
                "TPL003",
                "test",
                location="template:L2",
                artifact="tokenizer_config.json",
            ),
        ],
    ))

    assert list(payload)[:5] == [
        "file",
        "sha256",
        "template_sha256",
        "findings",
        "summary",
    ]
    assert list(payload["findings"][0])[:5] == [
        "rule_id",
        "severity",
        "title",
        "detail",
        "location",
    ]
    assert payload["findings"][0]["line"] == 2
    assert payload["artifacts"] == [
        {"uri": "model.gguf", "sha256": "abc"},
        {"uri": "tokenizer_config.json", "sha256": None},
    ]


def test_human_notes_are_coverage_derived(tmp_path, capsys) -> None:
    path = _clean(tmp_path)
    assert cli.main(["scan", str(path), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    coverage = {row["id"]: row for row in payload["coverage"]}

    assert captured.err.splitlines() == [
        f"note: {coverage['tokenizer.deep']['reason']}",
        f"note: {coverage['bundle']['reason']}",
    ]
    assert payload["rules_bundle_sha256"] == rules_bundle_sha256()


def test_exit_code_clean_is_zero(tmp_path, capsys):
    p = _clean(tmp_path)
    rc = cli.main(["scan", str(p)])
    capsys.readouterr()
    assert rc == 0


def test_exit_code_malicious_is_two(tmp_path, capsys):
    p = _malicious(tmp_path)
    rc = cli.main(["scan", str(p)])
    capsys.readouterr()
    assert rc == 2


def test_fail_on_none_returns_zero_for_fail(tmp_path, capsys):
    p = _malicious(tmp_path)
    assert cli.main(["scan", str(p), "--fail-on", "none"]) == 0
    capsys.readouterr()


def test_fail_on_warn_returns_one(tmp_path, capsys):
    # A URL in metadata is a WARN; default fail-on=fail -> 0, fail-on=warn -> 1.
    p = write_gguf(tmp_path / "warn.gguf", {
        "general.architecture": "llama",
        "llama.context_length": 4096,
        "general.description": "http://example.com/x",
        "tokenizer.chat_template": CHATML,
    }, tensors=[("a", (2, 2), 0)], tail=b"data")
    assert cli.main(["scan", str(p)]) == 0
    capsys.readouterr()
    assert cli.main(["scan", str(p), "--fail-on", "warn"]) == 1
    capsys.readouterr()


def test_diff_cli_exit_codes(tmp_path, capsys):
    a = write_gguf(tmp_path / "a.gguf", {
        "general.architecture": "llama", "llama.context_length": 4096,
        "tokenizer.chat_template": CHATML,
    }, tensors=[("a", (2, 2), 0)], tail=b"d")
    b = write_gguf(tmp_path / "b.gguf", {
        "general.architecture": "llama", "llama.context_length": 4096,
        "tokenizer.chat_template": CHATML.replace("im_start", "im_START"),
    }, tensors=[("a", (2, 2), 0)], tail=b"d")
    assert cli.main(["diff", str(a), str(a)]) == 0  # identical
    capsys.readouterr()
    assert cli.main(["diff", str(a), str(b)]) == 1  # template differs
    capsys.readouterr()


def test_parse_error_exit_code(tmp_path, capsys):
    bad = tmp_path / "notgguf.gguf"
    bad.write_bytes(b"NOPE not a gguf file at all")
    rc = cli.main(["scan", str(bad)])
    capsys.readouterr()
    assert rc == 3


def test_scan_json_carries_the_provenance_pair(tmp_path):
    """The ruleset digest says which rules ran; tool_version says which build ran them.

    README documents both as part of the --json envelope, and the MCP scan tool has
    always returned both -- the CLI was the odd one out.
    """

    import json as _json

    from c4nary import __version__, cli

    model = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "tokenizer.chat_template": "{{ messages[0]['content'] }}",
    }, tensors=[("a", (2, 2), 0)], tail=b"data")

    import contextlib
    import io

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli.main(["scan", str(model), "--json"])
    payload = _json.loads(out.getvalue())

    assert payload["tool_version"] == __version__
    assert len(payload["rules_bundle_sha256"]) == 64
