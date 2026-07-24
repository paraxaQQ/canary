import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import template_fail_gate


def test_rule_scope_is_all_registered_fail_template_rules() -> None:
    assert template_fail_gate.TEMPLATE_FAIL_RULES == {
        "TPL001",
        "TPL002",
        "TPL003",
        "TPL004",
        "TPL005",
        "TPL021",
        "TPL024",
        "TPL025",
    }


def test_inline_template_runs_fail_rules_without_header_fetch(monkeypatch) -> None:
    monkeypatch.setattr(
        template_fail_gate,
        "fetch_header_template",
        lambda *args: (_ for _ in ()).throw(AssertionError("header fetch was called")),
    )

    result = template_fail_gate.scan_repo(
        {"repo": "owner/model", "chat_template": "{{ cycler.__init__.__globals__ }}"},
        (1,),
        "",
    )

    assert result["status"] == "analyzed"
    assert result["source"] == "inline"
    assert any(hit["rule"] == "TPL001" for hit in result["hits"])


def test_missing_inline_template_uses_header(monkeypatch) -> None:
    monkeypatch.setattr(template_fail_gate, "pick_gguf", lambda _: "model.gguf")
    monkeypatch.setattr(
        template_fail_gate,
        "fetch_header_template",
        lambda *args: "{{ cycler.__init__.__globals__ }}",
    )

    result = template_fail_gate.scan_repo(
        {"repo": "owner/model", "chat_template": None, "gguf_files": ["model.gguf"]},
        (1,),
        "token",
    )

    assert result["status"] == "analyzed"
    assert result["source"] == "header"
    assert result["file"] == "model.gguf"
    assert any(hit["rule"] == "TPL001" for hit in result["hits"])


def test_header_without_template_is_not_an_exclusion(monkeypatch) -> None:
    monkeypatch.setattr(template_fail_gate, "pick_gguf", lambda _: "model.gguf")
    monkeypatch.setattr(template_fail_gate, "fetch_header_template", lambda *args: None)

    result = template_fail_gate.scan_repo(
        {"repo": "owner/model", "chat_template": None, "gguf_files": ["model.gguf"]},
        (1,),
        "token",
    )

    assert result["status"] == "no_template"
    assert result["hits"] == []


def test_fetch_header_template_escalates_incomplete_parse(monkeypatch) -> None:
    monkeypatch.setattr(
        template_fail_gate,
        "_fetch_prefixes",
        lambda *args: iter([b"short", b"enough"]),
    )

    def parse(data: bytes, **kwargs):
        if data == b"short":
            raise ValueError("unexpected end of file")
        return "{{ messages }}"

    monkeypatch.setattr(template_fail_gate, "extract_gguf_chat_template_bytes", parse)

    assert template_fail_gate.fetch_header_template(
        "owner/model", "model.gguf", (1, 4), "token"
    ) == "{{ messages }}"


def test_fetch_prefixes_uses_one_progressive_request(monkeypatch) -> None:
    calls = []

    class Response:
        status_code = 206
        headers = {}

        def iter_content(self, chunk_size):
            assert chunk_size == 64 * 1024
            yield b"ab"
            yield b"cde"

        def close(self):
            return None

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(template_fail_gate.requests, "get", get)
    monkeypatch.setattr(template_fail_gate.RESOLVER_LIMITER, "wait", lambda: None)

    prefixes = list(
        template_fail_gate._fetch_prefixes(
            "https://example.invalid/model.gguf", (2, 5), "token"
        )
    )

    assert prefixes == [b"ab", b"abcde"]
    assert len(calls) == 1
    assert calls[0][1]["headers"]["Range"] == "bytes=0-4"
