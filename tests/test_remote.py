"""Tests for the remote-scan helpers (no live network)."""

import pytest

from c4nary import remote
from c4nary.remote import RemoteError, pick_gguf, resolve_target


class _Response:
    def __init__(self, chunks, *, status_code=200, encoding="utf-8"):
        self.chunks = chunks
        self.status_code = status_code
        self.encoding = encoding
        self.closed = False

    def iter_content(self, chunk_size):
        yield from self.chunks

    def close(self):
        self.closed = True


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_pick_gguf_prefers_single_file():
    files = ["README.md", "model.Q4_K_M.gguf", "model.Q8_0.gguf", "mmproj.gguf"]
    assert pick_gguf(files) == "model.Q8_0.gguf"  # shortest single-file, no mmproj


def test_pick_gguf_handles_shards():
    files = ["m-00001-of-00003.gguf", "m-00002-of-00003.gguf"]
    assert pick_gguf(files) == "m-00001-of-00003.gguf"


def test_pick_gguf_none_when_absent():
    assert pick_gguf(["README.md", "config.json"]) is None


def test_resolve_url_passthrough():
    url = "https://huggingface.co/org/repo/resolve/main/model.gguf"
    assert resolve_target(url) == (url, url)


def test_resolve_repo_with_explicit_filename():
    url, display = resolve_target("org/repo", filename="model.gguf")
    assert url == "https://huggingface.co/org/repo/resolve/main/model.gguf"
    assert display == "org/repo/model.gguf"


def test_resolve_repo_picks_file_via_injected_lister():
    files = ["README.md", "model-Q4_K_M.gguf"]
    url, display = resolve_target("hf://org/repo", list_files=lambda r: files)
    assert url.endswith("/org/repo/resolve/main/model-Q4_K_M.gguf")
    assert display == "org/repo/model-Q4_K_M.gguf"


def test_resolve_repo_no_gguf_raises():
    with pytest.raises(RemoteError):
        resolve_target("org/repo", list_files=lambda r: ["config.json"])


def test_core_import_needs_no_requests():
    # The offline core must import without the optional network dependency.
    import importlib
    for mod in ("c4nary.parser", "c4nary.rules.template", "c4nary.cli"):
        importlib.import_module(mod)


def test_direct_url_never_receives_hf_token(monkeypatch):
    response = _Response([b"GGUF"], status_code=206)
    session = _Session(response)
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr(remote, "_session", lambda: session)

    assert remote._fetch_capped("https://attacker.example/model.gguf", 4) == b"GGUF"
    assert session.calls[0][1]["headers"] == {"Range": "bytes=0-3"}
    assert response.closed


def test_huggingface_url_receives_hf_token(monkeypatch):
    response = _Response([b"GGUF"], status_code=206)
    session = _Session(response)
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr(remote, "_session", lambda: session)

    remote._fetch_capped("https://huggingface.co/org/repo/model.gguf", 4)
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer secret"


def test_fetch_repo_text_streams_and_rejects_oversize(monkeypatch):
    response = _Response([b"abc", b"def"])
    session = _Session(response)
    monkeypatch.setattr(remote, "_session", lambda: session)

    assert remote.fetch_repo_text("org/repo", "README.md", max_bytes=5) is None
    _, kwargs = session.calls[0]
    assert kwargs["stream"] is True
    assert kwargs["headers"]["Range"] == "bytes=0-5"
    assert response.closed


def test_fetch_repo_text_decodes_bounded_response(monkeypatch):
    response = _Response(["café".encode()], status_code=206)
    session = _Session(response)
    monkeypatch.setattr(remote, "_session", lambda: session)

    assert remote.fetch_repo_text("org/repo", "README.md", max_bytes=5) == "café"
    assert response.closed
