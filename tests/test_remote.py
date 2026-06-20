"""Tests for the remote-scan helpers (no network: resolution logic only)."""

import pytest

from c4nary.remote import RemoteError, pick_gguf, resolve_target


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
