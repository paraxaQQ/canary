"""Safetensors **header** ingest: parsing, local/remote readers, TOK002, coverage.

Nothing here reads tensor data. The local header reader is exercised through a
spy stream to prove it requests only the 8-byte prefix and the header JSON, and
the remote reader through an injected range fetcher to prove it requests only
two byte ranges. A large trailing payload is simulated structurally, never
written or read.
"""

from __future__ import annotations

import json
import os
import struct

import pytest

from c4nary import cli
from c4nary.mcp_server import run_scan
from c4nary.repo import LocalRepo, RemoteRepo, scan_repo
from c4nary.safetensors import (
    MAX_HEADER_BYTES,
    PREFIX_BYTES,
    candidate_axes,
    header_length,
    is_safetensors_name,
    read_header,
    read_header_from_stream,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

def _payload(tensors: dict, metadata: dict | None = None) -> bytes:
    entries = {
        name: {"dtype": "F16", "shape": list(shape), "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }
    if metadata is not None:
        entries["__metadata__"] = metadata
    return json.dumps(entries).encode("utf-8")


def _file_bytes(tensors: dict, *, metadata: dict | None = None,
                data: bytes = b"") -> bytes:
    header = _payload(tensors, metadata)
    return struct.pack("<Q", len(header)) + header + data


def _write_safetensors(root, name: str, tensors: dict, *, data: bytes = b"") -> int:
    raw = _file_bytes(tensors, data=data)
    (root / name).write_bytes(raw)
    return len(raw) - len(data)


def _write_tokenizer(root, count: int) -> None:
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {f"t{index}": index for index in range(count)}}}),
        encoding="utf-8",
    )


def _scan(capsys, root, *extra: str):
    rc = cli.main(["scan", str(root), "--as", "repo", "--json", *extra])
    captured = capsys.readouterr()
    return rc, captured, json.loads(captured.out) if captured.out else None


def _coverage(payload) -> dict[str, dict[str, str]]:
    return {row["id"]: row for row in payload["coverage"]}


def _symlink_or_skip(target, link) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"OS symlinks unavailable: {exc}")


# --------------------------------------------------------------------------- #
# Header parser
# --------------------------------------------------------------------------- #

def test_parse_header_extracts_shapes_and_skips_metadata() -> None:
    raw = _file_bytes(
        {"model.embed_tokens.weight": [10, 4], "lm_head.weight": [10, 4]},
        metadata={"format": "pt"},
    )

    header = read_header("model.safetensors", raw[:8], raw[8:], file_size=len(raw))

    assert header.state == "examined"
    assert header.examined is True
    assert dict(header.shapes) == {
        "model.embed_tokens.weight": (10, 4),
        "lm_head.weight": (10, 4),
    }
    assert header.tensor_count == 2
    assert header.consumed == len(raw)


def test_zero_header_length_is_unparseable() -> None:
    header = read_header("model.safetensors", struct.pack("<Q", 0), b"")

    assert header.state == "unparseable"
    assert "must be positive" in header.reason


def test_short_prefix_is_unparseable() -> None:
    header = read_header("model.safetensors", b"\x01\x02", b"")

    assert header.state == "unparseable"
    assert "shorter than" in header.reason


def test_invalid_header_json_is_unparseable() -> None:
    raw = b"{not valid json"

    header = read_header(
        "model.safetensors", struct.pack("<Q", len(raw)), raw,
        file_size=8 + len(raw),
    )

    assert header.state == "unparseable"
    assert "valid JSON" in header.reason


def test_non_integer_shape_is_unparseable() -> None:
    raw = json.dumps({
        "model.embed_tokens.weight": {
            "dtype": "F16", "shape": [10, "four"], "data_offsets": [0, 0],
        },
    }).encode("utf-8")

    header = read_header(
        "model.safetensors", struct.pack("<Q", len(raw)), raw,
        file_size=8 + len(raw),
    )

    assert header.state == "unparseable"
    assert "non-integer shape" in header.reason


def test_header_over_its_cap_is_partial_without_reading_it() -> None:
    declared = MAX_HEADER_BYTES + 1

    header = read_header(
        "model.safetensors", struct.pack("<Q", declared), b"",
        file_size=8 + declared,
    )

    assert header.state == "partial"
    assert "cap" in header.reason
    assert header.consumed == PREFIX_BYTES


def test_header_past_end_of_file_is_unparseable() -> None:
    raw = _payload({"model.embed_tokens.weight": [4, 4]})
    prefix = struct.pack("<Q", len(raw))

    header = read_header(
        "model.safetensors", prefix, raw, file_size=8 + len(raw) - 1,
    )

    assert header.state == "unparseable"
    assert "past the end" in header.reason


def test_short_header_bytes_are_partial() -> None:
    raw = _payload({"model.embed_tokens.weight": [4, 4]})

    header = read_header(
        "model.safetensors", struct.pack("<Q", len(raw)), raw[:-3],
        truncated=True,
    )

    assert header.state == "partial"
    assert "incomplete" in header.reason


class _SpyStream:
    """Records every ``read(n)`` so the no-data boundary is observable."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.reads: list[int] = []
        self.offset = 0

    def read(self, size: int) -> bytes:
        self.reads.append(size)
        chunk = self.payload[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


def test_stream_reader_requests_only_prefix_and_header() -> None:
    header = _payload({"model.embed_tokens.weight": [10, 4]})
    payload = struct.pack("<Q", len(header)) + header
    spy = _SpyStream(payload)

    parsed = read_header_from_stream(
        "model.safetensors", spy, file_size=8 + len(header) + 8_000_000_000,
    )

    assert parsed.state == "examined"
    # Exactly two requests: the 8-byte prefix and the N-byte header. The
    # multi-GB tensor data after it is never requested.
    assert spy.reads == [PREFIX_BYTES, len(header)]
    assert spy.offset == len(payload)


def test_stream_reader_stops_after_prefix_when_over_cap() -> None:
    declared = MAX_HEADER_BYTES + 1
    spy = _SpyStream(struct.pack("<Q", declared) + b"x" * 16)

    parsed = read_header_from_stream("model.safetensors", spy, file_size=None)

    assert parsed.state == "partial"
    assert spy.reads == [PREFIX_BYTES]


def test_candidate_axes_cover_embedding_and_output_suffixes() -> None:
    shapes = {
        "model.embed_tokens.weight": (10, 4),
        "lm_head.weight": (10, 4),
        "output.weight": (12, 4),
        "wte.weight": (9, 3),
        "tok_embeddings.weight": (8, 3),
        "token_embd.weight": (7, 3),
        "transformer.h.0.attn.c_attn.weight": (4, 4),
        "model.norm.weight": (4,),
        "empty.embed_tokens.weight": (),
    }

    axes = candidate_axes(shapes)
    assert "empty.embed_tokens.weight" not in axes  # no axis to compare

    assert list(axes) == sorted(axes)
    assert set(axes) == {
        "model.embed_tokens.weight", "lm_head.weight", "output.weight",
        "wte.weight", "tok_embeddings.weight", "token_embd.weight",
    }


@pytest.mark.parametrize(
    "name",
    [
        ".hidden.safetensors",
        "model.safetensors.bak",
        "sub/model.safetensors",
        "sub\\model.safetensors",
        "../model.safetensors",
        "model..safetensors",
        "model\x00.safetensors",
        "C:model.safetensors",
        "model?.safetensors",
        "model#.safetensors",
        "model%.safetensors",
        ".safetensors",
        "",
    ],
)
def test_unsafe_safetensors_names_are_rejected(name) -> None:
    assert is_safetensors_name(name) is False


def test_header_length_helper_distinguishes_bounds() -> None:
    assert header_length(b"\x00" * 8) == (
        None, "header length must be positive, got 0")
    length, problem = header_length(struct.pack("<Q", MAX_HEADER_BYTES + 1))
    assert length == MAX_HEADER_BYTES + 1
    assert problem is not None


# --------------------------------------------------------------------------- #
# Local repo reader / TOK002 / coverage
# --------------------------------------------------------------------------- #

def test_local_safetensors_header_drives_tok002(tmp_path, capsys) -> None:
    _write_tokenizer(tmp_path, 12)
    _write_safetensors(tmp_path, "model.safetensors", {
        "model.embed_tokens.weight": [10, 4],
        "model.norm.weight": [4],
    })

    rc, _captured, payload = _scan(capsys, tmp_path)
    findings = [f for f in payload["findings"] if f["rule_id"] == "TOK002"]
    by_id = _coverage(payload)

    assert rc == cli.EXIT_FAIL
    assert len(findings) == 1
    assert findings[0]["severity"] == "FAIL"
    assert findings[0]["location"] == "safetensors:model.embed_tokens.weight"
    assert by_id["safetensors.model.safetensors"]["state"] == "examined"
    assert by_id["tensor_map.consistency"]["state"] == "examined"
    assert "only header bytes" in by_id["tensor_map.consistency"]["reason"]


def test_vocab_within_the_shape_emits_no_tok002(tmp_path, capsys) -> None:
    _write_tokenizer(tmp_path, 8)
    _write_safetensors(tmp_path, "model.safetensors", {
        "model.embed_tokens.weight": [10, 4],
    })

    rc, _captured, payload = _scan(capsys, tmp_path)

    assert rc == cli.EXIT_OK
    assert "TOK002" not in {f["rule_id"] for f in payload["findings"]}
    assert _coverage(payload)["tensor_map.consistency"]["state"] == "examined"


def test_headers_without_a_candidate_axis_are_partial(tmp_path, capsys) -> None:
    _write_tokenizer(tmp_path, 12)
    _write_safetensors(tmp_path, "model.safetensors", {
        "model.layers.0.mlp.weight": [10, 4],
    })

    rc, _captured, payload = _scan(capsys, tmp_path)
    row = _coverage(payload)["tensor_map.consistency"]

    assert rc == cli.EXIT_OK
    assert "TOK002" not in {f["rule_id"] for f in payload["findings"]}
    assert row["state"] == "partial"
    assert "no embedding or output tensor" in row["reason"]


def test_headers_without_a_tokenizer_vocab_are_partial(tmp_path, capsys) -> None:
    _write_safetensors(tmp_path, "model.safetensors", {
        "model.embed_tokens.weight": [10, 4],
    })

    rc, _captured, payload = _scan(capsys, tmp_path)
    row = _coverage(payload)["tensor_map.consistency"]

    assert rc == cli.EXIT_OK
    assert "TOK002" not in {f["rule_id"] for f in payload["findings"]}
    assert row["state"] == "partial"
    assert "no usable tokenizer.json vocabulary" in row["reason"]


def test_no_safetensors_is_absent_with_a_reason(tmp_path, capsys) -> None:
    _write_tokenizer(tmp_path, 12)

    _rc, captured, payload = _scan(capsys, tmp_path)
    row = _coverage(payload)["tensor_map.consistency"]

    assert row["state"] == "absent"
    assert "no top-level .safetensors file was present" in row["reason"]
    assert "no top-level .safetensors file was present" in captured.err
    assert not any(key.startswith("safetensors.") for key in _coverage(payload))


def test_sharded_headers_are_each_parsed_and_checked(tmp_path, capsys) -> None:
    _write_tokenizer(tmp_path, 12)
    _write_safetensors(tmp_path, "model-00001-of-00002.safetensors", {
        "model.layers.0.mlp.weight": [4, 4],
    })
    _write_safetensors(tmp_path, "model-00002-of-00002.safetensors", {
        "lm_head.weight": [10, 4],
    })

    rc, _captured, payload = _scan(capsys, tmp_path)
    by_id = _coverage(payload)
    tok002 = [f for f in payload["findings"] if f["rule_id"] == "TOK002"]

    assert rc == cli.EXIT_FAIL
    assert len(tok002) == 1
    assert tok002[0]["location"] == "safetensors:lm_head.weight"
    assert by_id["safetensors.model-00001-of-00002.safetensors"]["state"] == "examined"
    assert by_id["safetensors.model-00002-of-00002.safetensors"]["state"] == "examined"
    assert by_id["tensor_map.consistency"]["state"] == "examined"


def test_invalid_header_is_unparseable_not_absent(tmp_path, capsys) -> None:
    _write_tokenizer(tmp_path, 12)
    (tmp_path / "model.safetensors").write_bytes(b"{nope")

    _rc, _captured, payload = _scan(capsys, tmp_path)
    by_id = _coverage(payload)

    assert by_id["safetensors.model.safetensors"]["state"] == "unparseable"
    assert by_id["tensor_map.consistency"]["state"] == "unparseable"
    assert "TOK002" not in {f["rule_id"] for f in payload["findings"]}


def test_header_declaring_more_bytes_than_the_file_is_unparseable(
    tmp_path, capsys
) -> None:
    _write_tokenizer(tmp_path, 12)
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", 4096) + b"{}")

    _rc, _captured, payload = _scan(capsys, tmp_path)
    by_id = _coverage(payload)

    assert by_id["safetensors.model.safetensors"]["state"] == "unparseable"
    assert "past the end" in by_id["safetensors.model.safetensors"]["reason"]


def test_over_cap_header_is_partial_and_never_read(tmp_path, capsys) -> None:
    _write_tokenizer(tmp_path, 12)
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", MAX_HEADER_BYTES + 1)
    )

    _rc, _captured, payload = _scan(capsys, tmp_path)
    by_id = _coverage(payload)

    assert by_id["safetensors.model.safetensors"]["state"] == "partial"
    assert by_id["tensor_map.consistency"]["state"] == "partial"
    assert "cap" in by_id["safetensors.model.safetensors"]["reason"]


def test_out_of_root_safetensors_link_is_refused_and_never_read(
    tmp_path, capsys
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    secret = tmp_path / "secret.safetensors"
    secret.write_bytes(_file_bytes(
        {"model.embed_tokens.weight": [10, 4]},
        data=b"TOP SECRET TENSOR BYTES",
    ))
    _symlink_or_skip(secret, root / "model.safetensors")
    _write_tokenizer(root, 12)

    rc, captured, payload = _scan(capsys, root)
    by_id = _coverage(payload)
    rep = [f for f in payload["findings"] if f["rule_id"] == "REP001"]

    assert rc == cli.EXIT_OK  # REP001 is a WARN and the default fail gate is FAIL
    assert len(rep) == 1
    assert by_id["safetensors.model.safetensors"]["state"] == "skipped"
    assert by_id["tensor_map.consistency"]["state"] == "skipped"
    assert by_id["repo.links"]["state"] == "skipped"
    assert "model.safetensors -> " in by_id["repo.links"]["reason"]
    assert "TOP SECRET TENSOR BYTES" not in captured.out
    assert "TOP SECRET TENSOR BYTES" not in captured.err


def test_reader_lists_only_top_level_safetensors_names(tmp_path) -> None:
    _write_tokenizer(tmp_path, 12)
    _write_safetensors(tmp_path, "a.safetensors", {"a.weight": [1]})
    _write_safetensors(tmp_path, "b.safetensors", {"b.weight": [1]})
    (tmp_path / "sub").mkdir()
    _write_safetensors(tmp_path / "sub", "c.safetensors", {"c.weight": [1]})
    (tmp_path / ".hidden.safetensors").write_bytes(b"ignored")

    repo = LocalRepo(str(tmp_path))

    assert repo.safetensors_names() == ["a.safetensors", "b.safetensors"]


def test_more_than_the_file_cap_is_partial(tmp_path, capsys, monkeypatch) -> None:
    from c4nary import repo as repo_module

    _write_tokenizer(tmp_path, 12)
    for index in range(3):
        _write_safetensors(tmp_path, f"m{index}.safetensors", {
            f"t{index}": [1],
        })
    monkeypatch.setattr(repo_module, "MAX_HEADER_FILES", 2)

    _rc, _captured, payload = _scan(capsys, tmp_path)
    by_id = _coverage(payload)

    assert by_id["tensor_map.consistency"]["state"] == "partial"
    assert "only the first 2" in by_id["tensor_map.consistency"]["reason"]
    assert "safetensors.m0.safetensors" in by_id
    assert "safetensors.m2.safetensors" not in by_id


# --------------------------------------------------------------------------- #
# Remote repo reader
# --------------------------------------------------------------------------- #

class FakeRemote:
    """Injected listing/text/range trio; records every range request."""

    def __init__(self, text_files: dict[str, str] | None = None,
                 shard_files: dict[str, bytes] | None = None,
                 *, unfetchable=()):
        self.text_files = dict(text_files or {})
        self.shard_files = dict(shard_files or {})
        self.unfetchable = frozenset(unfetchable)
        self.ranges: list[tuple[str, int, int]] = []
        self.text_fetched: list[str] = []

    def list_files(self, repo: str) -> list[str]:
        return sorted({*self.text_files, *self.shard_files})

    def fetch_text(self, repo: str, name: str, *, max_bytes: int = 1 << 20,
                   python_only: bool = False):
        from c4nary.bundle import ReadResult

        self.text_fetched.append(name)
        text = self.text_files.get(name)
        if text is None:
            return None
        data = text.encode("utf-8")
        return ReadResult(
            data[:max_bytes].decode("utf-8", errors="replace"),
            truncated=len(data) > max_bytes,
        )

    def fetch_range(self, repo: str, name: str, start: int, length: int):
        self.ranges.append((name, start, length))
        if name in self.unfetchable:
            return None
        data = self.shard_files.get(name)
        if data is None:
            return None
        return data[start:start + length]

    def repo(self, target: str = "org/model") -> RemoteRepo:
        return RemoteRepo(
            target,
            list_files=self.list_files,
            fetch_text=self.fetch_text,
            fetch_range=self.fetch_range,
        )


def _remote_shard(tensors: dict, *, data: bytes = b"") -> bytes:
    return _file_bytes(tensors, data=data)


def _remote_files(vocab: int = 12) -> dict[str, str]:
    return {
        "tokenizer.json": json.dumps(
            {"model": {"vocab": {f"t{index}": index for index in range(vocab)}}}
        ),
    }


def test_remote_header_requests_only_prefix_and_header_ranges() -> None:
    shard = _remote_shard({"model.embed_tokens.weight": [10, 4]},
                          data=b"weights-not-fetched")
    fake = FakeRemote(_remote_files(), {"model.safetensors": shard})
    repo = fake.repo()

    findings, _coverage_rows = scan_repo(repo)
    header_len = len(shard) - PREFIX_BYTES - len(b"weights-not-fetched")

    # start=0,length=8 then start=8,length=N. The data range is never requested.
    assert fake.ranges == [
        ("model.safetensors", 0, PREFIX_BYTES),
        ("model.safetensors", PREFIX_BYTES, header_len),
    ]
    tok002 = [f for f in findings if f.rule_id == "TOK002"]
    assert len(tok002) == 1
    assert tok002[0].location == "safetensors:model.embed_tokens.weight"


def test_remote_coverage_reports_examined_tensor_map() -> None:
    shard = _remote_shard({"model.embed_tokens.weight": [10, 4]})
    fake = FakeRemote(_remote_files(), {"model.safetensors": shard})

    _findings, coverage = scan_repo(fake.repo())
    rows = {row.id: row for row in coverage.surfaces}

    assert rows["safetensors.model.safetensors"].state == "examined"
    assert rows["tensor_map.consistency"].state == "examined"


def test_remote_listed_but_unfetchable_header_is_skipped() -> None:
    fake = FakeRemote(
        _remote_files(), {"model.safetensors": b"unused"},
        unfetchable={"model.safetensors"},
    )

    findings, coverage = scan_repo(fake.repo())
    rows = {row.id: row for row in coverage.surfaces}

    assert rows["safetensors.model.safetensors"].state == "skipped"
    assert rows["tensor_map.consistency"].state == "skipped"
    assert "TOK002" not in {finding.rule_id for finding in findings}


def test_remote_no_safetensors_is_absent() -> None:
    fake = FakeRemote(_remote_files())

    _findings, coverage = scan_repo(fake.repo())
    rows = {row.id: row for row in coverage.surfaces}

    assert rows["tensor_map.consistency"].state == "absent"
    assert "no top-level .safetensors file was present" in \
        rows["tensor_map.consistency"].reason
    assert not any(row.id.startswith("safetensors.") for row in coverage.surfaces)


def test_remote_repo_never_selects_or_fetches_a_gguf(monkeypatch) -> None:
    import c4nary.remote as remote_module

    def forbidden(*args, **kwargs):
        raise AssertionError("repo ingest must not select or fetch a GGUF")

    monkeypatch.setattr(remote_module, "pick_gguf", forbidden)
    monkeypatch.setattr(remote_module, "fetch_remote_model", forbidden)
    shard = _remote_shard({"model.embed_tokens.weight": [10, 4]})
    fake = FakeRemote(
        {**_remote_files(), "README.md": "hello\n"},
        {"model.safetensors": shard},
    )

    findings, _coverage = scan_repo(fake.repo())

    assert "TOK002" in {finding.rule_id for finding in findings}
    assert all(not name.lower().endswith(".gguf") for name, _, _ in fake.ranges)


# --------------------------------------------------------------------------- #
# Remote range fetch helper
# --------------------------------------------------------------------------- #

class _Response:
    def __init__(self, chunks, *, status_code=206):
        self.chunks = chunks
        self.status_code = status_code
        self.closed = False
        self.body_read = False

    def iter_content(self, chunk_size):
        self.body_read = True
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


def test_fetch_remote_range_accepts_206(monkeypatch) -> None:
    from c4nary import remote

    response = _Response([b"abcd"])
    session = _Session(response)
    monkeypatch.setattr(remote, "_session", lambda: session)

    assert remote.fetch_remote_range("org/model", "model.safetensors", 0, 4) == b"abcd"
    _url, kwargs = session.calls[0]
    assert kwargs["headers"]["Range"] == "bytes=0-3"
    assert kwargs["stream"] is True


def test_fetch_remote_range_refuses_a_200_for_a_nonzero_start(monkeypatch) -> None:
    from c4nary import remote

    response = _Response([b"whole-file"], status_code=200)
    session = _Session(response)
    monkeypatch.setattr(remote, "_session", lambda: session)

    assert remote.fetch_remote_range(
        "org/model", "model.safetensors", 8, 4) is None
    assert response.closed is True
    assert response.body_read is False  # the full body was never consumed


def test_fetch_remote_range_accepts_a_200_for_the_prefix(monkeypatch) -> None:
    from c4nary import remote

    response = _Response([b"0123456789"], status_code=200)
    monkeypatch.setattr(remote, "_session", lambda: _Session(response))

    data = remote.fetch_remote_range("org/model", "model.safetensors", 0, 8)

    assert data == b"01234567"  # streamed hard cap, then only the range bytes
    assert response.closed is True


def test_fetch_remote_range_caps_a_lying_body(monkeypatch) -> None:
    from c4nary import remote

    response = _Response([b"x" * 100], status_code=206)
    monkeypatch.setattr(remote, "_session", lambda: _Session(response))

    assert len(remote.fetch_remote_range(
        "org/model", "model.safetensors", 0, 8)) == 8


def test_fetch_remote_range_scopes_auth_to_huggingface(monkeypatch) -> None:
    from c4nary import remote

    monkeypatch.setenv("HF_TOKEN", "secret")
    response = _Response([b"x"])
    session = _Session(response)
    monkeypatch.setattr(remote, "_session", lambda: session)

    remote.fetch_remote_range("org/model", "model.safetensors", 0, 1)

    _url, kwargs = session.calls[0]
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["headers"]["Accept-Encoding"] == "identity"


def test_fetch_remote_range_rejects_an_over_cap_length(monkeypatch) -> None:
    from c4nary import remote

    def forbidden(*args, **kwargs):
        raise AssertionError("an over-cap range must be rejected before any request")

    monkeypatch.setattr(remote, "_session", forbidden)
    with pytest.raises(ValueError, match="32 MiB"):
        remote.fetch_remote_range(
            "org/model", "model.safetensors", 0, (32 << 20) + 1)


@pytest.mark.parametrize(
    "filename",
    ["../model.safetensors", "sub/model.safetensors", "sub\\model.safetensors",
     ".hidden.safetensors", "model..safetensors", "model.bin",
     "model?.safetensors", "model#.safetensors", "model%.safetensors"],
)
def test_fetch_remote_range_rejects_unsafe_names(monkeypatch, filename) -> None:
    from c4nary import remote

    def forbidden(*args, **kwargs):
        raise AssertionError("unsafe name must be rejected before any request")

    monkeypatch.setattr(remote, "_session", forbidden)

    with pytest.raises(ValueError):
        remote.fetch_remote_range("org/model", filename, 0, 8)


# --------------------------------------------------------------------------- #
# MCP repo targets automatically include the new coverage/findings
# --------------------------------------------------------------------------- #

def test_mcp_local_repo_includes_safetensors_findings(tmp_path) -> None:
    _write_tokenizer(tmp_path, 12)
    _write_safetensors(tmp_path, "model.safetensors", {
        "model.embed_tokens.weight": [10, 4],
    })

    result = run_scan(str(tmp_path), target_type="repo")
    coverage = {row["id"]: row for row in result["coverage"]}
    tok002 = [f for f in result["findings"] if f["rule_id"] == "TOK002"]

    assert len(tok002) == 1
    assert coverage["safetensors.model.safetensors"]["state"] == "examined"
    assert coverage["tensor_map.consistency"]["state"] == "examined"


def test_mcp_remote_repo_includes_safetensors_findings(monkeypatch) -> None:
    import c4nary.repo as repo_module

    shard = _remote_shard({"model.embed_tokens.weight": [10, 4]})
    fake = FakeRemote(_remote_files(), {"model.safetensors": shard})
    monkeypatch.setattr(repo_module, "_hf_list_files", fake.list_files)
    monkeypatch.setattr(repo_module, "fetch_repo_text", fake.fetch_text)
    monkeypatch.setattr(repo_module, "fetch_remote_range", fake.fetch_range)

    result = run_scan("org/model", remote=True, target_type="repo")
    coverage = {row["id"]: row for row in result["coverage"]}
    ids = {f["rule_id"] for f in result["findings"]}

    assert "TOK002" in ids
    assert coverage["tensor_map.consistency"]["state"] == "examined"
    assert fake.ranges and all(start + length <= len(shard)
                               for _name, start, length in fake.ranges)
