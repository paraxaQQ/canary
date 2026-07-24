"""Parser round-trip and defensive-parsing tests."""

import struct

import pytest
from _ggufgen import write_gguf

from c4nary.parser import (
    GGUFParseError,
    MetaArray,
    extract_gguf_chat_template_bytes,
    parse_gguf,
    parse_gguf_metadata_bytes,
)


def test_roundtrip_metadata_and_tensors(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "llama.context_length": 4096,
        "general.name": "round-trip",
        "general.quantized": True,
    }, tensors=[("token_embd.weight", (64, 128), 0),
                ("blk.0.attn_q.weight", (64, 64), 8)])
    model = parse_gguf(p)
    assert model.version == 3
    assert model.architecture == "llama"
    assert model.metadata["llama.context_length"] == 4096
    assert model.metadata["general.quantized"] is True
    assert model.tensor_count == 2
    by_name = {t.name: t for t in model.tensors}
    assert by_name["token_embd.weight"].shape == (64, 128)
    assert by_name["token_embd.weight"].dtype == "F32"
    assert by_name["blk.0.attn_q.weight"].dtype == "Q8_0"


def test_large_string_array_is_previewed(tmp_path):
    tokens = [f"tok{i}" for i in range(500)]
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "tokenizer.ggml.tokens": tokens,
    })
    model = parse_gguf(p)
    arr = model.metadata["tokenizer.ggml.tokens"]
    assert isinstance(arr, MetaArray)
    assert arr.length == 500
    assert arr.truncated is True
    assert len(arr.preview) == 64


def test_chat_template_property(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "tokenizer.chat_template": "{{ messages }}",
    })
    assert parse_gguf(p).chat_template == "{{ messages }}"


def test_metadata_only_bytes_do_not_require_tensor_table(tmp_path):
    tensor_name = "tensor-name-that-is-not-in-metadata"
    p = write_gguf(
        tmp_path / "m.gguf",
        {"tokenizer.chat_template": "{{ messages }}"},
        tensors=[(tensor_name, (2, 2), 0)],
    )
    data = p.read_bytes()
    metadata_end = data.index(tensor_name.encode("utf-8")) - 8

    with pytest.raises(GGUFParseError):
        parse_gguf_metadata_bytes(data[:metadata_end - 1])
    model = parse_gguf_metadata_bytes(data[:metadata_end])

    assert model.chat_template == "{{ messages }}"
    assert model.tensor_count == 1
    assert model.tensors == ()
    assert model.data_start == 0


def test_chat_template_extractor_skips_large_vocab(tmp_path):
    p = write_gguf(
        tmp_path / "m.gguf",
        {
            "tokenizer.ggml.tokens": [f"token-{i}" for i in range(10_000)],
            "tokenizer.chat_template": "{{ messages }}",
        },
    )
    data = p.read_bytes()

    assert extract_gguf_chat_template_bytes(data) == "{{ messages }}"
    with pytest.raises(GGUFParseError):
        extract_gguf_chat_template_bytes(data[: len(data) // 2])


def test_bad_magic_raises(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"XXXX" + b"\x00" * 32)
    with pytest.raises(GGUFParseError):
        parse_gguf(bad)


def test_truncated_file_raises(tmp_path):
    good = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "general.name": "x",
    }, tensors=[("a", (2, 2), 0)])
    data = good.read_bytes()
    truncated = tmp_path / "trunc.gguf"
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(GGUFParseError):
        parse_gguf(truncated)


def test_absurd_array_length_rejected(tmp_path):
    # Hand-craft a header claiming a 1-billion element string array in a tiny
    # file: the bounds check must reject it rather than loop/allocate.
    blob = bytearray()
    blob += b"GGUF" + struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)  # tensor count
    blob += struct.pack("<Q", 1)  # metadata count
    key = b"tokenizer.ggml.tokens"
    blob += struct.pack("<Q", len(key)) + key
    blob += struct.pack("<I", 9)            # type = array
    blob += struct.pack("<I", 8)            # elem type = string
    blob += struct.pack("<Q", 1_000_000_000)  # absurd length
    p = tmp_path / "evil.gguf"
    p.write_bytes(bytes(blob))
    with pytest.raises(GGUFParseError, match="cannot fit"):
        parse_gguf(p)
    # remote.py's fetch escalation keys on this exact phrase to decide "header bigger
    # than this chunk -> fetch more" (vs a truncated big-vocab header); keep them coupled.
