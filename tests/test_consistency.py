"""Metadata/tokenizer/structural consistency rules (MET01x, TOK, STR, INT005-006)."""

import struct
from pathlib import Path

from _ggufgen import _gguf_str, _value, write_gguf

from c4nary.parser import parse_gguf
from c4nary.rules.metadata import analyze_metadata
from c4nary.rules.structure import analyze_structure
from c4nary.rules.tokenizer import analyze_tokenizer


def _meta_ids(model):
    return {f.rule_id for f in analyze_metadata(model)}


def _tok_ids(model):
    return {f.rule_id for f in analyze_tokenizer(model)}


def _str_ids(model):
    return {f.rule_id for f in analyze_structure(model)}


# --------------------------- metadata consistency ------------------------- #

def test_noncontiguous_layers_flagged(tmp_path):
    # A gap in the layer indices (missing layer 1) is the real tamper signal.
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "llama.block_count": 3,
    }, tensors=[("blk.0.attn.weight", (8, 8), 0),
                ("blk.2.attn.weight", (8, 8), 0)])
    assert "MET010" in _meta_ids(parse_gguf(p))


def test_partial_shard_not_flagged(tmp_path):
    # Fewer contiguous layers than declared = a deliberate shard, not tampering.
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "llama.block_count": 10,
    }, tensors=[(f"blk.{i}.attn.weight", (8, 8), 0) for i in range(5)])
    assert "MET010" not in _meta_ids(parse_gguf(p))


def test_shard_skips_layer_checks(tmp_path):
    # A multi-file shard holds a gappy subset of layers by design -> no MET010.
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "llama.block_count": 79,
        "split.count": 33, "split.no": 0,
    }, tensors=[(f"blk.{i}.attn.weight", (8, 8), 0) for i in (0, 1, 10, 11, 12)])
    assert "MET010" not in _meta_ids(parse_gguf(p))


def test_explicit_head_dim_skips_divisibility(tmp_path):
    # embedding_length % head_count != 0 is legal when head_dim is explicit.
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "llama.embedding_length": 5120,
        "llama.attention.head_count": 24,         # 5120 % 24 != 0
        "llama.attention.head_count_kv": 4,       # 24 % 4 == 0 (still checked)
        "llama.attention.key_length": 256,        # explicit head_dim
    }, tensors=[("token_embd.weight", (5120, 32), 0)])
    assert "MET012" not in _meta_ids(parse_gguf(p))


def test_embedding_length_mismatch(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "llama.embedding_length": 99,
    }, tensors=[("token_embd.weight", (8, 32), 0)])
    assert "MET011" in _meta_ids(parse_gguf(p))


def test_head_divisibility_violation(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "llama.embedding_length": 8,
        "llama.attention.head_count": 3,
    }, tensors=[("token_embd.weight", (8, 32), 0)])
    assert "MET012" in _meta_ids(parse_gguf(p))


def test_feed_forward_mismatch(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "llama.feed_forward_length": 99,
    }, tensors=[("blk.0.ffn_up.weight", (8, 32), 0)])
    assert "MET013" in _meta_ids(parse_gguf(p))


def test_consistent_model_no_consistency_findings(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "llama.block_count": 2,
        "llama.embedding_length": 8, "llama.attention.head_count": 2,
        "llama.attention.head_count_kv": 2, "llama.feed_forward_length": 32,
    }, tensors=[
        ("token_embd.weight", (8, 100), 0),
        ("blk.0.ffn_up.weight", (8, 32), 0),
        ("blk.1.ffn_up.weight", (8, 32), 0),
    ])
    ids = _meta_ids(parse_gguf(p))
    for rid in ("MET010", "MET011", "MET012", "MET013", "MET015"):
        assert rid not in ids


def test_duplicate_metadata_key(tmp_path):
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2)
    blob += _gguf_str("general.architecture") + _value("llama")
    blob += _gguf_str("general.architecture") + _value("llama")
    p = tmp_path / "dup.gguf"
    p.write_bytes(blob)
    model = parse_gguf(p)
    assert model.duplicate_keys == ("general.architecture",)
    assert "MET016" in _meta_ids(model)


# ------------------------------ tokenizer --------------------------------- #

def test_special_token_id_out_of_range(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "tokenizer.ggml.tokens": ["a", "b", "c"],
        "tokenizer.ggml.eos_token_id": 99,
    })
    assert "TOK001" in _tok_ids(parse_gguf(p))


def test_unset_special_token_id_not_flagged(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "tokenizer.ggml.tokens": ["a", "b", "c"],
        "tokenizer.ggml.eos_token_id": 2,           # valid
        "tokenizer.ggml.padding_token_id": -1,      # unset sentinel
    })
    assert "TOK001" not in _tok_ids(parse_gguf(p))


def test_vocab_desync_with_tensor(tmp_path):
    tokens = [f"t{i}" for i in range(100)]
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "tokenizer.ggml.tokens": tokens,
    }, tensors=[("token_embd.weight", (8, 16), 0)])  # 16 < 100
    assert "TOK002" in _tok_ids(parse_gguf(p))


def test_parallel_array_and_enum(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "tokenizer.ggml.tokens": ["a", "b", "c"],
        "tokenizer.ggml.scores": [0.1, 0.2],          # len 2 != 3
        "tokenizer.ggml.token_type": [1, 2, 9],       # 9 out of enum
    })
    assert "TOK003" in _tok_ids(parse_gguf(p))


def test_oversized_token(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "tokenizer.ggml.tokens": ["x" * 5000, "a"],
    })
    assert "TOK004" in _tok_ids(parse_gguf(p))


def test_add_bos_without_valid_id(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
        "tokenizer.ggml.tokens": ["a", "b"],
        "tokenizer.ggml.add_bos_token": True,        # no bos_token_id
    })
    assert "TOK005" in _tok_ids(parse_gguf(p))


# ------------------------------ structural -------------------------------- #

def test_offset_out_of_bounds(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
    }, tensors=[("a", (2, 2), 0)], offsets=[10 ** 9], data_len=16)
    assert "STR003" in _str_ids(parse_gguf(p))


def test_element_count_overflow(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
    }, tensors=[("a", (2 ** 40, 2 ** 40), 0)], offsets=[0], data_len=16)
    assert "STR001" in _str_ids(parse_gguf(p))


def test_bad_alignment(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "general.alignment": 7,  # not pow2
    }, tensors=[("a", (2, 2), 0)])
    assert "STR005" in _str_ids(parse_gguf(p))


def test_non_block_divisible_dim(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
    }, tensors=[("blk.0.attn.weight", (5, 8), 8)])  # Q8_0 block 32, 5%32 != 0
    assert "STR006" in _str_ids(parse_gguf(p))


def test_unknown_type(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
    }, tensors=[("a", (2, 2), 200)])  # unknown ggml type id
    assert "STR007" in _str_ids(parse_gguf(p))


def test_suspicious_tensor_name(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama",
    }, tensors=[("../etc/passwd", (2, 2), 0)])
    assert "STR008" in _str_ids(parse_gguf(p))


def test_adapter_tensor_flagged(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "llama.block_count": 1,
    }, tensors=[("blk.0.lora_a.weight", (8, 8), 0)])
    assert "INT005" in _str_ids(parse_gguf(p))


def test_shard_note(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "split.count": 3, "split.no": 0,
    }, tensors=[("a", (2, 2), 0)])
    assert "INT006" in _str_ids(parse_gguf(p))


def test_clean_model_no_structural_findings(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "general.alignment": 32,
    }, tensors=[("token_embd.weight", (8, 32), 0),
                ("blk.0.attn.weight", (64, 64), 8)])
    assert _str_ids(parse_gguf(p)) == set()
