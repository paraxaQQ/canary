"""Vocabulary-source abstraction (c4nary.vocab).

Covers the GGUF wrapper (byte-verbatim over ``MetaArray``), the ``tokenizer.json``
constructor reserved for repo ingest, and the two consumer regressions that depend
on the vocabulary being honest about what it actually holds.

Run: pytest tests/test_vocab.py
"""

from __future__ import annotations

from c4nary.coverage import build_scan_coverage
from c4nary.parser import GGUFModel, MetaArray, parse_gguf
from c4nary.rules.config import analyze_config
from c4nary.rules.template import analyze_templates
from c4nary.vocab import (
    SparseSurfaces,
    Vocab,
    vocab_from_gguf,
    vocab_from_metadata,
    vocab_from_tokenizer_json,
)

from _ggufgen import write_gguf


def _model(metadata: dict | None = None) -> GGUFModel:
    return GGUFModel(
        path="t",
        version=3,
        tensor_count=0,
        metadata=metadata or {},
        metadata_types={},
        tensors=(),
    )


def _coverage(model: GGUFModel, vocab: Vocab | None = None):
    return build_scan_coverage(
        model,
        remote=False,
        deep_tokenizer=True,
        bundle=False,
        manifest_requested=False,
        template_findings=analyze_templates(model),
        vocab=vocab,
    )


# --------------------------------------------------------------------------- #
# GGUF metadata constructor
# --------------------------------------------------------------------------- #

def test_vocab_from_metadata_mirrors_metaarray() -> None:
    tokens = MetaArray("string", 5, ("a", "b", "c", "d", "e"), False, max_elem_bytes=7)
    token_type = MetaArray("int32", 5, (1, 2, 3, 4, 5), False)
    vocab = vocab_from_metadata({
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": token_type,
    })

    assert vocab is not None
    assert vocab.surfaces == tokens.preview
    assert vocab.length == tokens.length
    assert vocab.complete == (not tokens.truncated)
    assert vocab.max_surface_bytes == tokens.max_elem_bytes == 7
    assert vocab.types == token_type.preview
    assert vocab.types_length == token_type.length
    assert vocab.types_complete is True
    assert vocab.source == "gguf:tokenizer.ggml.tokens"


def test_truncated_tokens_keep_true_length_and_hide_absent_ids() -> None:
    tokens = MetaArray("string", 100, ("a", "b"), True, max_elem_bytes=3)
    vocab = vocab_from_metadata({"tokenizer.ggml.tokens": tokens})

    assert vocab is not None
    assert vocab.complete is False
    assert vocab.length == 100
    assert vocab.surface(1) == "b"
    assert vocab.surface(2) is None          # not in the preview
    assert vocab.surface(99) is None         # in range of length, absent from surfaces
    assert vocab.max_surface_bytes == 3      # preserved even when truncated


def test_missing_or_scalar_tokens_returns_none() -> None:
    assert vocab_from_metadata({}) is None
    assert vocab_from_metadata({"tokenizer.ggml.tokens": "not-an-array"}) is None
    assert vocab_from_metadata({"tokenizer.ggml.tokens": ["a", "b"]}) is None
    assert vocab_from_metadata({"tokenizer.ggml.token_type": [1, 2]}) is None


def test_token_type_present_truncated_and_missing() -> None:
    tokens = MetaArray("string", 2, ("a", "b"), False)

    truncated = vocab_from_metadata({
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": MetaArray("int32", 100, (1, 2), True),
    })
    assert truncated is not None
    assert truncated.types == (1, 2)         # preview still answers the enum scan
    assert truncated.types_length == 100
    assert truncated.types_complete is False

    missing = vocab_from_metadata({"tokenizer.ggml.tokens": tokens})
    assert missing is not None
    assert missing.types is None
    assert missing.types_length is None
    assert missing.types_complete is False


def test_surface_rejects_out_of_range_non_str_and_bool_ids() -> None:
    vocab = Vocab(surfaces=("a", 7, None, "d"), length=4, complete=True)

    assert vocab.surface(0) == "a"
    assert vocab.surface(1) is None          # int element, not a surface string
    assert vocab.surface(2) is None
    assert vocab.surface(3) == "d"
    assert vocab.surface(-1) is None
    assert vocab.surface(4) is None
    assert vocab.surface(True) is None       # bool is an int subclass, still invalid
    assert vocab.surface("0") is None        # type: ignore[arg-type]


def test_surface_never_raises_on_odd_input() -> None:
    vocab = Vocab(surfaces=(None, object()), length=2, complete=True)
    for token_id in (None, 1.5, [], {}, object()):
        assert vocab.surface(token_id) is None  # type: ignore[arg-type]


def _legacy_reconstruct(ids, surfaces) -> str:
    parts = []
    for tid in ids:
        if 0 <= tid < len(surfaces) and isinstance(surfaces[tid], str):
            s = surfaces[tid]
            parts.append(" " + s[1:] if s[:1] in ("▁", "Ġ") else s)
    return "".join(parts)


def test_reconstruct_matches_legacy_behavior() -> None:
    surfaces = ("▁Hello", "Ġworld", "!", 12, None, "▁")
    vocab = Vocab(surfaces=surfaces, length=len(surfaces), complete=True)

    for ids in ([], [0, 1, 2], [3, 4], [99], [-1], [0, 5], [1, 0, 2]):
        assert vocab.reconstruct(ids) == _legacy_reconstruct(ids, surfaces)

    assert vocab.reconstruct([]) == ""
    assert vocab.reconstruct([0, 1, 2]) == " Hello world!"
    assert vocab.reconstruct([0, 5]) == " Hello "


def test_gguf_model_vocab_property(tmp_path) -> None:
    path = write_gguf(tmp_path / "m.gguf", {
        "tokenizer.ggml.tokens": ["<s>", "hi"],
        "tokenizer.ggml.token_type": [3, 1],
    })
    model = parse_gguf(path)

    vocab = model.vocab
    assert vocab is not None
    assert vocab.length == 2
    assert vocab.surfaces == ("<s>", "hi")
    assert vocab.types == (3, 1)
    assert vocab_from_gguf(model) == vocab


def test_gguf_model_vocab_property_is_none_without_tokens(tmp_path) -> None:
    model = parse_gguf(write_gguf(tmp_path / "m.gguf", {"general.architecture": "llama"}))
    assert model.vocab is None


# --------------------------------------------------------------------------- #
# tokenizer.json constructor (unused by the CLI for now)
# --------------------------------------------------------------------------- #

def test_tokenizer_json_dict_vocab() -> None:
    vocab = vocab_from_tokenizer_json({"model": {"vocab": {"a": 0, "b": 2}}})

    assert vocab is not None
    assert vocab.surfaces == ("a", None, "b")   # id 1 is a hole
    assert vocab.length == 3
    assert vocab.complete is True
    assert vocab.types is None
    assert vocab.types_complete is False
    assert vocab.max_surface_bytes == 1
    assert vocab.source == "tokenizer.json:model.vocab"


def test_tokenizer_json_list_of_pairs_vocab() -> None:
    vocab = vocab_from_tokenizer_json({
        "model": {"vocab": [["▁a", -1.0], ["b", -2.0]]},
    })

    assert vocab is not None
    assert vocab.surfaces == ("▁a", "b")
    assert vocab.length == 2
    assert vocab.max_surface_bytes == len("▁a".encode("utf-8"))


def test_tokenizer_json_ignores_bool_and_invalid_ids() -> None:
    vocab = vocab_from_tokenizer_json({
        "model": {"vocab": {"a": 0, "b": True, "c": "2", "d": -1}},
    })

    assert vocab is not None
    assert vocab.surfaces == ("a",)
    assert vocab.length == 1


def test_tokenizer_json_added_tokens_overlay_and_extend() -> None:
    vocab = vocab_from_tokenizer_json({
        "model": {"vocab": {"a": 0, "b": 1}},
        "added_tokens": [
            {"id": 1, "content": "<b>", "special": True},
            {"id": 3, "content": "<d>", "special": True},
        ],
    })

    assert vocab is not None
    assert vocab.surfaces == ("a", "<b>", None, "<d>")  # overlaid, extended, hole at 2
    assert vocab.length == 4


def test_tokenizer_json_added_token_with_bool_id_is_ignored() -> None:
    vocab = vocab_from_tokenizer_json({
        "model": {"vocab": {"a": 0}},
        "added_tokens": [{"id": True, "content": "x"}],
    })

    assert vocab is not None
    assert vocab.surfaces == ("a",)


def test_tokenizer_json_missing_empty_or_invalid_returns_none() -> None:
    assert vocab_from_tokenizer_json({}) is None
    assert vocab_from_tokenizer_json({"model": "nope"}) is None
    assert vocab_from_tokenizer_json({"model": {}}) is None
    assert vocab_from_tokenizer_json({"model": {"vocab": {}}}) is None
    assert vocab_from_tokenizer_json({"model": {"vocab": []}}) is None
    assert vocab_from_tokenizer_json({
        "model": {"vocab": {"a": True}},
    }) is None


def test_tokenizer_json_huge_id_is_sparse() -> None:
    # A 50M id would be a 400MB list if materialised. Only the two present
    # entries are stored; every hole reads as None.
    vocab = vocab_from_tokenizer_json({
        "model": {"vocab": {"a": 0, "z": 50_000_000}},
    })

    assert vocab is not None
    assert isinstance(vocab.surfaces, SparseSurfaces)
    assert vocab.length == 50_000_001
    assert vocab.surface(0) == "a"
    assert vocab.surface(1) is None
    assert vocab.surface(50_000_000) == "z"
    assert vocab.surface(50_000_001) is None
    assert vocab.surfaces[:2] == ("a", None)
    assert vocab.surfaces[-1] == "z"


def test_tokenizer_json_added_tokens_can_supply_an_empty_vocab() -> None:
    vocab = vocab_from_tokenizer_json({
        "model": {"vocab": {}},
        "added_tokens": [{"id": 0, "content": "<pad>"}],
    })

    assert vocab is not None
    assert vocab.surfaces == ("<pad>",)
    assert vocab.length == 1


# --------------------------------------------------------------------------- #
# Consumer regressions
# --------------------------------------------------------------------------- #

def test_cfg002_fires_and_incomplete_vocab_claims_nothing() -> None:
    model = _model({
        "tokenizer.ggml.eos_token_id": 0,
        "tokenizer.ggml.tokens": MetaArray(
            "string", 4, ("<pad>", "hello", "▁Sorry", "world"), False),
    })

    assert "CFG002" in {f.rule_id for f in analyze_config(
        model, {"suppress_tokens": [2]})}

    truncated = Vocab(surfaces=("<pad>",), length=4, complete=False)
    assert "CFG002" not in {f.rule_id for f in analyze_config(
        model, {"suppress_tokens": [2]}, vocab=truncated)}
    assert truncated.surface(2) is None


def test_cfg002_injected_complete_vocab_is_used() -> None:
    model = _model()  # no GGUF token array at all
    vocab = Vocab(
        surfaces=("<pad>", "hello", "ĠSorry", "world"),
        length=4,
        complete=True,
    )

    assert "CFG002" in {f.rule_id for f in analyze_config(
        model, {"suppress_tokens": [2]}, vocab=vocab)}


def test_deep_tokenizer_coverage_examined_only_when_both_arrays_complete() -> None:
    model = _model()
    complete = Vocab(
        surfaces=("a", "b"), length=2, complete=True,
        types=(1, 1), types_length=2, types_complete=True,
    )
    partial_types = Vocab(
        surfaces=("a", "b"), length=2, complete=True,
        types=(1,), types_length=2, types_complete=False,
    )
    no_types = Vocab(surfaces=("a", "b"), length=2, complete=True)
    partial_surfaces = Vocab(surfaces=("a",), length=2, complete=False)

    rows = {
        name: {row["id"]: row for row in _coverage(model, vocab).to_dict()}
        for name, vocab in (
            ("complete", complete),
            ("partial_types", partial_types),
            ("no_types", no_types),
            ("partial_surfaces", partial_surfaces),
        )
    }

    assert rows["complete"]["tokenizer.deep"]["state"] == "examined"
    for name in ("partial_types", "no_types", "partial_surfaces"):
        assert rows[name]["tokenizer.deep"]["state"] == "partial"


def test_deep_tokenizer_coverage_from_gguf_metadata_arrays() -> None:
    model = _model({
        "tokenizer.ggml.tokens": MetaArray("string", 2, ("a", "b"), False),
        "tokenizer.ggml.token_type": MetaArray("int32", 2, (1, 1), False),
    })
    examined = {row["id"]: row for row in _coverage(model).to_dict()}
    assert examined["tokenizer.deep"]["state"] == "examined"

    model.metadata["tokenizer.ggml.token_type"] = MetaArray(
        "int32", 2, (1,), True)
    partial = {row["id"]: row for row in _coverage(model).to_dict()}
    assert partial["tokenizer.deep"]["state"] == "partial"
