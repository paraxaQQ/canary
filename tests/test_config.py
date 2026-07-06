"""Decode-time config-lever rules CFG001 / CFG002. Run: pytest tests/test_config.py

Synthetic model + config dicts (no network). Covers the Whisper false-positive guard
(begin_suppress_tokens is legitimate) and the bad_words singleton-vs-phrase distinction.
"""

from c4nary.parser import GGUFModel, MetaArray
from c4nary.rules.config import analyze_config


def _model(tokens=None, eos=None):
    meta = {}
    if eos is not None:
        meta["tokenizer.ggml.eos_token_id"] = eos
    if tokens is not None:
        meta["tokenizer.ggml.tokens"] = MetaArray("string", len(tokens), tuple(tokens), False)
    return GGUFModel(path="t", version=3, tensor_count=0, metadata=meta,
                     metadata_types={}, tensors=())


def _ids(model, cfg):
    return {f.rule_id for f in analyze_config(model, cfg)}


def test_cfg001_suppress_eos_fires():
    assert "CFG001" in _ids(_model(eos=2), {"suppress_tokens": [5, 2, 9]})


def test_cfg001_begin_suppress_eos_is_benign_whisper():
    # first-step-only suppression of eos is a legit anti-empty-output measure (Whisper)
    assert "CFG001" not in _ids(_model(eos=2), {"begin_suppress_tokens": [220, 2]})


def test_cfg001_no_stop_suppressed_is_clean():
    assert "CFG001" not in _ids(_model(eos=2), {"suppress_tokens": [5, 9, 13]})


def test_cfg001_bad_words_singleton_eos_fires():
    assert "CFG001" in _ids(_model(eos=2), {"bad_words_ids": [[2]]})


def test_cfg001_bad_words_phrase_not_flagged():
    # [7, 2] bans the sequence, not the eos token itself
    assert "CFG001" not in _ids(_model(eos=2), {"bad_words_ids": [[7, 2]]})


def test_cfg002_refusal_surface_suppressed_fires():
    m = _model(tokens=["<pad>", "hello", "▁Sorry", "world"], eos=0)
    assert "CFG002" in _ids(m, {"suppress_tokens": [2]})


def test_cfg002_benign_suppress_is_clean():
    m = _model(tokens=["<pad>", "hello", "▁Sorry", "world"], eos=0)
    assert "CFG002" not in _ids(m, {"suppress_tokens": [1, 3]})


def test_cfg002_matches_bpe_gpt2_prefix():
    # cross-family coverage: BPE/gpt2 word-start is Ġ, not SPM's ▁ -- both must match
    m = _model(tokens=["<pad>", "ĠSorry", "Ġcannot", "x"], eos=0)
    assert "CFG002" in _ids(m, {"suppress_tokens": [1]})   # ĠSorry
    assert "CFG002" in _ids(m, {"suppress_tokens": [2]})   # Ġcannot


def test_cfg001_config_eos_list_captured():
    # multi-stop models (Llama-3) declare eos as a LIST in the config; suppressing the
    # stop the GGUF didn't declare must still be caught via the union.
    m = _model(eos=128009)
    assert "CFG001" in _ids(m, {"suppress_tokens": [128001],
                                "eos_token_id": [128001, 128009]})


def test_cfg002_multitoken_bad_words_phrase_reconstructs():
    # a banned SEQUENCE that spells a refusal -- split apostrophe form and a phrase --
    # is caught by reconstruction, which single-token matching would miss.
    m = _model(tokens=["<pad>", "Ġcan", "'t", "ĠI", "Ġcannot"], eos=0)
    assert "CFG002" in _ids(m, {"bad_words_ids": [[1, 2]]})   # Ġcan + 't  -> "can't"
    assert "CFG002" in _ids(m, {"bad_words_ids": [[3, 4]]})   # ĠI + Ġcannot -> "I cannot"


def test_cfg002_benign_multitoken_bad_words_is_clean():
    m = _model(tokens=["<pad>", "Ġfoo", "Ġbar"], eos=0)
    assert "CFG002" not in _ids(m, {"bad_words_ids": [[1, 2]]})   # "foo bar" -> not refusal


def test_empty_or_unrelated_config_no_findings():
    assert analyze_config(_model(eos=2), {}) == []
    assert analyze_config(_model(eos=2), {"temperature": 0.7, "top_p": 0.9}) == []


def test_scalar_tokens_does_not_crash():
    # a crafted model with a SCALAR at tokenizer.ggml.tokens (not a MetaArray) must not
    # crash the whole --bundle scan -- was an uncaught AttributeError on toks.truncated.
    m = GGUFModel(path="t", version=3, tensor_count=0,
                  metadata={"tokenizer.ggml.tokens": "not-an-array"},
                  metadata_types={}, tensors=())
    assert analyze_config(m, {"suppress_tokens": [1, 2], "bad_words_ids": [[3]]}) is not None
