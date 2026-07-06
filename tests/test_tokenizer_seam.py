"""Deep tokenizer-seam pass (TOK015 + the deep/materialization gating).

Run: pytest tests/test_tokenizer_seam.py

The seam pass is opt-in (``deep=True``) and needs the FULL vocab materialized. TOK010
(static "delimiter typed NORMAL") was calibrated out (~6% FP incl. Gemma) -- a single-
token NORMAL delimiter still resolves to its id, so it is not a broken boundary; the
real seam WARNs are encoder-gated (Piece B). TOK015 is the informational surface metric
that confirms the pass ran.
"""

from c4nary.parser import GGUFModel, MetaArray
from c4nary.rules.tokenizer import analyze_tokenizer

NORMAL, CONTROL = 1, 3


def _model(tokens, types, template, *, truncated=False):
    meta = {
        "tokenizer.chat_template": template,
        "tokenizer.ggml.tokens": MetaArray("string", len(tokens), tuple(tokens), truncated),
        "tokenizer.ggml.token_type": MetaArray("int32", len(types), tuple(types), False),
    }
    return GGUFModel(path="t", version=3, tensor_count=0, metadata=meta,
                     metadata_types={}, tensors=())


def _rule(model, rid, deep=True):
    return [f for f in analyze_tokenizer(model, deep=deep) if f.rule_id == rid]


def _ids(model, deep=True):
    return [f.rule_id for f in analyze_tokenizer(model, deep=deep)]


def test_tok015_counts_reachable_excludes_noise():
    # 2 real role specials + a whitespace CONTROL + a <pad> reserved -> reachable == 2
    toks = ["<|im_start|>", "<|im_end|>", "\n", "<pad>", "hello"]
    typ = [CONTROL, CONTROL, CONTROL, CONTROL, NORMAL]
    f = _rule(_model(toks, typ, "{{ '<|im_start|>' }}"), "TOK015")
    assert len(f) == 1 and "2 reachable" in f[0].detail


def test_tok015_only_with_deep():
    m = _model(["a", "<|im_start|>"], [NORMAL, CONTROL], "{{ '<|im_start|>' }}")
    assert "TOK015" not in _ids(m, deep=False)
    assert "TOK015" in _ids(m, deep=True)


def test_truncated_vocab_no_seam_even_when_deep():
    # silence != clean: a truncated preview can't answer reachability -> pass no-ops
    m = _model(["a", "<|im_start|>"], [NORMAL, CONTROL], "{{ '<|im_start|>' }}",
               truncated=True)
    assert "TOK015" not in _ids(m, deep=True)


def test_gemma_style_normal_delimiter_is_not_flagged():
    # regression against the killed TOK010: a NORMAL single-token turn delimiter that the
    # template emits must NOT raise any WARN/FAIL -- only the INFO summary.
    m = _model(["<start_of_turn>", "<end_of_turn>", "hi"], [NORMAL, NORMAL, NORMAL],
               "{{ '<start_of_turn>' + x + '<end_of_turn>' }}")
    sev = {f.severity for f in analyze_tokenizer(m, deep=True)}
    assert sev <= {"INFO"}


def test_tok012_confusable_pair_is_info():
    # ASCII <|User|> + fullwidth <｜User｜> both registered special -> INFO, never WARN
    ascii_u, full_u = "<|User|>", "<｜User｜>"
    m = _model([ascii_u, full_u, "hi"], [CONTROL, CONTROL, NORMAL], "{{ x }}")
    f = _rule(m, "TOK012")
    assert len(f) == 1
    assert f[0].severity == "INFO"
    assert ascii_u in f[0].detail and full_u in f[0].detail  # raw forms preserved


def test_tok012_single_form_no_flag():
    # distinct role tokens with distinct skeletons -> no confusable collision
    m = _model(["<|User|>", "<|Assistant|>"], [CONTROL, CONTROL], "{{ x }}")
    assert _rule(m, "TOK012") == []
