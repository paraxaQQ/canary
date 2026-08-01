"""tokenizer.json normalizer/decoder rule NRM001. Run: pytest tests/test_tokenizer_json.py

Standard normalizers only map ' ' <-> '▁' (SPM) or collapse whitespace -- no letters. A
Replace that rewrites word content is the anomaly: it runs on every input/output.
"""

from c4nary.rules.tokenizer_json import analyze_special_tokens, analyze_tokenizer_json


def _ids(d):
    return {f.rule_id for f in analyze_tokenizer_json(d)}


def test_benign_spm_space_meta():
    d = {"normalizer": {"type": "Sequence", "normalizers": [
        {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "}]},
         "decoder": {"type": "ByteLevel"}}
    assert _ids(d) == set()


def test_benign_whitespace_regex():
    d = {"normalizer": {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}}
    assert _ids(d) == set()


def test_flags_literal_content_rewrite():
    d = {"normalizer": {"type": "Replace",
                        "pattern": {"String": "I cannot"}, "content": "I can"}}
    assert "NRM001" in _ids(d)


def test_flags_regex_refusal_strip():
    d = {"normalizer": {"type": "Replace",
                        "pattern": {"Regex": "(?i)sorry"}, "content": ""}}
    assert "NRM001" in _ids(d)


def test_flags_decoder_output_rewrite():
    d = {"decoder": {"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "unsafe"}, "content": "safe"}]}}
    assert "NRM001" in _ids(d)


def test_no_normalizer_clean():
    assert _ids({"normalizer": None, "decoder": {"type": "ByteLevel"}}) == set()
    assert analyze_tokenizer_json({}) == []


def test_nrm001_pretokenizer_and_postprocessor():
    assert "NRM001" in _ids(
        {"pre_tokenizer": {"type": "Replace", "pattern": {"String": "cannot"},
                           "content": "can"}})
    assert "NRM001" in _ids(
        {"post_processor": {"type": "Sequence", "processors": [
            {"type": "Replace", "pattern": {"String": "unsafe"}, "content": "safe"}]}})


def _stok_ids(*datas):
    return {f.rule_id for f in analyze_special_tokens(*datas)}


def test_nrm002_concealed_special_token():
    assert "NRM002" in _stok_ids({"additional_special_tokens": ["<|im_start|>​"]})
    assert "NRM002" in _stok_ids({"<|evil‮|>": 32000})   # added_tokens.json key form


def test_nrm002_benign_special_tokens_clean():
    assert analyze_special_tokens(
        {"bos_token": "<s>",
         "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}) == []


def test_deeply_nested_tokenizer_json_fails_closed_not_crashed():
    """A ~7 KB attacker-authored tokenizer.json must not abort the scan.

    The reader cap is 48 MiB but JSON nesting is unbounded, so the recursive walkers
    blew the stack five orders of magnitude under the cap. An uncaught RecursionError
    left stdout empty at exit 1 -- the code this tool's own table defines as "WARN
    findings present".
    """

    from c4nary.rules.tokenizer_json import analyze_special_tokens, analyze_tokenizer_json

    deep = inner = {}
    for _ in range(2000):
        inner["n"] = inner = {}
    payload = {"normalizer": deep, "post_processor": deep}

    assert analyze_tokenizer_json(payload) == []
    assert analyze_special_tokens(payload) == []
