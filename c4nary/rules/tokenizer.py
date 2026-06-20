"""Tokenizer consistency rules (TOK001-005).

Deterministic cross-checks over ``tokenizer.ggml.*`` metadata and the tensor
map: special-token ids in range, vocabulary synchronized with the embedding /
output tensors, parallel arrays the same length, and BOS/EOS flags consistent.
Array *contents* are previewed (first 64), but ``MetaArray.length`` is the true
count, so all length/range checks are exact. No weight data is read.
"""

from __future__ import annotations

from ..parser import INT32_MAX, GGUFModel, MetaArray
from ..report import Finding
from .registry import finding

# Sentinels some exporters use for "unset" (uint32 -1 / int32 -1).
_UNSET = frozenset({-1, 0xFFFFFFFF})

_SPECIAL_ID_KEYS = (
    "bos_token_id", "eos_token_id", "eot_token_id", "eom_token_id",
    "unknown_token_id", "separator_token_id", "padding_token_id",
    "cls_token_id", "mask_token_id",
)

# tokenizer.ggml.token_type enum (NORMAL..BYTE).
_TOKEN_TYPE_RANGE = range(1, 7)


def _vocab_size(model: GGUFModel) -> int | None:
    tokens = model.metadata.get("tokenizer.ggml.tokens")
    return tokens.length if isinstance(tokens, MetaArray) else None


def analyze_tokenizer(model: GGUFModel) -> list[Finding]:
    findings: list[Finding] = []
    meta = model.metadata
    vocab = _vocab_size(model)
    by_name = {t.name: t for t in model.tensors}

    # TOK001 - special-token ids within the vocabulary.
    if vocab is not None:
        for key in _SPECIAL_ID_KEYS:
            tid = meta.get(f"tokenizer.ggml.{key}")
            if isinstance(tid, int) and not isinstance(tid, bool) and tid not in _UNSET:
                if tid < 0 or tid >= vocab:
                    findings.append(finding(
                        "TOK001",
                        f"{key}={tid} is outside the vocabulary [0, {vocab}).",
                        location=f"tokenizer.ggml.{key}"))

    # TOK002 - vocabulary synchronized with embedding/output tensors. Skipped on
    # shards: the embedding/output tensor may live in a different shard file.
    split_count = meta.get("split.count")
    is_shard = isinstance(split_count, int) and split_count > 1
    if vocab is not None and not is_shard:
        for tname in ("token_embd.weight", "output.weight"):
            t = by_name.get(tname)
            if t and t.shape and vocab > max(t.shape):
                findings.append(finding(
                    "TOK002",
                    f"token count {vocab} exceeds every axis of {tname} {t.shape}; "
                    f"tokens would index outside the embedding table.",
                    location="tokenizer.ggml.tokens"))

    # TOK003 - parallel arrays length + token_type enum.
    if vocab is not None:
        for key in ("scores", "token_type"):
            arr = meta.get(f"tokenizer.ggml.{key}")
            if isinstance(arr, MetaArray) and arr.length != vocab:
                findings.append(finding(
                    "TOK003",
                    f"tokenizer.ggml.{key} has {arr.length} entries but there are "
                    f"{vocab} tokens.",
                    location=f"tokenizer.ggml.{key}"))
    ttype = meta.get("tokenizer.ggml.token_type")
    if isinstance(ttype, MetaArray):
        bad = sorted({v for v in ttype.preview
                      if isinstance(v, int) and v not in _TOKEN_TYPE_RANGE})
        if bad:
            findings.append(finding(
                "TOK003",
                f"token_type contains values outside the enum 1..6: {bad[:8]}.",
                location="tokenizer.ggml.token_type"))

    # TOK004 - implausibly large vocabulary token.
    tokens = meta.get("tokenizer.ggml.tokens")
    if isinstance(tokens, MetaArray) and tokens.max_elem_bytes > 4096:
        sev_note = " (exceeds INT32_MAX - signed-cast overflow risk)" \
            if tokens.max_elem_bytes > INT32_MAX else ""
        findings.append(finding(
            "TOK004",
            f"a vocabulary token is {tokens.max_elem_bytes} bytes{sev_note}.",
            location="tokenizer.ggml.tokens"))

    # TOK005 - add_bos/add_eos flags consistent with the ids they reference.
    for flag, id_key in (("add_bos_token", "bos_token_id"),
                         ("add_eos_token", "eos_token_id")):
        if meta.get(f"tokenizer.ggml.{flag}") is True:
            tid = meta.get(f"tokenizer.ggml.{id_key}")
            ok = (isinstance(tid, int) and not isinstance(tid, bool)
                  and tid not in _UNSET and tid >= 0
                  and (vocab is None or tid < vocab))
            if not ok:
                findings.append(finding(
                    "TOK005",
                    f"{flag} is set but {id_key} is missing or out of range.",
                    location=f"tokenizer.ggml.{flag}"))

    return findings
