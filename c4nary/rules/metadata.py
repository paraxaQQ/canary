"""Metadata sanity rules.

Lightweight, deterministic checks over the parsed GGUF metadata KV store and
tensor map: embedded URLs/IPs, oversized fields, non-standard keys, and a
best-effort architecture/quantization consistency check. These are advisory —
metadata anomalies are review prompts, not proof of anything.
"""

from __future__ import annotations

import dataclasses
import re

from ..parser import CHAT_TEMPLATE_KEY, GGUFModel, MetaArray
from ..report import Finding
from ..template_ast import IP_RE, URL_RE
from .registry import finding
from .template import analyze_embedded_template, scan_injection_text

# GGUF metadata key namespaces considered standard.
_BASE_PREFIXES = ("general.", "tokenizer.", "quantize.", "split.", "gguf.")

MAX_METADATA_STRING = 8192

_BLK_RE = re.compile(r"^blk\.(\d+)\.")

# general.file_type enum -> the dtype its weights should predominantly be. Only
# the unambiguous "pure" types; mixed *_K types vary by tensor and are skipped.
_FILE_TYPE_EXPECTED = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 32: "BF16",
}


def analyze_metadata(model: GGUFModel) -> list[Finding]:
    findings: list[Finding] = []
    recognized = _recognized_prefixes(model)

    for key in sorted(model.metadata):
        value = model.metadata[key]
        if key == CHAT_TEMPLATE_KEY:
            continue  # owned by the template rules

        # Non-standard key namespace.
        if not key.startswith(recognized):
            findings.append(finding(
                "MET003",
                f"Metadata key {key!r} is outside standard GGUF namespaces.",
                location=key,
            ))

        if isinstance(value, str):
            findings.extend(_scalar_string_checks(key, value))
        elif isinstance(value, MetaArray) and value.elem_type == "string":
            findings.extend(_array_string_checks(key, value))

    findings.extend(_consistency_checks(model))
    findings.extend(_tensor_dtype_checks(model))
    findings.extend(_consistency_v2(model))
    return findings


def _consistency_v2(model: GGUFModel) -> list[Finding]:
    """Deterministic metadata-vs-tensor-map cross-checks (MET010-016).

    These compare declared scalars against tensor *shapes* (never weight data)
    and are the high-confidence, near-zero-false-positive backbone: a mismatch
    is a structural impossibility, not a heuristic.
    """

    findings: list[Finding] = []

    # MET016 - duplicate metadata keys (parser-differential evasion).
    for key in dict.fromkeys(model.duplicate_keys):
        findings.append(finding(
            "MET016", f"Metadata key {key!r} is defined more than once.", location=key))

    arch = model.architecture
    if not arch:
        return findings
    meta = model.metadata
    by_name = {t.name: t for t in model.tensors}
    # A multi-file shard holds only a subset of tensors, so any check that
    # compares the tensor map against declared scalars is unreliable on it.
    split_count = meta.get("split.count")
    is_shard = isinstance(split_count, int) and split_count > 1

    def num(suffix):
        v = meta.get(f"{arch}.{suffix}")
        return v if isinstance(v, int) else None

    # MET010 - blk.N layers must be contiguous from 0. A FAIL is reserved for
    # *gaps* (a removed/added middle layer) or indices beyond block_count; a file
    # with fewer contiguous layers than declared is almost always a deliberate
    # shard (e.g. "...Layers00-30.gguf") and is left to INT006/INT005, not failed.
    block_count = num("block_count")
    if block_count is not None and not is_shard:
        idxs = {int(m.group(1)) for t in model.tensors
                if (m := _BLK_RE.match(t.name))}
        if idxs:
            top = max(idxs)
            gaps = sorted(set(range(top + 1)) - idxs)
            if gaps:
                findings.append(finding(
                    "MET010",
                    f"blk layer indices are non-contiguous (missing {gaps[:8]} "
                    f"below layer {top}).",
                    location=f"{arch}.block_count"))

    # MET011 - embedding_length vs token_embd shape.
    emb = num("embedding_length")
    te = by_name.get("token_embd.weight")
    if emb is not None and te is not None and not is_shard and emb not in te.shape:
        findings.append(finding(
            "MET011",
            f"embedding_length={emb} matches neither axis of token_embd.weight "
            f"{te.shape}.",
            location=f"{arch}.embedding_length"))

    # MET012 - attention head divisibility. The embedding/head_count relation
    # only holds when head_dim is *implied* (= embedding_length / head_count);
    # modern archs (Qwen3, etc.) set an explicit head_dim via key_length, which
    # legitimately decouples it. The GQA invariant head_count % head_count_kv is
    # robust and always checked.
    hc = num("attention.head_count")
    hc_kv = num("attention.head_count_kv")
    explicit_head_dim = num("attention.key_length") is not None
    if emb is not None and hc and not explicit_head_dim and emb % hc != 0:
        findings.append(finding(
            "MET012",
            f"embedding_length={emb} is not divisible by head_count={hc} "
            f"(no explicit head_dim).",
            location=f"{arch}.attention.head_count"))
    if hc and hc_kv and hc % hc_kv != 0:
        findings.append(finding(
            "MET012",
            f"head_count={hc} is not divisible by head_count_kv={hc_kv}.",
            location=f"{arch}.attention.head_count_kv"))

    # MET013 - feed_forward_length vs a clean 2-D ffn tensor.
    ffl = num("feed_forward_length")
    if ffl is not None and not is_shard:
        for t in model.tensors:
            base = t.name.split(".")[-2] if "." in t.name else ""
            if base in ("ffn_up", "ffn_gate", "ffn_down") and len(t.shape) == 2:
                if ffl not in t.shape:
                    findings.append(finding(
                        "MET013",
                        f"feed_forward_length={ffl} matches neither axis of "
                        f"{t.name} {t.shape}.",
                        location=f"{arch}.feed_forward_length"))
                break

    if not is_shard:
        findings.extend(_quant_label_check(model))  # dominant-dtype needs full map
    findings.extend(_rope_checks(model, arch))
    return findings


def _quant_label_check(model: GGUFModel) -> list[Finding]:
    file_type = model.metadata.get("general.file_type")
    expected = _FILE_TYPE_EXPECTED.get(file_type) if isinstance(file_type, int) else None
    if expected is None:
        return []
    counts: dict[str, int] = {}
    for t in model.tensors:
        if not t.name.endswith(".weight") or "norm" in t.name:
            continue
        if t.name in ("token_embd.weight", "output.weight"):
            continue
        counts[t.dtype] = counts.get(t.dtype, 0) + 1
    if not counts:
        return []
    dominant = max(counts, key=lambda k: counts[k])
    if dominant != expected:
        return [finding(
            "MET014",
            f"general.file_type implies {expected} weights but the dominant "
            f"weight dtype is {dominant}.",
            location="general.file_type")]
    return []


def _rope_checks(model: GGUFModel, arch: str) -> list[Finding]:
    out: list[Finding] = []
    meta = model.metadata
    cl = meta.get(f"{arch}.context_length")
    ocl = meta.get(f"{arch}.rope.scaling.original_context_length")
    if isinstance(cl, int) and isinstance(ocl, int) and cl < ocl:
        out.append(finding(
            "MET015",
            f"context_length={cl} is below original_context_length={ocl}.",
            location=f"{arch}.context_length"))
    fb = meta.get(f"{arch}.rope.freq_base")
    if isinstance(fb, (int, float)) and not isinstance(fb, bool) and fb <= 0:
        out.append(finding(
            "MET015", f"rope.freq_base={fb} is not positive.",
            location=f"{arch}.rope.freq_base"))
    return out


def _recognized_prefixes(model: GGUFModel) -> tuple[str, ...]:
    prefixes = list(_BASE_PREFIXES)
    arch = model.architecture
    if arch:
        prefixes.append(f"{arch}.")
    return tuple(prefixes)


_PROVENANCE_KEY_TERMS = ("url", "repo", "source", "homepage", "website",
                         "huggingface", "doi")


def _is_provenance_key(key: str) -> bool:
    k = key.lower()
    return any(term in k for term in _PROVENANCE_KEY_TERMS)


def _scalar_string_checks(key: str, value: str) -> list[Finding]:
    findings: list[Finding] = []

    # Provenance keys (general.source.url, *.repo_url, ...) are *expected* to hold
    # a URL; flagging those is pure noise (it fired on most real models). Still
    # flag IPs anywhere, and URLs in any non-provenance field.
    urls = URL_RE.findall(value)
    ips = IP_RE.findall(value)
    flag_urls = urls and not _is_provenance_key(key)
    if flag_urls or ips:
        findings.append(finding(
            "MET001",
            f"Value contains {len(urls)} URL(s) / {len(ips)} IP(s) "
            f"(e.g. {(urls + ips)[0]!r}).",
            location=key,
        ))

    if len(value) > MAX_METADATA_STRING:
        findings.append(finding(
            "MET002",
            f"String value is {len(value)} chars (> {MAX_METADATA_STRING}).",
            location=key,
        ))
    else:
        preview = value if len(value) <= 120 else value[:117] + "..."
        findings.append(finding(
            "MET004",
            f"{key} = {preview!r}",
            location=key,
        ))

    # MET020 / MET021 - hidden codepoints or instruction text in metadata strings.
    concealed, hits = scan_injection_text(value)
    if concealed:
        findings.append(finding(
            "MET020",
            f"Value contains invisible / zero-width / bidi codepoints "
            f"({', '.join(f'U+{cp:04X}' for cp in concealed)}) that hide text; metadata "
            f"should be plain printable.",
            location=key,
        ))
    if hits:
        findings.append(finding(
            "MET021",
            f"Value contains injection-idiom instruction text (e.g. {hits[0]!r}) not tied "
            f"to the conversation - a hidden instruction stashed in metadata.",
            location=key,
        ))
    # Also route Jinja-carrying values through the AST rules (a second template in metadata).
    for f in analyze_embedded_template(value):
        findings.append(dataclasses.replace(
            f, location=f"{key}:{f.location}" if f.location else key))

    return findings


def _array_string_checks(key: str, value: MetaArray) -> list[Finding]:
    findings: list[Finding] = []
    for item in value.preview:
        if not isinstance(item, str):
            continue
        if URL_RE.search(item) or IP_RE.search(item):
            findings.append(finding(
                "MET001",
                f"String array element contains a URL/IP (e.g. {item!r}).",
                location=key,
            ))
            break  # one finding per key is enough

    if key == "tokenizer.ggml.merges":
        return findings

    # MET021 - a vocab token whose string IS an instruction (rare, but possible).
    for item in value.preview:
        if not isinstance(item, str) or len(item) < 16:
            continue
        _, hits = scan_injection_text(item)
        if hits:
            findings.append(finding(
                "MET021",
                f"String array element in {key} contains injection-idiom text "
                f"(e.g. {hits[0]!r}): {item[:60]!r}... -- a vocab token carrying an "
                f"instruction (if referenced by bos_token_id / a post_processor) is a "
                f"decode-time injection surface. Manual review.",
                location=key,
            ))
            break  # one finding per key is enough
    return findings


def _consistency_checks(model: GGUFModel) -> list[Finding]:
    arch = model.architecture
    if not arch:
        return []
    if not model.tensors:
        return []
    has_arch_keys = any(k.startswith(f"{arch}.") for k in model.metadata)
    if not has_arch_keys:
        return [finding(
            "MET005",
            f"Architecture {arch!r} is declared but no '{arch}.*' configuration "
            f"keys are present, which is unusual for a genuine model.",
            location="general.architecture",
        )]
    return []


def _tensor_dtype_checks(model: GGUFModel) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[str] = set()
    for t in model.tensors:
        if t.dtype.startswith("GGML_TYPE_") and t.dtype not in seen:
            seen.add(t.dtype)
            findings.append(finding(
                "MET006",
                f"Tensor {t.name!r} uses unrecognized dtype {t.dtype}.",
                location=t.name,
            ))
    return findings
