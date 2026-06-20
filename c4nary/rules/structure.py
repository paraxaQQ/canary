"""Structural / parser-exploitation rules (STR001-008) and provenance (INT005-006).

These flag GGUF files crafted to crash or exploit naive C loaders rather than to
manipulate model behavior: element-count/size overflows that wrap a 64-bit
allocation, tensor data offsets that point outside the file, overlapping tensor
regions, implausible alignment, non-block-divisible dimensions, unknown types,
and abusive tensor names. All checks use the parsed header and the file size;
no tensor data is read. Sizes are computed in Python bignums so a value that
would wrap in C is visible here.
"""

from __future__ import annotations

from ..parser import (
    INT63_MAX,
    GGUFModel,
    ggml_block_size,
    ggml_nbytes,
    ne_product,
)
from ..report import Finding
from .registry import finding

_MAX_NAME_BYTES = 64
_MAX_ALIGNMENT = 1 << 20


def analyze_structure(model: GGUFModel) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(_alignment_check(model))
    findings.extend(_tensor_checks(model))
    findings.extend(_overlap_check(model))
    findings.extend(_provenance_checks(model))
    return findings


def _alignment_check(model: GGUFModel) -> list[Finding]:
    raw = model.metadata.get("general.alignment")
    if raw is None:
        return []
    bad = (not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0
           or (raw & (raw - 1)) != 0 or raw < 4 or raw > _MAX_ALIGNMENT)
    if bad:
        return [finding(
            "STR005",
            f"general.alignment={raw!r} is not a sane power of two in [4, {_MAX_ALIGNMENT}].",
            location="general.alignment")]
    return []


def _tensor_checks(model: GGUFModel) -> list[Finding]:
    findings: list[Finding] = []
    fs = model.file_size
    ds = model.data_start

    for t in model.tensors:
        loc = t.name or "<tensor>"

        # STR001 - element-count / byte-size overflow (computed in bignums).
        ne = ne_product(t.shape)
        nb = ggml_nbytes(t.type_id, t.shape)
        if ne > INT63_MAX or (nb is not None and nb > INT63_MAX):
            findings.append(finding(
                "STR001",
                f"tensor {t.name!r} element count/size overflows int64 "
                f"(ne={ne}, shape={t.shape}).",
                location=loc))

        # STR003 - data offset (plus computed size) within the file.
        start = ds + t.offset
        if start > fs:
            findings.append(finding(
                "STR003",
                f"tensor {t.name!r} data starts at {start} which is past EOF "
                f"(file_size={fs}).",
                location=loc))
        elif nb is not None and start + nb > fs:
            findings.append(finding(
                "STR003",
                f"tensor {t.name!r} data [{start}, {start + nb}) extends past EOF "
                f"(file_size={fs}).",
                location=loc))

        # STR006 - zero / non-block-divisible dimension.
        if t.shape and any(d == 0 for d in t.shape):
            findings.append(finding(
                "STR006", f"tensor {t.name!r} has a zero dimension {t.shape}.",
                location=loc))
        else:
            bs = ggml_block_size(t.type_id)
            if bs and bs > 1 and t.shape and t.shape[0] % bs != 0:
                findings.append(finding(
                    "STR006",
                    f"tensor {t.name!r} innermost dim {t.shape[0]} is not a "
                    f"multiple of the {t.dtype} block size {bs}.",
                    location=loc))

        # STR007 - unknown ggml type (byte size unknown, untested loader path).
        if t.dtype.startswith("GGML_TYPE_"):
            findings.append(finding(
                "STR007",
                f"tensor {t.name!r} uses unknown ggml type id {t.type_id}.",
                location=loc))

        # STR008 - abusive tensor name.
        reasons = _name_problems(t.name)
        if reasons:
            findings.append(finding(
                "STR008",
                f"tensor name {t.name!r} is suspicious: {', '.join(reasons)}.",
                location=loc))

    return findings


def _name_problems(name: str) -> list[str]:
    reasons = []
    if len(name.encode("utf-8", errors="replace")) > _MAX_NAME_BYTES:
        reasons.append(f">{_MAX_NAME_BYTES} bytes")
    if any(ord(c) < 32 for c in name):
        reasons.append("control/NUL bytes")
    if "�" in name:
        reasons.append("invalid UTF-8")
    if "/" in name or "\\" in name or ".." in name:
        reasons.append("path-traversal characters")
    return reasons


def _overlap_check(model: GGUFModel) -> list[Finding]:
    ds = model.data_start
    intervals = []
    for t in model.tensors:
        nb = ggml_nbytes(t.type_id, t.shape)
        if nb is None or nb == 0:
            continue
        start = ds + t.offset
        intervals.append((start, start + nb, t.name))
    intervals.sort()
    findings: list[Finding] = []
    for (s1, e1, n1), (s2, e2, n2) in zip(intervals, intervals[1:]):
        if s2 < e1:
            findings.append(finding(
                "STR004",
                f"tensors {n1!r} and {n2!r} have overlapping data regions "
                f"([{s1},{e1}) vs [{s2},{e2})).",
                location=n2))
    return findings


def _provenance_checks(model: GGUFModel) -> list[Finding]:
    findings: list[Finding] = []

    # INT006 - sharded model: a clean verdict only covers this shard.
    split_count = model.metadata.get("split.count")
    if isinstance(split_count, int) and split_count > 1:
        findings.append(finding(
            "INT006",
            f"this file is shard {model.metadata.get('split.no', '?')} of "
            f"{split_count}; the verdict covers only this shard.",
            location="split.count"))

    # INT005 - injected / anomalous tensors.
    seen: set[str] = set()
    arch = model.architecture
    block_count = model.metadata.get(f"{arch}.block_count") if arch else None
    for t in model.tensors:
        if t.name in seen:
            findings.append(finding(
                "INT005", f"duplicate tensor name {t.name!r}.", location=t.name))
        seen.add(t.name)
        lname = t.name.lower()
        if "lora" in lname or "adapter" in lname:
            findings.append(finding(
                "INT005",
                f"tensor {t.name!r} matches adapter/LoRA naming (possible graft).",
                location=t.name))
        if isinstance(block_count, int) and t.name.startswith("blk."):
            try:
                idx = int(t.name.split(".")[1])
            except (IndexError, ValueError):
                idx = None
            if idx is not None and idx >= block_count:
                findings.append(finding(
                    "INT005",
                    f"tensor {t.name!r} is in layer {idx} beyond declared "
                    f"block_count={block_count}.",
                    location=t.name))
    return findings
