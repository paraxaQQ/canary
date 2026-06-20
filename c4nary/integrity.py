"""Integrity & provenance: hashing, manifest compare, structural diff.

All comparisons are over **structure** — metadata, template text, and the
tensor map (names/shapes/dtypes). Raw weight bytes are never read or compared
(spec §5c). Everything here is deterministic and offline.
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any

from .parser import CHAT_TEMPLATE_KEY, GGUFModel, MetaArray
from .report import Finding
from .rules.registry import finding
from .template_ast import template_sha256

_CHUNK = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of the whole file (read-only)."""

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def model_template_sha256(model: GGUFModel) -> str | None:
    tpl = model.chat_template
    return template_sha256(tpl) if tpl is not None else None


def _serialize_value(value: Any) -> Any:
    if isinstance(value, MetaArray):
        return f"<array:{value.elem_type}[{value.length}]>"
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _serialize_metadata(model: GGUFModel) -> dict[str, Any]:
    """Scalar/summary view of metadata, excluding the chat template."""

    return {
        k: _serialize_value(model.metadata[k])
        for k in sorted(model.metadata)
        if k != CHAT_TEMPLATE_KEY
    }


def _serialize_tensors(model: GGUFModel) -> dict[str, dict[str, Any]]:
    return {
        t.name: {"shape": list(t.shape), "dtype": t.dtype}
        for t in sorted(model.tensors, key=lambda t: t.name)
    }


def build_manifest(model: GGUFModel, file_sha256: str) -> dict[str, Any]:
    """A known-good reference snapshot for later drift detection."""

    return {
        "file_sha256": file_sha256,
        "template_sha256": model_template_sha256(model),
        "metadata": _serialize_metadata(model),
        "tensors": _serialize_tensors(model),
    }


def compare_manifest(
    model: GGUFModel, file_sha256: str, manifest: dict[str, Any]
) -> list[Finding]:
    """Findings describing how ``model`` drifts from ``manifest``."""

    findings: list[Finding] = []

    expected_file = manifest.get("file_sha256")
    if expected_file is not None and expected_file != file_sha256:
        findings.append(finding(
            "INT001",
            f"File SHA-256 {file_sha256} != manifest {expected_file}.",
            location="file",
        ))

    expected_tpl = manifest.get("template_sha256")
    actual_tpl = model_template_sha256(model)
    if expected_tpl != actual_tpl:
        findings.append(finding(
            "INT002",
            f"Template SHA-256 {actual_tpl} != manifest {expected_tpl}.",
            location=CHAT_TEMPLATE_KEY,
        ))

    exp_meta = manifest.get("metadata", {}) or {}
    act_meta = _serialize_metadata(model)
    for key in sorted(set(exp_meta) | set(act_meta)):
        if key not in act_meta:
            findings.append(finding(
                "INT003", f"Metadata key {key!r} removed (was {exp_meta[key]!r}).",
                location=key))
        elif key not in exp_meta:
            findings.append(finding(
                "INT003", f"Metadata key {key!r} added (now {act_meta[key]!r}).",
                location=key))
        elif exp_meta[key] != act_meta[key]:
            findings.append(finding(
                "INT003",
                f"Metadata key {key!r} changed: {exp_meta[key]!r} -> {act_meta[key]!r}.",
                location=key))

    exp_tensors = manifest.get("tensors", {}) or {}
    act_tensors = _serialize_tensors(model)
    for name in sorted(set(exp_tensors) | set(act_tensors)):
        if name not in act_tensors:
            findings.append(finding(
                "INT004", f"Tensor {name!r} removed.", location=name))
        elif name not in exp_tensors:
            findings.append(finding(
                "INT004", f"Tensor {name!r} added.", location=name))
        elif exp_tensors[name] != act_tensors[name]:
            findings.append(finding(
                "INT004",
                f"Tensor {name!r} changed: {exp_tensors[name]} -> {act_tensors[name]}.",
                location=name))

    return findings


# --------------------------------------------------------------------------- #
# Two-file structural diff (`canary diff a.gguf b.gguf`)
# --------------------------------------------------------------------------- #

def diff_models(a: GGUFModel, b: GGUFModel) -> dict[str, Any]:
    """Structural diff of two models. Deterministic, structure-only."""

    meta_a = _serialize_metadata(a)
    meta_b = _serialize_metadata(b)
    metadata_diff = {"added": {}, "removed": {}, "changed": {}}
    for key in sorted(set(meta_a) | set(meta_b)):
        if key not in meta_a:
            metadata_diff["added"][key] = meta_b[key]
        elif key not in meta_b:
            metadata_diff["removed"][key] = meta_a[key]
        elif meta_a[key] != meta_b[key]:
            metadata_diff["changed"][key] = {"a": meta_a[key], "b": meta_b[key]}

    tpl_a = a.chat_template
    tpl_b = b.chat_template
    template_changed = template_sha256(tpl_a or "") != template_sha256(tpl_b or "") \
        if (tpl_a is not None or tpl_b is not None) else False
    template_diff = list(difflib.unified_diff(
        (tpl_a or "").splitlines(),
        (tpl_b or "").splitlines(),
        fromfile="a:chat_template",
        tofile="b:chat_template",
        lineterm="",
    )) if template_changed else []

    tens_a = _serialize_tensors(a)
    tens_b = _serialize_tensors(b)
    tensor_diff = {"added": [], "removed": [], "changed": []}
    for name in sorted(set(tens_a) | set(tens_b)):
        if name not in tens_a:
            tensor_diff["added"].append(name)
        elif name not in tens_b:
            tensor_diff["removed"].append(name)
        elif tens_a[name] != tens_b[name]:
            tensor_diff["changed"].append(
                {"name": name, "a": tens_a[name], "b": tens_b[name]})

    return {
        "metadata": metadata_diff,
        "template_changed": template_changed,
        "template_diff": template_diff,
        "tensors": tensor_diff,
    }


def diff_is_empty(diff: dict[str, Any]) -> bool:
    md = diff["metadata"]
    td = diff["tensors"]
    return (
        not md["added"] and not md["removed"] and not md["changed"]
        and not diff["template_changed"]
        and not td["added"] and not td["removed"] and not td["changed"]
    )
