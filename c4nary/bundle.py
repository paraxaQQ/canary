"""Shared opt-in repo-bundle audit.

Given a parsed model and a text-reader callable, fetch/read the repo's decode-time config,
tokenizer.json, special-token files, divergent template sources, and model card, and route
them through the CFG / NRM / DOC / TPL030 rules. Both the CLI (`--bundle`) and the MCP `scan`
tool call this, so they run the identical audit -- the reader is the only thing that differs
(HTTP range-fetch for a remote repo vs a local sibling file).
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .parser import GGUFModel
    from .report import Finding

# Decode-time config files, in precedence order, routed through the CFG rules.
BUNDLE_CONFIGS = ("generation_config.json", "config.json")

# Materialize these vocab arrays for a bundle scan -- CFG002 reconstructs a refusal from the
# token surfaces, so the full vocab is needed (same keys the deep-tokenizer pass uses).
DEEP_TOK_KEYS = frozenset({"tokenizer.ggml.tokens", "tokenizer.ggml.token_type"})

Reader = Callable[..., "str | None"]  # read_text(name, max_bytes=...) -> text or None


def bundle_findings(model: "GGUFModel", read_text: Reader) -> "list[Finding]":
    """Run every repo-bundle rule against the files ``read_text`` can supply."""
    from .parser import MetaArray
    from .rules.config import analyze_config
    from .rules.template import analyze_card, analyze_repo_templates
    from .rules.tokenizer_json import (
        _iter_token_strings,
        _post_processor_tokens,
        analyze_special_tokens,
        analyze_tokenizer_json,
    )

    out: list = []

    for name in BUNDLE_CONFIGS:
        raw = read_text(name)
        if not raw:
            continue
        try:
            cfg = json.loads(raw)
        except ValueError:
            continue
        for f in analyze_config(model, cfg):
            loc = f"{name}:{f.location}" if f.location else name
            out.append(dataclasses.replace(f, location=loc))

    def _read_json(name: str, max_bytes: int = 4 << 20):
        raw = read_text(name, max_bytes)
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            return None

    tokenizer_data = _read_json("tokenizer.json", 48 << 20)
    if isinstance(tokenizer_data, dict):
        out.extend(analyze_tokenizer_json(tokenizer_data))

    special_tokens = _read_json("special_tokens_map.json")
    added_tokens = _read_json("added_tokens.json")
    tcj = _read_json("tokenizer_config.json")

    reachable = _post_processor_tokens(tokenizer_data)
    for token_name in ("bos_token", "eos_token"):
        add_name = f"add_{token_name}"
        enabled = ((isinstance(tcj, dict) and tcj.get(add_name) is True)
                   or model.metadata.get(f"tokenizer.ggml.{add_name}") is True)
        if not enabled:
            continue
        value = tcj.get(token_name) if isinstance(tcj, dict) else None
        if value is None and isinstance(special_tokens, dict):
            value = special_tokens.get(token_name)
        reachable.update(_iter_token_strings(value))

        token_id = model.metadata.get(f"tokenizer.ggml.{token_name}_id")
        tokens = model.metadata.get("tokenizer.ggml.tokens")
        if (isinstance(token_id, int) and not isinstance(token_id, bool)
                and isinstance(tokens, MetaArray) and not tokens.truncated
                and 0 <= token_id < len(tokens.preview)
                and isinstance(tokens.preview[token_id], str)):
            reachable.add(tokens.preview[token_id])

    tokenizer_added = {
        entry.get("content")
        for entry in tokenizer_data.get("added_tokens", [])
        if isinstance(tokenizer_data, dict) and isinstance(entry, dict)
        and entry.get("special") is True and isinstance(entry.get("content"), str)
        and entry.get("content") in reachable
    } if isinstance(tokenizer_data, dict) else set()
    out.extend(analyze_special_tokens(
        special_tokens, added_tokens, reachable=reachable - tokenizer_added))

    # repo template sources + divergence from the GGUF's embedded template.
    # tokenizer_config.json is the primary source; processor_config.json carries a
    # chat_template for multimodal models (LLaVA, Qwen-VL, etc.) -- a divergent template
    # parked there is invisible to a tokenizer_config-only audit.
    pcj = _read_json("processor_config.json")
    extra_templates: tuple[tuple[str, str], ...] = ()
    if isinstance(pcj, dict):
        pc_template = pcj.get("chat_template")
        if isinstance(pc_template, str) and pc_template.strip():
            extra_templates = (("processor_config.json", pc_template),)
    out.extend(analyze_repo_templates(
        model, tcj, read_text("chat_template.jinja"), extra_templates))

    readme = read_text("README.md")
    if readme:
        out.extend(analyze_card(readme))  # findings already tagged README.md
    return out
