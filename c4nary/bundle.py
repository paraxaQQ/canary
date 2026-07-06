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
    from .rules.config import analyze_config
    from .rules.template import analyze_card, analyze_repo_templates
    from .rules.tokenizer_json import analyze_special_tokens, analyze_tokenizer_json

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

    # tokenizer.json holds the vocab -> large cap; the normalizer/decoder we care about sit at
    # the top, but valid JSON needs the whole file.
    raw = read_text("tokenizer.json", 48 << 20)
    if raw:
        try:
            out.extend(analyze_tokenizer_json(json.loads(raw)))
        except ValueError:
            pass

    # special / added token files -- concealed (hidden/bidi) privileged tokens
    def _read_json(name: str):
        r = read_text(name, 4 << 20)
        try:
            return json.loads(r) if r else None
        except ValueError:
            return None

    out.extend(analyze_special_tokens(_read_json("special_tokens_map.json"),
                                      _read_json("added_tokens.json")))

    # repo template sources + divergence from the GGUF's embedded template
    tcj = None
    raw_tc = read_text("tokenizer_config.json", 4 << 20)
    if raw_tc:
        try:
            tcj = json.loads(raw_tc)
        except ValueError:
            tcj = None
    out.extend(analyze_repo_templates(model, tcj, read_text("chat_template.jinja")))

    readme = read_text("README.md")
    if readme:
        out.extend(analyze_card(readme))  # findings already tagged README.md
    return out
