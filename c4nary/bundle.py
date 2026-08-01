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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .coverage import Surface, SurfaceState

if TYPE_CHECKING:
    from .parser import GGUFModel
    from .report import Finding

# Decode-time config files, in precedence order, routed through the CFG rules.
BUNDLE_CONFIGS = ("generation_config.json", "config.json")
BUNDLE_FILES = (
    *BUNDLE_CONFIGS,
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "README.md",
)

# Materialize these vocab arrays for a bundle scan -- CFG002 reconstructs a refusal from the
# token surfaces, so the full vocab is needed (same keys the deep-tokenizer pass uses).
DEEP_TOK_KEYS = frozenset({"tokenizer.ggml.tokens", "tokenizer.ggml.token_type"})

@dataclass(frozen=True)
class ReadResult:
    text: str
    truncated: bool = False


Reader = Callable[..., "ReadResult | str | None"]


# Windows resolves these regardless of extension, so `CON.py` is a device, not a file.
_RESERVED_STEMS = frozenset({
    "con", "prn", "aux", "nul",
    *(f"com{n}" for n in range(1, 10)),
    *(f"lpt{n}" for n in range(1, 10)),
})


def _safe_bundle_name(name: str) -> str:
    stem = name[:-3] if name.endswith(".py") else ""
    if (
        not name
        or name.startswith(".")
        or ".." in name
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or not name.endswith(".py")
        # A module name is an identifier; anything else is not a name a loader could
        # import, only an attempt to reach a file. `C:foo.py` has no separator and clears
        # every check above, but os.path.join drops the base for a drive-qualified name.
        or not stem.isidentifier()
        or stem.lower() in _RESERVED_STEMS
    ):
        raise ValueError(f"unsafe Python bundle filename: {name!r}")
    return name


def bundle_findings(
    model: "GGUFModel",
    read_text: Reader,
) -> "tuple[list[Finding], list[Surface]]":
    """Run every repo-bundle rule against the files ``read_text`` can supply."""
    from .parser import MetaArray
    from .rules.config import analyze_auto_map, analyze_config
    from .rules.template import (
        analyze_card,
        analyze_repo_templates,
        embedded_template_unparseable,
    )
    from .rules.tokenizer_json import (
        _iter_token_strings,
        _post_processor_tokens,
        analyze_special_tokens,
        analyze_tokenizer_json,
    )

    out: list[Finding] = []
    surfaces: dict[str, Surface] = {}
    auto_map_configs: list[tuple[str, dict]] = []

    def _record(name: str, state: SurfaceState, reason: str) -> None:
        surfaces[name] = Surface(f"bundle.{name}", state, reason)

    def _read(
        name: str,
        max_bytes: int = 1 << 20,
        *,
        python_only: bool = False,
    ) -> ReadResult | None:
        raw = (
            read_text(name, max_bytes, python_only=True)
            if python_only
            else read_text(name, max_bytes)
        )
        if raw is None or raw == "":
            _record(name, "absent", f"{name} was absent or inaccessible.")
            return None
        result = raw if isinstance(raw, ReadResult) else ReadResult(raw)
        _record(
            name,
            "partial" if result.truncated else "examined",
            (
                f"{name} exceeded the reader cap and was truncated."
                if result.truncated
                else f"{name} was read and examined."
            ),
        )
        return result

    def _read_json(name: str, max_bytes: int = 4 << 20) -> object:
        result = _read(name, max_bytes)
        if result is None:
            return None
        try:
            return json.loads(result.text)
        except (ValueError, RecursionError):
            _record(
                name,
                "partial" if result.truncated else "unparseable",
                (
                    f"{name} was truncated before JSON parsing completed."
                    if result.truncated
                    else f"{name} was present but invalid JSON."
                ),
            )
            return None

    for name in BUNDLE_CONFIGS:
        cfg = _read_json(name)
        if not isinstance(cfg, dict):
            continue
        if name == "config.json":
            auto_map_configs.append((name, cfg))
        config_findings = analyze_config(
            model,
            cfg,
            include_auto_map=False,
            source=name,
        )
        if any(
            isinstance(value, str)
            and len(value) >= 16
            and embedded_template_unparseable(value)
            for value in cfg.values()
        ):
            _record(
                name,
                "partial",
                f"{name} was valid JSON, but an embedded Jinja value was unparseable.",
            )
        for result in config_findings:
            loc = f"{name}:{result.location}" if result.location else name
            out.append(dataclasses.replace(result, location=loc, artifact=name))

    tokenizer_data = _read_json("tokenizer.json", 48 << 20)
    if isinstance(tokenizer_data, dict):
        out.extend(analyze_tokenizer_json(tokenizer_data))

    special_tokens = _read_json("special_tokens_map.json")
    added_tokens = _read_json("added_tokens.json")
    tcj = _read_json("tokenizer_config.json")
    if isinstance(tcj, dict):
        auto_map_configs.append(("tokenizer_config.json", tcj))

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
    if isinstance(pcj, dict):
        auto_map_configs.append(("processor_config.json", pcj))
    preprocessor_config = _read_json("preprocessor_config.json")
    if isinstance(preprocessor_config, dict):
        auto_map_configs.append(("preprocessor_config.json", preprocessor_config))
    out.extend(analyze_auto_map(auto_map_configs))
    from .rules.python_code import analyze_auto_map_python

    def _read_python(name: str, max_bytes: int, *, python_only: bool) -> ReadResult | None:
        # Not `_read`: that stamps `bundle.<name>.py = examined` before any AST work, so
        # a file that then fails ast.parse gets two coverage rows contradicting each
        # other. `rmt.<name>` is the single authority for Python surfaces.
        raw = read_text(name, max_bytes, python_only=python_only)
        if raw is None or raw == "":
            return None
        return raw if isinstance(raw, ReadResult) else ReadResult(raw)

    rmt_findings, rmt_surfaces = analyze_auto_map_python(auto_map_configs, _read_python)
    out.extend(rmt_findings)

    extra_templates: tuple[tuple[str, str], ...] = ()
    if isinstance(pcj, dict):
        pc_template = pcj.get("chat_template")
        if isinstance(pc_template, str) and pc_template.strip():
            extra_templates = (("processor_config.json", pc_template),)
    chat_template_result = _read("chat_template.jinja")
    chat_template_jinja = (
        chat_template_result.text if chat_template_result is not None else None
    )
    repo_findings = analyze_repo_templates(
        model,
        tcj if isinstance(tcj, dict) else None,
        chat_template_jinja,
        extra_templates,
    )
    for result in repo_findings:
        if result.rule_id != "TPL000" or result.artifact is None:
            continue
        state = "unparseable" if result.artifact == "chat_template.jinja" else "partial"
        _record(
            result.artifact,
            state,
            f"{result.artifact} contained a chat template that failed Jinja parsing.",
        )
    out.extend(repo_findings)

    readme = _read("README.md")
    if readme:
        out.extend(analyze_card(readme.text))  # findings already tagged README.md
    return out, sorted(
        [*surfaces.values(), *rmt_surfaces],
        key=lambda surface: surface.id,
    )
