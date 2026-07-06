"""Static analysis rules for the embedded Jinja2 chat template.

The detector walks the parsed AST (never rendering it) and flags the SSTI /
sandbox-escape primitives behind CVE-2024-34359 and its variants, plus a few
heuristic behavioral red flags. A template whose normalized hash matches a
vetted reference short-circuits to a single INFO finding.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from typing import TYPE_CHECKING

import jinja2
from jinja2 import nodes

from ..report import Finding

if TYPE_CHECKING:
    from ..parser import GGUFModel
from ..template_ast import (
    IP_RE,
    URL_RE,
    ast_depth,
    iter_nodes,
    load_known_templates,
    node_location,
    parse_template,
    reconstruct_const_string,
    template_sha256,
)
from .registry import finding

# Python dunders that form SSTI sandbox-escape chains. __class__ is deliberately
# EXCLUDED: on its own (obj.__class__ / obj.__class__.__name__) it is inert type
# introspection that real tool-calling templates use to type-check arguments -- a FAIL
# there false-positives on legitimate models (Darkhn Gemma-Animus type-checks tool args;
# an SSTI-probe repo's messages[0].content.__class__ escapes nothing). A GENUINE escape via
# __class__ always continues into one of the escape dunders below (kept flagged) or starts
# from a bare literal (''.__class__), which _ast_checks catches as a pivot.
DUNDERS = frozenset({
    "__base__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__builtins__", "__init__", "__import__", "__dict__",
    "__getattribute__", "__getitem__", "__reduce__", "__reduce_ex__",
    "__code__", "__func__", "__closure__", "__self__", "mro",
})

# Jinja2 BUILT-IN globals used to reach arbitrary Python. These exist in every
# Jinja environment and do not occur in legitimate chat templates -> FAIL.
# (request/config are Flask globals, NOT Jinja built-ins: absent from
# apply_chat_template's sandbox and colliding with benign 'config.x' variables,
# so excluded. A genuine config.__class__... exploit is still caught by TPL001.)
GADGETS = frozenset({"lipsum", "cycler", "joiner"})

# Exec / introspection primitives: dangerous in ANY position (Name, attribute, or
# subscript key) and never a plausible benign field name. ('system'/'open'/'input'
# are excluded -- common chat words; their exec form is reached via 'os'/builtins,
# which are flagged, and the dunder path is caught by TPL001.)
DANGEROUS_NAMES = frozenset({
    "popen", "eval", "exec", "getattr", "setattr", "__import__", "import",
    "compile", "builtins", "globals", "locals", "vars", "breakpoint",
    "importlib", "execfile", "memoryview", "pty",
})
# Module names: the SSTI target, but also plausible benign FIELD names
# (terminal_state.os, device.platform). Flagged only as a bare Name or a subscript
# key (__globals__['os']) -- NOT as a plain attribute, where it is almost always a
# benign field. Any genuine escape reaches them through a dunder (TPL001).
# ('sys' is excluded entirely: commonly a variable for the system message.)
DANGEROUS_MODULES = frozenset({
    "os", "subprocess", "socket", "platform", "commands",
})

# Lowercased tokens that, if reconstructed from split string literals, signal a
# hidden payload.
# Only tokens that never occur benignly in a concatenated template string. Common
# words (system, eval, exec) are deliberately excluded: a template assembling the
# "system" role header from constants is not SSTI, and direct os.system / eval
# references are already caught as AST names by TPL003. (Calibrated on a real
# corpus where 'system' role-header concatenation was the sole TPL005 FAIL.)
RECON_TOKENS = (
    "popen", "subprocess", "importlib", "__globals__", "__builtins__",
    "__class__", "__subclasses__", "__import__", "__mro__", "__bases__",
    "__base__", "__init__", "__dict__", "__reduce__", "__getattribute__",
    "__name__", "getattr", "setattr",
)

# Introspection dunders that essentially never appear in a benign string literal
# (unlike __class__/__init__/__name__, which show up in code-example templates).
# A Const string containing one of these is a laundered sandbox-escape key.
INTROSPECTION_DUNDERS = (
    "__globals__", "__builtins__", "__subclasses__", "__mro__", "__bases__",
    "__base__", "__reduce__", "__getattribute__",
)

# Literals that, when used as a branch/comparison key, may indicate a
# behavioral trigger (flagged WARN, never as proof of malice).
SUSPICIOUS_LITERALS = frozenset({
    "password", "passwd", "passphrase", "login", "secret", "seed", "apikey",
    "api_key", "credential", "credentials", "token", "private_key", "privatekey",
})

MAX_TEMPLATE_BYTES = 50_000
MAX_AST_DEPTH = 25

# --- Behavioral / silent-hijack detection -------------------------------- #
# Bidirectional-override controls (Trojan Source): rendered order != token order.
# Only the embedding/override (LRE/RLE/PDF/LRO/RLO) and isolate (LRI/RLI/FSI/PDI) chars
# can reorder STRONG text (letters) -- the actual Trojan-Source primitive. The directional
# MARKS LRM/RLM/ALM are deliberately EXCLUDED (see _BIDI_MARKS_ALLOWED): zero-width,
# strongly-typed hints that only affect neutral-character placement, Unicode-recommended,
# and ubiquitous + editor-auto-inserted in any RTL (Arabic/Hebrew/Persian/Urdu) prose --
# flagging them as FAIL false-positives a whole class of legitimate localized models.
BIDI_CODEPOINTS = frozenset({
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
    0x2066, 0x2067, 0x2068, 0x2069,
})
# Zero-width / format controls that conceal text. Excludes the joiners
# U+200C/U+200D (see _JOINERS_ALLOWED) which are required in many scripts.
EXPLICIT_INVISIBLE = frozenset({
    0x200B, 0x2060, 0xFEFF, 0x00AD, 0x180E, 0x2028, 0x2029,
})
# ZWNJ / ZWJ are essential in Persian/Arabic/Indic text and emoji sequences --
# far too common in legitimate templates to treat as concealment.
_JOINERS_ALLOWED = frozenset({0x200C, 0x200D})
# LRM / RLM / ALM: legitimate RTL direction marks (same rationale as the joiners) --
# not a Trojan-Source override (they can't reorder strong text) and not concealment.
_BIDI_MARKS_ALLOWED = frozenset({0x200E, 0x200F, 0x061C})

# A chat template legitimately branches only on conversation STRUCTURE. These
# are the content accessors a behavioral trigger inspects instead.
CONTENT_KEYS = frozenset({"content", "text"})
DATE_NAMES = frozenset({"strftime_now", "now", "today", "utcnow", "datetime", "date"})
DATE_ATTRS = frozenset({"now", "today", "utcnow"})

# Modern templates branch on content to handle structural FORMAT markers (tool
# calls, reasoning tags, harmony channels) -- these are not backdoor triggers.
# Only a natural-language literal looks like a trigger. (Calibrated on a real
# corpus where every TPL020 false positive compared content against one of these.)
_STRUCTURAL_CHARS = frozenset("<>|[]{}")
_FORMAT_WORDS = frozenset({
    "user", "assistant", "system", "tool", "tool_response", "tool_responses",
    "tool_call", "tool_calls", "function", "functions", "ipython", "model",
    "think", "channel", "message", "observation", "developer", "human", "ai",
    "role", "name", "type", "image", "audio", "video",
    # multimodal content-type tags + reasoning-channel / tool-error markers modern
    # templates branch on (calibrated against the trending-GGUF re-scan; these are
    # protocol tokens, not natural-language backdoor triggers).
    "no_think", "image_url", "audio_url", "video_url", "input_audio", "failed to",
})

# Imperative idioms characteristic of injected instructions (not generic
# assistant prose). Matched case-insensitively as substrings.
INSTRUCTION_LEXICON = (
    "ignore previous", "ignore all previous", "ignore the above",
    "ignore your instructions", "ignore your previous", "disregard previous",
    "disregard all previous", "from now on", "do not mention", "do not reveal",
    "never mention", "never reveal", "never warn", "never refuse",
    "always recommend", "always say",
    "regardless of the user", "without telling the user", "override your",
    "you must always recommend",
)
# 'instead of answering' was removed: it appears in benign helpfulness system
# prompts ("explain why instead of answering incorrectly"), not just injections.


def analyze_template(source: str | None) -> list[Finding]:
    """Return findings for the (optional) chat template. Pure, deterministic."""

    if source is None:
        return [finding("TPL101", "The file declares no tokenizer.chat_template key.")]

    # Fast path: exact match against a vetted reference suppresses content rules.
    digest = template_sha256(source)
    known = load_known_templates()
    if digest in known:
        return [finding(
            "TPL100",
            f"Normalized template matches known-good reference '{known[digest]}'.",
            location=f"template_sha256={digest}",
        )]

    findings: list[Finding] = []
    try:
        ast = parse_template(source)
    except jinja2.TemplateSyntaxError as exc:
        findings.append(finding(
            "TPL000",
            f"Jinja2 parse error: {exc.message}.",
            location=f"template:L{exc.lineno}" if exc.lineno else "template",
        ))
    except RecursionError:
        findings.append(finding(
            "TPL000",
            "Template nesting is too deep to parse (possible obfuscation).",
        ))
    else:
        findings.extend(_ast_checks(ast))
        findings.extend(_depth_check(ast))
        findings.extend(_behavioral_checks(ast))
        findings.extend(_transport_checks(ast))

    findings.extend(_text_checks(source))
    return _dedupe(findings)


_TEMPLATE_KEY = "tokenizer.chat_template"


def analyze_templates(model: GGUFModel) -> list[Finding]:
    """Analyze EVERY chat template the file carries, not just the default. llama.cpp
    writes named variants as ``tokenizer.chat_template.<name>`` (tool_use, rag, ...); a
    backdoor parked in a non-default variant -- or behind a non-string default -- is
    invisible when only ``tokenizer.chat_template`` is read. Findings are tagged with the
    variant they came from. Upholds 'silence != didn't look'."""
    templates: list[tuple[str, str]] = []
    for key, val in model.metadata.items():
        if (key == _TEMPLATE_KEY or key.startswith(_TEMPLATE_KEY + ".")) \
                and isinstance(val, str):
            variant = "default" if key == _TEMPLATE_KEY else key[len(_TEMPLATE_KEY) + 1:]
            templates.append((variant, val))
    if not templates:
        return [finding("TPL101", "The file declares no tokenizer.chat_template key.")]
    templates.sort()
    multi = len(templates) > 1
    out: list[Finding] = []
    for variant, source in templates:
        for f in analyze_template(source):
            if multi:
                tag = f"chat_template[{variant}]"
                f = dataclasses.replace(
                    f, location=f"{tag} {f.location}" if f.location else tag)
            out.append(f)
    return out


def _template_key(t: str) -> str:
    """Whitespace-insensitive comparison key: a reformatted-only template is not a
    meaningful divergence, so collapse runs of whitespace before comparing."""
    return re.sub(r"\s+", " ", t.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _gguf_templates(model: GGUFModel) -> list[str]:
    return [v for k, v in model.metadata.items()
            if (k == _TEMPLATE_KEY or k.startswith(_TEMPLATE_KEY + "."))
            and isinstance(v, str) and v.strip()]


def _repo_templates(tokenizer_config: dict | None,
                    chat_template_jinja: str | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(tokenizer_config, dict):
        ct = tokenizer_config.get("chat_template")
        if isinstance(ct, str) and ct.strip():
            out.append(("tokenizer_config.json", ct))
        elif isinstance(ct, list):                       # [{name, template}, ...]
            for item in ct:
                if isinstance(item, dict) and isinstance(item.get("template"), str):
                    out.append((f"tokenizer_config.json[{item.get('name', '?')}]",
                                item["template"]))
    if isinstance(chat_template_jinja, str) and chat_template_jinja.strip():
        out.append(("chat_template.jinja", chat_template_jinja))
    return out


def analyze_repo_templates(model: GGUFModel, tokenizer_config: dict | None,
                           chat_template_jinja: str | None) -> list[Finding]:
    """Scan the repo's template SOURCES and flag divergence from the GGUF's embedded
    template. Transformers reads ``tokenizer_config.json`` / ``chat_template.jinja``; a GGUF
    loader reads the embedded template -- a divergent repo template is a place to hide a
    backdoor from a GGUF-only audit. A divergent template is also scanned for backdoors."""
    out: list[Finding] = []
    gguf_keys = {_template_key(t) for t in _gguf_templates(model)}
    for source, tmpl in _repo_templates(tokenizer_config, chat_template_jinja):
        if _template_key(tmpl) in gguf_keys:
            continue                                     # identical -> already scanned
        if gguf_keys:                                    # GGUF has a template; this differs
            out.append(finding(
                "TPL030",
                f"The repo chat template in {source} differs from the GGUF's embedded "
                f"template: a transformers loader (reading {source}) sees a different "
                f"template than a GGUF loader. A place to hide a backdoor from a GGUF-only "
                f"audit; the divergent template was scanned - review the diff.",
                location=source))
        for f in analyze_template(tmpl):                 # scan the divergent / extra template
            loc = f"{source}:{f.location}" if f.location else source
            out.append(dataclasses.replace(f, location=loc))
    return out


def _ast_checks(ast: nodes.Template) -> list[Finding]:
    findings: list[Finding] = []

    for node in iter_nodes(ast):
        loc = node_location(node)

        # Attribute / subscript access to dunders, gadgets, dangerous names.
        if isinstance(node, nodes.Getattr):
            _classify_name(findings, node.attr, loc, context="attribute")
            # ''.__class__ / (0).__class__ -- accessing __class__ on a bare literal is the
            # canonical SSTI pivot start; a variable's .__class__ is benign type introspection.
            if _fold(node.attr) == "__class__" and _is_const_root(node.node):
                findings.append(finding(
                    "TPL001",
                    f"Accesses '__class__' on the literal {_pivot_repr(node.node)} - a "
                    f"Jinja2 SSTI sandbox-escape pivot.",
                    location=loc))
        elif isinstance(node, nodes.Getitem):
            key = _const_str(node.arg)
            if key is not None:
                _classify_name(findings, key, loc, context="subscript key")
            # Subscripting a bare literal with a (possibly computed) key is the
            # classic SSTI pivot ''[...] / ()[...] / (0)[...] -- never benign, and
            # it defeats key-must-be-const detection.
            if _is_literal_pivot(node.node):
                findings.append(finding(
                    "TPL001",
                    f"Subscripts the literal {_pivot_repr(node.node)} - a Jinja2 "
                    f"SSTI sandbox-escape pivot (often with a computed key).",
                    location=loc))
        elif isinstance(node, nodes.Name):
            _classify_name(findings, node.name, loc, context="name")
        elif isinstance(node, nodes.Const) and isinstance(node.value, str):
            # Introspection dunder hidden inside a string literal (var-keyed
            # chains, str.format spec injection).
            low = _fold(node.value).lower()
            for tok in INTROSPECTION_DUNDERS:
                if tok in low:
                    findings.append(finding(
                        "TPL001",
                        f"String literal contains introspection dunder {tok!r} "
                        f"(laundered sandbox-escape key).",
                        location=loc))
                    break

        # Abusable filters.
        if isinstance(node, nodes.Filter) and node.name == "attr":
            findings.append(finding(
                "TPL004",
                "Uses the |attr filter, which bypasses Jinja2's attribute sandbox.",
                location=loc,
            ))
        if isinstance(node, nodes.Filter) and node.name == "map":
            if any(_const_str(a) == "attr" for a in node.args):
                findings.append(finding(
                    "TPL004",
                    "Uses map('attr'), an attribute-sandbox bypass primitive.",
                    location=loc,
                ))
            # map(attribute='X') is a benign field extractor in function-calling
            # templates (map(attribute='function'/'role')); only flag when X is a
            # dunder / dangerous name, via the classifier.
            kw_attr = next((_const_str(kw.value) for kw in node.kwargs
                            if kw.key == "attribute"), None)
            if kw_attr is not None:
                _classify_name(findings, kw_attr, loc, context="map attribute")
                # map(attribute='__class__') has no benign use (real templates map
                # 'role'/'function') and is the pivot of the |map(attribute=...) escape
                # chain -- flag it even though a plain foo.__class__ type-check is allowed.
                if _fold(kw_attr) == "__class__":
                    findings.append(finding(
                        "TPL001",
                        "Uses map(attribute='__class__') - extracts each element's type "
                        "object, the pivot of a map-based SSTI escape chain.",
                        location=loc))

        # Split-string reconstruction of dangerous tokens.
        if isinstance(node, (nodes.Concat, nodes.Add)):
            assembled = reconstruct_const_string(node)
            _check_reconstructed(findings, assembled, loc)
        if isinstance(node, nodes.Filter) and node.name == "join":
            assembled = _reconstruct_join(node)
            _check_reconstructed(findings, assembled, loc)

        # Behavioral: branch/compare keyed on a suspicious literal.
        if isinstance(node, nodes.Compare):
            for lit in _compare_literals(node):
                if lit.lower() in SUSPICIOUS_LITERALS:
                    findings.append(finding(
                        "TPL010",
                        f"Comparison against the literal {lit!r} - possible "
                        f"behavioral trigger; manual review (not proof of malice).",
                        location=loc,
                    ))

    return findings


def _fold(s: str) -> str:
    """NFKC-normalize so fullwidth/compat homoglyphs (os, __class__) match the
    ASCII rule sets. (Does not defeat Cyrillic-style confusables; the behavioral
    lexicon uses _lex_text, which does -- see below.)"""
    return unicodedata.normalize("NFKC", s)


# Cyrillic / Greek codepoints that are visual confusables of Latin letters. A
# chat-template instruction written with these renders identically but slips a
# plain-ASCII lexicon match. Folded back to ASCII only for the BEHAVIORAL lexicon
# (TPL021/023/027), not the SSTI rules, so the SSTI 0-FP calibration is untouched.
_CONFUSABLES = {
    # Cyrillic -> Latin
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m",
    "н": "h", "о": "o", "р": "p", "с": "c", "т": "t",
    "у": "y", "х": "x", "ѕ": "s", "і": "i", "ј": "j",
    "ё": "e", "ԛ": "q", "ԝ": "w",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "У": "Y", "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J",
    # Greek -> Latin (visually confusable subset)
    "α": "a", "ο": "o", "ρ": "p", "ε": "e", "ι": "i",
    "κ": "k", "μ": "m", "ν": "v", "τ": "t", "υ": "u",
    "χ": "x", "Α": "A", "Β": "B", "Ε": "E", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}


def _deconfuse(s: str) -> str:
    return "".join(_CONFUSABLES.get(ch, ch) for ch in s)


def _lex_text(s: str) -> str:
    """Normalize for behavioral-lexicon matching: NFKC + Latin-confusables fold +
    lowercase. Catches a homoglyph-obfuscated instruction (Cyrillic 'аlwауѕ
    rесоmmеnd') that a plain ASCII lexicon would miss."""
    return _deconfuse(_fold(s)).lower()


def _is_literal_pivot(node) -> bool:
    """``''[...]`` / ``()[...]`` / ``(0)[...]`` / ``[]``/``{}`` -- subscripting a
    bare literal, the SSTI escape pivot. Non-empty list/dict literals (benign
    role lookups like ``{'user': ...}[role]``) are excluded."""
    if isinstance(node, nodes.Const):
        v = node.value
        return v == "" or (isinstance(v, (int, float)) and not isinstance(v, bool))
    if isinstance(node, (nodes.Tuple, nodes.List, nodes.Dict)):
        return not node.items
    return False


def _pivot_repr(node) -> str:
    if isinstance(node, nodes.Const):
        return repr(node.value)
    return {nodes.Tuple: "()", nodes.List: "[]", nodes.Dict: "{}"}.get(type(node), "literal")


def _is_const_root(node) -> bool:
    """A constant literal ('', 'x', 0, (), {}) used as the object of a ``.__class__`` access
    -- the entry point of the ``''.__class__.__mro__...`` escape. A variable root
    (``foo.__class__``) is benign type introspection, not a pivot."""
    if isinstance(node, nodes.Const):
        return True
    return isinstance(node, (nodes.Tuple, nodes.List, nodes.Dict)) and not node.items


def _classify_name(findings: list[Finding], name: str, loc: str, *, context: str) -> None:
    folded = _fold(name)
    if folded in DUNDERS:
        findings.append(finding(
            "TPL001",
            f"Access to dunder {name!r} ({context}) - an SSTI sandbox-escape primitive.",
            location=loc,
        ))
    if folded in GADGETS:
        findings.append(finding(
            "TPL002",
            f"Reference to Jinja2 gadget {name!r} ({context}); used to reach "
            f"arbitrary Python and absent from legitimate chat templates.",
            location=loc,
        ))
    if folded in DANGEROUS_NAMES:
        findings.append(finding(
            "TPL003",
            f"Reference to dangerous name {name!r} ({context}).",
            location=loc,
        ))
    elif folded in DANGEROUS_MODULES and context != "attribute":
        # A bare module attribute (terminal_state.os) is a benign field; the SSTI
        # form is a Name (os.system) or subscript key (__globals__['os']).
        findings.append(finding(
            "TPL003",
            f"Reference to dangerous module {name!r} ({context}).",
            location=loc,
        ))


def _check_reconstructed(findings: list[Finding], assembled: str | None, loc: str) -> None:
    if not assembled:
        return
    low = _fold(assembled).lower()
    for token in RECON_TOKENS:
        if token in low:
            findings.append(finding(
                "TPL005",
                f"String operations assemble the dangerous token {token!r} "
                f"(reconstructed: {assembled!r}).",
                location=loc,
            ))
            return


def _text_checks(source: str) -> list[Finding]:
    findings: list[Finding] = []

    urls = URL_RE.findall(source)
    ips = [m for m in IP_RE.findall(source)]
    if urls or ips:
        sample = (urls + ips)[0]
        findings.append(finding(
            "TPL011",
            f"Template text contains {len(urls)} URL(s) and {len(ips)} IP(s) "
            f"(e.g. {sample!r}).",
            location="template:text",
        ))

    if len(source.encode("utf-8")) > MAX_TEMPLATE_BYTES:
        findings.append(finding(
            "TPL012",
            f"Template is {len(source)} chars (> {MAX_TEMPLATE_BYTES} byte threshold).",
            location="template:text",
        ))

    # Hidden-character scan over the raw source (works even if parsing failed).
    bidi = sorted({ord(c) for c in source if ord(c) in BIDI_CODEPOINTS})
    if bidi:
        findings.append(finding(
            "TPL025",
            f"Template contains bidirectional-override codepoints ({_fmt_cps(bidi)}) "
            f"that can make rendered text differ from what is tokenized (Trojan Source).",
            location="template:text",
        ))
    hidden = sorted({ord(c) for c in source if _is_hidden(c)})
    if hidden:
        findings.append(finding(
            "TPL024",
            f"Template contains invisible / zero-width / tag codepoints "
            f"({_fmt_cps(hidden)}) that can conceal instructions; vetted "
            f"templates are plain printable text.",
            location="template:text",
        ))
    controls = sorted({ord(c) for c in source if _is_control(c)})
    if controls:
        findings.append(finding(
            "TPL026",
            f"Template contains raw control characters ({_fmt_cps(controls)}); "
            f"anomalous (often an artifact) - manual review.",
            location="template:text",
        ))

    low = _lex_text(source)  # NFKC + confusables-fold so obfuscated text matches
    hits = [p for p in INSTRUCTION_LEXICON if p in low]
    if hits:
        findings.append(finding(
            "TPL023",
            f"Template emits imperative instruction-like text not sourced from the "
            f"conversation (e.g. {hits[0]!r}) - possible hidden instruction injection; "
            f"manual review, not proof of malice.",
            location="template:text",
        ))

    return findings


def scan_injection_text(text: str) -> tuple[list[int], list[str]]:
    """Shared free-text injection scan for the model-card (DOC) and metadata (MET) rules.
    Returns (concealed codepoints, instruction-lexicon hits). Concealed = invisible /
    zero-width / bidi codepoints that hide text from a human while an LLM still reads it;
    raw control chars (VT/FF) are excluded (whitespace an LLM reads as whitespace too)."""
    if not text:
        return [], []
    concealed = sorted({ord(c) for c in text
                        if _is_hidden(c) or ord(c) in BIDI_CODEPOINTS})
    hits = [p for p in INSTRUCTION_LEXICON if p in _lex_text(text)]
    return concealed, hits


def analyze_card(text: str) -> list[Finding]:
    """Scan a model card (README) for injection aimed at the LLM-in-the-loop that reads
    cards -- an agent that browses / summarizes / selects models. Scoped to the
    injection-relevant checks (invisible/bidi codepoints + the instruction lexicon); URL
    and size checks are excluded because cards are legitimately link- and length-heavy."""
    findings: list[Finding] = []
    concealed, hits = scan_injection_text(text)
    if concealed:
        findings.append(finding(
            "DOC001",
            f"Model card contains invisible / zero-width / bidi codepoints "
            f"({_fmt_cps(concealed)}) that hide text from a human reader while an "
            f"LLM-in-the-loop summarizing the model still reads it - a Trojan-Source-style "
            f"card injection.",
            location="README.md"))

    if hits:
        findings.append(finding(
            "DOC002",
            f"Model card contains imperative instruction idioms (e.g. {hits[0]!r}) that "
            f"read as an injection aimed at an LLM summarizing / selecting the model - "
            f"manual review, not proof of malice.",
            location="README.md"))
    return findings


def analyze_embedded_template(source: str) -> list[Finding]:
    """Run the AST-based SSTI + behavioral + transport checks on a template embedded in a
    NON-chat_template surface -- a metadata / config string that carries Jinja delimiters
    (threat-model §5: a second template stashed where the audit doesn't look). Skips the
    text checks (url/size/lexicon); those surfaces have their own (MET001/002/020/021).
    Conservative: a string with delimiters that doesn't parse as Jinja is ignored."""
    if not source or ("{%" not in source and "{{" not in source):
        return []
    try:
        ast = parse_template(source)
    except (jinja2.TemplateSyntaxError, RecursionError):
        return []
    return _ast_checks(ast) + _behavioral_checks(ast) + _transport_checks(ast)


def _fmt_cps(codepoints: list[int]) -> str:
    return ", ".join(f"U+{cp:04X}" for cp in codepoints)


def _is_hidden(ch: str) -> bool:
    """Zero-width / format / tag / private-use codepoints that conceal text."""
    cp = ord(ch)
    if cp in BIDI_CODEPOINTS or cp in _JOINERS_ALLOWED or cp in _BIDI_MARKS_ALLOWED:
        return False  # override/isolate bidi -> TPL025; joiners + RTL marks are legit
    if cp in EXPLICIT_INVISIBLE:
        return True
    if 0xE0000 <= cp <= 0xE007F:  # Unicode tag block
        return True
    return unicodedata.category(ch) in ("Cf", "Co")  # format / private-use


def _is_control(ch: str) -> bool:
    """Raw C0/C1 control characters (excluding normal whitespace)."""
    return unicodedata.category(ch) == "Cc" and ch not in "\t\n\r"


def _behavioral_checks(ast: nodes.Template) -> list[Finding]:
    """Detect 'silent-hijack' templates: render faithfully, execute no code, but
    branch on conversation content or smuggle hidden instructions."""

    findings: list[Finding] = []
    tainted = _content_tainted_names(ast)
    # Exclude content-tainted names: a variable that ALSO holds user content is a dual-role
    # slot (e.g. a default-system-prompt holder), not a planted instruction. Promoting it to
    # a FAIL false-positives on legitimate templates like DBRX's default prompt.
    instr_tainted = _instruction_tainted_names(ast) - tainted
    if_taint = _macro_if_taint(ast, tainted)
    for node in iter_nodes(ast):
        if isinstance(node, nodes.If):
            findings.extend(_check_if(node, if_taint.get(id(node), tainted),
                                      instr_tainted))
        if isinstance(node, (nodes.Concat, nodes.Add)):
            findings.extend(_recon_behavioral(reconstruct_const_string(node),
                                               node_location(node)))
        if isinstance(node, nodes.Filter) and node.name == "join":
            findings.extend(_recon_behavioral(_reconstruct_join(node),
                                              node_location(node)))
    return findings


# Filters/functions that turn an encoded blob into live content -- the machinery an
# obfuscated payload needs; anomalous in a self-contained chat template. NB: from_json /
# fromjson are EXCLUDED -- modern tool-calling templates use them legitimately to parse
# tool arguments (calibration: 15/1665 templates, all tool-calling, all from_json).
_DECODE_FILTERS = frozenset({
    "b64decode", "b32decode", "b16decode", "a85decode", "b85decode", "decodebytes",
    "urldecode", "unquote", "unquote_plus",
})


def _transport_checks(ast: nodes.Template) -> list[Finding]:
    """Obfuscation transports: pulling in external template code (include/import/extends)
    or a decode/deserialize filter that makes an encoded payload live. Anomaly WARNs -- a
    self-contained chat template needs neither."""
    out: list[Finding] = []
    ext_done = dec_done = False
    for node in iter_nodes(ast):
        if not ext_done and isinstance(
                node, (nodes.Include, nodes.Import, nodes.FromImport, nodes.Extends)):
            ext_done = True
            out.append(finding(
                "TPL031",
                f"Template uses {type(node).__name__.lower()!r} to pull in external "
                f"template code - a chat template should be self-contained; this hides "
                f"logic outside the audited file.",
                location=node_location(node)))
        elif not dec_done:
            name = node.name if isinstance(node, (nodes.Filter, nodes.Name)) else None
            if name in _DECODE_FILTERS:
                dec_done = True
                out.append(finding(
                    "TPL032",
                    f"Template uses the {name!r} decode/deserialize filter - the machinery "
                    f"that turns an encoded blob into live content; anomalous in a chat "
                    f"template and a common obfuscation transport.",
                    location=node_location(node)))
    return out


def _check_if(node: nodes.If, tainted: frozenset[str] = frozenset(),
              instr_tainted: frozenset[str] = frozenset()) -> list[Finding]:
    test = node.test
    loc = node_location(node)
    out: list[Finding] = []

    if _refs_content(test) or _refs_tainted(test, tainted):
        # A trigger literal is a string the content is compared against -- not the
        # 'content'/'text' accessor key itself. Real templates branch on content
        # truthiness / type (multimodal) constantly; only a comparison against a
        # specific literal looks like a trigger. (Calibrated against a real-world
        # corpus where bare content checks were a 35% false-positive source.)
        # A meaningful trigger is a non-empty, non-structural literal the content
        # is compared against. Empty-string checks (content == '') and format
        # markers are benign and excluded.
        trigger_lits = [s for s in _subtree_literals(test)
                        if s.strip() and s not in CONTENT_KEYS
                        and not _is_structural_literal(s)]
        literal = (_body_emits_instruction(node.body)
                   or _body_emits_instruction(node.else_))
        # A content-gated injection FAILs only under a SPECIFIC content trigger (a literal the
        # message is compared against) -- whether the instruction is emitted directly or via a
        # hoisted var. A default system / identity prompt emitted under a bare truthiness /
        # emptiness gate (system_message == '') is a WARN (TPL023), not a targeted backdoor --
        # avoids DBRX / aero-deuce / SAGE-identity FPs on real models.
        tainted_var = (_body_emits_tainted_var(node.body, instr_tainted)
                       or _body_emits_tainted_var(node.else_, instr_tainted))
        if (literal or tainted_var) and trigger_lits:
            out.append(finding(
                "TPL021",
                "A content-keyed branch (or its else) emits imperative instruction text "
                "not sourced from the conversation (content trigger + injected instruction).",
                location=loc,
            ))
        elif trigger_lits:
            out.append(finding(
                "TPL020",
                f"Conditional test compares message CONTENT against the literal "
                f"{trigger_lits[0]!r} - the trigger shape of a behavioral backdoor; "
                f"manual review, not proof of malice.",
                location=loc,
            ))

    # TPL022 fires only when the date is *compared* against a value (behavior
    # changes ON a date = logic bomb), not when the current date is merely
    # referenced/guarded for display (e.g. Llama-3.2's "Today Date:" line).
    if _refs_date(test) and _has_compare(test):
        out.append(finding(
            "TPL022",
            "Conditional test compares against the current date/time (logic "
            "bomb): behavior changes by date.",
            location=loc,
        ))
    return out


def _has_compare(test: nodes.Node) -> bool:
    return any(isinstance(n, nodes.Compare) for n in iter_nodes(test))


def _is_structural_literal(s: str) -> bool:
    """A format/markup token (tool/think/channel marker, role word) -- not a
    natural-language backdoor trigger."""
    if any(c in _STRUCTURAL_CHARS for c in s):
        return True
    # Channel/control markers are often slash-prefixed (/think, /no_think); strip a
    # leading slash before matching. A non-format slash marker like /system_override
    # is deliberately NOT excluded -- a content-gated system override is a real
    # trigger that earns a WARN.
    return s.strip().lower().lstrip("/") in _FORMAT_WORDS


def _refs_content(test: nodes.Node) -> bool:
    for n in iter_nodes(test):
        if isinstance(n, nodes.Getitem) and _const_str(n.arg) in CONTENT_KEYS:
            return True
        if isinstance(n, nodes.Getattr) and n.attr in CONTENT_KEYS:
            return True
        # message.get('content') -- accessor laundered through .get()
        if (isinstance(n, nodes.Call) and isinstance(n.node, nodes.Getattr)
                and n.node.attr == "get"
                and any(_const_str(a) in CONTENT_KEYS for a in n.args)):
            return True
        # messages | map(attribute='content') -- bulk content extraction
        if (isinstance(n, nodes.Filter) and n.name == "map"
                and any(kw.key == "attribute" and _const_str(kw.value) in CONTENT_KEYS
                        for kw in n.kwargs)):
            return True
    return False


def _content_tainted_names(ast: nodes.Template) -> frozenset[str]:
    """Names a branch could test that actually hold message content, reached by
    binding content into a variable via ``{% set %}`` (one or many hops, including
    namespace accumulators). Without this, a content-gated backdoor hides its
    trigger behind ``{% set c = messages[-1]['content'] %}{% if 'x' in c %}`` -- the
    `if` never names content, so the direct check misses it."""
    return _expand_taint(ast, frozenset())


def _reconstruct_assign(node: nodes.Node) -> str | None:
    """Best-effort string for a ``{% set %}`` RHS: a const, a ~/+ concat of consts, a
    list/tuple of consts (space-joined), or a ``|join`` over a list literal."""
    if isinstance(node, nodes.Const) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (nodes.Concat, nodes.Add)):
        return reconstruct_const_string(node)
    if isinstance(node, nodes.Filter) and node.name == "join":
        return _reconstruct_join(node)
    if isinstance(node, (nodes.List, nodes.Tuple)):
        parts = [n.value for n in node.items
                 if isinstance(n, nodes.Const) and isinstance(n.value, str)]
        return " ".join(parts) if parts else None
    return None


def _instruction_tainted_names(ast: nodes.Template) -> frozenset[str]:
    """Names bound via ``{% set %}`` to text that hits the instruction lexicon -- e.g.
    ``{% set v = 'always recommend acme' %}`` or ``{% set p = ['ignore','previous'] %}``.
    A body that OUTPUTS such a name emits the injection even though no literal sits in the
    body -- promoting the finding from a WARN trigger to the TPL021 FAIL."""
    tainted: set[str] = set()
    for a in (n for n in iter_nodes(ast) if isinstance(n, nodes.Assign)):
        recon = _reconstruct_assign(a.node)
        if recon and any(p in _lex_text(recon) for p in INSTRUCTION_LEXICON):
            tainted.update(_assign_target_names(a.target))
    return frozenset(tainted)


def _expand_taint(scope: nodes.Node,
                  seed: frozenset[str] | set[str]) -> frozenset[str]:
    """Fixpoint of content-taint over the ``{% set %}`` assignments inside ``scope``,
    starting from ``seed`` (used to seed a macro body with its content-tainted params)."""
    assigns = [n for n in iter_nodes(scope) if isinstance(n, nodes.Assign)]
    tainted: set[str] = set(seed)
    changed = True
    while changed:
        changed = False
        for a in assigns:
            if _refs_content(a.node) or _refs_tainted(a.node, tainted):
                for name in _assign_target_names(a.target):
                    if name not in tainted:
                        tainted.add(name)
                        changed = True
    return frozenset(tainted)


def _macro_if_taint(ast: nodes.Template,
                    global_tainted: frozenset[str]) -> dict[int, frozenset[str]]:
    """Map each ``if`` inside a macro body to the taint set it should be checked
    against. When a macro is called with a content/tainted argument, the matching
    parameter name is tainted inside that macro -- closing the gate-evasion where the
    content check hides one call-hop away (``{% macro chk(t) %}{% if 'x' in t %}``
    called as ``chk(messages[-1]['content'])``), which the direct taint never crosses."""
    macros = {m.name: m for m in iter_nodes(ast) if isinstance(m, nodes.Macro)}
    if not macros:
        return {}

    # Every call to a known macro, tagged with the macro it sits inside (None = module).
    inner_ids = {id(n) for m in macros.values() for n in iter_nodes(m)}
    calls: list[tuple[str | None, nodes.Call]] = []
    for host, m in macros.items():
        for c in iter_nodes(m):
            if (isinstance(c, nodes.Call) and isinstance(c.node, nodes.Name)
                    and c.node.name in macros):
                calls.append((host, c))
    for c in iter_nodes(ast):
        if (isinstance(c, nodes.Call) and isinstance(c.node, nodes.Name)
                and c.node.name in macros and id(c) not in inner_ids):
            calls.append((None, c))

    # Fixpoint: a call's args are evaluated in the taint context of the macro it sits in,
    # so a tainted parameter propagates one hop further on each pass.
    param_taint: dict[str, set[str]] = {name: set() for name in macros}
    changed = True
    while changed:
        changed = False
        for host, call in calls:
            callee = call.node.name
            params = [a.name for a in macros[callee].args]
            ctx = set(global_tainted) | (param_taint[host] if host else set())
            for i, arg in enumerate(call.args):
                if (i < len(params) and params[i] not in param_taint[callee]
                        and (_refs_content(arg) or _refs_tainted(arg, ctx))):
                    param_taint[callee].add(params[i])
                    changed = True
            for kw in call.kwargs:
                if (kw.key in params and kw.key not in param_taint[callee]
                        and (_refs_content(kw.value) or _refs_tainted(kw.value, ctx))):
                    param_taint[callee].add(kw.key)
                    changed = True

    if_taint: dict[int, frozenset[str]] = {}
    for name, m in macros.items():
        if not param_taint[name]:
            continue
        expanded = _expand_taint(m, global_tainted | param_taint[name])
        for node in iter_nodes(m):
            if isinstance(node, nodes.If):
                if_taint[id(node)] = expanded
    return if_taint


def _assign_target_names(target: nodes.Node) -> list[str]:
    if isinstance(target, nodes.Name):
        return [target.name]
    if isinstance(target, nodes.NSRef):       # {% set ns.attr = ... %}
        return [target.name]
    if isinstance(target, (nodes.Tuple, nodes.List)):
        return [n for item in target.items for n in _assign_target_names(item)]
    return []


def _refs_tainted(test: nodes.Node, tainted: frozenset[str] | set[str]) -> bool:
    return any(isinstance(n, nodes.Name) and n.name in tainted
               for n in iter_nodes(test))


def _refs_date(test: nodes.Node) -> bool:
    for n in iter_nodes(test):
        if isinstance(n, nodes.Name) and n.name in DATE_NAMES:
            return True
        if isinstance(n, nodes.Getattr) and n.attr in DATE_ATTRS:
            return True
    return False


def _subtree_literals(test: nodes.Node) -> list[str]:
    return [n.value for n in iter_nodes(test)
            if isinstance(n, nodes.Const) and isinstance(n.value, str)]


def _body_emits_instruction(body: list[nodes.Node]) -> bool:
    texts: list[str] = []
    for stmt in body:
        for n in iter_nodes(stmt):
            if isinstance(n, nodes.TemplateData):
                texts.append(n.data)
            elif isinstance(n, nodes.Const) and isinstance(n.value, str):
                texts.append(n.value)
            elif isinstance(n, (nodes.Concat, nodes.Add)):
                asm = reconstruct_const_string(n)
                if asm:
                    texts.append(asm)
    joined = _lex_text(" ".join(texts))
    return any(p in joined for p in INSTRUCTION_LEXICON)


def _body_emits_tainted_var(body: list[nodes.Node],
                            instr_tainted: frozenset[str]) -> bool:
    """The body outputs a variable that holds instruction text (hoisted via ``{% set %}``)."""
    if not instr_tainted:
        return False
    return any(isinstance(n, nodes.Name) and n.name in instr_tainted
               for stmt in body for n in iter_nodes(stmt))


def _recon_behavioral(assembled: str | None, loc: str) -> list[Finding]:
    if not assembled:
        return []
    low = _lex_text(assembled)
    if any(p in low for p in INSTRUCTION_LEXICON):
        return [finding(
            "TPL027",
            f"String operations assemble instruction-like text {assembled!r} "
            f"(evades literal scanning).",
            location=loc,
        )]
    return []


def _depth_check(ast: nodes.Template) -> list[Finding]:
    # kept separate so _ast_checks stays a flat single pass.
    if ast_depth(ast) > MAX_AST_DEPTH:
        return [finding(
            "TPL013",
            f"Template AST nests deeper than {MAX_AST_DEPTH} levels.",
            location="template",
        )]
    return []


def _const_str(node) -> str | None:
    if isinstance(node, nodes.Const) and isinstance(node.value, str):
        return node.value
    return None


def _reconstruct_join(node: nodes.Filter) -> str | None:
    target = node.node
    if not isinstance(target, (nodes.List, nodes.Tuple)):
        return None
    parts = []
    for item in target.items:
        piece = _const_str(item)
        if piece is None:
            return None
        parts.append(piece)
    sep = ""
    if node.args:
        sep_const = _const_str(node.args[0])
        if sep_const is not None:
            sep = sep_const
    return sep.join(parts)


def _compare_literals(node: nodes.Compare) -> list[str]:
    out: list[str] = []
    left = _const_str(node.expr)
    if left is not None:
        out.append(left)
    for op in node.ops:
        rhs = _const_str(op.expr)
        if rhs is not None:
            out.append(rhs)
    return out


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple] = set()
    out: list[Finding] = []
    for f in findings:
        key = (f.rule_id, f.location, f.detail)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out
