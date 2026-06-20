"""Static analysis rules for the embedded Jinja2 chat template.

The detector walks the parsed AST (never rendering it) and flags the SSTI /
sandbox-escape primitives behind CVE-2024-34359 and its variants, plus a few
heuristic behavioral red flags. A template whose normalized hash matches a
vetted reference short-circuits to a single INFO finding.
"""

from __future__ import annotations

import unicodedata

import jinja2
from jinja2 import nodes

from ..report import Finding
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

# Python dunders that form SSTI sandbox-escape chains.
DUNDERS = frozenset({
    "__class__", "__base__", "__bases__", "__subclasses__", "__mro__",
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
BIDI_CODEPOINTS = frozenset({
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
    0x2066, 0x2067, 0x2068, 0x2069, 0x200E, 0x200F, 0x061C,
})
# Zero-width / format controls that conceal text. Excludes the joiners
# U+200C/U+200D (see _JOINERS_ALLOWED) which are required in many scripts.
EXPLICIT_INVISIBLE = frozenset({
    0x200B, 0x2060, 0xFEFF, 0x00AD, 0x180E, 0x2028, 0x2029,
})
# ZWNJ / ZWJ are essential in Persian/Arabic/Indic text and emoji sequences --
# far too common in legitimate templates to treat as concealment.
_JOINERS_ALLOWED = frozenset({0x200C, 0x200D})

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

    findings.extend(_text_checks(source))
    return _dedupe(findings)


def _ast_checks(ast: nodes.Template) -> list[Finding]:
    findings: list[Finding] = []

    for node in iter_nodes(ast):
        loc = node_location(node)

        # Attribute / subscript access to dunders, gadgets, dangerous names.
        if isinstance(node, nodes.Getattr):
            _classify_name(findings, node.attr, loc, context="attribute")
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
    ASCII rule sets. (Does not defeat Cyrillic-style confusables -- documented.)"""
    return unicodedata.normalize("NFKC", s)


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

    low = _fold(source).lower()  # NFKC so fullwidth injected text matches
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


def _fmt_cps(codepoints: list[int]) -> str:
    return ", ".join(f"U+{cp:04X}" for cp in codepoints)


def _is_hidden(ch: str) -> bool:
    """Zero-width / format / tag / private-use codepoints that conceal text."""
    cp = ord(ch)
    if cp in BIDI_CODEPOINTS or cp in _JOINERS_ALLOWED:
        return False  # bidi -> TPL025; joiners are legitimate in many scripts
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
    for node in iter_nodes(ast):
        if isinstance(node, nodes.If):
            findings.extend(_check_if(node))
        if isinstance(node, (nodes.Concat, nodes.Add)):
            findings.extend(_recon_behavioral(reconstruct_const_string(node),
                                               node_location(node)))
        if isinstance(node, nodes.Filter) and node.name == "join":
            findings.extend(_recon_behavioral(_reconstruct_join(node),
                                              node_location(node)))
    return findings


def _check_if(node: nodes.If) -> list[Finding]:
    test = node.test
    loc = node_location(node)
    out: list[Finding] = []

    if _refs_content(test):
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
        if _body_emits_instruction(node.body):
            out.append(finding(
                "TPL021",
                "A content-keyed branch also emits imperative instruction text not "
                "sourced from the conversation (content trigger + injected instruction).",
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
    return s.strip().lower() in _FORMAT_WORDS


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
    return False


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
    joined = _fold(" ".join(texts)).lower()
    return any(p in joined for p in INSTRUCTION_LEXICON)


def _recon_behavioral(assembled: str | None, loc: str) -> list[Finding]:
    if not assembled:
        return []
    low = _fold(assembled).lower()
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
