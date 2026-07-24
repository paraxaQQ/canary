"""tokenizer.json normalizer / decoder audit (NRM rules).

The HF fast-tokenizer file carries a ``normalizer`` (runs on every input BEFORE tokenizing)
and a ``decoder`` (runs on every output). A ``Replace`` rule that rewrites CONTENT text --
not just the standard whitespace <-> SentencePiece meta-space handling -- silently alters
prompts or responses: map a refusal trigger away, strip an apology, or route benign text to
a special-token surface. Deterministic, no execution. Opt-in bundle scan only.

This is a transformers-side surface (llama.cpp rebuilds its own normalizer from the GGUF
tokenizer type), but the bundled tokenizer.json is a controllable artifact anyone loading
the repo with transformers inherits.
"""

from __future__ import annotations

import re

from ..report import Finding
from .registry import finding


def _collect_replaces(node, out: list[dict]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "Replace":
            out.append(node)
        for v in node.values():
            _collect_replaces(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_replaces(v, out)


def _pat_content(repl: dict) -> tuple[str, str, bool]:
    pat = repl.get("pattern", {})
    if isinstance(pat, dict):
        if "Regex" in pat:
            return str(pat.get("Regex", "")), str(repl.get("content", "")), True
        return str(pat.get("String", "")), str(repl.get("content", "")), False
    return str(pat), str(repl.get("content", "")), False


def _rewrites_words(pat: str, content: str, is_regex: bool) -> bool:
    """True when a Replace touches CONTENT (letters), not just whitespace / SentencePiece
    meta-space. The standard normalizers only map ' ' <-> '▁' (no letters), so any letter in
    the pattern or replacement means it rewrites actual words."""
    def has_alpha(s: str, regex: bool) -> bool:
        if regex:
            s = re.sub(r"\(\?[a-zA-Z]+\)", "", s)   # inline flags (?i) (?m)
            s = re.sub(r"\\[A-Za-z]", "", s)          # class escapes \s \d \w \b
        return any(c.isalpha() for c in s)
    return has_alpha(pat, is_regex) or has_alpha(content, False)


def analyze_tokenizer_json(data: dict) -> list[Finding]:
    if not isinstance(data, dict):
        return []
    findings: list[Finding] = []
    for section in ("normalizer", "pre_tokenizer", "decoder", "post_processor"):
        replaces: list[dict] = []
        _collect_replaces(data.get(section), replaces)
        for repl in replaces:
            pat, content, is_regex = _pat_content(repl)
            if _rewrites_words(pat, content, is_regex):
                where = "output" if section == "decoder" else "input"
                findings.append(finding(
                    "NRM001",
                    f"tokenizer.json {section} has a {'regex' if is_regex else 'literal'} "
                    f"Replace that rewrites content text ({pat!r} -> {content!r}): it runs "
                    f"on every {where} and can silently alter prompts/responses beyond "
                    f"whitespace / meta-space. Manual review.",
                    location=f"tokenizer.json:{section}"))

    # NRM003 -- instruction text in an added special token that the post-processor uses.
    from .template import scan_injection_text
    added = data.get("added_tokens")
    reachable = _post_processor_tokens(data)
    if isinstance(added, list):
        for entry in added:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if (not isinstance(content, str) or entry.get("special") is not True
                    or content not in reachable):
                continue
            _, hits = scan_injection_text(content)
            if hits:
                findings.append(finding(
                    "NRM003",
                    f"tokenizer.json post_processor references the added special token "
                    f"{content!r}, which contains instruction-idiom text (e.g. {hits[0]!r}). "
                    f"The tokenizer can insert this instruction-bearing token into encoded "
                    f"prompts. Manual review.",
                    location="tokenizer.json:added_tokens"))
    return findings


def _iter_token_strings(node):
    """Token content strings from special_tokens_map.json / added_tokens.json -- values,
    ``content`` fields, and added_tokens.json keys (``{token: id}``)."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str):
                yield k
            yield from _iter_token_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_token_strings(v)


def _post_processor_tokens(data: dict | None) -> set[str]:
    if not isinstance(data, dict):
        return set()
    return set(_iter_token_strings(data.get("post_processor")))


def analyze_special_tokens(
        *datas,
        reachable: frozenset[str] | set[str] = frozenset(),
) -> list[Finding]:
    """Flag hidden/bidi tokens and instruction text on a proven insertion path."""
    from .template import scan_injection_text

    findings: list[Finding] = []
    seen: set[str] = set()
    for data in datas:
        for s in _iter_token_strings(data):
            if not s or s in seen:
                continue
            seen.add(s)
            concealed, hits = scan_injection_text(s)
            if concealed:
                findings.append(finding(
                    "NRM002",
                    f"a special/added token {s!r} contains hidden / bidi codepoints "
                    f"({', '.join(f'U+{cp:04X}' for cp in concealed)}) - a concealed "
                    f"privileged token a human reader cannot see.",
                    location="special_tokens"))
            if hits and s in reachable:
                findings.append(finding(
                    "NRM003",
                    f"a declared BOS/EOS or post-processor token {s!r} contains "
                    f"instruction-idiom text (e.g. {hits[0]!r}). The tokenizer has a "
                    f"declared path that inserts this token into prompts. Manual review.",
                    location="special_tokens"))
    return findings
