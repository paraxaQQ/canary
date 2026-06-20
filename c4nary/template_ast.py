"""Jinja2 template parsing helpers — **AST only, never rendered**.

This module owns the single allowed entry point to Jinja2 (``Environment().parse``)
plus normalization/hashing and iterative AST traversal helpers. Nothing here
calls ``.render()`` or compiles the template (invariant §7.1). Traversal is
iterative so a maliciously deep template cannot blow the Python stack.
"""

from __future__ import annotations

import functools
import hashlib
import re
from pathlib import Path

import jinja2
from jinja2 import nodes
from jinja2.ext import Extension

_KNOWN_DIR = Path(__file__).parent / "known_templates"


class _GenerationExtension(Extension):
    """Parse Hugging Face's ``{% generation %}...{% endgeneration %}`` block.

    Transformers' ``apply_chat_template`` registers a custom Jinja tag that marks
    assistant-generated spans (for ``return_assistant_tokens_mask``). A bare
    Jinja2 environment does not know that tag and raises ``TemplateSyntaxError``,
    which would make many legitimate modern templates un-analyzable. We register
    a no-op equivalent that simply keeps the block body in the AST so the walker
    still sees the inner nodes. It is parsed, never rendered.
    """

    tags = {"generation"}

    def parse(self, parser):  # pragma: no cover - exercised via parse_template
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body, lineno=lineno)

URL_RE = re.compile(r"""https?://[^\s"'<>`)]+""", re.IGNORECASE)
# IPv4 dotted quad with plausible octets, not part of a longer number/version.
IP_RE = re.compile(
    r"(?<![\d.])"
    r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?![\d.])"
)


def normalize_template(source: str) -> str:
    """Canonical form for hashing/comparison: unify newlines, strip ends.

    Deliberately minimal and lossless-ish so that hashes are stable across
    platforms (CRLF vs LF) without masking meaningful content differences.
    """

    s = source.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()


def template_sha256(source: str) -> str:
    return hashlib.sha256(normalize_template(source).encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=1)
def load_known_templates() -> dict[str, str]:
    """Map normalized-template-hash -> reference name from ``known_templates/``.

    Cached: the reference set is static data, and this is called once per scan
    (which matters when auditing thousands of models in one process).
    """

    out: dict[str, str] = {}
    if _KNOWN_DIR.is_dir():
        for p in sorted(_KNOWN_DIR.glob("*.jinja")):
            text = p.read_text(encoding="utf-8")
            out[template_sha256(text)] = p.stem
    return out


def parse_template(source: str) -> nodes.Template:
    """Parse to a Jinja2 AST. Raises ``TemplateSyntaxError`` (or RecursionError).

    Registers the same extensions Hugging Face's ``apply_chat_template`` uses --
    the ``generation`` block and ``loopcontrols`` (``{% break %}`` /
    ``{% continue %}``) -- so modern templates parse. No compilation, no rendering.
    """

    env = jinja2.Environment(
        autoescape=False,
        extensions=[_GenerationExtension, "jinja2.ext.loopcontrols"],
    )
    return env.parse(source)


def iter_nodes(root: nodes.Node):
    """Iterative pre-order walk over all AST nodes (stack-safe)."""

    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.iter_child_nodes())


def ast_depth(root: nodes.Node) -> int:
    """Maximum nesting depth of the AST, computed iteratively."""

    max_depth = 0
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            max_depth = depth
        for child in node.iter_child_nodes():
            stack.append((child, depth + 1))
    return max_depth


def node_location(node: nodes.Node) -> str:
    lineno = getattr(node, "lineno", None)
    return f"template:L{lineno}" if lineno else "template"


def reconstruct_const_string(node: nodes.Node, _depth: int = 0) -> str | None:
    """If ``node`` is purely constant strings combined with ``~`` / ``+``,
    return the assembled literal; otherwise ``None``.

    Used to defeat split-string obfuscation like ``'po' ~ 'pen'``.
    """

    if _depth > 256:  # guard against pathological chains
        return None
    if isinstance(node, nodes.Const):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, nodes.Concat):
        parts = []
        for child in node.nodes:
            piece = reconstruct_const_string(child, _depth + 1)
            if piece is None:
                return None
            parts.append(piece)
        return "".join(parts)
    if isinstance(node, nodes.Add):
        left = reconstruct_const_string(node.left, _depth + 1)
        right = reconstruct_const_string(node.right, _depth + 1)
        if left is None or right is None:
            return None
        return left + right
    return None
