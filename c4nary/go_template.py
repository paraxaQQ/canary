"""Structural parser and rules for Ollama's Go ``text/template`` chat templates.

Ollama's ``TEMPLATE`` is Go ``text/template``, not Jinja2. The Jinja AST family
cannot read it, and rendering it would violate c4nary's no-execution invariant.
This module tokenizes and parses the template *without* executing it, then runs a
small rule family over the structure:

* ``GOT000`` -- the template did not parse;
* ``GOT001`` -- a ``.Prompt``/``.Messages``-keyed branch emits instruction text;
* ``GOT002`` -- ``currentDate``/``yesterdayDate`` is used in a test or comparison;
* ``GOT003`` -- ``call`` invokes a function value from the data;
* ``GOT004`` -- named template composition (``define``/``template``/``block``);
* ``GOT005`` -- ``printf``/``print`` reconstructs instruction-like text.

It is deliberately conservative. Anything it cannot classify is a parse error
(``GOT000``), never a silent "clean".
"""

from __future__ import annotations

import bisect
import dataclasses
import re
from typing import Iterable

from .report import Finding
from .rules.registry import finding
from .rules.template import instruction_text_evidence, is_structural_literal

MAX_TEMPLATE_BYTES = 256 * 1024
MAX_NODES = 20_000
MAX_DEPTH = 25

_DATE_FUNCS = frozenset({"currentDate", "yesterdayDate"})
_COMPARISONS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
_PRINT_FUNCS = frozenset({"printf", "print", "println"})
_CONTENT_ROOTS = (".Prompt", ".Messages")
_CONTENT_KEYS = frozenset({"content", "text"})
_KEYWORDS = frozenset({
    "if", "else", "range", "with", "end", "template", "define", "block",
    "break", "continue",
})
_LITERAL_WORDS = frozenset({"nil", "true", "false"})


class _ParseError(Exception):
    def __init__(self, message: str, line: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Expr:
    kind: str  # field | var | literal | call | method | pipeline | paren
    name: str = ""
    value: object = None
    args: tuple["Expr", ...] = ()
    receiver: "Expr | None" = None


@dataclasses.dataclass
class Branch:
    test: Expr
    line: int
    body: list["Node"] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Node:
    kind: str  # text | action | if | range | with | define | block | template
    line: int
    text: str = ""
    expr: Expr | None = None
    body: list["Node"] = dataclasses.field(default_factory=list)
    branches: list[Branch] = dataclasses.field(default_factory=list)
    else_body: list["Node"] = dataclasses.field(default_factory=list)
    name: str | None = None
    bindings: tuple[str, ...] = ()
    target: str | None = None


@dataclasses.dataclass
class GoTemplate:
    source: str
    nodes: tuple[Node, ...] = ()
    error: str | None = None
    error_line: int | None = None
    oversize: bool = False
    action_count: int = 0

    @property
    def parsed(self) -> bool:
        return self.error is None

    def findings(self) -> list[Finding]:
        if self.error is not None:
            location = f"TEMPLATE:L{self.error_line}" if self.error_line else "TEMPLATE"
            if self.error.startswith("rule evaluation"):
                detail = (f"Go template {self.error}; the template was not fully "
                          f"analyzed.")
            else:
                detail = (f"Go template failed to parse: {self.error}. Only "
                          f"text-level checks ran; control flow and expressions "
                          f"were not analyzed.")
            return [finding("GOT000", detail, location=location)]
        out: list[Finding] = []
        try:
            _walk(tuple(self.nodes), _Context(), out)
        except RecursionError:
            self.error = "rule evaluation exceeded the nesting budget"
            return [finding(
                "GOT000",
                "Go template rule evaluation exceeded the nesting budget; the "
                "template was not fully analyzed.",
                location="TEMPLATE")]
        except Exception as exc:  # noqa: BLE001 - fail closed on crafted input
            self.error = f"rule evaluation raised {type(exc).__name__}: {exc}"
            return [finding(
                "GOT000",
                f"Go template rule evaluation raised {type(exc).__name__}: {exc}; "
                f"the template was not fully analyzed.",
                location="TEMPLATE")]
        out = _dedupe(out)
        compositions = [f for f in out if f.rule_id == "GOT004"]
        if len(compositions) > 1:
            out = [f for f in out if f.rule_id != "GOT004"] + [compositions[0]]
        return out


@dataclasses.dataclass
class _Context:
    tainted: set[str] = dataclasses.field(default_factory=set)
    dot_tainted: bool = False

    def copy(self) -> "_Context":
        return _Context(set(self.tainted), self.dot_tainted)


# --------------------------------------------------------------------------- #
# Action and text splitting
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _Event:
    kind: str  # text | action | comment
    payload: str
    line: int


def _newline_index(source: str) -> list[int]:
    return [i for i, ch in enumerate(source) if ch == "\n"]


def _find_action_end(source: str, start: int) -> int:
    i = start
    quote: str | None = None
    while i < len(source) - 1:
        ch = source[i]
        if quote is not None:
            if quote in ('"', "'") and ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ('"', "`", "'"):
            quote = ch
            i += 1
            continue
        if ch == "}" and source[i + 1] == "}":
            return i
        i += 1
    return -1


def _split_events(source: str) -> list[_Event]:
    newlines = _newline_index(source)
    events: list[_Event] = []
    i = 0
    while i < len(source):
        start = source.find("{{", i)
        if start < 0:
            events.append(_Event("text", source[i:], _line_at(newlines, i)))
            break
        if start > i:
            events.append(_Event("text", source[i:start], _line_at(newlines, i)))

        line = _line_at(newlines, start)
        j = start + 2
        if j < len(source) and source[j] == "-" and j + 1 < len(source) \
                and source[j + 1] in " \t\r\n":
            j += 1
        probe = j
        while probe < len(source) and source[probe] in " \t\r\n":
            probe += 1
        if source.startswith("/*", probe):
            end = source.find("*/", probe + 2)
            if end < 0:
                raise _ParseError("unclosed template comment", line)
            after = end + 2
            while after < len(source) and source[after] in " \t\r\n":
                after += 1
            if after < len(source) and source[after] == "-":
                after += 1
                while after < len(source) and source[after] in " \t\r\n":
                    after += 1
            if not source.startswith("}}", after):
                raise _ParseError("comment does not end with the action", line)
            events.append(_Event("comment", "", line))
            i = after + 2
            continue

        end = _find_action_end(source, j)
        if end < 0:
            raise _ParseError("unclosed action", line)
        inner = source[j:end]
        stripped = inner.rstrip()
        if stripped.endswith("-") and len(stripped) >= 2 \
                and stripped[-2] in " \t\r\n":
            inner = stripped[:-1]
        events.append(_Event("action", inner.strip(), line))
        i = end + 2
    return events


def _line_at(newlines: list[int], offset: int) -> int:
    return bisect.bisect_right(newlines, offset - 1) + 1


# --------------------------------------------------------------------------- #
# Lexer
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _Token:
    kind: str
    value: object
    pos: int


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(
    r"-?(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _decode_go_string(body: str, line: int) -> str:
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= len(body):
            raise _ParseError("unterminated string escape", line)
        esc = body[i]
        simple = {
            "a": "\a", "b": "\b", "f": "\f", "n": "\n", "r": "\r",
            "t": "\t", "v": "\v", "\\": "\\", '"': '"', "'": "'",
        }
        if esc in simple:
            out.append(simple[esc])
            i += 1
        elif esc in "xuU" or esc in "01234567":
            if esc in "xuU":
                start = i + 1
                end = start + {"x": 2, "u": 4, "U": 8}[esc]
                base = 16
            else:
                start = i
                end = i + 3
                base = 8
            try:
                out.append(chr(int(body[start:end], base)))
            except ValueError:
                raise _ParseError(f"invalid string escape \\{esc}", line) from None
            i = end
        else:
            raise _ParseError(f"unknown string escape \\{esc}", line)
    return "".join(out)


def _lex(inner: str, line: int) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if inner.startswith(":=", i):
            tokens.append(_Token("DECLARE", ":=", i))
            i += 2
            continue
        if ch == "=":
            tokens.append(_Token("ASSIGN", "=", i))
            i += 1
            continue
        if ch == "|":
            tokens.append(_Token("PIPE", "|", i))
            i += 1
            continue
        if ch == ",":
            tokens.append(_Token("COMMA", ",", i))
            i += 1
            continue
        if ch == "(":
            tokens.append(_Token("LPAREN", "(", i))
            i += 1
            continue
        if ch == ")":
            tokens.append(_Token("RPAREN", ")", i))
            i += 1
            continue
        if ch in ('"', "`", "'"):
            quote = ch
            j = i + 1
            while j < len(inner):
                if quote == '"' and inner[j] == "\\":
                    j += 2
                    continue
                if inner[j] == quote:
                    break
                j += 1
            if j >= len(inner):
                raise _ParseError("unterminated quoted value", line)
            raw = inner[i + 1:j]
            if quote == "'":
                value = _decode_go_string(raw, line)
                if len(value) != 1:
                    raise _ParseError("rune constant must be one character", line)
            elif quote == "`":
                value = raw
            else:
                value = _decode_go_string(raw, line)
            kind = "RAW" if quote == "`" else "CHAR" if quote == "'" else "STRING"
            tokens.append(_Token(kind, value, i))
            i = j + 1
            continue
        if ch == ".":
            j = i
            while j < len(inner) and inner[j] == ".":
                j += 1
            while j < len(inner) and (inner[j].isalnum() or inner[j] == "_"):
                j += 1
                while j < len(inner) and (inner[j].isalnum() or inner[j] == "_"):
                    j += 1
                if j < len(inner) and inner[j] == "." and j + 1 < len(inner) \
                        and (inner[j + 1].isalnum() or inner[j + 1] == "_"):
                    j += 1
                    continue
                break
            tokens.append(_Token("FIELD", inner[i:j], i))
            i = j
            continue
        if ch == "$":
            j = i + 1
            if j < len(inner) and (inner[j].isalpha() or inner[j] == "_"):
                match = _IDENT.match(inner, j)
                assert match is not None
                j = match.end()
            while j < len(inner) and inner[j] == "." and j + 1 < len(inner) \
                    and (inner[j + 1].isalnum() or inner[j + 1] == "_"):
                j += 1
                match = _IDENT.match(inner, j)
                assert match is not None
                j = match.end()
            tokens.append(_Token("VAR", inner[i:j], i))
            i = j
            continue
        if ch.isdigit() or (ch == "-" and i + 1 < len(inner) and inner[i + 1].isdigit()):
            match = _NUMBER.match(inner, i)
            if match is None:
                raise _ParseError("invalid number", line)
            tokens.append(_Token("NUMBER", match.group(0), i))
            i = match.end()
            continue
        if ch.isalpha() or ch == "_":
            match = _IDENT.match(inner, i)
            if match is None:
                raise _ParseError(f"unexpected character {ch!r}", line)
            tokens.append(_Token("IDENT", match.group(0), i))
            i = match.end()
            continue
        raise _ParseError(f"unexpected character {ch!r}", line)
    return tokens


# --------------------------------------------------------------------------- #
# Expression parser
# --------------------------------------------------------------------------- #


class _ExprParser:
    def __init__(self, tokens: list[_Token], line: int) -> None:
        self.tokens = tokens
        self.line = line
        self.pos = 0
        self.depth = 0

    def parse(self) -> Expr:
        if not self.tokens:
            raise _ParseError("empty action", self.line)
        expr = self._pipeline()
        if self.pos != len(self.tokens):
            raise _ParseError(
                f"unexpected token {self.tokens[self.pos].value!r}", self.line)
        return expr

    def _peek(self) -> _Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _take(self) -> _Token:
        token = self._peek()
        if token is None:
            raise _ParseError("unexpected end of action", self.line)
        self.pos += 1
        return token

    def _pipeline(self) -> Expr:
        commands = [self._command()]
        while True:
            token = self._peek()
            if token is None or token.kind != "PIPE":
                break
            self._take()
            commands.append(self._command())
        if len(commands) == 1:
            return commands[0]
        return Expr("pipeline", args=tuple(commands))

    def _command(self) -> Expr:
        first = self._operand()
        args: list[Expr] = []
        while self._starts_operand():
            args.append(self._operand())
        if args:
            if first.kind == "call":
                return Expr("call", name=first.name, args=tuple(args))
            if first.kind in ("field", "var"):
                # Go templates call methods as `{{ .Receiver.Method arg }}`.
                return Expr("method", name=first.name, args=tuple(args),
                            receiver=first)
            raise _ParseError("value cannot take arguments", self.line)
        return first

    def _starts_operand(self) -> bool:
        token = self._peek()
        return token is not None and token.kind in {
            "FIELD", "VAR", "STRING", "RAW", "CHAR", "NUMBER", "IDENT", "LPAREN",
        }

    def _operand(self) -> Expr:
        token = self._take()
        if token.kind == "FIELD":
            return Expr("field", name=str(token.value))
        if token.kind == "VAR":
            return Expr("var", name=str(token.value))
        if token.kind in ("STRING", "RAW", "CHAR"):
            return Expr("literal", value=token.value)
        if token.kind == "NUMBER":
            return Expr("literal", value=token.value)
        if token.kind == "LPAREN":
            self.depth += 1
            if self.depth > MAX_DEPTH:
                raise _ParseError("expression nesting is too deep", self.line)
            inner = self._pipeline()
            token = self._take()
            if token.kind != "RPAREN":
                raise _ParseError("missing closing parenthesis", self.line)
            self.depth -= 1
            return Expr("paren", args=(inner,))
        if token.kind == "IDENT":
            word = str(token.value)
            if word in _LITERAL_WORDS:
                if word == "nil":
                    return Expr("literal", value=None)
                return Expr("literal", value=(word == "true"))
            if word in _KEYWORDS:
                raise _ParseError(f"unexpected keyword {word!r}", self.line)
            return Expr("call", name=word, args=())
        raise _ParseError(f"unexpected token {token.value!r}", self.line)


# --------------------------------------------------------------------------- #
# Structural parser
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _Frame:
    node: Node
    kind: str
    target: list[Node]
    seen_else: bool = False


def parse_go_template(source: str | None) -> GoTemplate | None:
    """Parse an Ollama ``TEMPLATE`` without rendering it. ``None`` means no template."""
    if source is None:
        return None
    if len(source.encode("utf-8")) > MAX_TEMPLATE_BYTES:
        return GoTemplate(
            source,
            error=f"template exceeds the {MAX_TEMPLATE_BYTES} byte parser cap",
            oversize=True,
        )
    try:
        events = _split_events(source)
        nodes, actions = _parse_events(events)
    except _ParseError as exc:
        return GoTemplate(source, error=exc.message, error_line=exc.line)
    except RecursionError:
        return GoTemplate(
            source, error="template nesting is too deep to parse", error_line=None)
    except Exception as exc:  # noqa: BLE001 - fail closed on crafted input
        return GoTemplate(
            source,
            error=f"template analysis raised {type(exc).__name__}: {exc}",
            error_line=None,
        )
    return GoTemplate(source, nodes=tuple(nodes), action_count=actions)


def _parse_events(events: list[_Event]) -> tuple[list[Node], int]:
    root: list[Node] = []
    stack: list[_Frame] = []
    actions = 0

    def current() -> list[Node]:
        return stack[-1].target if stack else root

    for event in events:
        if event.kind == "text":
            if event.payload:
                current().append(Node("text", event.line, text=event.payload))
            continue
        if event.kind == "comment":
            continue
        actions += 1
        if actions > MAX_NODES:
            raise _ParseError(f"more than {MAX_NODES} template actions", event.line)
        tokens = _lex(event.payload, event.line)
        if not tokens:
            raise _ParseError("empty action", event.line)

        keyword = str(tokens[0].value) if tokens[0].kind == "IDENT" else None
        if keyword == "end":
            if len(tokens) != 1:
                raise _ParseError("unexpected token after end", event.line)
            if not stack:
                raise _ParseError("unexpected {{end}}", event.line)
            stack.pop()
            continue
        if keyword == "else":
            if not stack or stack[-1].kind not in ("if", "range", "with"):
                raise _ParseError("unexpected {{else}}", event.line)
            frame = stack[-1]
            if frame.seen_else:
                raise _ParseError("multiple {{else}} blocks", event.line)
            if len(tokens) > 1 and tokens[1].kind == "IDENT" \
                    and tokens[1].value == "if":
                if frame.kind != "if":
                    raise _ParseError("{{else if}} is only valid in an if", event.line)
                test = _ExprParser(tokens[2:], event.line).parse()
                branch = Branch(test, event.line)
                frame.node.branches.append(branch)
                frame.target = branch.body
            else:
                if len(tokens) != 1:
                    raise _ParseError("unexpected token after else", event.line)
                frame.seen_else = True
                frame.target = frame.node.else_body
            continue
        if keyword in ("if", "with"):
            test = _ExprParser(tokens[1:], event.line).parse()
            node = Node(keyword, event.line)
            if keyword == "if":
                node.branches.append(Branch(test, event.line))
                current().append(node)
                stack.append(_Frame(node, "if", node.branches[0].body))
            else:
                node.expr = test
                current().append(node)
                stack.append(_Frame(node, "with", node.body))
            if len(stack) > MAX_DEPTH:
                raise _ParseError("template nesting is too deep", event.line)
            continue
        if keyword == "range":
            node = _parse_range(tokens, event.line)
            current().append(node)
            stack.append(_Frame(node, "range", node.body))
            if len(stack) > MAX_DEPTH:
                raise _ParseError("template nesting is too deep", event.line)
            continue
        if keyword in ("define", "block", "template"):
            node = _parse_named(tokens, event.line)
            current().append(node)
            if keyword != "template":
                stack.append(_Frame(node, keyword, node.body))
                if len(stack) > MAX_DEPTH:
                    raise _ParseError("template nesting is too deep", event.line)
            continue
        if keyword in ("break", "continue"):
            if len(tokens) != 1:
                raise _ParseError(f"unexpected token after {keyword}", event.line)
            current().append(Node(keyword, event.line))
            continue

        if tokens[0].kind == "VAR" and len(tokens) >= 2 \
                and tokens[1].kind in ("DECLARE", "ASSIGN"):
            target = str(tokens[0].value)
            expr = _ExprParser(tokens[2:], event.line).parse()
            current().append(Node("action", event.line, expr=expr, target=target))
        else:
            expr = _ExprParser(tokens, event.line).parse()
            current().append(Node("action", event.line, expr=expr))

    if stack:
        frame = stack[-1]
        raise _ParseError(f"unclosed {{{{{frame.kind}}}}} block", frame.node.line)
    return root, actions


def _parse_range(tokens: list[_Token], line: int) -> Node:
    index = 1
    bindings: list[str] = []
    if index + 1 < len(tokens) and tokens[index].kind == "VAR":
        if tokens[index + 1].kind == "COMMA":
            if index + 3 >= len(tokens) or tokens[index + 2].kind != "VAR" \
                    or tokens[index + 3].kind != "DECLARE":
                raise _ParseError("invalid range variable declaration", line)
            bindings = [str(tokens[index].value), str(tokens[index + 2].value)]
            index += 4
        elif tokens[index + 1].kind == "DECLARE":
            bindings = [str(tokens[index].value)]
            index += 2
    expr = _ExprParser(tokens[index:], line).parse()
    return Node("range", line, expr=expr, bindings=tuple(bindings))


def _parse_named(tokens: list[_Token], line: int) -> Node:
    kind = str(tokens[0].value)
    if len(tokens) < 2 or tokens[1].kind != "STRING":
        raise _ParseError(f"{kind} requires a quoted template name", line)
    name = str(tokens[1].value)
    node = Node(kind, line, name=name)
    if len(tokens) > 2:
        node.expr = _ExprParser(tokens[2:], line).parse()
    return node


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


def _walk(nodes: Iterable[Node], ctx: _Context, out: list[Finding]) -> None:
    for node in nodes:
        if node.kind == "text":
            continue
        if node.kind == "action":
            if node.expr is not None:
                _scan_expr(node.expr, line=node.line, out=out)
                if node.target is not None and _expr_tainted(node.expr, ctx):
                    ctx.tainted.add(node.target)
            continue
        if node.kind == "if":
            for branch in node.branches:
                _scan_expr(branch.test, line=branch.line, out=out)
                if _expr_has_date(branch.test):
                    out.append(_date_finding(branch.line))
                triggers = _content_triggers(branch.test, ctx)
                if triggers:
                    evidence = _body_evidence(branch.body)
                    if evidence is not None:
                        hit, shape = evidence
                        detail = (
                            f"a content-keyed Go template branch (trigger "
                            f"{', '.join(repr(t) for t in sorted(triggers))}) emits "
                            f"instruction text not sourced from the conversation "
                            f"({hit or 'structural shape ' + repr(shape)}); possible "
                            f"content-gated injection.")
                        out.append(finding(
                            "GOT001", detail,
                            location=f"TEMPLATE:L{branch.line}",
                            subject="trigger:" + ",".join(sorted(triggers))))
                _walk(branch.body, ctx.copy(), out)
            _walk(node.else_body, ctx.copy(), out)
            continue
        if node.kind == "range":
            if node.expr is not None:
                _scan_expr(node.expr, line=node.line, out=out)
            inner = ctx.copy()
            if node.expr is not None and _expr_tainted(node.expr, ctx):
                inner.tainted.update(node.bindings)
                inner.dot_tainted = True
            _walk(node.body, inner, out)
            continue
        if node.kind == "with":
            if node.expr is not None:
                _scan_expr(node.expr, line=node.line, out=out)
            inner = ctx.copy()
            if node.expr is not None and _expr_tainted(node.expr, ctx):
                inner.dot_tainted = True
            _walk(node.body, inner, out)
            continue
        if node.kind in ("define", "block"):
            out.append(finding(
                "GOT004",
                f"uses a named Go template {node.kind} ({node.name!r}); the audit "
                f"covers the whole TEMPLATE string, but the indirection is easy to "
                f"miss - review the definition.",
                location=f"TEMPLATE:L{node.line}",
                subject=f"{node.kind}:{node.name}"))
            if node.expr is not None:
                _scan_expr(node.expr, line=node.line, out=out)
            _walk(node.body, ctx.copy(), out)
            continue
        if node.kind == "template":
            out.append(finding(
                "GOT004",
                f"invokes the named Go template {node.name!r}; review that its "
                f"definition is present in the audited TEMPLATE string.",
                location=f"TEMPLATE:L{node.line}",
                subject=f"template:{node.name}"))
            if node.expr is not None:
                _scan_expr(node.expr, line=node.line, out=out)


def _scan_expr(expr: Expr, *, line: int, out: list[Finding]) -> None:
    location = f"TEMPLATE:L{line}"
    if expr.kind == "call":
        if expr.name in _COMPARISONS and any(
                _expr_has_date(arg) for arg in expr.args):
            out.append(_date_finding(line))
        if expr.name == "call":
            out.append(finding(
                "GOT003",
                "uses the Go template call builtin to invoke a function value from "
                "the template data; Ollama's template data has no function values, so "
                "this construct is anomalous.",
                location=location,
                subject="call"))
        if expr.name in _PRINT_FUNCS:
            text = " ".join(
                str(arg.value) for arg in expr.args
                if arg.kind == "literal" and isinstance(arg.value, str))
            if text:
                evidence = instruction_text_evidence(text)
                if evidence is not None:
                    hit, shape = evidence
                    out.append(finding(
                        "GOT005",
                        f"{expr.name} assembles instruction-like text "
                        f"({hit or 'structural shape ' + repr(shape)}); possible "
                        f"obfuscated instruction text.",
                        location=location,
                        subject=f"recon:{hit or shape}"))
    if expr.receiver is not None:
        _scan_expr(expr.receiver, line=line, out=out)
    for arg in expr.args:
        _scan_expr(arg, line=line, out=out)


def _date_finding(line: int) -> Finding:
    return finding(
        "GOT002",
        "currentDate/yesterdayDate is used in a test or comparison, so template "
        "behavior can change by date.",
        location=f"TEMPLATE:L{line}",
        subject="date-gated")


def _content_triggers(test: Expr, ctx: _Context) -> set[str]:
    out: set[str] = set()

    def collect(expr: Expr) -> None:
        for arg in expr.args:
            if arg.kind != "literal" or not isinstance(arg.value, str):
                continue
            literal = arg.value
            stripped = literal.strip()
            if not stripped or stripped.lower() in _CONTENT_KEYS:
                continue
            if is_structural_literal(literal):
                continue
            out.add(literal)

    def visit(expr: Expr) -> None:
        if expr.kind == "call" and expr.name in _COMPARISONS:
            if any(_expr_tainted(arg, ctx) for arg in expr.args):
                collect(expr)
        if expr.kind == "method" and _expr_tainted(expr.receiver, ctx):
            collect(expr)
        if expr.kind == "pipeline":
            # In a pipeline the previous command's value becomes the last
            # argument, so `{{ .Prompt | eq "deploy" }}` is a content gate.
            if any(_expr_tainted(command, ctx) for command in expr.args):
                for command in expr.args:
                    if command.kind == "call" and command.name in _COMPARISONS:
                        collect(command)
        for arg in expr.args:
            visit(arg)

    visit(test)
    return out


def _body_evidence(body: Iterable[Node]) -> tuple[str | None, str | None] | None:
    parts: list[str] = []

    def visit(nodes: Iterable[Node]) -> None:
        for node in nodes:
            if node.kind == "text":
                parts.append(node.text)
                continue
            if node.kind == "if":
                for branch in node.branches:
                    visit(branch.body)
                visit(node.else_body)
                continue
            if node.kind in ("range", "with", "define", "block"):
                visit(node.body)

    visit(body)
    text = "\n".join(parts)
    if not text:
        return None
    return instruction_text_evidence(text)


def _expr_tainted(expr: Expr | None, ctx: _Context) -> bool:
    if expr is None:
        return False
    if expr.kind == "literal":
        return False
    if expr.kind == "field":
        name = expr.name
        if ctx.dot_tainted:
            return True
        return any(name == root or name.startswith(root + ".")
                   for root in _CONTENT_ROOTS)
    if expr.kind == "var":
        name = expr.name
        root = name.split(".", 1)[0]
        if root in ctx.tainted:
            return True
        return any(token in name for token in _CONTENT_ROOTS)
    if expr.kind == "method":
        return (_expr_tainted(expr.receiver, ctx)
                or any(_expr_tainted(arg, ctx) for arg in expr.args))
    return any(_expr_tainted(arg, ctx) for arg in expr.args)


def _expr_has_date(expr: Expr) -> bool:
    if expr.kind == "call" and expr.name in _DATE_FUNCS:
        return True
    return any(_expr_has_date(arg) for arg in expr.args)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str | None, str | None]] = set()
    out: list[Finding] = []
    for item in findings:
        key = (item.rule_id, item.location, item.subject)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
