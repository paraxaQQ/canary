"""Sandbox regression probes outside the scanner path.

The shipped scanner only parses Jinja ASTs. These tests render inert access
gadgets under both Jinja sandbox environments to measure their current block
points. This differs from tools/oracle/render.py, which deliberately uses the
mutable SandboxedEnvironment for benign list-mutation compatibility.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TypeAlias

import jinja2
import pytest
from jinja2.exceptions import SecurityError
from jinja2.sandbox import ImmutableSandboxedEnvironment, SandboxedEnvironment


Sandbox: TypeAlias = type[SandboxedEnvironment]
ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
MINIMUM_PATCHED_JINJA = (3, 1, 6)


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:3])


@pytest.mark.parametrize(
    "environment",
    [SandboxedEnvironment, ImmutableSandboxedEnvironment],
)
@pytest.mark.parametrize(
    "source",
    [
        (FIXTURES / "trigger_ssti.jinja").read_text(encoding="utf-8"),
        "{{ ().__class__.__base__.__subclasses__() }}",
        "{{ lipsum.__globals__['os'] }}",
    ],
)
def test_patched_sandboxes_block_inert_ssti_gadgets(
    environment: Sandbox,
    source: str,
) -> None:
    version = jinja2.__version__
    if _version_tuple(version) < MINIMUM_PATCHED_JINJA:
        pytest.skip(f"Jinja2 {version} predates the patched sandbox baseline")

    with pytest.raises(SecurityError) as caught:
        environment().from_string(source).render(
            messages=[{"role": "user", "content": "my password is x"}],
        )
    assert type(caught.value) is SecurityError, f"Jinja2 {version}"


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _dotted(call: ast.Call) -> str | None:
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


# The guard exists so a future refactor cannot quietly introduce execution into the
# scanner, which means its value is exactly the set of spellings it knows. Split in two
# because the right test differs: some names are dangerous whatever the receiver, while
# `loads`/`compile` are only dangerous from specific modules -- `json.loads` and
# `re.compile` are used throughout and must stay legal.
_BANNED_CALLS = frozenset({
    "SandboxedEnvironment",
    "ImmutableSandboxedEnvironment",
    "from_string",
    "render",
    "render_async",
    "run_path",
    "run_module",
    "system",
    "popen",
    "spawn",
    "check_output",
    "Popen",
    "CDLL",
})
_BANNED_DOTTED = frozenset({
    "builtins.exec", "builtins.eval", "builtins.compile", "builtins.__import__",
    "pickle.loads", "pickle.load", "marshal.loads", "marshal.load",
    "importlib.import_module", "codecs.decode",
    "subprocess.run", "subprocess.call", "subprocess.check_call",
})


def _violations(tree: ast.Module, relative: str) -> tuple[list[str], list[str]]:
    """Every banned execution spelling in ``tree``, plus the allow-listed parse-only ones.

    One implementation, used both to audit the shipped package and to prove the guard
    catches what it claims. A probe that reimplements the walk drifts from the guard it
    is supposed to certify.
    """

    violations: list[str] = []
    parse_only: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "jinja2.sandbox" for alias in node.names):
                violations.append(f"{relative}:{node.lineno}: import jinja2.sandbox")
        elif isinstance(node, ast.ImportFrom) and node.module == "jinja2.sandbox":
            violations.append(f"{relative}:{node.lineno}: from jinja2.sandbox import")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            dotted = _dotted(node)
            if name == "Environment":
                if (
                    relative == "c4nary/template_ast.py"
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "jinja2"
                ):
                    parse_only.append(f"{relative}:{node.lineno}")
                else:
                    violations.append(f"{relative}:{node.lineno}: {name}()")
            elif name in _BANNED_CALLS or dotted in _BANNED_DOTTED:
                violations.append(f"{relative}:{node.lineno}: {dotted or name}()")
            elif (
                isinstance(node.func, ast.Name)
                and name in {"compile", "exec", "eval", "__import__"}
            ):
                violations.append(f"{relative}:{node.lineno}: {name}()")
            elif name == "getattr" and any(
                isinstance(arg, ast.Constant)
                and arg.value in (_BANNED_CALLS | {"exec", "eval", "compile"})
                for arg in node.args[1:]
            ):
                # Spelling the name in a string is the obvious way past a
                # spelling-based guard: getattr(template, "render")().
                violations.append(f"{relative}:{node.lineno}: getattr(..., banned)")
    return violations, parse_only


def test_scanner_source_cannot_render_or_execute() -> None:
    violations: list[str] = []
    parse_only_environments: list[str] = []

    for path in sorted((ROOT / "c4nary").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found, parse_only = _violations(tree, path.relative_to(ROOT).as_posix())
        violations.extend(found)
        parse_only_environments.extend(parse_only)

    assert parse_only_environments == ["c4nary/template_ast.py:89"]
    assert not violations, violations


@pytest.mark.parametrize(
    "source",
    [
        "import jinja2.sandbox",
        "from jinja2.sandbox import SandboxedEnvironment",
        "env = Environment()",
        "SandboxedEnvironment()",
        "ImmutableSandboxedEnvironment()",
        "t.from_string(src)",
        "t.render(ctx)",
        "await t.render_async(ctx)",
        "exec(src)",
        "eval(src)",
        "compile(src, 'f', 'exec')",
        "__import__('os')",
        "builtins.exec(src)",
        "builtins.eval(src)",
        "pickle.loads(blob)",
        "marshal.loads(blob)",
        "runpy.run_path(p)",
        "runpy.run_module(m)",
        "os.system('id')",
        "os.popen('id')",
        "pty.spawn(['sh'])",
        "subprocess.run(['sh'])",
        "subprocess.Popen(['sh'])",
        "subprocess.check_output(['sh'])",
        "ctypes.CDLL('x.so')",
        "importlib.import_module('os')",
        "getattr(template, 'render')()",
        "getattr(builtins, 'exec')(src)",
    ],
)
def test_the_render_guard_catches_each_banned_spelling(source: str) -> None:
    """A guard nobody has watched fail is not evidence.

    Its worth is exactly the set of spellings it knows, so each one is asserted. The
    previous version knew only bare `exec` and a trailing `render`, which `builtins.exec`
    and `getattr(t, "render")` both walk straight past.
    """

    found, _ = _violations(ast.parse(source), "probe.py")
    assert found, f"render guard does not catch: {source}"


@pytest.mark.parametrize(
    "source",
    [
        "json.loads(text)",
        "json.load(fh)",
        "re.compile(pattern)",
        "jinja2.Environment()",
        "data.get('render')",
        "getattr(node, 'body', None)",
        "ast.parse(source)",
        "hashlib.sha256(b'').hexdigest()",
    ],
)
def test_the_render_guard_leaves_ordinary_calls_alone(source: str) -> None:
    """`json.loads` and `re.compile` are used throughout; a guard that trips on them
    would be turned off."""

    found, _ = _violations(ast.parse(source), "c4nary/template_ast.py")
    assert found == [], source
