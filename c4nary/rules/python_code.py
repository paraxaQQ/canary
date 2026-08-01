"""Static ast-only audit of auto_map-selected Python source."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Iterable, NamedTuple

from ..bundle import ReadResult, Reader, _safe_bundle_name
from ..coverage import Surface
from ..report import Finding
from .registry import finding


MAX_PYTHON_BYTES = 1 << 20
MAX_PYTHON_FETCHES = 32
# Uncapped, a hostile file yields ~69,800 findings and gigabytes of SARIF -- past what any
# consumer ingests, so the flood would delete the report. Benign modules produce single
# digits. Overflow is always reported as a count, never dropped.
MAX_FINDINGS_PER_ARTIFACT = 100

_BUILTIN_SINKS = frozenset({"eval", "exec", "compile", "__import__"})
_EXECUTION_SINKS = frozenset({
    "os.system",
    "os.popen",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "pty.spawn",
    "subprocess.Popen",
    "subprocess.run",
    "subprocess.call",
})
_CAPABILITY_SINKS = frozenset({
    "ctypes.CDLL",
    "subprocess.check_output",
    "socket.socket",
    "importlib.import_module",
})
_DECODERS = frozenset({
    "base64.b64decode",
    "base64.standard_b64decode",
    "base64.urlsafe_b64decode",
    "binascii.a2b_base64",
    "binascii.unhexlify",
    "bytes.fromhex",
    "bytearray.fromhex",
    "codecs.decode",
})
_BASE_OR_HEX_ENCODINGS = frozenset({
    "base16",
    "base32",
    "base64",
    "base85",
    "hex",
    "hex_codec",
})


@dataclass(frozen=True)
class _Analysis:
    findings: list[Finding]
    tree: ast.Module | None


def _rmt000(artifact: str, detail: str) -> Finding:
    return finding(
        "RMT000",
        detail,
        location=artifact,
        artifact=artifact,
    )


class _ImportBindingVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.bindings: dict[str, str] = {}
        # local name -> (sibling filename, name inside it, or None when the local name
        # binds the module object). Resolved in _resolve_relative, once the one-hop
        # follower has parsed the sibling.
        self.relative: dict[str, tuple[str, str | None]] = {}
        self.star: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".", 1)[0]
            self.bindings[local] = (
                alias.name if alias.asname else alias.name.split(".", 1)[0]
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level != 0:
            if node.module:
                child = f"{node.module}.py"
                for alias in node.names:
                    if alias.name == "*":
                        self.star.append(child)
                    else:
                        self.relative[alias.asname or alias.name] = (child, alias.name)
            else:
                for alias in node.names:  # `from . import sibling`
                    if alias.name != "*":
                        self.relative[alias.asname or alias.name] = (
                            f"{alias.name}.py", None,
                        )
            return
        if not node.module:
            return
        for alias in node.names:
            if alias.name != "*":
                self.bindings[alias.asname or alias.name] = (
                    f"{node.module}.{alias.name}"
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _import_bindings(nodes: Iterable[ast.stmt]) -> dict[str, str]:
    visitor = _ImportBindingVisitor()
    for node in nodes:
        visitor.visit(node)
    return visitor.bindings


def _relative_bindings(
    nodes: Iterable[ast.stmt],
) -> tuple[dict[str, tuple[str, str | None]], list[str]]:
    visitor = _ImportBindingVisitor()
    for node in nodes:
        visitor.visit(node)
    return visitor.relative, visitor.star


class _BindingVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.names.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _bound_names(nodes: Iterable[ast.stmt]) -> set[str]:
    visitor = _BindingVisitor()
    for node in nodes:
        visitor.visit(node)
    return visitor.names


def _dotted_parts(node: ast.expr) -> list[str] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return list(reversed(parts))


def _qualified_path(
    node: ast.expr,
    bindings: dict[str, str],
    shadowed: set[str],
    modules: dict[str, dict[str, str]] | None = None,
) -> str | None:
    parts = _dotted_parts(node)
    if not parts:
        return None
    root = parts[0]
    # `from . import sibling` binds a module object, so `sibling.system(...)` resolves
    # only through that sibling's own bindings; a root->string map cannot express it.
    if modules and root in modules and len(parts) > 1:
        target = modules[root].get(parts[1])
        if target is not None:
            return ".".join([target, *parts[2:]])
    if root in bindings:
        return ".".join([bindings[root], *parts[1:]])
    if root in {"bytes", "bytearray"} and root not in shadowed:
        return ".".join(parts)
    return None


def _builtin_sink(node: ast.expr, shadowed: set[str]) -> str | None:
    if (
        isinstance(node, ast.Name)
        and node.id in _BUILTIN_SINKS
        and node.id not in shadowed
    ):
        return node.id
    return None


def _decoder_path(
    call: ast.Call,
    bindings: dict[str, str],
    shadowed: set[str],
) -> str | None:
    path = _qualified_path(call.func, bindings, shadowed)
    if path not in _DECODERS:
        return None
    if path == "codecs.decode" and len(call.args) >= 2:
        encoding = call.args[1]
        if (
            isinstance(encoding, ast.Constant)
            and isinstance(encoding.value, str)
            and encoding.value.lower().replace("-", "_") not in _BASE_OR_HEX_ENCODINGS
        ):
            return None
    return path


def _is_module_dict(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("globals", "vars")
        and not node.args
    )


def _main_guard_defeated(tree: ast.Module) -> str | None:
    """Reason the ``__main__`` reachability premise fails for this module, else None.

    A file that writes ``__name__`` makes the main guard fire on import, so the skip
    would hide every sink behind one extra line of source. Only *writes* count: matching
    any mention of ``globals()`` fires on 1.9% of real site-packages modules.
    """

    if "__name__" in _bound_names(tree.body):
        return "rebinds `__name__` at module scope"
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        if any(
            isinstance(t, ast.Subscript) and _is_module_dict(t.value) for t in targets
        ):
            return "assigns into its own module dict"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("update", "setdefault")
            and _is_module_dict(node.func.value)
        ):
            return "mutates its own module dict"
    return None


def _is_main_guard(test: ast.expr) -> bool:
    """True only when ``test`` provably gates the body on ``__name__ == "__main__"``.

    Exact shape, never a structural walk: ``if __name__ != "__main__":`` is the branch
    that *always* runs under ``exec_module``, as are ``... or True`` and ``not (...)``.
    Only an ``and`` chain preserves the guarantee, since every operand must hold.
    """

    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return any(_is_main_guard(operand) for operand in test.values)
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


class _ImportTimeVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        artifact: str,
        bindings: dict[str, str],
        shadowed: set[str],
        main_guard_sound: bool = True,
        unresolved: dict[str, str] | None = None,
        module_bindings: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.artifact = artifact
        self.bindings = bindings
        self.shadowed = shadowed
        self.main_guard_sound = main_guard_sound
        # Unresolvable relative-import names, mapped to why. Reported only where one is
        # actually called: most relative imports pull in classes, and flagging every
        # unresolved name buries the signal.
        self.unresolved = unresolved or {}
        self.module_bindings = module_bindings or {}
        self.suppressed = 0
        self.findings: list[Finding] = []
        self._seen: set[tuple[str, int, str]] = set()

    # No early bail once capped: stopping the walk saves ~0.6s but makes the overflow
    # count always read 1, and a report that understates what it withheld is worse than
    # a slow one.
    def _append(self, rule_id: str, node: ast.AST, sink: str, detail: str) -> None:
        line = getattr(node, "lineno", 1)
        key = (rule_id, line, sink)
        if key in self._seen:
            return
        if len(self.findings) >= MAX_FINDINGS_PER_ARTIFACT:
            # Counted, not dropped: the caller turns this into one RMT000 naming the
            # total. A truncation that does not announce itself is a false clean.
            self.suppressed += 1
            return
        self._seen.add(key)
        self.findings.append(finding(
            rule_id,
            detail,
            location=f"{self.artifact}:L{line}",
            artifact=self.artifact,
            line=line,
            subject=sink,
        ))

    def _sink(self, expression: ast.expr) -> tuple[str, str] | None:
        builtin = _builtin_sink(expression, self.shadowed)
        if builtin is not None:
            return "RMT010", builtin
        path = _qualified_path(
            expression, self.bindings, self.shadowed, self.module_bindings,
        )
        if path in _EXECUTION_SINKS:
            return "RMT010", path
        if path in _CAPABILITY_SINKS:
            return "RMT011", path
        return None

    def _reference(self, expression: ast.expr) -> None:
        sink = self._sink(expression)
        if sink is None:
            return
        rule_id, path = sink
        self._append(
            rule_id,
            expression,
            path,
            f"Import-time decorator references {path}.",
        )

    def visit_Call(self, node: ast.Call) -> None:
        # No `shadowed` guard: the import binds these names, so every one of them is in
        # `shadowed` and checking it would suppress the breadcrumb entirely.
        root = _dotted_parts(node.func)
        if root and root[0] in self.unresolved:
            self._append(
                "RMT000",
                node,
                f"unresolved-import:{root[0]}",
                f"Import-time evaluation calls {'.'.join(root)}, bound by a relative "
                f"import this audit could not follow: {self.unresolved[root[0]]}. "
                f"Whatever it resolves to was not checked against the sink lists.",
            )
        sink = self._sink(node.func)
        if sink is not None:
            rule_id, path = sink
            self._append(
                rule_id,
                node,
                path,
                f"Import-time evaluation calls {path}.",
            )
            if rule_id == "RMT010":
                decoder = next(
                    (
                        decoder
                        for argument in [*node.args, *(kw.value for kw in node.keywords)]
                        for nested in ast.walk(argument)
                        if isinstance(nested, ast.Call)
                        and (decoder := _decoder_path(
                            nested,
                            self.bindings,
                            self.shadowed,
                        )) is not None
                    ),
                    None,
                )
                if decoder is not None:
                    self._append(
                        "RMT020",
                        node,
                        f"{decoder}->{path}",
                        f"Import-time evaluation passes content from {decoder} to {path}.",
                    )
        self.generic_visit(node)

    def _visit_function_header(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self._reference(decorator)
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_header(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_header(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)

    def visit_If(self, node: ast.If) -> None:
        # transformers loads the file with exec_module, so __name__ is the module's own
        # name and a main-guard body is unreachable. The test and the else branch still
        # run, so only the guarded body is skipped -- and only while the module leaves
        # __name__ alone, which is what main_guard_sound tracks.
        self.visit(node.test)
        if not self.main_guard_sound or not _is_main_guard(node.test):
            for statement in node.body:
                self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self._reference(decorator)
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        previous_shadowed = self.shadowed
        previous_bindings = self.bindings
        self.shadowed = previous_shadowed | _bound_names(node.body)
        self.bindings = {**previous_bindings, **_import_bindings(node.body)}
        for statement in node.body:
            self.visit(statement)
        self.bindings = previous_bindings
        self.shadowed = previous_shadowed


def _analyze_python_source(
    source: str,
    *,
    artifact: str,
    truncated: bool = False,
    max_bytes: int = MAX_PYTHON_BYTES,
) -> _Analysis:
    if truncated:
        return _Analysis(
            [_rmt000(
                artifact,
                f"{artifact} was truncated at the {max_bytes}-byte reader cap.",
            )],
            None,
        )
    try:
        size = len(source.encode("utf-8"))
    except MemoryError:
        return _Analysis(
            [_rmt000(artifact, f"{artifact} could not be sized because encoding exhausted memory.")],
            None,
        )
    if size > max_bytes:
        return _Analysis(
            [_rmt000(
                artifact,
                f"{artifact} is {size} bytes, above the {max_bytes}-byte audit cap.",
            )],
            None,
        )
    try:
        tree = ast.parse(source, filename=artifact)
    except (SyntaxError, RecursionError, MemoryError) as exc:
        return _Analysis(
            [_rmt000(
                artifact,
                f"{artifact} failed ast.parse with {type(exc).__name__}.",
            )],
            None,
        )

    try:
        rebind = _main_guard_defeated(tree)
        visitor = _ImportTimeVisitor(
            artifact=artifact,
            bindings=_import_bindings(tree.body),
            shadowed=_bound_names(tree.body),
            main_guard_sound=rebind is None,
        )
        visitor.visit(tree)
    except (RecursionError, MemoryError) as exc:
        # ast.parse builds left-nested expressions iteratively, so a file that parses
        # fine can still exhaust these walks -- ~1 KB of `1+1+...` does it. Fail closed
        # like the parse above, or the whole scan aborts and takes the template findings
        # with it.
        return _Analysis(
            [_rmt000(
                artifact,
                f"{artifact} exhausted the import-time walker with {type(exc).__name__}.",
            )],
            None,
        )

    findings = visitor.findings
    if visitor.suppressed:
        findings = [*findings, _rmt000(
            artifact,
            f"{artifact} produced more than {MAX_FINDINGS_PER_ARTIFACT} import-time "
            f"findings; {visitor.suppressed} further occurrence(s) were counted but not "
            f"listed. A file this dense with import-time execution needs reading, not a "
            f"longer report.",
        )]
    # The shadow set is flow-insensitive, so `exec = exec` -- or any Store of a sink name
    # anywhere at module scope, including after the call site -- silently retires the
    # whole builtin-sink family for the file. Leave the suppression (rebinding really can
    # make the name benign) but never let it happen without a row in the report.
    shadowed_sinks = sorted(_BUILTIN_SINKS & _bound_names(tree.body))
    if shadowed_sinks:
        findings = [_rmt000(
            artifact,
            f"{artifact} rebinds the builtin sink name(s) {', '.join(shadowed_sinks)} at "
            f"module scope; calls to them were not treated as import-time execution.",
        ), *findings]
    if rebind is not None:
        findings = [_rmt000(
            artifact,
            f"{artifact} {rebind}, so `if __name__ == \"__main__\":` bodies were audited "
            f"as import-time scope rather than skipped.",
        ), *findings]
    return _Analysis(findings, tree)


def analyze_python_source(
    source: str,
    *,
    artifact: str,
    truncated: bool = False,
    max_bytes: int = MAX_PYTHON_BYTES,
) -> list[Finding]:
    return _analyze_python_source(
        source,
        artifact=artifact,
        truncated=truncated,
        max_bytes=max_bytes,
    ).findings


def _auto_map_references(
    configs: Iterable[tuple[str, dict]],
) -> list[tuple[str, str, str]]:
    references: list[tuple[str, str, str]] = []
    for config_file, config in configs:
        auto_map = config.get("auto_map")
        if not isinstance(auto_map, dict):
            continue
        for key in sorted(auto_map):
            value = auto_map[key]
            values = value if isinstance(value, (list, tuple)) else (value,)
            for reference in values:
                if isinstance(reference, str) and reference:
                    references.append((config_file, str(key), reference))
    return references


def _reference_filename(reference: str) -> tuple[str | None, str | None]:
    if "--" in reference:
        repository, _local_reference = reference.split("--", 1)
        return None, repository
    if "." not in reference:
        raise ValueError(f"auto_map reference has no class separator: {reference!r}")
    module_name, _class_name = reference.rsplit(".", 1)
    if "." in module_name:
        # A dotted module (`pkg.sub.modeling_x.MyModel`) is a package path, not a
        # sibling file. Flattening it to `pkg.sub.modeling_x.py` would audit a file
        # the loader never opens, which is worse than not auditing -- same reasoning
        # as the first-separator `--` split above. Report the boundary instead.
        raise ValueError(
            f"auto_map reference names a dotted module path, which does not resolve "
            f"to a sibling bundle file: {reference!r}"
        )
    return _safe_bundle_name(f"{module_name}.py"), None


def _relative_imports(tree: ast.Module) -> list[tuple[int, str]]:
    edges: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level < 1:
            continue
        candidates = [node.module] if node.module else [
            alias.name
            for alias in node.names
            if alias.name != "*"
        ]
        for module_name in candidates:
            if module_name:
                edges.append((node.lineno, f"{module_name}.py"))
    return sorted(set(edges))


class _Relative(NamedTuple):
    resolved: dict[str, str]
    modules: dict[str, dict[str, str]]
    unresolved: dict[str, str]
    incomplete: bool


def _resolve_relative(
    tree: ast.Module,
    children: dict[str, ast.Module],
) -> _Relative:
    """Split a module's relative-import names into resolved bindings and dead ends.

    Without this, a sink split across two files -- ``from .utils import system`` beside
    ``from os import system`` -- resolves to nothing in either while both report
    ``examined``. Dead ends are returned so a call through one can still be reported.
    """

    resolved: dict[str, str] = {}
    modules: dict[str, dict[str, str]] = {}
    unresolved: dict[str, str] = {}
    incomplete = False
    relative, star = _relative_bindings(tree.body)

    # `from .sibling import *` re-exports the sibling's own import bindings under the
    # same local names, so folding them in resolves the star exactly rather than
    # guessing. Applied first so an explicit import below can override.
    for child_file in star:
        child = children.get(child_file)
        if child is None:
            incomplete = True
            continue
        resolved.update(_import_bindings(child.body))

    for local, (child_file, original) in sorted(relative.items()):
        child = children.get(child_file)
        if child is None:
            # No finding here. The sibling already carries its own absent / skipped /
            # unparseable coverage row, and the parent drops to `partial` below, so a
            # per-call-site breadcrumb would restate coverage 85 times per 1,185 real
            # modules without telling an operator anything new.
            incomplete = True
            continue
        child_relative, _ = _relative_bindings(child.body)
        if original is None:
            modules[local] = _import_bindings(child.body)
        elif original in child_relative:
            # The sibling got it from a relative import of its own -- that is the second
            # hop, past the documented bound. Not silently clean.
            unresolved[local] = (
                f"{original!r} reaches {child_file} through a further relative import"
            )
        elif (target := _import_bindings(child.body).get(original)) is not None:
            resolved[local] = target
        elif original in _bound_names(child.body):
            # Defined in the sibling rather than re-exported -- a class or function,
            # which is what nearly every real relative import pulls in. Not a dead end.
            continue
        else:
            unresolved[local] = (
                f"{original!r} is not bound at the top level of {child_file}"
            )
    return _Relative(resolved, modules, unresolved, incomplete)


def analyze_auto_map_python(
    configs: Iterable[tuple[str, dict]],
    read_text: Reader,
) -> tuple[list[Finding], list[Surface]]:
    findings: list[Finding] = []
    surfaces: dict[str, Surface] = {}
    primary_files: list[str] = []

    for config_file, key, reference in _auto_map_references(configs):
        try:
            filename, cross_repository = _reference_filename(reference)
        except ValueError as exc:
            findings.append(_rmt000(config_file, str(exc)))
            continue
        if cross_repository is not None:
            digest = hashlib.sha256(reference.encode("utf-8")).hexdigest()[:12]
            findings.append(finding(
                "RMT001",
                f"auto_map entry {key!r} targets repository {cross_repository!r}; "
                f"cross-repository Python was not fetched.",
                location=f"{config_file}:auto_map.{key}",
                artifact=config_file,
            ))
            surfaces[f"rmt.cross_repo.{digest}"] = Surface(
                f"rmt.cross_repo.{digest}",
                "skipped",
                f"cross-repository auto_map reference {reference!r} was not fetched.",
            )
        elif filename is not None:
            primary_files.append(filename)

    seen: set[str] = set()
    fetches = 0

    def analyze_file(filename: str) -> _Analysis | None:
        nonlocal fetches
        surface_id = f"rmt.{filename}"
        if filename in seen:
            return None
        seen.add(filename)
        if fetches >= MAX_PYTHON_FETCHES:
            surfaces[surface_id] = Surface(
                surface_id,
                "skipped",
                f"{filename} was not read because the {MAX_PYTHON_FETCHES}-file budget was exhausted.",
            )
            findings.append(_rmt000(
                filename,
                f"{filename} was not audited because the Python fetch budget was exhausted.",
            ))
            return None
        fetches += 1
        try:
            raw = read_text(
                _safe_bundle_name(filename),
                MAX_PYTHON_BYTES,
                python_only=True,
            )
        except ValueError as exc:
            surfaces[surface_id] = Surface(surface_id, "skipped", str(exc))
            findings.append(_rmt000(filename, str(exc)))
            return None
        if raw is None or raw == "":
            surfaces[surface_id] = Surface(
                surface_id,
                "absent",
                f"{filename} was referenced by auto_map or a relative import but was absent.",
            )
            return None
        result = raw if isinstance(raw, ReadResult) else ReadResult(raw)
        analysis = _analyze_python_source(
            result.text,
            artifact=filename,
            truncated=result.truncated,
        )
        if result.truncated:
            state = "partial"
            reason = f"{filename} was truncated and refused by the Python auditor."
        elif analysis.tree is None:
            state = "unparseable"
            reason = f"{filename} failed closed before a complete AST audit."
        elif any(
            f.rule_id == "RMT000" and "not listed" in f.detail
            for f in analysis.findings
        ):
            state = "partial"
            reason = (
                f"{filename} was parsed with ast.parse, but produced more findings than "
                f"the {MAX_FINDINGS_PER_ARTIFACT}-per-file report cap; the rest were "
                f"counted, not listed."
            )
        else:
            state = "examined"
            reason = f"{filename} was parsed with ast.parse and examined."
        surfaces[surface_id] = Surface(surface_id, state, reason)
        findings.extend(analysis.findings)
        return analysis

    primary_analyses: list[tuple[str, _Analysis]] = []
    for filename in sorted(set(primary_files)):
        analysis = analyze_file(filename)
        if analysis is not None and analysis.tree is not None:
            primary_analyses.append((filename, analysis))

    child_trees: dict[str, ast.Module] = {}
    for parent, analysis in primary_analyses:
        for line, raw_filename in _relative_imports(analysis.tree):
            edge_id = f"rmt.edge.{parent}.L{line}"
            try:
                filename = _safe_bundle_name(raw_filename)
            except ValueError as exc:
                surfaces[edge_id] = Surface(
                    edge_id,
                    "skipped",
                    f"relative import from {parent}:L{line} was not followed: {exc}",
                )
                continue
            if filename == parent:
                surfaces[edge_id] = Surface(
                    edge_id,
                    "skipped",
                    f"relative import cycle {parent}:L{line} -> {filename} was not followed.",
                )
                continue
            child = analyze_file(filename)
            if child is not None and child.tree is not None:
                child_trees[filename] = child.tree

    # Second walk of each primary, now that the siblings it imports from have been
    # parsed. Additive: only findings the first walk could not reach are kept.
    for parent, analysis in primary_analyses:
        rel = _resolve_relative(analysis.tree, child_trees)
        if not (rel.resolved or rel.modules or rel.unresolved or rel.incomplete):
            continue
        try:
            revisit = _ImportTimeVisitor(
                artifact=parent,
                bindings={**_import_bindings(analysis.tree.body), **rel.resolved},
                shadowed=_bound_names(analysis.tree.body),
                main_guard_sound=_main_guard_defeated(analysis.tree) is None,
                unresolved=rel.unresolved,
                module_bindings=rel.modules,
            )
            revisit.visit(analysis.tree)
        except (RecursionError, MemoryError) as exc:
            findings.append(_rmt000(
                parent,
                f"{parent} exhausted the import-time walker with {type(exc).__name__} "
                f"while resolving relative imports.",
            ))
            continue
        already = {(f.rule_id, f.line, f.subject) for f in analysis.findings}
        fresh = [
            f for f in revisit.findings
            if (f.rule_id, f.line, f.subject) not in already
        ]
        findings.extend(fresh)
        surface_id = f"rmt.{parent}"
        if (rel.incomplete or rel.unresolved) and surfaces.get(surface_id) is not None:
            surfaces[surface_id] = Surface(
                surface_id,
                "partial",
                f"{parent} was parsed with ast.parse and examined, but names it takes "
                f"from relative imports could not be resolved, so calls through them "
                f"were not checked against the sink lists.",
            )

    return findings, [surfaces[key] for key in sorted(surfaces)]
