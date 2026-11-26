"""Ollama ``Modelfile`` target for ``canary scan ... --as ollama``.

A Modelfile is a portable model definition: a required ``FROM`` plus optional
``PARAMETER``, ``TEMPLATE``, ``SYSTEM``, ``LICENSE``, ``MESSAGE``, ``REQUIRES``,
and ``CAPABILITY`` instructions. Its ``TEMPLATE`` is Go ``text/template``, not
Jinja2, so this target runs only the engine-agnostic text checks from the
template family; Go semantics are a later band.

The parser is deliberately narrow and read-only:

* ``FROM`` is resolved to a kind (model reference, GGUF file, GGUF shard glob,
  or Safetensors directory) but the referenced artifact is never opened. The
  coverage row says so rather than implying the model was audited.
* A value may be triple-quoted across lines, single-line, or quoted; a ``#``
  comment is only a comment at the start of a line outside a value.
* Ambiguity is recorded, not hidden: duplicate single-valued instructions keep
  the last value and add a coverage note. Only shapes Ollama cannot interpret at
  all raise :class:`OllamaError`.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass, replace

from .go_template import GoTemplate, parse_go_template
from .report import Finding
from .rules.template import analyze_template_text
from .template_ast import template_sha256

MAX_MODELFILE_BYTES = 4 << 20

# The documented set plus the instructions Ollama's parser also accepts. The
# extras are recorded but not resolved by A4a; rejecting a file Ollama accepts
# would be worse than an honest "not analyzed" note.
_INSTRUCTION_NAMES = (
    "FROM", "PARAMETER", "TEMPLATE", "SYSTEM", "LICENSE", "MESSAGE",
    "REQUIRES", "CAPABILITY", "ADAPTER", "DRAFT", "RENDERER", "PARSER",
)
_UNRESOLVED_INSTRUCTIONS = frozenset({"ADAPTER", "DRAFT", "RENDERER", "PARSER"})
_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})

# Names come from Ollama's ``api.Options``/``api.Runner`` json tags plus the
# parser's accepted-deprecated list. The reference page documents a subset, so
# validating only against the page mislabels valid files (num_gpu,
# presence_penalty, penalize_newline in the real-world corpus).
_PARAMETER_TYPES = {
    "num_keep": "int",
    "seed": "int",
    "num_predict": "int",
    "top_k": "int",
    "top_p": "float",
    "min_p": "float",
    "repeat_last_n": "int",
    "temperature": "float",
    "repeat_penalty": "float",
    "presence_penalty": "float",
    "frequency_penalty": "float",
    "stop": "string",
    "num_ctx": "int",
    "num_batch": "int",
    "num_gpu": "int",
    "main_gpu": "int",
    "use_mmap": "bool",
    "num_thread": "int",
    "draft_num_predict": "int",
    # Accepted by the parser with a deprecation warning.
    "penalize_newline": "bool",
    "low_vram": "bool",
    "f16_kv": "bool",
    "logits_all": "bool",
    "vocab_only": "bool",
    "use_mlock": "bool",
    "mirostat": "int",
    "mirostat_tau": "float",
    "mirostat_eta": "float",
}
_REPEATABLE_PARAMETERS = frozenset({"stop"})
_BOOL_VALUES = frozenset({"1", "0", "t", "f", "true", "false"})
_REJECTED_PARAMETERS = {
    "typical_p": "rejected by Ollama as a model parameter; pass it as a request option",
}

_REQUIRES_RE = re.compile(r"\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.-]+)?")
_CAPABILITY_RE = re.compile(r"[a-z][a-z0-9_-]*")
_MODEL_REF_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"(?:/[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)*"
    r"(?::[A-Za-z0-9._-]+)?"
)
_GLOB_CHARS = frozenset("*?[")


class OllamaError(ValueError):
    """A Modelfile target that cannot be scanned."""


@dataclass(frozen=True)
class Instruction:
    name: str
    value: str
    line: int
    role: str | None = None
    parameter: str | None = None


@dataclass(frozen=True)
class FromSpec:
    """One resolved ``FROM`` value. ``kind`` is model/gguf/gguf_glob/
    safetensors_dir/unresolved."""

    raw: str
    kind: str
    paths: tuple[str, ...]
    reason: str
    line: int
    outside_root: bool = False
    partial: bool = False


@dataclass(frozen=True)
class Modelfile:
    display: str
    path: str
    root: str
    sha256: str
    instructions: tuple[Instruction, ...]
    from_specs: tuple[FromSpec, ...]
    template: str | None
    template_line: int | None
    template_analysis: GoTemplate | None
    system: str | None
    system_line: int | None
    messages: tuple[Instruction, ...]
    license: str | None
    requires: str | None
    capabilities: tuple[str, ...]
    parameters: tuple[Instruction, ...]
    parameter_issues: tuple[str, ...]
    notes: tuple[str, ...]
    invalid_utf8: bool
    line_count: int

    @property
    def template_sha256(self) -> str | None:
        return template_sha256(self.template) if self.template is not None else None


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


_LINE_RE = re.compile(r"(\S+)(?:\s+(.*))?$", re.DOTALL)


def _split_head(value: str) -> tuple[str, str]:
    """Split the first whitespace-separated token from the rest (space or tab)."""
    parts = re.split(r"[ \t]+", value.strip(), maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _split_instruction(line: str, line_no: int) -> tuple[str, str]:
    match = _LINE_RE.match(line)
    name = match.group(1).upper() if match is not None else ""
    rest = (match.group(2) or "") if match is not None else ""
    if name not in _INSTRUCTION_NAMES:
        if name == "STOP":
            raise OllamaError(
                f"line {line_no}: STOP is not a Modelfile instruction; use "
                f"PARAMETER stop")
        raise OllamaError(
            f"line {line_no}: unknown Modelfile instruction {name!r}; expected one "
            f"of {', '.join(_INSTRUCTION_NAMES)}")
    return name, rest.strip()


def _closed_triple(text: str) -> str | None:
    """Return the text before a line-ending ``\"\"\"``, or None.

    Ollama's parser closes a quoted value only when the whole accumulated value
    ends with the quote (after trimming), so an internal ``\"\"\"`` stays
    content. Matching that keeps a template that prints triple quotes intact.
    """
    stripped = text.rstrip()
    return stripped[:-3] if stripped.endswith('"""') else None


def _read_value(
    rest: str,
    lines: list[str],
    index: int,
    line_no: int,
) -> tuple[str, int]:
    """Read an instruction value, consuming lines for a quoted block."""
    rest = rest.strip()
    if not rest:
        raise OllamaError(f"line {line_no}: instruction requires a value")
    if not rest.startswith('"'):
        return rest, index

    if rest.startswith('"""'):
        body = rest[3:]
        closed = _closed_triple(body)
        if closed is not None:
            return closed, index
        chunks: list[str] = [body] if body else []
        while index < len(lines):
            nxt = lines[index]
            index += 1
            closed = _closed_triple(nxt)
            if closed is not None:
                chunks.append(closed)
                return "\n".join(chunks), index
            chunks.append(nxt)
        raise OllamaError(f"line {line_no}: unterminated triple-quoted value")

    stripped = rest.rstrip()
    if len(stripped) >= 2 and stripped.endswith('"'):
        return stripped[1:-1], index
    chunks = [rest[1:]]
    while index < len(lines):
        nxt = lines[index]
        index += 1
        stripped = nxt.rstrip()
        if stripped.endswith('"'):
            chunks.append(stripped[:-1])
            return "\n".join(chunks), index
        chunks.append(nxt)
    raise OllamaError(f"line {line_no}: unterminated quoted value")


def _normalize_lines(text: str) -> list[str]:
    """Split only on newlines; U+2028 and other separators stay as content."""
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _count_lines(text: str) -> int:
    lines = _normalize_lines(text)
    return len(lines) - 1 if lines and lines[-1] == "" else len(lines)


def _parse_instructions(text: str) -> list[Instruction]:
    lines = _normalize_lines(text)
    out: list[Instruction] = []
    index = 0
    while index < len(lines):
        raw = lines[index]
        index += 1
        line_no = index
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith('"""'):
            raise OllamaError(
                f"line {line_no}: triple-quoted block has no instruction")

        name, rest = _split_instruction(stripped, line_no)
        if name == "PARAMETER":
            parameter, remainder = _split_head(rest)
            parameter = parameter.lower()
            if not parameter:
                raise OllamaError(f"line {line_no}: PARAMETER requires a name")
            value, index = _read_value(remainder, lines, index, line_no)
            out.append(Instruction(name, value, line_no, parameter=parameter))
        elif name == "MESSAGE":
            role, remainder = _split_head(rest)
            role = role.lower()
            if role not in _MESSAGE_ROLES:
                raise OllamaError(
                    f"line {line_no}: MESSAGE role {role!r} is not one of "
                    f"{', '.join(sorted(_MESSAGE_ROLES))}")
            value, index = _read_value(remainder, lines, index, line_no)
            out.append(Instruction(name, value, line_no, role=role))
        else:
            value, index = _read_value(rest, lines, index, line_no)
            out.append(Instruction(name, value, line_no))
    return out


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath(
            [os.path.normcase(path), os.path.normcase(root)]
        ) == os.path.normcase(root)
    except ValueError:
        return False


def _resolve_from(raw: str, root: str, line: int) -> FromSpec:
    if "/blobs/sha256-" in raw.replace("\\", "/"):
        return FromSpec(
            raw, "unresolved", (),
            "an Ollama model-store blob path; the blob was not read by A4a", line)
    if os.name == "nt" and raw.startswith("/"):
        return FromSpec(
            raw, "unresolved", (),
            "a POSIX absolute path cannot be resolved on this platform; the "
            "referenced artifact was not read", line)
    expanded = os.path.expanduser(raw)
    absolute = expanded if os.path.isabs(expanded) else os.path.join(root, expanded)
    resolved = os.path.normpath(absolute)

    if os.path.isdir(resolved):
        try:
            names = sorted(
                name for name in os.listdir(resolved)
                if name.lower().endswith(".safetensors") and not name.startswith(".")
            )
        except OSError as exc:
            return FromSpec(raw, "unresolved", (), f"directory cannot be listed: {exc}", line)
        paths = tuple(os.path.join(resolved, name) for name in names)
        if paths:
            return FromSpec(
                raw, "safetensors_dir", paths,
                f"directory contains {len(paths)} top-level .safetensors file(s); "
                f"the weights were not read", line)
        return FromSpec(
            raw, "safetensors_dir", (resolved,),
            "directory contains no top-level .safetensors file; nothing would be "
            "imported from it", line, partial=True)

    if any(ch in expanded for ch in _GLOB_CHARS):
        matches = sorted(glob.glob(resolved))
        files = [path for path in matches if os.path.isfile(path)]
        ggufs = [path for path in files if path.lower().endswith(".gguf")]
        if not files:
            return FromSpec(raw, "unresolved", (), "the wildcard matched no file", line)
        if not ggufs:
            return FromSpec(
                raw, "unresolved", (),
                f"the wildcard matched {len(files)} file(s), but none end in .gguf",
                line)
        if len(ggufs) != len(files):
            return FromSpec(
                raw, "gguf_glob", tuple(ggufs),
                f"the wildcard matched {len(files)} file(s), but only {len(ggufs)} "
                f"end in .gguf; non-GGUF matches were ignored", line,
                partial=True)
        return FromSpec(
            raw, "gguf_glob", tuple(ggufs),
            f"the wildcard matched {len(ggufs)} GGUF shard(s); the shards were not read",
            line)

    if resolved.lower().endswith(".gguf"):
        if os.path.isfile(resolved):
            return FromSpec(
                raw, "gguf", (resolved,),
                "the GGUF file exists; it was not read by the Modelfile scan", line)
        return FromSpec(raw, "unresolved", (), "the GGUF file does not exist", line)

    if _MODEL_REF_RE.fullmatch(raw):
        return FromSpec(
            raw, "model", (),
            "an Ollama model reference; local model-store resolution is not "
            "implemented in A4a, so the referenced model was not read", line)
    return FromSpec(raw, "unresolved", (), "the value is not a file, directory, or model reference", line)


def _outside_root(raw: str, root: str) -> bool:
    expanded = os.path.expanduser(raw)
    absolute = expanded if os.path.isabs(expanded) else os.path.join(root, expanded)
    return not _is_within(os.path.normpath(absolute), root)


_PATH_KINDS = frozenset({"gguf", "gguf_glob", "safetensors_dir"})


def _resolve_from_specs(raw_specs: list[tuple[str, int]], root: str) -> tuple[FromSpec, ...]:
    def resolved(raw: str, line: int) -> FromSpec:
        spec = _resolve_from(raw, root, line)
        return replace(
            spec,
            outside_root=(
                spec.kind in _PATH_KINDS and _outside_root(raw, root)),
        )

    specs = tuple(resolved(raw, line) for raw, line in raw_specs)
    if len(specs) > 1 and any(spec.kind != "gguf" for spec in specs):
        raise OllamaError(
            "multiple FROM instructions are only supported for GGUF shards; "
            f"found {len(specs)} values with kinds "
            f"{', '.join(sorted({spec.kind for spec in specs}))}")
    return specs


def _validate_parameters(
    parameters: tuple[Instruction, ...],
) -> tuple[str, ...]:
    issues: list[str] = []
    seen: dict[str, int] = {}
    for parameter in parameters:
        name = parameter.parameter or ""
        value = parameter.value.strip()
        seen[name] = seen.get(name, 0) + 1
        if name in _REJECTED_PARAMETERS:
            issues.append(f"{name}: {_REJECTED_PARAMETERS[name]}")
            continue
        kind = _PARAMETER_TYPES.get(name)
        if kind == "int":
            try:
                int(value)
            except ValueError:
                issues.append(f"{name}: expected an integer, got {value!r}")
        elif kind == "float":
            try:
                parsed = float(value)
            except ValueError:
                issues.append(f"{name}: expected a number, got {value!r}")
            else:
                if parsed != parsed or parsed in (float("inf"), float("-inf")):
                    issues.append(f"{name}: expected a finite number, got {value!r}")
        elif kind == "bool":
            if value.lower() not in _BOOL_VALUES:
                issues.append(f"{name}: expected a boolean, got {value!r}")
        elif kind is None:
            issues.append(f"{name}: not a known Ollama parameter name")
    for name, count in sorted(seen.items()):
        if count > 1 and name not in _REPEATABLE_PARAMETERS:
            issues.append(f"{name}: repeated {count} times; the last value wins")
    return tuple(issues)


def _single(
    instructions: list[Instruction],
    name: str,
    notes: list[str],
) -> Instruction | None:
    found = [instruction for instruction in instructions if instruction.name == name]
    if not found:
        return None
    if len(found) > 1:
        notes.append(
            f"{len(found)} {name} instructions; the last (line {found[-1].line}) is "
            f"the effective one")
    return found[-1]


def load_modelfile(target: str) -> Modelfile:
    """Parse a local Modelfile. Never resolves a model-store reference."""
    if target.startswith("ollama://"):
        raise OllamaError(
            f"local Ollama model-store resolution is not implemented "
            f"({target}); pass a Modelfile path instead")
    path = os.fspath(target)
    if not os.path.exists(path) and _MODEL_REF_RE.fullmatch(path):
        raise OllamaError(
            f"Ollama model references are not implemented in A4a ({path}); "
            f"pass a Modelfile path instead")
    if os.path.isdir(path):
        raise OllamaError(f"an Ollama target must be a Modelfile, not a directory: {path}")
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_MODELFILE_BYTES + 1)
    except OSError as exc:
        raise OllamaError(f"cannot read Modelfile {path}: {exc}") from exc
    if len(data) > MAX_MODELFILE_BYTES:
        raise OllamaError(
            f"Modelfile exceeds the {MAX_MODELFILE_BYTES} byte reader cap: {path}")
    if data[:4] == b"GGUF":
        raise OllamaError(
            f"{path} is a GGUF file, not an Ollama Modelfile; drop --as ollama")
    if b"\x00" in data:
        raise OllamaError(f"{path} is not a text Modelfile (NUL bytes present)")
    try:
        text = data.decode("utf-8")
        invalid_utf8 = False
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        invalid_utf8 = True
    if text.startswith("\ufeff"):
        text = text[1:]

    instructions = _parse_instructions(text)
    froms = [
        (instruction.value, instruction.line)
        for instruction in instructions if instruction.name == "FROM"
    ]
    if not froms:
        raise OllamaError("the Modelfile has no FROM instruction, which Ollama requires")
    for raw, line in froms:
        if not raw.strip():
            raise OllamaError(f"line {line}: FROM requires a value")

    root = os.path.dirname(os.path.abspath(path))
    notes: list[str] = []
    unresolved = sorted({
        instruction.name for instruction in instructions
        if instruction.name in _UNRESOLVED_INSTRUCTIONS
    })
    if unresolved:
        notes.append(
            f"{', '.join(unresolved)} present but not resolved or analyzed by A4a")
    template = _single(instructions, "TEMPLATE", notes)
    system = _single(instructions, "SYSTEM", notes)
    requires = _single(instructions, "REQUIRES", notes)
    parameters = tuple(
        instruction for instruction in instructions if instruction.name == "PARAMETER")

    # Ollama keeps every LICENSE command; concatenate them rather than pretending
    # the last one replaced the rest.
    licenses = [
        instruction.value for instruction in instructions if instruction.name == "LICENSE"
    ]
    license_text = "\n\n".join(licenses) if licenses else None

    requires_value: str | None = None
    if requires is not None:
        requires_value = requires.value.strip()
        if requires_value[:1] in ("v", "V"):
            requires_value = requires_value[1:]
        if not _REQUIRES_RE.fullmatch(requires_value):
            notes.append(f"REQUIRES value {requires.value!r} is not a version number")
    capabilities = tuple(
        instruction.value.strip() for instruction in instructions
        if instruction.name == "CAPABILITY")
    for capability in capabilities:
        if not _CAPABILITY_RE.fullmatch(capability):
            notes.append(f"CAPABILITY value {capability!r} is not a plain token")

    template_text = template.value if template is not None else None
    return Modelfile(
        display=path,
        path=path,
        root=root,
        sha256=hashlib.sha256(data).hexdigest(),
        instructions=tuple(instructions),
        from_specs=_resolve_from_specs(froms, root),
        template=template_text,
        template_line=template.line if template is not None else None,
        template_analysis=parse_go_template(template_text),
        system=system.value if system is not None else None,
        system_line=system.line if system is not None else None,
        messages=tuple(
            instruction for instruction in instructions if instruction.name == "MESSAGE"),
        license=license_text,
        requires=requires_value,
        capabilities=capabilities,
        parameters=parameters,
        parameter_issues=_validate_parameters(parameters),
        notes=tuple(notes),
        invalid_utf8=invalid_utf8,
        line_count=_count_lines(text),
    )


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


def analyze_modelfile(modelfile: Modelfile) -> list[Finding]:
    """Run the text checks and Go template rules on every model-facing value."""
    findings: list[Finding] = []

    if modelfile.template is not None:
        findings += analyze_template_text(
            modelfile.template,
            prefix=f"TEMPLATE:L{modelfile.template_line}",
            label="TEMPLATE",
        )
        if modelfile.template_analysis is not None:
            findings += modelfile.template_analysis.findings()
    if modelfile.system is not None:
        findings += analyze_template_text(
            modelfile.system,
            prefix=f"SYSTEM:L{modelfile.system_line}",
            label="SYSTEM",
            include_size=False,
        )
    for message in modelfile.messages:
        # Example messages legitimately contain links and instruction-like text;
        # only concealment is meaningful on this surface.
        findings += analyze_template_text(
            message.value,
            prefix=f"MESSAGE[{message.role}]:L{message.line}",
            label=f"MESSAGE[{message.role}]",
            include_size=False,
            include_urls=False,
        )
    if modelfile.license is not None:
        # A license legitimately carries URLs and imperative legal prose. Only
        # concealment (TPL024/025/026) is meaningful on this surface.
        findings += analyze_template_text(
            modelfile.license,
            prefix="LICENSE",
            label="LICENSE",
            include_size=False,
            include_urls=False,
            include_lexicon=False,
        )
    return findings


# --------------------------------------------------------------------------- #
# Shared rule pass
# --------------------------------------------------------------------------- #


def scan_ollama(
    modelfile: Modelfile,
    *,
    policy=None,
    baseline=None,
    policy_requested: bool = False,
    baseline_requested: bool = False,
):
    """Run the Modelfile rule pass shared by the CLI and the MCP ``scan`` tool."""
    from .coverage import build_ollama_coverage
    from .policy import apply_report_policy, load_baseline, load_policy

    if policy is None:
        policy = load_policy(None)
    if baseline is None:
        baseline = load_baseline(None)

    findings = analyze_modelfile(modelfile)
    registered = [finding.severity for finding in findings]
    findings = apply_report_policy(findings, policy, baseline)
    suppressed_count = sum(finding.suppressed for finding in findings)
    retiered_count = sum(
        before != after.severity for before, after in zip(registered, findings)
    )
    coverage = build_ollama_coverage(
        modelfile,
        baseline_requested=baseline_requested,
        suppressed_count=suppressed_count,
        policy_requested=policy_requested,
        retiered_count=retiered_count,
    )
    return findings, coverage
