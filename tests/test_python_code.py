from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from _ggufgen import write_gguf

from c4nary import cli
from c4nary.bundle import ReadResult, _safe_bundle_name
from c4nary.report import FAIL
from c4nary.rules import python_code
from c4nary.rules.python_code import (
    MAX_FINDINGS_PER_ARTIFACT,
    MAX_PYTHON_BYTES,
    _reference_filename,
    analyze_auto_map_python,
    analyze_python_source,
)
from c4nary.rules.registry import all_rules


FIXTURES = Path(__file__).parent / "fixtures" / "rmt"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_benign_transformers_style_source_is_clean() -> None:
    assert analyze_python_source(
        _read("benign_modeling.py"),
        artifact="benign_modeling.py",
    ) == []


def test_hostile_import_time_fixture_has_multiple_hits() -> None:
    findings = analyze_python_source(
        _read("hostile_modeling.py"),
        artifact="hostile_modeling.py",
    )

    assert len(findings) >= 5
    assert "RMT010" in {finding.rule_id for finding in findings}
    assert "RMT020" in {finding.rule_id for finding in findings}
    assert all(finding.severity != FAIL for finding in findings)


def test_disjoint_builtin_and_import_binding_matchers() -> None:
    source = """
import os as operating_system
import re
import torch
import pandas
from os import system

re.compile("x")
torch.compile(lambda: None)
pandas.eval("1 + 1")
system("id")
operating_system.popen("id")
"""
    findings = analyze_python_source(source, artifact="modeling.py")
    details = [finding.detail for finding in findings]

    assert len(findings) == 2
    assert any("os.system" in detail for detail in details)
    assert any("os.popen" in detail for detail in details)
    assert all("re.compile" not in detail for detail in details)
    assert all("torch.compile" not in detail for detail in details)
    assert all("pandas.eval" not in detail for detail in details)


def test_shadowed_builtin_and_function_body_are_not_import_time_hits() -> None:
    """Shadowing suppresses the sink, but never silently.

    The shadow set is flow-insensitive, so `exec = exec` would otherwise retire the
    whole builtin-sink family for a file while coverage still reported it examined.
    The suppression stands; it just leaves an RMT000 row behind.
    """

    source = """
eval = lambda value: value
eval("clean")

def later():
    import os
    os.system("not import time")
"""
    findings = analyze_python_source(source, artifact="modeling.py")

    assert [f.rule_id for f in findings] == ["RMT000"]
    assert "eval" in findings[0].detail


def test_a_file_that_shadows_nothing_stays_completely_clean() -> None:
    source = """
import os

VALUE = os.getcwd()

def later():
    os.system("not import time")
"""
    assert analyze_python_source(source, artifact="modeling.py") == []


def test_function_local_import_does_not_bind_module_scope() -> None:
    source = """
def later():
    import os as helper

helper.system("not a resolved module binding")
"""
    assert analyze_python_source(source, artifact="modeling.py") == []


def test_class_body_import_is_resolved_in_class_scope() -> None:
    source = """
class Loader:
    import os as operating_system
    result = operating_system.system("id")
"""
    findings = analyze_python_source(source, artifact="modeling.py")

    assert len(findings) == 1
    assert findings[0].rule_id == "RMT010"
    assert "os.system" in findings[0].detail


def test_permanently_warn_capability_sinks_are_separate() -> None:
    source = """
import ctypes
import importlib
import socket
import subprocess

ctypes.CDLL("kernel.so")
importlib.import_module("optional")
socket.socket()
subprocess.check_output(["nvcc", "--version"])
"""
    findings = analyze_python_source(source, artifact="modeling.py")

    assert len(findings) == 4
    assert {finding.rule_id for finding in findings} == {"RMT011"}


@pytest.mark.parametrize(
    ("source", "truncated", "max_bytes"),
    [
        ("if :", False, MAX_PYTHON_BYTES),
        ("x" * 64, False, 32),
        ("value = 1", True, MAX_PYTHON_BYTES),
    ],
)
def test_fail_closed_source_states(
    source: str,
    truncated: bool,
    max_bytes: int,
) -> None:
    findings = analyze_python_source(
        source,
        artifact="modeling.py",
        truncated=truncated,
        max_bytes=max_bytes,
    )
    assert [finding.rule_id for finding in findings] == ["RMT000"]


@pytest.mark.parametrize("error", [RecursionError, MemoryError])
def test_ast_parse_resource_errors_fail_closed(monkeypatch, error) -> None:
    def fail(*args, **kwargs):
        raise error()

    monkeypatch.setattr(python_code.ast, "parse", fail)
    findings = analyze_python_source("value = 1", artifact="modeling.py")
    assert [finding.rule_id for finding in findings] == ["RMT000"]


@pytest.mark.parametrize(
    "name",
    ["..", "../x.py", "a/b.py", "a\\b.py", ".hidden.py", "model.json"],
)
def test_safe_bundle_name_rejects_unsafe_python_names(name: str) -> None:
    with pytest.raises(ValueError):
        _safe_bundle_name(name)


def test_auto_map_uses_first_cross_repo_separator() -> None:
    assert _reference_filename("org/repo--modeling--Model") == (None, "org/repo")
    assert _reference_filename("modeling_x.Model") == ("modeling_x.py", None)


def test_one_hop_relative_import_is_audited() -> None:
    requested: list[str] = []

    def reader(
        name: str,
        max_bytes: int,
        *,
        python_only: bool,
    ) -> ReadResult | None:
        requested.append(name)
        path = FIXTURES / name
        return ReadResult(path.read_text(encoding="utf-8")) if path.is_file() else None

    findings, surfaces = analyze_auto_map_python(
        [("config.json", {"auto_map": {"AutoModel": "modeling_x.Model"}})],
        reader,
    )

    assert requested == ["modeling_x.py", "utils_x.py"]
    assert any(
        finding.rule_id == "RMT010" and "os.system" in finding.detail
        for finding in findings
    )
    assert {surface.state for surface in surfaces} == {"examined"}


def test_cross_repo_reference_is_recorded_but_never_fetched() -> None:
    def reader(*args, **kwargs):
        raise AssertionError("cross-repository source must not be fetched")

    findings, surfaces = analyze_auto_map_python(
        [("config.json", {"auto_map": {
            "AutoModel": "other/repo--modeling_x.Model",
        }})],
        reader,
    )

    assert [finding.rule_id for finding in findings] == ["RMT001"]
    assert [surface.state for surface in surfaces] == ["skipped"]


def test_python_syntax_is_version_bounded() -> None:
    findings = analyze_python_source(
        _read("syntax_312.py"),
        artifact="syntax_312.py",
    )
    if sys.version_info >= (3, 12):
        assert findings == []
    else:
        assert [finding.rule_id for finding in findings] == ["RMT000"]


def test_rmt_registers_no_fail_rule() -> None:
    rmt_rules = [rule for rule in all_rules() if rule.rule_id.startswith("RMT")]
    assert rmt_rules
    assert all(rule.severity != FAIL for rule in rmt_rules)


def test_all_three_bundle_sinks_call_the_python_name_validator() -> None:
    for relative in ("c4nary/cli.py", "c4nary/mcp_server.py", "c4nary/remote.py"):
        tree = ast.parse(Path(relative).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_safe_bundle_name"
        ]
        assert calls, relative


def test_local_reader_reports_cleanly_parsed_truncation(tmp_path, capsys) -> None:
    model = write_gguf(tmp_path / "model.gguf", {})
    (tmp_path / "config.json").write_text(json.dumps({
        "auto_map": {"AutoModel": "modeling_x.Model"},
    }), encoding="utf-8")
    (tmp_path / "modeling_x.py").write_text(
        "value = 1\n" + ("x = 1\n" * (MAX_PYTHON_BYTES // 3)),
        encoding="utf-8",
    )

    assert cli.main(["scan", str(model), "--bundle", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert any(
        finding["rule_id"] == "RMT000"
        and finding["artifact"] == "modeling_x.py"
        for finding in payload["findings"]
    )
    assert not list(tmp_path.rglob("__pycache__"))


def test_main_guard_body_is_not_import_time_scope() -> None:
    """`if __name__ == "__main__":` never fires under trust_remote_code.

    exec_module leaves __name__ as the module's own name, so the guarded body is
    unreachable. Calibration found this false positive twice, both `pty.spawn` under a
    compound guard -- and pty.spawn is a FAIL promotion candidate.
    """

    plain = 'import pty\nif __name__ == "__main__":\n    pty.spawn(["sh"])\n'
    compound = (
        'import sys, pty\n'
        'if sys.platform != "win32" and __name__ == "__main__":\n'
        '    pty.spawn(["sh"])\n'
    )
    for source in (plain, compound):
        assert analyze_python_source(source, artifact="m.py") == []

    # the else branch and the test itself DO run on import.
    with_else = (
        'import os\n'
        'if __name__ == "__main__":\n'
        '    pass\n'
        'else:\n'
        '    os.system("id")\n'
    )
    assert [f.rule_id for f in analyze_python_source(with_else, artifact="m.py")] == ["RMT010"]

    # an unguarded conditional is still import-time scope.
    unguarded = 'import sys, os\nif sys.platform == "linux":\n    os.system("id")\n'
    assert [f.rule_id for f in analyze_python_source(unguarded, artifact="m.py")] == ["RMT010"]


@pytest.mark.parametrize(
    "test_source",
    [
        'if __name__ != "__main__":',            # the branch that ALWAYS runs
        'if __name__ == "__main__" or True:',    # or-chain does not gate the body
        'if not (__name__ == "__main__"):',      # negated
        'if str(__name__ == "__main__"):',       # comparison buried in a call
        'if __name__ is not "__main__":',        # identity, not equality
    ],
)
def test_main_guard_lookalikes_do_not_suppress(test_source: str) -> None:
    """The guard must match an exact shape, never a structural walk.

    A walk-based match reads `__name__ != "__main__"` as a main guard -- but under
    exec_module that is the branch that always runs, so it would be a two-character
    bypass of the entire family, and the coverage manifest would still certify the
    file as examined.
    """

    source = f'import os\n{test_source}\n    os.system("id")\n'
    assert [f.rule_id for f in analyze_python_source(source, artifact="m.py")] == ["RMT010"]


def test_dotted_module_reference_is_reported_not_flattened() -> None:
    """A package path must not be flattened into a sibling filename.

    `pkg.sub.modeling_x.MyModel` previously derived `pkg.sub.modeling_x.py`, a file the
    loader never opens -- the same failure the first-separator `--` split avoids. The
    caller turns this into an RMT000 breadcrumb rather than auditing the wrong file.
    """

    assert _reference_filename("modeling_x.MyModel") == ("modeling_x.py", None)
    for reference in ("pkg.sub.modeling_x.MyModel", "a.b.c.MyModel"):
        with pytest.raises(ValueError, match="dotted module path"):
            _reference_filename(reference)


def test_dotted_module_surfaces_as_rmt000() -> None:
    findings, _surfaces = analyze_auto_map_python(
        [("config.json", {"auto_map": {"AutoModel": "pkg.sub.modeling_x.MyModel"}})],
        lambda name, max_bytes, python_only=False: ReadResult(None, False),
    )
    assert [f.rule_id for f in findings] == ["RMT000"]


@pytest.mark.parametrize(
    ("label", "rebind"),
    [
        ("assignment", '__name__ = "__main__"'),
        ("subscript", 'globals()["__name__"] = "__main__"'),
        ("update", 'vars().update(__name__="__main__")'),
    ],
)
def test_main_guard_skip_is_withdrawn_when_the_module_writes_its_own_name(
    label: str,
    rebind: str,
) -> None:
    """One extra line must not blind the whole family.

    The skip is sound only because `exec_module` leaves `__name__` set to the module's
    own name. A file that writes `__name__` makes the guard fire on import, so the
    skipped body is the code that actually runs -- previously that suppressed every
    RMT finding in the file while coverage still certified it `examined`.
    """

    source = f'import os\n{rebind}\nif __name__ == "__main__":\n    os.system("id")\n'
    rule_ids = [f.rule_id for f in analyze_python_source(source, artifact="m.py")]
    assert rule_ids == ["RMT000", "RMT010"], label


def test_reading_globals_does_not_withdraw_the_main_guard_skip() -> None:
    """Only writes count. Matching any `globals()` mention fires on 1.9% of real
    site-packages modules that merely read it, which is the false-positive rate this
    project exists to avoid."""

    source = (
        'import pty\n'
        'CONFIG = globals().get("CONFIG")\n'
        'if __name__ == "__main__":\n'
        '    pty.spawn(["sh"])\n'
    )
    assert analyze_python_source(source, artifact="m.py") == []


def test_import_time_walk_fails_closed_on_a_deep_ast() -> None:
    """A file that parses fine can still exhaust the walkers that run after it.

    `ast.parse` builds left-nested expressions iteratively, so ~1 KB of `1+1+...`
    parses and then blows the stack in `_import_bindings`. Uncaught, that propagated
    past the CLI handler and aborted the entire scan -- discarding the template FAILs
    too, at the exit code the docs define as "WARN findings present".
    """

    source = "x = " + "+".join(["1"] * 1000) + "\n"
    assert len(source) < MAX_PYTHON_BYTES
    assert [f.rule_id for f in analyze_python_source(source, artifact="m.py")] == ["RMT000"]


def _repo(files: dict[str, str]):
    def reader(name: str, max_bytes: int, *, python_only: bool = False):
        return ReadResult(files[name]) if name in files else None

    return analyze_auto_map_python(
        [("config.json", {"auto_map": {"AutoModel": "modeling_x.Model"}})],
        reader,
    )


@pytest.mark.parametrize(
    ("label", "modeling", "utils"),
    [
        ("direct", "from .utils_x import system\nsystem('id')\n", "from os import system\n"),
        ("aliased", "from .utils_x import system as s\ns('id')\n", "from os import system\n"),
        ("star", "from .utils_x import *\nsystem('id')\n", "from os import system\n"),
        ("module object", "from . import utils_x\nutils_x.system('id')\n", "from os import system\n"),
    ],
)
def test_a_sink_re_exported_across_the_one_hop_is_resolved(
    label: str,
    modeling: str,
    utils: str,
) -> None:
    """Neither file is dangerous alone; together they run os.system on import.

    The one-hop follower already read the sibling, but its bindings were discarded, so
    every form here resolved to nothing while BOTH surfaces reported `examined` -- a
    clean bill of health with a receipt, on a file that executes a shell command.
    """

    findings, surfaces = _repo({"modeling_x.py": modeling, "utils_x.py": utils})

    assert [f.rule_id for f in findings] == ["RMT010"], label
    assert "os.system" in findings[0].detail
    assert {s.state for s in surfaces} == {"examined"}


def test_decode_then_execute_composes_across_the_hop() -> None:
    findings, _ = _repo({
        "modeling_x.py": (
            "from .utils_x import run\n"
            "import base64\n"
            "run(base64.b64decode('aWQ='))\n"
        ),
        "utils_x.py": "from os import system as run\n",
    })

    assert sorted({f.rule_id for f in findings}) == ["RMT010", "RMT020"]


@pytest.mark.parametrize(
    ("label", "files"),
    [
        ("class defined in the sibling", {
            "modeling_x.py": "from .configuration_x import XConfig\ncfg = XConfig()\n",
            "configuration_x.py": "class XConfig:\n    pass\n",
        }),
        ("function defined in the sibling", {
            "modeling_x.py": "from .utils_x import helper\nhelper()\n",
            "utils_x.py": "def helper():\n    return 1\n",
        }),
        ("re-export never called", {
            "modeling_x.py": "from .utils_x import system\nVALUE = 1\n",
            "utils_x.py": "from os import system\n",
        }),
    ],
)
def test_ordinary_relative_imports_stay_silent(label: str, files: dict[str, str]) -> None:
    """Nearly every real relative import pulls in a class or a function.

    Flagging those would bury the signal: reporting each unresolved name fired on 114
    call sites across 1,185 real site-packages modules that use relative imports.
    """

    findings, _ = _repo(files)
    assert findings == [], label


def test_a_second_hop_is_reported_rather_than_passed_as_clean() -> None:
    """Resolution stops after one hop, so say so where it matters.

    The sibling was read and reports `examined`, so nothing else in the report would
    tell an operator that a name reached it through a further relative import.
    """

    findings, surfaces = _repo({
        "modeling_x.py": "from .utils_x import system\nsystem('id')\n",
        "utils_x.py": "from .deeper_x import system\n",
    })
    states = {s.id: s.state for s in surfaces}

    assert [f.rule_id for f in findings] == ["RMT000"]
    assert "further relative import" in findings[0].detail
    assert states["rmt.modeling_x.py"] == "partial"


def test_an_unreadable_sibling_downgrades_coverage_without_a_finding() -> None:
    """The absent sibling already has its own coverage row; do not restate it per call.

    The parent drops to `partial`, which is the honest signal that its audit is
    incomplete, and the operator reads which file was missing off the manifest.
    """

    findings, surfaces = _repo({
        "modeling_x.py": "from .missing_x import system\nsystem('id')\n",
    })
    states = {s.id: s.state for s in surfaces}

    assert findings == []
    assert states["rmt.modeling_x.py"] == "partial"
    assert states["rmt.missing_x.py"] == "absent"


@pytest.mark.parametrize(
    "reference",
    [
        "C:foo.Model",        # drive-relative: os.path.join DROPS the base directory
        "Z:evil.Model",
        "COM1.Model",         # Windows resolves device names regardless of extension
        "nul.Model",
        "modeling-x.Model",   # not an importable module name
        "modeling x.Model",
    ],
)
def test_auto_map_names_that_are_not_module_names_are_refused(reference: str) -> None:
    """`auto_map` values are Python module paths, so anything else is only ever an
    attempt to reach a file.

    `C:foo.py` contains no separator, no `..`, and ends in `.py`, so it passed every
    check -- and then `os.path.join(bundle_dir, "C:foo.py")` returns `"C:foo.py"`,
    reading from the current directory of drive C: instead of the model's directory.
    """

    read = lambda *a, **k: pytest.fail("unsafe reference reached the reader")
    findings, surfaces = analyze_auto_map_python(
        [("config.json", {"auto_map": {"AutoModel": reference}})], read,
    )

    assert [f.rule_id for f in findings] == ["RMT000"]
    assert [s.state for s in surfaces] in ([], ["skipped"])


def test_finding_flood_is_capped_and_the_overflow_is_counted() -> None:
    """A hostile repo must not turn the report into a denial of service.

    One 1 MiB file of `os.system(...)` per line yields ~69,800 findings and 68 MiB of
    SARIF; the 32-file bundle projected to ~2 GiB and took minutes. GitHub rejects SARIF
    over 10 MB, so the flood is unusable to the operator as well as expensive.
    """

    source = "import os\n" + 'os.system("x")\n' * 250
    findings = analyze_python_source(source, artifact="modeling_x.py")
    overflow = [f for f in findings if f.rule_id == "RMT000"]

    assert len(findings) == MAX_FINDINGS_PER_ARTIFACT + 1
    assert len(overflow) == 1
    # Counted, not silently dropped: the exact number withheld has to be in the report.
    assert "150 further occurrence(s)" in overflow[0].detail


def test_a_capped_file_is_never_reported_as_fully_examined() -> None:
    """Truncating the list while still claiming `examined` is the false clean this
    family exists to prevent."""

    dense = "import os\n" + 'os.system("x")\n' * 250
    files = {"modeling_x.py": dense}

    def reader(name, max_bytes, *, python_only=False):
        return ReadResult(files[name]) if name in files else None

    _findings, surfaces = analyze_auto_map_python(
        [("config.json", {"auto_map": {"AutoModel": "modeling_x.Model"}})], reader,
    )
    states = {s.id: s.state for s in surfaces}

    assert states["rmt.modeling_x.py"] == "partial"
    assert "counted, not listed" in {s.id: s.reason for s in surfaces}["rmt.modeling_x.py"]


def test_the_cap_leaves_ordinary_files_untouched() -> None:
    """Real modules produce single digits; the cap is ~10x the worst honest case."""

    source = "import os\n" + "".join(
        f'os.system("cmd{i}")\n' for i in range(MAX_FINDINGS_PER_ARTIFACT - 1)
    )
    findings = analyze_python_source(source, artifact="modeling_x.py")

    assert len(findings) == MAX_FINDINGS_PER_ARTIFACT - 1
    assert not [f for f in findings if f.rule_id == "RMT000"]
