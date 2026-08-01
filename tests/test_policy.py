from __future__ import annotations

import json
from pathlib import Path

import pytest

from _ggufgen import write_gguf

from c4nary import cli
from c4nary.policy import (
    PolicyError,
    apply_report_policy,
    finding_fingerprint,
    load_baseline,
    load_policy,
)
from c4nary.rules.registry import finding, get_rule


CVE = (
    Path(__file__).parent / "fixtures" / "cve_llama_drama.jinja"
).read_text(encoding="utf-8")


def _malicious(tmp_path: Path) -> Path:
    return write_gguf(tmp_path / "evil.gguf", {
        "general.architecture": "llama",
        "llama.context_length": 4096,
        "tokenizer.chat_template": CVE,
    }, tensors=[("a", (2, 2), 0)], tail=b"data")


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_fingerprint_excludes_detail_and_line() -> None:
    first = finding(
        "TPL001",
        "old wording",
        location="template:dangerous node",
        artifact="chat_template",
        line=2,
    )
    second = finding(
        "TPL001",
        "new wording",
        location="template:dangerous node",
        artifact="chat_template",
        line=99,
    )

    assert finding_fingerprint(first) == finding_fingerprint(second)


def test_fingerprint_separates_repeat_occurrences_at_one_location() -> None:
    """Subject keeps one baseline entry from retiring a whole (rule, location) group.

    jinja2 reports only a construct's start line and shipped templates are largely
    minified, so an SSTI chain emits several TPL001 findings that all share
    ``template:L1``. Without ``subject`` they collapse to one fingerprint and a
    justification written about ``__globals__`` would also suppress ``__init__`` --
    including a different payload appearing there later.
    """

    globals_access, init_access = (
        finding(
            "TPL001",
            f"Access to dunder {name!r} (attribute) - an SSTI sandbox-escape primitive.",
            location="template:L1",
            artifact="chat_template",
            subject=f"dunder:{name}:attribute",
        )
        for name in ("__globals__", "__init__")
    )
    subscript_form = finding(
        "TPL001",
        "Access to dunder '__globals__' (subscript key) - an SSTI sandbox-escape primitive.",
        location="template:L1",
        artifact="chat_template",
        subject="dunder:__globals__:subscript key",
    )

    prints = {
        finding_fingerprint(f)
        for f in (globals_access, init_access, subscript_form)
    }
    assert len(prints) == 3

    # subject is structured, so rewording the prose must still not move the print.
    assert finding_fingerprint(globals_access) == finding_fingerprint(finding(
        "TPL001",
        "completely different wording",
        location="template:L1",
        artifact="chat_template",
        line=42,
        subject="dunder:__globals__:attribute",
    ))


def test_every_corpus_fail_finding_has_its_own_fingerprint() -> None:
    """The 28 frozen corpus templates must yield one fingerprint per finding.

    A regression here means a baseline entry silently covers findings its
    justification was never reviewed against.
    """

    from dataclasses import replace

    from c4nary.rules.template import analyze_template

    total = 0
    collisions: list[str] = []
    for path in sorted((Path(__file__).parent / "fixtures" / "corpus-fail").glob("*.jinja")):
        # analyze_template leaves artifact unset; the scan retags it per file, so
        # uniqueness is a within-artifact property. Stamp it the same way here.
        findings = [
            replace(item, artifact=path.name)
            for item in analyze_template(path.read_text(encoding="utf-8"))
            if item.severity == "FAIL"
        ]
        total += len(findings)
        if len({finding_fingerprint(f) for f in findings}) != len(findings):
            collisions.append(path.name)

    assert total == 140
    assert collisions == []


def test_baseline_requires_a_justification(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "baseline.json", {
        "suppressions": [{
            "fingerprint": "a" * 64,
        }],
    })

    with pytest.raises(PolicyError, match="requires a justification"):
        load_baseline(path)


@pytest.mark.parametrize("rule_id", ["TPL*", "TPL001.artifact", "*"])
def test_policy_rejects_nonliteral_rule_ids(
    tmp_path: Path,
    rule_id: str,
) -> None:
    path = _write_json(tmp_path / "policy.json", {
        "severity_overrides": {rule_id: "WARN"},
    })

    with pytest.raises(PolicyError, match="unregistered rule id"):
        load_policy(path)


def test_policy_does_not_mutate_registry_severity(
    tmp_path: Path,
    capsys,
) -> None:
    original = get_rule("TPL001").severity
    policy_path = _write_json(tmp_path / "policy.json", {
        "severity_overrides": {"TPL001": "WARN"},
    })
    policy = load_policy(policy_path)
    processed = apply_report_policy(
        [finding("TPL001", "test")],
        policy,
        load_baseline(None),
    )

    assert processed[0].severity == "WARN"
    assert get_rule("TPL001").severity == original == "FAIL"

    model = _malicious(tmp_path)
    cli.main(["scan", str(model), "--policy", str(policy_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert all(
        item["severity"] == "WARN"
        for item in payload["findings"]
        if item["rule_id"] == "TPL001"
    )
    assert get_rule("TPL001").severity == original


def test_suppressed_fails_remain_visible_and_exit_zero(
    tmp_path: Path,
    capsys,
) -> None:
    model = _malicious(tmp_path)
    assert cli.main(["scan", str(model), "--json"]) == 2
    initial = json.loads(capsys.readouterr().out)
    fail_findings = [
        item
        for item in initial["findings"]
        if item["severity"] == "FAIL"
    ]
    fingerprints = sorted({item["fingerprint"] for item in fail_findings})
    baseline = _write_json(tmp_path / "baseline.json", {
        "suppressions": [
            {
                "fingerprint": fingerprint,
                "justification": "reviewed fixture finding",
            }
            for fingerprint in fingerprints
        ],
    })

    assert cli.main([
        "scan",
        str(model),
        "--baseline",
        str(baseline),
        "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    suppressed = [item for item in payload["findings"] if item["suppressed"]]
    coverage = {row["id"]: row for row in payload["coverage"]}

    assert len(suppressed) == len(fail_findings)
    assert all(
        item["suppression_justification"] == "reviewed fixture finding"
        for item in suppressed
    )
    assert payload["summary"]["fail"] == 0
    assert payload["summary"]["suppressed"] == len(fail_findings)
    assert coverage["baseline"]["state"] == "examined"
    assert str(len(fail_findings)) in coverage["baseline"]["reason"]

    assert cli.main([
        "scan",
        str(model),
        "--baseline",
        str(baseline),
    ]) == 0
    human = capsys.readouterr().out
    assert f"{len(fail_findings)} suppressed by baseline" in human
    assert "risk indicators are suppressed by the supplied baseline" in human
    assert "baseline justification: reviewed fixture finding" in human


def test_fail_on_none_matrix_and_cli_precedence(tmp_path: Path, capsys) -> None:
    model = _malicious(tmp_path)
    policy = _write_json(tmp_path / "policy.json", {"fail_on": "none"})

    assert cli.main(["scan", str(model), "--fail-on", "none"]) == 0
    capsys.readouterr()
    assert cli.main(["scan", str(model), "--policy", str(policy)]) == 0
    capsys.readouterr()
    assert cli.main([
        "scan",
        str(model),
        "--policy",
        str(policy),
        "--fail-on",
        "fail",
    ]) == 2
    capsys.readouterr()


def test_fingerprint_components_are_nul_separated() -> None:
    """Without a separator, component boundaries are ambiguous and collide.

    A collision silently suppresses a finding nobody reviewed, and suppression removes
    findings from the exit-code count -- so an ambiguous boundary can flip a CI gate.
    """

    split_one = finding(
        "TPL001", "x", location="a.py:L1", artifact="modeling_",
    )
    split_two = finding(
        "TPL001", "x", location="py:L1", artifact="modeling_a.",
    )
    assert finding_fingerprint(split_one) != finding_fingerprint(split_two)

    # subject must not be able to impersonate a location either.
    assert finding_fingerprint(
        finding("TPL001", "x", location="a", artifact="b", subject="c")
    ) != finding_fingerprint(
        finding("TPL001", "x", location="a", artifact="b\x00c")
    )


def test_fail_rules_that_repeat_at_one_location_carry_a_subject() -> None:
    """Every FAIL rule able to fire twice at one location must discriminate.

    TOK002 loops two tensor names onto `tokenizer.ggml.tokens`; TOK003 emits both the
    length mismatch and the enum violation at `tokenizer.ggml.token_type`; TPL004 and
    TPL021 each have two emit sites that land on the same minified line. Without a
    subject, one baseline entry retires the sibling FAIL unreviewed.
    """

    import ast
    from pathlib import Path as _Path

    from c4nary.rules.registry import all_rules

    fail_ids = {rule.rule_id for rule in all_rules() if rule.severity == "FAIL"}
    repeatable = {"TOK002", "TOK003", "TPL004", "TPL021"}
    assert repeatable <= fail_ids

    missing: list[str] = []
    # Anchored to __file__, not cwd: a cwd-relative glob yields nothing from anywhere but
    # the repo root, and this assertion then certifies a source tree it never opened.
    sources = sorted((_Path(__file__).parents[1] / "c4nary").rglob("*.py"))
    assert len(sources) > 20, "rule sources not found -- this test would pass vacuously"
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "finding"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in repeatable
                and not any(kw.arg == "subject" for kw in node.keywords)
            ):
                missing.append(f"{path}:{node.lineno} {node.args[0].value}")

    assert missing == []


def test_same_named_tensors_do_not_share_a_str_fingerprint(tmp_path: Path) -> None:
    """Tensor names are attacker-controlled and need not be unique.

    STR001/STR003 are the FAIL rules in structure.py and both located on the tensor
    name alone, so two same-named out-of-bounds tensors hashed to one fingerprint and a
    single reviewed baseline entry retired both -- taking the gate from exit 2 to 0.
    """

    from dataclasses import replace as _replace

    from c4nary.parser import parse_gguf
    from c4nary.rules.structure import analyze_structure

    model = write_gguf(tmp_path / "dup.gguf", {"general.architecture": "llama"}, tensors=[
        ("dup", (4, 4), 0),
        ("dup", (4, 4), 0),
    ], tail=b"data")
    parsed = parse_gguf(str(model))
    # Point both same-named tensors past EOF -> two STR003 FAILs at one location. Patched
    # here rather than written, because a GGUF that really declares terabytes has to be
    # terabytes on disk.
    parsed = _replace(parsed, tensors=[
        _replace(t, offset=(1 << 40) + index)
        for index, t in enumerate(parsed.tensors)
    ])
    findings = [
        f for f in analyze_structure(parsed)
        if get_rule(f.rule_id).severity == "FAIL"
    ]

    assert len(findings) == 2
    assert len({finding_fingerprint(f) for f in findings}) == 2


def test_dedupe_keeps_occurrences_that_differ_only_in_subject() -> None:
    """_dedupe must not be coarser than the fingerprint it feeds.

    TPL021 carries a constant detail and puts the trigger literal only in `subject`,
    so on a minified single-line template -- the normal shape for a GGUF embedded
    template -- two distinct content gates collapsed into one and the second was
    never emitted at all.
    """

    from c4nary.rules.template import _dedupe

    findings = [
        finding(
            "TPL021",
            "Content-gated instruction injection.",
            location="template:L1",
            artifact="chat_template",
            subject=f"trigger:{trigger}",
        )
        for trigger in ("ACTIVATE", "DEPLOY")
    ]

    assert len(_dedupe(findings)) == 2
    assert len({finding_fingerprint(f) for f in _dedupe(findings)}) == 2


def test_repeatable_fail_subjects_actually_discriminate() -> None:
    """The AST grep above proves a `subject=` kwarg exists, not that its value differs.

    A constant subject satisfies the grep and still collapses the two TOK003 FAILs at
    `tokenizer.ggml.token_type` into one fingerprint, so this asserts the behaviour the
    grep only approximates.
    """

    from c4nary.parser import MetaArray
    from c4nary.rules.tokenizer import analyze_tokenizer

    class _Model:
        metadata = {
            "tokenizer.ggml.tokens": MetaArray("string", 4, ("a", "b", "c", "d"), False),
            "tokenizer.ggml.token_type": MetaArray("int32", 3, (1, 2, 99), False),
        }
        tensors: list = []

    at_token_type = [
        f for f in analyze_tokenizer(_Model())
        if f.rule_id == "TOK003" and f.location == "tokenizer.ggml.token_type"
    ]

    assert len(at_token_type) == 2, [f.detail for f in at_token_type]
    assert len({finding_fingerprint(f) for f in at_token_type}) == 2


def test_hostile_lone_surrogate_does_not_abort_the_scan() -> None:
    r"""A `\udXXX` escape reaching the fingerprint must not kill every scan.

    Without `surrogatepass` this raises UnicodeEncodeError out of a function that runs
    on every finding of every scan.
    """

    digest = finding_fingerprint(finding(
        "TPL001",
        "x",
        location="template:L1",
        artifact="chat_template",
        subject="dunder:\ud800:attribute",
    ))
    assert len(digest) == 64 and int(digest, 16) >= 0


def test_a_policy_downgrade_leaves_a_coverage_row(tmp_path: Path, capsys) -> None:
    """A re-tiered report must not be indistinguishable from a clean one.

    `--policy` can turn every FAIL into a WARN, changing the verdict and the exit code.
    The baseline already recorded its suppressions in coverage; the policy recorded
    nothing, so a downgraded scan looked exactly like a model with no FAIL findings.
    """

    model = _malicious(tmp_path)
    policy = _write_json(tmp_path / "policy.json", {
        "severity_overrides": {"TPL001": "WARN"},
    })

    assert cli.main(["scan", str(model), "--json"]) == 2
    clean = {r["id"]: r for r in json.loads(capsys.readouterr().out)["coverage"]}
    assert clean["policy"]["state"] == "absent"

    cli.main(["scan", str(model), "--policy", str(policy), "--json"])
    row = {r["id"]: r for r in json.loads(capsys.readouterr().out)["coverage"]}["policy"]

    assert row["state"] == "examined"
    assert "re-tiered" in row["reason"]
    assert row["reason"].split("re-tiered ")[1].split()[0] != "0"


def test_no_finding_shares_a_fingerprint_with_another_in_the_same_artifact() -> None:
    """Derived, not hardcoded: every shipped Jinja fixture, every severity.

    The earlier version of this check enumerated four rule ids known to repeat, so it
    could only ever catch the collisions already known about.
    """

    from collections import defaultdict
    from dataclasses import replace

    from c4nary.rules.template import analyze_template

    fixtures = sorted((Path(__file__).parent / "fixtures").rglob("*.jinja"))
    fixtures += sorted(
        (Path(__file__).parents[1] / "c4nary" / "known_templates").glob("*.jinja")
    )
    assert len(fixtures) > 30, "fixture corpus not found -- this would pass vacuously"

    collisions: list[str] = []
    for path in fixtures:
        groups: dict[str, list] = defaultdict(list)
        for item in analyze_template(path.read_text(encoding="utf-8")):
            groups[finding_fingerprint(replace(item, artifact=path.name))].append(item)
        collisions += [
            f"{path.name}: {group[0].rule_id} x{len(group)} at {group[0].location}"
            for group in groups.values()
            if len(group) > 1
        ]

    assert collisions == []
