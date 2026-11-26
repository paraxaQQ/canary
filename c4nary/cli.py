"""Command-line interface: ``scan``, ``diff``, ``hash``, ``rules``.

Exit codes (spec §6):
  0  no unsuppressed findings at/above the threshold, or --fail-on none
  1  WARN-level findings present (only when --fail-on warn)
  2  FAIL-level findings present; argparse also uses 2 for invalid syntax
  >2 tool error (bad file, parse failure)

For ``diff``: 0 = structurally identical, 1 = differences found, >2 = error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

from . import __version__
from .coverage import Coverage, Surface, build_scan_coverage
from .ingest import TARGETS, classify_target
from .ollama import OllamaError, load_modelfile, scan_ollama
from .integrity import (
    write_manifest,
    compare_manifest,
    diff_is_empty,
    diff_models,
    model_template_sha256,
    sha256_file,
)
from .parser import GGUFParseError, parse_gguf
from .policy import PolicyError, apply_report_policy, load_baseline, load_policy
# RemoteError only; remote.py imports requests lazily, so the offline core stays
# free of a network dependency (pinned by tests/test_remote.py).
from .remote import RemoteError
from .repo import RepoError
from .provenance import rules_bundle_sha256
from .report import (
    FAIL,
    WARN,
    Finding,
    render_human,
    render_json,
    summarize,
)
from .rules.metadata import analyze_metadata
from .rules.registry import all_rules
from .rules.structure import analyze_structure
from .rules.template import analyze_templates
from .rules.tokenizer import analyze_tokenizer
from .sarif import render_sarif

EXIT_OK = 0
EXIT_WARN = 1
EXIT_FAIL = 2
EXIT_ERROR = 3


def main(argv: list[str] | None = None) -> int:
    # Defensive: a GGUF metadata value may contain arbitrary Unicode. Never let
    # a non-UTF-8 console turn a print into a crash. (JSON output is already
    # ASCII via ensure_ascii=True; this only affects terminal display bytes.)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        # No subcommand: drop into the guided menu on a real terminal, else
        # print help (so pipes/CI never hang waiting on input()).
        if _is_interactive():
            return interactive()
        parser.print_help()
        return EXIT_OK
    try:
        return args.func(args)
    except (
        GGUFParseError, OSError, json.JSONDecodeError, PolicyError, RemoteError,
        RepoError, OllamaError,
        # Backstop for any recursive walker over attacker-shaped input without its own
        # depth budget. Escaping instead gives CPython's exit 1, which this tool's table
        # defines as "WARN findings present" -- a crash would read as a passing scan.
        RecursionError,
    ) as exc:
        print(f"c4nary: error: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="canary",
        description="Deterministic, offline, read-only GGUF and repository security "
                    "auditor. Detects risk indicators; does not prove a model safe "
                    "or malicious.",
    )
    p.add_argument("--version", action="version", version=f"c4nary {__version__}")
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("scan",
                        help="audit a GGUF model, repository, or Ollama Modelfile")
    sp.add_argument("file", help="path to a .gguf file, a repository directory "
                                 "(--as repo), a local Modelfile (--as ollama), or "
                                 "(with --remote) a Hugging Face repo id; a GGUF "
                                 "URL is accepted only for a GGUF target")
    output = sp.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="emit deterministic JSON")
    output.add_argument("--sarif", action="store_true", help="emit deterministic SARIF 2.1.0")
    sp.add_argument("--manifest", metavar="M.JSON",
                    help="compare a local GGUF against a known-good manifest "
                         "(local GGUF only)")
    sp.add_argument("--remote", action="store_true",
                    help="the target is on Hugging Face: a GGUF header for a GGUF "
                         "target, or the repository's fixed files and safetensors "
                         "headers with --as repo; no weights")
    sp.add_argument("--gguf-file", dest="hf_filename", metavar="NAME.gguf",
                    help="with --remote and a GGUF target, the specific .gguf "
                         "filename to fetch")
    # Compatibility alias; omitted from --help.
    sp.add_argument("--file", dest="hf_filename", help=argparse.SUPPRESS)
    sp.add_argument("--fail-on", choices=("warn", "fail", "none"), default=None,
                    help="exit non-zero threshold (default: policy, then fail)")
    sp.add_argument("--policy", metavar="POLICY.JSON",
                    help="apply per-rule severity overrides and a fail threshold")
    sp.add_argument("--baseline", metavar="BASELINE.JSON",
                    help="suppress matching finding fingerprints with required justifications")
    sp.add_argument("--deep-tokenizer", action="store_true",
                    help="GGUF: materialize the full tokenizer vocab and run the "
                         "seam checks (TOK012/TOK015). Repository: tokenizer.deep is "
                         "partial or skipped because tokenizer.json has no token "
                         "types; no seam findings")
    sp.add_argument("--bundle", action="store_true",
                    help="GGUF target only: also audit repository surfaces -- "
                         "decode-time levers (CFG*), tokenizer.json (NRM*), model "
                         "card (DOC*), a divergent repo template (TPL030), and any "
                         "auto_map custom Python (RMT*; ast only, never executed). "
                         "Repository targets read these by default")
    sp.add_argument("--as", dest="as_type", choices=TARGETS, default=None,
                    help="target type; a local directory defaults to repo")
    sp.add_argument("--ollama", action="store_true",
                    help="alias for --as ollama: the target is a local Ollama "
                         "Modelfile; model-store references are not implemented yet")
    sp.set_defaults(func=_cmd_scan)

    dp = sub.add_parser("diff", help="structural diff of two GGUF files")
    dp.add_argument("a")
    dp.add_argument("b")
    dp.add_argument("--json", action="store_true")
    dp.set_defaults(func=_cmd_diff)

    hp = sub.add_parser("hash", help="print file + template SHA-256")
    hp.add_argument("file")
    hp.add_argument("--json", action="store_true")
    hp.add_argument("--manifest", metavar="M.JSON",
                    help="write a known-good manifest to this path")
    hp.set_defaults(func=_cmd_hash)

    rp = sub.add_parser("rules", help="list all rule ids and descriptions")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(func=_cmd_rules)

    return p


_DEEP_TOK_KEYS = frozenset({"tokenizer.ggml.tokens", "tokenizer.ggml.token_type"})


def _scan_bundle(args, model) -> tuple[list, list[Surface]]:
    """Opt-in repo-bundle audit -- fetch (remote) or read (local sibling) the repo's config /
    tokenizer.json / special-token / card / divergent-template surfaces and route them through
    the CFG / NRM / DOC / TPL030 rules. Shared with the MCP scan tool (bundle.bundle_findings)."""
    from .bundle import ReadResult, _safe_bundle_name

    def _read(
        name: str,
        max_bytes: int = 1 << 20,
        *,
        python_only: bool = False,
    ) -> ReadResult | None:
        if python_only:
            name = _safe_bundle_name(name)
        if args.remote:
            from .remote import fetch_repo_text
            return fetch_repo_text(
                args.file,
                name,
                max_bytes=max_bytes,
                python_only=python_only,
            )
        path = os.path.join(os.path.dirname(os.path.abspath(args.file)), name)
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                data = fh.read(max_bytes + 1)
            return ReadResult(
                data[:max_bytes].decode("utf-8", errors="replace"),
                truncated=len(data) > max_bytes,
            )
        return None

    from .bundle import bundle_findings
    return bundle_findings(model, _read)


@dataclass(frozen=True)
class _ScanResult:
    findings: list[Finding]
    coverage: Coverage
    display: str
    file_sha: str | None
    template_sha: str | None
    target: str = "gguf"


def _scan_gguf(args, policy, baseline) -> _ScanResult:
    """The GGUF ingest: parse (or range-fetch) a .gguf and run every GGUF rule family.

    RemoteError propagates to ``main``, which prints it in the same form the inline
    handler used to.
    """
    deep = getattr(args, "deep_tokenizer", False)
    bundle = getattr(args, "bundle", False)
    materialize = _DEEP_TOK_KEYS if (deep or bundle) else None
    if args.remote:
        from .remote import fetch_remote_model
        model, display = fetch_remote_model(
            args.file, args.hf_filename, materialize=materialize)
        file_sha = None
        template_findings = analyze_templates(model)
        findings = (
            template_findings
            + analyze_metadata(model)
            + analyze_tokenizer(model, deep=deep)
        )
    else:
        model = parse_gguf(args.file, materialize=materialize)
        display = model.path
        file_sha = sha256_file(args.file)
        template_findings = analyze_templates(model)
        findings = (
            template_findings
            + analyze_metadata(model)
            + analyze_tokenizer(model, deep=deep)
            + analyze_structure(model)
        )
        if args.manifest:
            with open(args.manifest, encoding="utf-8") as fh:
                manifest = json.load(fh)
            findings += compare_manifest(model, file_sha, manifest)

    bundle_surfaces: list[Surface] = []
    if bundle:
        bundle_results, bundle_surfaces = _scan_bundle(args, model)
        findings += bundle_results

    registered = [finding.severity for finding in findings]
    findings = apply_report_policy(findings, policy, baseline)
    suppressed_count = sum(finding.suppressed for finding in findings)
    retiered_count = sum(
        before != after.severity for before, after in zip(registered, findings)
    )
    coverage = build_scan_coverage(
        model,
        remote=args.remote,
        deep_tokenizer=deep,
        bundle=bundle,
        manifest_requested=bool(args.manifest),
        baseline_requested=getattr(args, "baseline", None) is not None,
        suppressed_count=suppressed_count,
        policy_requested=getattr(args, "policy", None) is not None,
        retiered_count=retiered_count,
        template_findings=template_findings,
        bundle_surfaces=bundle_surfaces,
    )
    return _ScanResult(findings, coverage, display, file_sha,
                       model_template_sha256(model))


def _scan_repo(args, policy, baseline) -> _ScanResult:
    """The repository ingest: load a local directory or remote repository and run
    the shared rule pass (``repo.scan_repo``), which the MCP tool also calls."""
    from .repo import load_local_repo, load_remote_repo, scan_repo

    repo = load_remote_repo(args.file) if args.remote else load_local_repo(args.file)
    findings, coverage = scan_repo(
        repo,
        deep=getattr(args, "deep_tokenizer", False),
        policy=policy,
        baseline=baseline,
        policy_requested=getattr(args, "policy", None) is not None,
        baseline_requested=getattr(args, "baseline", None) is not None,
    )
    return _ScanResult(findings, coverage, repo.display, None, None, "repo")


def _scan_ollama(args, policy, baseline) -> _ScanResult:
    """The Modelfile ingest: parse a local Modelfile and run the shared pass."""
    modelfile = load_modelfile(args.file)
    findings, coverage = scan_ollama(
        modelfile,
        policy=policy,
        baseline=baseline,
        policy_requested=getattr(args, "policy", None) is not None,
        baseline_requested=getattr(args, "baseline", None) is not None,
    )
    return _ScanResult(
        findings, coverage, modelfile.display, modelfile.sha256,
        modelfile.template_sha256, "ollama",
    )


def _cmd_scan(args) -> int:
    policy = load_policy(getattr(args, "policy", None))
    baseline = load_baseline(getattr(args, "baseline", None))

    target = classify_target(
        args.file,
        remote=args.remote,
        ollama=getattr(args, "ollama", False),
        override=getattr(args, "as_type", None),
    )
    if target == "ollama":
        if args.remote:
            print("c4nary: error: --remote means Hugging Face and is not supported "
                  "for an Ollama Modelfile; scan the Modelfile path directly.",
                  file=sys.stderr)
            return EXIT_ERROR
        if getattr(args, "hf_filename", None):
            print("c4nary: error: --gguf-file (--file) names a single GGUF and has "
                  "no meaning for an Ollama Modelfile.", file=sys.stderr)
            return EXIT_ERROR
        if getattr(args, "manifest", None):
            print("c4nary: error: --manifest compares a single GGUF and is not "
                  "supported for an Ollama Modelfile.", file=sys.stderr)
            return EXIT_ERROR
        if getattr(args, "bundle", False):
            print("c4nary: error: --bundle is a GGUF/repository surface and has no "
                  "meaning for an Ollama Modelfile.", file=sys.stderr)
            return EXIT_ERROR
        if getattr(args, "deep_tokenizer", False):
            print("note: --deep-tokenizer applies to GGUF targets; a Modelfile has "
                  "no tokenizer surface.", file=sys.stderr)
        result = _scan_ollama(args, policy, baseline)
    elif target == "repo":
        if getattr(args, "hf_filename", None):
            print("c4nary: error: --gguf-file (--file) names a single GGUF and has "
                  "no meaning for a repository target.", file=sys.stderr)
            return EXIT_ERROR
        if getattr(args, "manifest", None):
            print("c4nary: error: --manifest compares a single GGUF and is not "
                  "supported for a repository target.", file=sys.stderr)
            return EXIT_ERROR
        if args.remote:
            # A repo id such as 'gpt2' or 'org/model' may also name a local
            # directory. The explicit --remote flag wins; only an absolute,
            # drive-qualified, or backslash path is treated as a local-target
            # mistake. A forward slash is the repo-id separator, not a path tell.
            path_like = (
                os.path.isabs(args.file)
                or "\\" in args.file
                or (os.name == "nt" and len(args.file) > 1 and args.file[1] == ":")
            )
            if path_like and os.path.isdir(args.file):
                print(f"c4nary: error: --remote expects a Hugging Face repo id, but "
                      f"{args.file} is a local directory; drop --remote to scan it.",
                      file=sys.stderr)
                return EXIT_ERROR
        elif not os.path.isdir(args.file):
            print(f"c4nary: error: repository target must be a local directory: "
                  f"{args.file}", file=sys.stderr)
            return EXIT_ERROR
        if getattr(args, "bundle", False):
            print("note: repo bundle surfaces are already read; --bundle is implied.",
                  file=sys.stderr)
        result = _scan_repo(args, policy, baseline)
    else:
        result = _scan_gguf(args, policy, baseline)

    findings, coverage = result.findings, result.coverage
    for note in coverage.notes():
        print(f"note: {note}", file=sys.stderr)

    if getattr(args, "sarif", False):
        print(render_sarif(file=result.display, sha256=result.file_sha,
                           findings=findings))
    elif args.json:
        print(render_json(
            file=result.display, sha256=result.file_sha,
            template_sha256=result.template_sha, findings=findings,
            coverage=coverage))
    else:
        print(render_human(
            file=result.display, sha256=result.file_sha,
            template_sha256=result.template_sha, findings=findings,
            target=result.target))

    counts = summarize(findings)
    fail_on = getattr(args, "fail_on", None) or policy.fail_on
    if fail_on == "none":
        return EXIT_OK
    if counts[FAIL]:
        return EXIT_FAIL
    if fail_on == "warn" and counts[WARN]:
        return EXIT_WARN
    return EXIT_OK


def _cmd_diff(args) -> int:
    a = parse_gguf(args.a)
    b = parse_gguf(args.b)
    diff = diff_models(a, b)

    if args.json:
        print(json.dumps(diff, indent=2, ensure_ascii=True))
    else:
        print(_render_diff_human(args.a, args.b, diff))

    return EXIT_OK if diff_is_empty(diff) else EXIT_WARN


def _cmd_hash(args) -> int:
    model = parse_gguf(args.file)
    file_sha = sha256_file(args.file)
    template_sha = model_template_sha256(model)

    if args.manifest:
        write_manifest(model, file_sha, args.manifest, args.file)
        print(f"wrote manifest: {args.manifest}", file=sys.stderr)

    if args.json:
        print(json.dumps(
            {"file": model.path, "sha256": file_sha,
             "template_sha256": template_sha},
            indent=2, ensure_ascii=True))
    else:
        print(f"sha256          {file_sha}")
        print(f"template_sha256 {template_sha or '(none)'}")
    return EXIT_OK


def _cmd_rules(args) -> int:
    rules = all_rules()
    if args.json:
        print(json.dumps({
            "tool_version": __version__,
            "rules_bundle_sha256": rules_bundle_sha256(),
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "severity": rule.severity,
                    "title": rule.title,
                    "description": rule.description,
                }
                for rule in rules
            ],
        }, indent=2, ensure_ascii=True))
    else:
        for r in rules:
            print(f"{r.rule_id}  {r.severity:4}  {r.title}")
            print(f"          {r.description}")
    return EXIT_OK


def _render_diff_human(path_a: str, path_b: str, diff: dict) -> str:
    lines = [f"c4nary diff", f"  a: {path_a}", f"  b: {path_b}", ""]
    if diff_is_empty(diff):
        lines.append("No structural differences.")
        return "\n".join(lines) + "\n"

    md = diff["metadata"]
    if md["added"] or md["removed"] or md["changed"]:
        lines.append("[metadata]")
        for k, v in md["added"].items():
            lines.append(f"  + {k} = {v!r}")
        for k, v in md["removed"].items():
            lines.append(f"  - {k} = {v!r}")
        for k, v in md["changed"].items():
            lines.append(f"  ~ {k}: {v['a']!r} -> {v['b']!r}")
        lines.append("")

    if diff["template_changed"]:
        lines.append("[chat_template]")
        lines.extend("  " + ln for ln in diff["template_diff"])
        lines.append("")

    td = diff["tensors"]
    if td["added"] or td["removed"] or td["changed"]:
        lines.append("[tensors]")
        for name in td["added"]:
            lines.append(f"  + {name}")
        for name in td["removed"]:
            lines.append(f"  - {name}")
        for ch in td["changed"]:
            lines.append(f"  ~ {ch['name']}: {ch['a']} -> {ch['b']}")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Interactive (menu) mode -- run `canary` with no arguments on a terminal.
# Every action still has a flag-based subcommand; this is just a friendlier door.
# --------------------------------------------------------------------------- #

_BANNER = (
    "c4nary - GGUF, repository, and Ollama Modelfile security auditor\n"
    "Flags risk indicators; never renders the template or runs the model.\n"
)


def interactive() -> int:
    print(_BANNER)
    menu = [
        ("1", "Scan a local .gguf file", _i_scan_local),
        ("2", "Scan a Hugging Face GGUF (header only)", _i_scan_remote),
        ("3", "Scan a local repository", _i_scan_repo_local),
        ("4", "Scan a Hugging Face repository", _i_scan_repo_remote),
        ("5", "Diff two .gguf files", _i_diff),
        ("6", "Hash a .gguf / write a manifest", _i_hash),
        ("7", "List the detection rules", _i_rules),
        ("8", "Scan a local Ollama Modelfile", _i_scan_ollama),
    ]
    while True:
        print("What would you like to do?")
        for key, label, _ in menu:
            print(f"  {key}) {label}")
        print("  q) Quit")
        choice = _ask(">")
        if choice in ("q", "quit", "exit", None):
            return EXIT_OK
        action = next((fn for key, _, fn in menu if key == choice), None)
        if action is None:
            print("  (choose 1-8 or q)\n")
            continue
        try:
            action()
        except (
            GGUFParseError, OSError, json.JSONDecodeError,
            PolicyError, RemoteError, RepoError, OllamaError,
        ) as exc:
            print(f"  error: {exc}")
        print()


def _ask(label: str) -> str | None:
    try:
        return input(f"{label} ").strip().strip("'\"").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _scan_args(target: str, *, remote: bool, filename: str | None = None,
               as_type: str | None = None):
    return argparse.Namespace(
        file=target,
        json=False,
        sarif=False,
        manifest=None,
        remote=remote,
        hf_filename=filename,
        fail_on=None,
        policy=None,
        baseline=None,
        deep_tokenizer=False,
        bundle=False,
        as_type=as_type,
    )


def _i_scan_local() -> None:
    path = _ask("Path to .gguf file:")
    if path:
        _cmd_scan(_scan_args(path, remote=False))


def _i_scan_remote() -> None:
    repo = _ask("Hugging Face repo id or GGUF URL:")
    if not repo:
        return
    fn = _ask("Specific .gguf filename (blank = auto):") or None
    _cmd_scan(_scan_args(repo, remote=True, filename=fn))


def _i_scan_repo_local() -> None:
    path = _ask("Path to repository directory:")
    if path:
        _cmd_scan(_scan_args(path, remote=False, as_type="repo"))


def _i_scan_repo_remote() -> None:
    repo = _ask("Hugging Face repo id:")
    if repo:
        _cmd_scan(_scan_args(repo, remote=True, as_type="repo"))


def _i_scan_ollama() -> None:
    path = _ask("Path to Modelfile:")
    if path:
        _cmd_scan(_scan_args(path, remote=False, as_type="ollama"))


def _i_diff() -> None:
    a = _ask("First .gguf:")
    b = _ask("Second .gguf:") if a else None
    if a and b:
        _cmd_diff(argparse.Namespace(a=a, b=b, json=False))


def _i_hash() -> None:
    path = _ask("Path to .gguf file:")
    if not path:
        return
    manifest = _ask("Manifest output path (blank = skip):") or None
    _cmd_hash(argparse.Namespace(file=path, json=False, manifest=manifest))


def _i_rules() -> None:
    _cmd_rules(argparse.Namespace(json=False))


if __name__ == "__main__":
    sys.exit(main())
