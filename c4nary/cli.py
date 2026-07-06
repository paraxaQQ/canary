"""Command-line interface: ``scan``, ``diff``, ``hash``, ``rules``.

Exit codes (spec §6):
  0  no findings at/above the fail threshold
  1  WARN-level findings present (only when --fail-on warn)
  2  FAIL-level findings present
  >2 tool error (bad file, parse failure)

For ``diff``: 0 = structurally identical, 1 = differences found, >2 = error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .integrity import (
    build_manifest,
    compare_manifest,
    diff_is_empty,
    diff_models,
    model_template_sha256,
    sha256_file,
)
from .parser import GGUFParseError, parse_gguf
from .report import (
    FAIL,
    WARN,
    render_human,
    render_json,
    summarize,
)
from .rules.metadata import analyze_metadata
from .rules.registry import all_rules
from .rules.structure import analyze_structure
from .rules.template import analyze_templates
from .rules.tokenizer import analyze_tokenizer

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
    except (GGUFParseError, OSError, json.JSONDecodeError) as exc:
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
        description="Deterministic, offline, read-only GGUF security auditor. "
                    "Detects risk indicators; does not prove a model safe or malicious.",
    )
    p.add_argument("--version", action="version", version=f"c4nary {__version__}")
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("scan", help="audit a GGUF file's template + metadata")
    sp.add_argument("file", help="path to a .gguf file, or (with --remote) a "
                                 "Hugging Face repo id or URL")
    sp.add_argument("--json", action="store_true", help="emit deterministic JSON")
    sp.add_argument("--manifest", metavar="M.JSON",
                    help="compare against a known-good manifest (local scans only)")
    sp.add_argument("--remote", action="store_true",
                    help="audit a model's header over the network without "
                         "downloading its weights (Hugging Face repo id or URL)")
    sp.add_argument("--file", dest="hf_filename", metavar="NAME.gguf",
                    help="with --remote, the specific .gguf filename to fetch")
    sp.add_argument("--fail-on", choices=("warn", "fail"), default="fail",
                    help="exit non-zero threshold (default: fail)")
    sp.add_argument("--deep-tokenizer", action="store_true",
                    help="materialize the full tokenizer vocab and run the seam / "
                         "reachability checks (TOK010+); off by default")
    sp.add_argument("--bundle", action="store_true",
                    help="also audit the repo bundle (generation_config.json / "
                         "config.json) for decode-time levers (CFG*); opt-in, "
                         "materializes the vocab")
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


def _scan_bundle(args, model) -> list:
    """Opt-in repo-bundle audit -- fetch (remote) or read (local sibling) the repo's config /
    tokenizer.json / special-token / card / divergent-template surfaces and route them through
    the CFG / NRM / DOC / TPL030 rules. Shared with the MCP scan tool (bundle.bundle_findings)."""
    def _read(name: str, max_bytes: int = 1 << 20) -> str | None:
        if args.remote:
            from .remote import fetch_repo_text
            return fetch_repo_text(args.file, name, max_bytes=max_bytes)
        path = os.path.join(os.path.dirname(os.path.abspath(args.file)), name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                return fh.read(max_bytes)
        return None

    from .bundle import bundle_findings
    return bundle_findings(model, _read)


def _cmd_scan(args) -> int:
    deep = getattr(args, "deep_tokenizer", False)
    bundle = getattr(args, "bundle", False)
    materialize = _DEEP_TOK_KEYS if (deep or bundle) else None
    if args.remote:
        from .remote import RemoteError, fetch_remote_model
        try:
            model, display = fetch_remote_model(
                args.file, args.hf_filename, materialize=materialize)
        except RemoteError as exc:
            print(f"c4nary: error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        file_sha = None
        findings = (analyze_templates(model)
                    + analyze_metadata(model)
                    + analyze_tokenizer(model, deep=deep))
        # The header is truncated, so the file is the wrong size: structural and
        # whole-file integrity checks cannot run on a remote scan.
        print("note: remote header scan - structural (STR*) and whole-file "
              "integrity checks need the complete file and were skipped.",
              file=sys.stderr)
        if args.manifest:
            print("note: --manifest ignored for remote scans (needs the full file).",
                  file=sys.stderr)
    else:
        model = parse_gguf(args.file, materialize=materialize)
        display = model.path
        file_sha = sha256_file(args.file)
        findings = (analyze_templates(model)
                    + analyze_metadata(model)
                    + analyze_tokenizer(model, deep=deep)
                    + analyze_structure(model))
        if args.manifest:
            with open(args.manifest, encoding="utf-8") as fh:
                manifest = json.load(fh)
            findings += compare_manifest(model, file_sha, manifest)

    # State whether the deep tokenizer pass ran, so a clean verdict never silently
    # means "didn't look" (mirrors the STR* skipped note).
    if deep:
        print("note: deep tokenizer pass ran - full vocab materialized; "
              "seam checks (TOK010+) active.", file=sys.stderr)
    else:
        print("note: deep tokenizer seam checks (TOK010+) not run; pass "
              "--deep-tokenizer to enable.", file=sys.stderr)

    if bundle:
        findings += _scan_bundle(args, model)
        print("note: bundle scan ran - generation_config / config levers (CFG*) + model "
              "card README (DOC*) audited.", file=sys.stderr)

    template_sha = model_template_sha256(model)
    if args.json:
        print(render_json(
            file=display, sha256=file_sha,
            template_sha256=template_sha, findings=findings))
    else:
        print(render_human(
            file=display, sha256=file_sha,
            template_sha256=template_sha, findings=findings))

    counts = summarize(findings)
    if counts[FAIL]:
        return EXIT_FAIL
    if args.fail_on == "warn" and counts[WARN]:
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
        manifest = build_manifest(model, file_sha)
        with open(args.manifest, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(manifest, indent=2, ensure_ascii=True))
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
        print(json.dumps(
            [{"rule_id": r.rule_id, "severity": r.severity,
              "title": r.title, "description": r.description} for r in rules],
            indent=2, ensure_ascii=True))
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
    "c4nary - GGUF chat-template security auditor\n"
    "Flags it for risk indicators; never renders the template or runs the model.\n"
)


def interactive() -> int:
    print(_BANNER)
    menu = [
        ("1", "Scan a local .gguf file", _i_scan_local),
        ("2", "Scan a Hugging Face model (no download)", _i_scan_remote),
        ("3", "Diff two .gguf files", _i_diff),
        ("4", "Hash a file / write a manifest", _i_hash),
        ("5", "List the detection rules", _i_rules),
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
            print("  (choose 1-5 or q)\n")
            continue
        try:
            action()
        except (GGUFParseError, OSError, json.JSONDecodeError) as exc:
            print(f"  error: {exc}")
        print()


def _ask(label: str) -> str | None:
    try:
        return input(f"{label} ").strip().strip("'\"").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _scan_args(target: str, *, remote: bool, filename: str | None = None):
    return argparse.Namespace(file=target, json=False, manifest=None,
                              remote=remote, hf_filename=filename, fail_on="fail")


def _i_scan_local() -> None:
    path = _ask("Path to .gguf file:")
    if path:
        _cmd_scan(_scan_args(path, remote=False))


def _i_scan_remote() -> None:
    repo = _ask("Hugging Face repo id or URL:")
    if not repo:
        return
    fn = _ask("Specific .gguf filename (blank = auto):") or None
    _cmd_scan(_scan_args(repo, remote=True, filename=fn))


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
