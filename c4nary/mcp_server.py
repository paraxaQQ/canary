"""MCP server: expose c4nary's auditing over the Model Context Protocol.

A thin wrapper around the same engine the CLI uses, so an MCP-capable agent
(Claude Desktop / Claude Code / any MCP client) can audit GGUF chat templates
and metadata as tools. The invariants are unchanged: the tools parse only
(never render or run a model), are read-only on inputs, and are deterministic.
The single network path is the opt-in ``remote`` flag on ``scan`` -- exactly
as on the CLI.

``mcp`` is an optional dependency (``pip install c4nary[mcp]``); like
``remote.py``'s use of ``requests`` it is imported lazily, so the offline core
never depends on it.

Run it (stdio transport):

    python -m c4nary.mcp_server      # or the `c4nary-mcp` console script

Register with an MCP client, e.g. Claude Desktop ``claude_desktop_config.json``:

    {"mcpServers": {"c4nary": {"command": "c4nary-mcp"}}}
"""

from __future__ import annotations

import json
import os

from . import __version__
from .coverage import Surface, build_scan_coverage
from .integrity import (
    write_manifest,
    compare_manifest,
    diff_is_empty,
    diff_models,
    model_template_sha256,
    sha256_file,
)
from .parser import parse_gguf
from .provenance import rules_bundle_sha256
from .report import FAIL, INFO, WARN, findings_to_dicts, summarize, verdict_line
from .rules.metadata import analyze_metadata
from .rules.registry import all_rules
from .rules.structure import analyze_structure
from .rules.template import analyze_templates
from .rules.tokenizer import analyze_tokenizer

_DEEP_TOK_KEYS = frozenset({"tokenizer.ggml.tokens", "tokenizer.ggml.token_type"})

# The published logo, referenced by URL so it is not base64-inlined into the
# serverInfo of every handshake. Resolves once the asset is pushed to the repo.
ICON_URL = "https://raw.githubusercontent.com/paraxaQQ/canary/main/assets/canary-logo.png"

SERVER_INSTRUCTIONS = (
    "c4nary statically audits a GGUF model's embedded Jinja2 chat template, "
    "metadata, tokenizer, and structure for risk indicators (Jinja2 SSTI / "
    "sandbox-escape primitives, behavioral 'silent-hijack' shapes, hidden "
    "characters, and structural tampering). It parses only -- it never renders "
    "the template or runs the model -- and is read-only and deterministic. "
    "Findings are heuristic risk indicators, NOT proof that a model is safe or "
    "malicious; surface them as 'review recommended', not as a verdict."
)


def _summary(findings: list) -> dict[str, int]:
    counts = summarize(findings)
    return {"fail": counts[FAIL], "warn": counts[WARN], "info": counts[INFO]}


def run_scan(
    path: str,
    remote: bool = False,
    hf_filename: str | None = None,
    manifest_path: str | None = None,
    deep_tokenizer: bool = False,
    bundle: bool = False,
) -> dict:
    """Audit a GGUF model's chat template, metadata, and tokenizer for risk indicators.

    This is the primary tool. It runs the template (TPL*), metadata (MET*),
    tokenizer (TOK*), and -- for local files -- structural (STR*) rule sets; with
    ``bundle`` it also audits the repo's config, tokenizer.json, model card, and any
    ``auto_map`` custom Python (CFG*/NRM*/DOC*/RMT*/TPL030). It returns every finding
    with its stable rule id, severity
    (FAIL/WARN/INFO), plain-language detail, and location. Nothing is rendered or executed.

    Args:
        path: Path to a local ``.gguf`` file. With ``remote=True`` this is instead
            a Hugging Face repo id (``org/name``), an ``hf://org/name`` ref, or a
            direct ``http(s)://`` URL to a ``.gguf``.
        remote: If true, range-fetch only the model's GGUF *header* over the
            network (metadata + template + tensor map, never the weights) instead
            of reading a local file. This is the only network-touching option and
            needs the ``requests`` extra. Structural (STR*) and whole-file
            integrity checks are skipped on remote scans (the file is truncated).
        hf_filename: With ``remote=True``, the specific ``.gguf`` filename to fetch
            from the repo. Omit to auto-pick the smallest single-file quant.
        manifest_path: Optional path to a known-good manifest JSON (from the
            ``hash`` tool). Local scans compare against it and emit INT* drift
            findings. Ignored for remote scans.
        deep_tokenizer: If true, materialize the full tokenizer vocab and run the
            template<->tokenizer seam checks (TOK012+). Off by default (extra bandwidth).
        bundle: If true, also fetch (remote) or read (local sibling) the repo's config
            (generation_config.json / config.json), tokenizer.json, special-token files,
            model card (README.md), any divergent template source, and the custom Python
            named by an ``auto_map`` entry -- which is read and parsed with ``ast`` only,
            never imported or executed -- and run the CFG / NRM / DOC / TPL030 / RMT
            rules over them. Off by default (more fetching).

    Returns:
        A dict with ``file``, ``sha256`` (null for remote), ``template_sha256``,
        ``findings`` (sorted, deterministic), a ``summary`` count, an honest
        ``verdict`` line, any ``notes`` about skipped checks, a ``coverage`` manifest
        naming what was examined, skipped, absent, truncated or unparseable, plus
        ``tool_version`` and ``rules_bundle_sha256`` identifying the ruleset that ran.
    """
    materialize = _DEEP_TOK_KEYS if (deep_tokenizer or bundle) else None
    if remote:
        from .remote import fetch_remote_model

        model, display = fetch_remote_model(path, hf_filename, materialize=materialize)
        file_sha = None
        template_findings = analyze_templates(model)
        findings = (
            template_findings
            + analyze_metadata(model)
            + analyze_tokenizer(model, deep=deep_tokenizer)
        )
    else:
        model = parse_gguf(path, materialize=materialize)
        display = model.path
        file_sha = sha256_file(path)
        template_findings = analyze_templates(model)
        findings = (
            template_findings
            + analyze_metadata(model)
            + analyze_tokenizer(model, deep=deep_tokenizer)
            + analyze_structure(model)
        )
        if manifest_path:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            findings += compare_manifest(model, file_sha, manifest)

    bundle_surfaces: list[Surface] = []
    if bundle:
        from .bundle import ReadResult, _safe_bundle_name, bundle_findings

        def _bundle_read(
            name: str,
            max_bytes: int = 1 << 20,
            *,
            python_only: bool = False,
        ) -> "ReadResult | None":
            if python_only:
                name = _safe_bundle_name(name)
            if remote:
                from .remote import fetch_repo_text
                return fetch_repo_text(
                    path,
                    name,
                    max_bytes=max_bytes,
                    python_only=python_only,
                )
            sib = os.path.join(os.path.dirname(os.path.abspath(path)), name)
            if os.path.isfile(sib):
                with open(sib, "rb") as fh:
                    data = fh.read(max_bytes + 1)
                return ReadResult(
                    data[:max_bytes].decode("utf-8", errors="replace"),
                    truncated=len(data) > max_bytes,
                )
            return None

        bundle_results, bundle_surfaces = bundle_findings(model, _bundle_read)
        findings += bundle_results

    coverage = build_scan_coverage(
        model,
        remote=remote,
        deep_tokenizer=deep_tokenizer,
        bundle=bundle,
        manifest_requested=manifest_path is not None,
        template_findings=template_findings,
        bundle_surfaces=bundle_surfaces,
    )

    return {
        "file": display,
        "sha256": file_sha,
        "template_sha256": model_template_sha256(model),
        "findings": findings_to_dicts(findings),
        "summary": _summary(findings),
        "verdict": verdict_line(findings),
        "notes": coverage.notes(),
        "coverage": coverage.to_dict(),
        "tool_version": __version__,
        "rules_bundle_sha256": rules_bundle_sha256(),
    }


def run_diff(a: str, b: str) -> dict:
    """Structurally diff two local GGUF files (metadata, chat template, tensor map).

    Compares structure only -- raw weight bytes are never read. Useful for
    spotting tampering between a trusted baseline and a candidate model.

    Args:
        a: Path to the first ``.gguf`` file.
        b: Path to the second ``.gguf`` file.

    Returns:
        A dict with ``identical`` (bool) plus ``metadata`` (added/removed/changed),
        ``template_changed`` + a unified ``template_diff``, and ``tensors``
        (added/removed/changed).
    """
    diff = diff_models(parse_gguf(a), parse_gguf(b))
    return {"identical": diff_is_empty(diff), **diff}


def run_hash(path: str, write_manifest_to: str | None = None) -> dict:
    """Hash a GGUF file and optionally write a known-good manifest for later drift checks.

    Args:
        path: Path to a local ``.gguf`` file.
        write_manifest_to: If set, write a known-good manifest (file hash,
            normalized template hash, metadata + tensor-map snapshot) to this
            path. This is the only operation that writes a file, and it only ever
            creates this output artifact -- the input model is never modified.

    Returns:
        A dict with ``file``, ``sha256``, ``template_sha256``, and
        ``manifest_written`` (the path, when a manifest was written).
    """
    model = parse_gguf(path)
    file_sha = sha256_file(path)
    out: dict = {
        "file": model.path,
        "sha256": file_sha,
        "template_sha256": model_template_sha256(model),
    }
    if write_manifest_to:
        write_manifest(model, file_sha, write_manifest_to, path)
        out["manifest_written"] = write_manifest_to
    return out


def list_rules() -> dict:
    """List every detection rule c4nary can emit (id, severity, title, description).

    Each finding from ``scan`` maps to exactly one of these stable rule ids, so
    this is the legend for interpreting results.
    """
    return {
        "tool_version": __version__,
        "rules_bundle_sha256": rules_bundle_sha256(),
        "rules": [
            {
                "rule_id": rule.rule_id,
                "severity": rule.severity,
                "title": rule.title,
                "description": rule.description,
            }
            for rule in all_rules()
        ],
    }


def build_server():
    """Construct the FastMCP server with the four c4nary tools registered.

    ``mcp`` is imported here (not at module load) so the offline core and the
    tool functions above stay importable without the optional dependency.
    """
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import Icon
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "c4nary's MCP server needs the 'mcp' package "
            "(pip install c4nary[mcp])."
        ) from exc

    server = FastMCP(
        "c4nary",
        instructions=SERVER_INSTRUCTIONS,
        icons=[Icon(src=ICON_URL, mimeType="image/png")],
    )
    # FastMCP exposes no `version` kwarg; without this the handshake reports the
    # mcp SDK's version as ours. Set it on the underlying low-level server.
    server._mcp_server.version = __version__
    server.tool(name="scan")(run_scan)
    server.tool(name="diff")(run_diff)
    server.tool(name="hash")(run_hash)
    server.tool(name="rules")(list_rules)
    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
