"""Deterministic coverage manifest for every scan surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal

from .vocab import Vocab, resolve_vocab

if TYPE_CHECKING:
    from .parser import GGUFModel
    from .report import Finding


SurfaceState = Literal["examined", "absent", "skipped", "unparseable", "partial"]
_STATES = frozenset({"examined", "absent", "skipped", "unparseable", "partial"})
_NOTE_SURFACES = (
    "structure", "manifest", "tokenizer.deep", "bundle", "repo.links",
    # Repo-only summaries: the GGUF path keeps its historical stderr exactly.
    "chat_templates", "tokenizer", "metadata", "tensor_map.consistency",
)
_REPO_SUMMARY_SURFACES = frozenset({
    "chat_templates", "tokenizer", "metadata", "tensor_map.consistency",
})


@dataclass(frozen=True)
class Surface:
    id: str
    state: SurfaceState
    reason: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("coverage surface id cannot be empty")
        if self.state not in _STATES:
            raise ValueError(f"invalid coverage state: {self.state!r}")
        if not self.reason:
            raise ValueError(f"coverage surface {self.id!r} needs a reason")

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "state": self.state, "reason": self.reason}


@dataclass(frozen=True)
class Coverage:
    surfaces: tuple[Surface, ...]

    @classmethod
    def from_surfaces(cls, surfaces: Iterable[Surface]) -> "Coverage":
        by_id: dict[str, Surface] = {}
        for surface in surfaces:
            if surface.id in by_id:
                raise ValueError(f"duplicate coverage surface id: {surface.id!r}")
            by_id[surface.id] = surface
        return cls(tuple(by_id[key] for key in sorted(by_id)))

    def to_dict(self) -> list[dict[str, str]]:
        return [surface.to_dict() for surface in self.surfaces]

    def notes(self) -> list[str]:
        by_id = {surface.id: surface for surface in self.surfaces}
        repo_mode = "repo.source" in by_id
        notes: list[str] = []
        for surface_id in _NOTE_SURFACES:
            surface = by_id.get(surface_id)
            if surface is None:
                continue
            if surface_id == "structure" and surface.state == "examined":
                continue
            if surface_id == "manifest" and surface.state in {"absent", "examined"}:
                # A repo manifest is unsupported, not silently absent.
                if not (repo_mode and surface.state == "absent"):
                    continue
            if surface_id == "repo.links" and surface.state == "examined":
                continue
            if surface_id in _REPO_SUMMARY_SURFACES:
                if not repo_mode or surface.state == "examined":
                    continue
            notes.append(surface.reason)
        return notes


def build_scan_coverage(
    model: "GGUFModel",
    *,
    remote: bool,
    deep_tokenizer: bool,
    bundle: bool,
    manifest_requested: bool,
    template_findings: list["Finding"],
    baseline_requested: bool = False,
    suppressed_count: int = 0,
    policy_requested: bool = False,
    retiered_count: int = 0,
    bundle_surfaces: Iterable[Surface] = (),
    vocab: Vocab | None = None,
) -> Coverage:
    from .rules.template import embedded_template_unparseable

    template_count = sum(
        isinstance(value, str)
        for key, value in model.metadata.items()
        if key == "tokenizer.chat_template"
        or key.startswith("tokenizer.chat_template.")
    )
    parse_failures = sum(finding.rule_id == "TPL000" for finding in template_findings)
    # TPL100 is the vetted-reference fast path: the template matched a known-good hash,
    # so it was never parsed and no content rule ran. "examined" would claim work the
    # tool deliberately skipped.
    vetted = sum(finding.rule_id == "TPL100" for finding in template_findings)
    if template_count == 0:
        template_surface = Surface(
            "chat_templates",
            "absent",
            "no string-valued embedded chat template was present.",
        )
    elif parse_failures == 0 and vetted:
        template_surface = Surface(
            "chat_templates",
            "skipped" if vetted >= template_count else "partial",
            f"{vetted} of {template_count} embedded chat template(s) matched a vetted "
            f"reference; content rules were skipped for those.",
        )
    elif parse_failures == 0:
        template_surface = Surface(
            "chat_templates",
            "examined",
            f"all {template_count} embedded chat template(s) were parsed and examined.",
        )
    elif parse_failures == template_count:
        template_surface = Surface(
            "chat_templates",
            "unparseable",
            f"all {template_count} embedded chat template(s) failed Jinja parsing.",
        )
    else:
        template_surface = Surface(
            "chat_templates",
            "partial",
            f"{parse_failures} of {template_count} embedded chat template(s) failed parsing.",
        )

    vocab_value = resolve_vocab(model, vocab)
    deep_ready = (
        vocab_value is not None
        and vocab_value.complete
        and vocab_value.types is not None
        and vocab_value.types_complete
    )
    if not deep_tokenizer:
        deep_surface = Surface(
            "tokenizer.deep",
            "skipped",
            "deep tokenizer seam checks (TOK012/TOK015) not run; enable the deep tokenizer pass.",
        )
    elif deep_ready:
        deep_surface = Surface(
            "tokenizer.deep",
            "examined",
            "deep tokenizer pass ran; TOK012/TOK015 seam checks were active.",
        )
    else:
        deep_surface = Surface(
            "tokenizer.deep",
            "partial",
            "deep tokenizer pass requested, but full token and token_type arrays were unavailable "
            "or truncated; TOK012/TOK015 were skipped.",
        )

    split_count = model.metadata.get("split.count")
    is_shard = isinstance(split_count, int) and split_count > 1
    consistency_surface = Surface(
        "tensor_map.consistency",
        "partial" if is_shard else "examined",
        (
            "multi-file shard: MET010/MET011/MET013/MET014 and TOK002 need a complete "
            "tensor map and were skipped."
            if is_shard
            else "metadata, tokenizer, and tensor-map consistency checks were examined."
        ),
    )
    structure_surface = Surface(
        "structure",
        "skipped" if remote else "examined",
        (
            "remote header scan: structural (STR*) and whole-file integrity checks need "
            "the complete file and were skipped."
            if remote
            else "local structural checks examined the complete file."
        ),
    )
    if not manifest_requested:
        manifest_surface = Surface(
            "manifest",
            "absent",
            "no comparison manifest was supplied.",
        )
    elif remote:
        manifest_surface = Surface(
            "manifest",
            "skipped",
            "manifest comparison ignored for remote scans because it needs the full file.",
        )
    else:
        manifest_surface = Surface(
            "manifest",
            "examined",
            "the supplied manifest was compared against the complete local file.",
        )
    bundle_surface = Surface(
        "bundle",
        "examined" if bundle else "skipped",
        (
            "bundle scan ran across repo config, tokenizer, template, and model-card surfaces."
            if bundle
            else "bundle scan not run; enable it to audit repo config, tokenizer, template, "
            "and model-card surfaces."
        ),
    )
    baseline_surface = Surface(
        "baseline",
        "examined" if baseline_requested else "absent",
        (
            f"the supplied baseline suppressed {suppressed_count} finding(s); "
            "all suppressed findings remain in the report."
            if baseline_requested
            else "no suppression baseline was supplied."
        ),
    )

    # A policy that downgrades FAIL to WARN changes the verdict and the exit code, so
    # without a row a re-tiered report is indistinguishable from a clean one.
    policy_surface = Surface(
        "policy",
        "examined" if policy_requested else "absent",
        (
            f"the supplied policy re-tiered {retiered_count} finding(s) from their "
            "registered severity."
            if policy_requested
            else "no severity policy was supplied; registered severities applied."
        ),
    )

    unparseable_metadata = [
        Surface(
            f"metadata.{key}.embedded_jinja",
            "unparseable",
            f"metadata value {key!r} contained Jinja delimiters but failed parsing.",
        )
        for key, value in sorted(model.metadata.items())
        if isinstance(value, str)
        and key != "tokenizer.chat_template"
        and not key.startswith("tokenizer.chat_template.")
        and embedded_template_unparseable(value)
    ]

    return Coverage.from_surfaces([
        template_surface,
        Surface("metadata", "examined", "GGUF metadata keys and values were examined."),
        Surface("tokenizer", "examined", "GGUF tokenizer metadata was examined."),
        deep_surface,
        consistency_surface,
        structure_surface,
        manifest_surface,
        baseline_surface,
        policy_surface,
        bundle_surface,
        *unparseable_metadata,
        *bundle_surfaces,
    ])


def _repo_links_surface(links) -> Surface:
    refused = sorted(links, key=lambda link: (link.name, link.target, link.kind))
    if not refused:
        return Surface(
            "repo.links",
            "examined",
            "no allowlisted repo file resolved outside the repository root.",
        )

    from .rules.repo import MAX_REP001_FINDINGS

    escapes = [link for link in refused if link.kind == "escape"]
    others = [link for link in refused if link.kind != "escape"]
    described = [
        *escapes[:MAX_REP001_FINDINGS],
        *others,
    ]
    parts = [
        f"{link.name} -> {link.target} ({link.reason})" for link in described
    ]
    omitted = max(0, len(escapes) - MAX_REP001_FINDINGS)
    if omitted:
        parts.append(f"{omitted} more escaping file(s) were not listed.")
    return Surface(
        "repo.links",
        "skipped",
        "repo link containment refused surface(s): " + "; ".join(parts),
    )


def _tensor_map_surface(artifact) -> Surface:
    """Honest tensor-map row for a repo: absent, or header/axis state."""
    headers = artifact.safetensors
    if not headers:
        # Keep the historical phrase (GGUF-only checks not applicable) while
        # naming the concrete reason a repo scan has no tensor axis at all.
        return Surface(
            "tensor_map.consistency",
            "absent",
            "a repo target has no GGUF tensor map; no top-level .safetensors "
            "file was present.",
        )

    state = artifact.safetensors_state
    count = len(headers)
    if state == "skipped":
        return Surface(
            "tensor_map.consistency",
            "skipped",
            f"{count} top-level safetensors header(s) were not read; "
            f"tensor-map consistency is unknown. {artifact.safetensors_reason}",
        )
    if state == "unparseable":
        return Surface(
            "tensor_map.consistency",
            "unparseable",
            f"{artifact.safetensors_reason} TOK002 needs a tensor axis and none "
            f"could be parsed.",
        )
    if state == "partial":
        return Surface(
            "tensor_map.consistency",
            "partial",
            f"{artifact.safetensors_reason} Only the axes that were parsed could "
            f"be checked.",
        )

    # Every header parsed. The axis check still needs a candidate and a vocab.
    if not artifact.tensor_axes:
        return Surface(
            "tensor_map.consistency",
            "partial",
            f"all {count} top-level safetensors header(s) were parsed, but no "
            f"embedding or output tensor axis was found; TOK002 was not "
            f"applicable.",
        )
    if artifact.vocab is None:
        return Surface(
            "tensor_map.consistency",
            "partial",
            f"all {count} top-level safetensors header(s) were parsed, but no "
            f"usable tokenizer.json vocabulary exists to compare against the "
            f"tensor axes.",
        )
    return Surface(
        "tensor_map.consistency",
        "examined",
        f"all {count} top-level safetensors header(s) were parsed and "
        f"{len(artifact.tensor_axes)} embedding/output tensor axis/axes were "
        f"compared against the {artifact.vocab.length}-token vocabulary; only "
        f"header bytes were read.",
    )


def _safetensors_surfaces(artifact) -> list[Surface]:
    """One row per top-level header. An absent header adds no per-file row."""
    return [
        Surface(
            f"safetensors.{header.name}",
            header.state,
            header.reason,
        )
        for header in artifact.safetensors
    ]


def build_repo_coverage(
    artifact,
    *,
    template_findings: list["Finding"],
    deep_tokenizer: bool,
    bundle_surfaces: Iterable[Surface] = (),
    baseline_requested: bool = False,
    suppressed_count: int = 0,
    policy_requested: bool = False,
    retiered_count: int = 0,
) -> Coverage:
    """Coverage manifest for a repo target, local or remote (GGUF-less ingest)."""
    template_count = artifact.template_count
    parse_failures = sum(finding.rule_id == "TPL000" for finding in template_findings)
    if artifact.template_source_refused:
        template_surface = Surface(
            "chat_templates",
            "skipped",
            "a repo chat template source was not read (refused or unavailable); "
            "its template status is unknown.",
        )
    elif artifact.template_source_partial:
        template_surface = Surface(
            "chat_templates",
            "partial",
            "a repo chat template source exceeded the reader cap; it was examined "
            "partially.",
        )
    elif artifact.template_source_unparseable:
        template_surface = Surface(
            "chat_templates",
            "unparseable",
            "a repo chat template source was present but could not be parsed; its "
            "template status is unknown.",
        )
    elif template_count == 0:
        template_surface = Surface(
            "chat_templates",
            "absent",
            "no repo chat template source was present.",
        )
    elif parse_failures == 0:
        template_surface = Surface(
            "chat_templates",
            "examined",
            f"all {template_count} repo chat template source(s) were parsed and examined.",
        )
    elif parse_failures >= template_count:
        template_surface = Surface(
            "chat_templates",
            "unparseable",
            f"all {template_count} repo chat template source(s) failed Jinja parsing.",
        )
    else:
        template_surface = Surface(
            "chat_templates",
            "partial",
            f"{parse_failures} of {template_count} repo chat template source(s) failed parsing.",
        )

    if artifact.tokenizer_truncated:
        tokenizer_surface = Surface(
            "tokenizer",
            "partial",
            "tokenizer.json exceeded the reader cap; no complete vocabulary was built.",
        )
    elif artifact.tokenizer_unparseable:
        tokenizer_surface = Surface(
            "tokenizer",
            "unparseable",
            "tokenizer.json was present but invalid JSON; no vocabulary was built.",
        )
    elif artifact.vocab is not None:
        tokenizer_surface = Surface(
            "tokenizer",
            "examined",
            f"tokenizer.json supplied a usable vocabulary of {artifact.vocab.length} token(s).",
        )
    else:
        tokenizer_file = artifact.files.get("tokenizer.json")
        if tokenizer_file is None:
            tokenizer_surface = Surface(
                "tokenizer", "absent", "tokenizer.json was absent.")
        elif tokenizer_file.skipped:
            tokenizer_surface = Surface(
                "tokenizer",
                "skipped",
                f"tokenizer.json was not read: {tokenizer_file.skipped}",
            )
        elif tokenizer_file.truncated:
            tokenizer_surface = Surface(
                "tokenizer",
                "partial",
                "tokenizer.json exceeded the reader cap; no complete vocabulary was built.",
            )
        else:
            tokenizer_surface = Surface(
                "tokenizer",
                "absent",
                "tokenizer.json was present but declared no usable vocabulary.",
            )

    if not deep_tokenizer:
        deep_surface = Surface(
            "tokenizer.deep",
            "skipped",
            "deep tokenizer seam checks (TOK012/TOK015) not run; enable the deep tokenizer pass.",
        )
    elif artifact.vocab is None:
        deep_surface = Surface(
            "tokenizer.deep",
            "skipped",
            "deep tokenizer pass requested, but no usable tokenizer.json vocabulary was "
            "available; TOK012/TOK015 were skipped.",
        )
    else:
        deep_surface = Surface(
            "tokenizer.deep",
            "partial",
            "deep tokenizer pass requested, but tokenizer.json carries no token-type split; "
            "TOK012/TOK015 were skipped.",
        )

    baseline_surface = Surface(
        "baseline",
        "examined" if baseline_requested else "absent",
        (
            f"the supplied baseline suppressed {suppressed_count} finding(s); "
            "all suppressed findings remain in the report."
            if baseline_requested
            else "no suppression baseline was supplied."
        ),
    )
    policy_surface = Surface(
        "policy",
        "examined" if policy_requested else "absent",
        (
            f"the supplied policy re-tiered {retiered_count} finding(s) from their "
            "registered severity."
            if policy_requested
            else "no severity policy was supplied; registered severities applied."
        ),
    )

    return Coverage.from_surfaces([
        (
            Surface(
                "repo.source",
                "examined",
                f"remote repository {artifact.display} was the scan target.",
            )
            if artifact.remote
            else Surface(
                "repo.source",
                "examined",
                f"local directory {artifact.display} was the scan root.",
            )
        ),
        (
            Surface(
                "repo.links",
                "absent",
                "local link containment does not apply to a remote repo; "
                "server-served bytes are audited as-is.",
            )
            if artifact.remote
            else _repo_links_surface(artifact.links)
        ),
        template_surface,
        tokenizer_surface,
        deep_surface,
        Surface(
            "metadata",
            "absent",
            "a repo target has no GGUF metadata; MET001-MET016 were not applicable.",
        ),
        Surface(
            "structure",
            "absent",
            "a repo target has no GGUF container; STR001-STR008 were not applicable.",
        ),
        _tensor_map_surface(artifact),
        Surface(
            "manifest",
            "absent",
            "manifest comparison is not supported for a repo target in A3.",
        ),
        baseline_surface,
        policy_surface,
        *_safetensors_surfaces(artifact),
        *bundle_surfaces,
    ])
