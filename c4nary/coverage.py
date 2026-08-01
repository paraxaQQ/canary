"""Deterministic coverage manifest for every scan surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal

if TYPE_CHECKING:
    from .parser import GGUFModel
    from .report import Finding


SurfaceState = Literal["examined", "absent", "skipped", "unparseable", "partial"]
_STATES = frozenset({"examined", "absent", "skipped", "unparseable", "partial"})
_NOTE_SURFACES = ("structure", "manifest", "tokenizer.deep", "bundle")


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
        notes: list[str] = []
        for surface_id in _NOTE_SURFACES:
            surface = by_id.get(surface_id)
            if surface is None:
                continue
            if surface_id == "structure" and surface.state == "examined":
                continue
            if surface_id == "manifest" and surface.state in {"absent", "examined"}:
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
) -> Coverage:
    from .parser import MetaArray
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

    tokens = model.metadata.get("tokenizer.ggml.tokens")
    token_types = model.metadata.get("tokenizer.ggml.token_type")
    deep_ready = (
        isinstance(tokens, MetaArray)
        and isinstance(token_types, MetaArray)
        and not tokens.truncated
        and not token_types.truncated
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
