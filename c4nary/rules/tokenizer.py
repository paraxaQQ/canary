"""Tokenizer consistency rules (TOK001-005).

Deterministic cross-checks over ``tokenizer.ggml.*`` metadata and the tensor
map: special-token ids in range, vocabulary synchronized with the embedding /
output tensors, parallel arrays the same length, and BOS/EOS flags consistent.
Array *contents* are previewed (first 64), but ``MetaArray.length`` is the true
count, so all length/range checks are exact. No weight data is read.
"""

from __future__ import annotations

import re

from ..parser import INT32_MAX, GGUFModel, MetaArray
from ..report import Finding
from ..vocab import Vocab, resolve_vocab
from .registry import finding

# Sentinels some exporters use for "unset" (uint32 -1 / int32 -1).
_UNSET = frozenset({-1, 0xFFFFFFFF})

_SPECIAL_ID_KEYS = (
    "bos_token_id", "eos_token_id", "eot_token_id", "eom_token_id",
    "unknown_token_id", "separator_token_id", "padding_token_id",
    "cls_token_id", "mask_token_id",
)

# tokenizer.ggml.token_type enum (NORMAL..BYTE).
_TOKEN_TYPE_RANGE = range(1, 7)
_NORMAL, _CONTROL, _USER_DEFINED = 1, 3, 4

# Role/turn delimiter shapes -- family-agnostic and conservative (no whitespace,
# bounded length: structural boundary markers, not prose). Pipe-wrapped angle forms
# (<|x|>, plus the fullwidth/confusable pipe set), bare angle tags (<x>), and bracket
# forms ([INST] / [/INST]).
_PIPES = "|｜│ǀ∣"
_ROLE_DELIM_RE = re.compile(
    rf"<[{_PIPES}][^<>\n]{{0,46}}?[{_PIPES}]>"
    r"|<[/A-Za-z][^<>\s]{0,46}?>"
    r"|\[/?[A-Z][A-Z0-9_]{0,30}\]"
)


def _is_role_delim(s: str) -> bool:
    return bool(s) and _ROLE_DELIM_RE.fullmatch(s) is not None


def _skeleton(s: str) -> str:
    """Collapse confusable role-token codepoints to an ASCII canonical (pipe family -> |,
    SentencePiece meta ▁ -> _) so homoglyph twins group together. RAW forms are preserved
    and reported; only the grouping key is folded."""
    return "".join("|" if ch in _PIPES else ("_" if ch == "▁" else ch) for ch in s)


def _is_noise_surface(s: str) -> bool:
    """Whitespace or reserved/padding placeholder -- excluded from the surface metric."""
    if not s or s.isspace():
        return True
    return s.strip("<>[]|｜ ▁").lower() in {"", "pad", "unk", "unused", "reserved"}


def _seam_findings(
    model: GGUFModel, deep: bool, vocab: Vocab | None
) -> list[Finding]:
    """Deep tokenizer-seam pass (opt-in ``--deep-tokenizer``). Needs the FULL vocab +
    token_type materialized; a truncated preview cannot answer reachability, so the pass
    no-ops rather than guessing (the CLI surfaces that it was skipped -- silence != clean)."""
    if not deep:
        return []
    if vocab is None or not vocab.complete or vocab.types is None or not vocab.types_complete:
        return []

    surfaces, types = vocab.surfaces, vocab.types
    all_surf: set[str] = set()
    special_surf: set[str] = set()
    control_count = 0
    for i in range(min(len(surfaces), len(types))):
        s, t = surfaces[i], types[i]
        if not isinstance(s, str):
            continue
        all_surf.add(s)
        if t == _CONTROL:
            control_count += 1
        if t in (_CONTROL, _USER_DEFINED):
            special_surf.add(s)

    findings: list[Finding] = []

    # TOK012 (INFO) - confusable / legacy twin forms: two DISTINCT role-token surfaces that
    # collapse to one homoglyph skeleton (ASCII <|User|> + fullwidth <｜User｜>, both
    # registered special). INFO, not WARN: whether the model still honors the legacy twin
    # as a boundary is unverifiable statically -- that needs runtime testing (the v3 sandbox).
    by_skel: dict[str, set[str]] = {}
    for s in special_surf:
        if _is_role_delim(s):
            by_skel.setdefault(_skeleton(s), set()).add(s)
    for skel, forms in sorted(by_skel.items()):
        if len(forms) >= 2:
            findings.append(finding(
                "TOK012",
                f"role-token skeleton {skel!r} has {len(forms)} registered confusable "
                f"forms {sorted(forms)}; a legacy/homoglyph twin may still be honored by "
                f"the model as a boundary (unverifiable statically -- needs runtime).",
                location="tokenizer.ggml.tokens"))

    # TOK015 (INFO) - reachable role-surface summary; confirms the deep pass ran.
    # Whitespace / reserved-padding excluded (never key severity on raw n_control). The
    # broken-boundary WARN was calibrated out (a single-token NORMAL delimiter still
    # resolves to its id, Gemma's <start_of_turn>); reachability stays runtime-gated.
    reachable = sorted(s for s in special_surf
                       if _is_role_delim(s) and not _is_noise_surface(s))
    findings.append(finding(
        "TOK015",
        f"deep tokenizer pass ran: {len(reachable)} reachable role/turn special "
        f"surface(s) in a {len(all_surf)}-token vocab ({control_count} CONTROL); "
        f"whitespace / reserved excluded.",
        location="tokenizer.ggml.tokens"))
    return findings


def analyze_tokenizer(
    model: GGUFModel, deep: bool = False, *, vocab: Vocab | None = None
) -> list[Finding]:
    findings: list[Finding] = []
    meta = model.metadata
    v = resolve_vocab(model, vocab)
    by_name = {t.name: t for t in model.tensors}

    # TOK001 - special-token ids within the vocabulary.
    if v is not None:
        for key in _SPECIAL_ID_KEYS:
            tid = meta.get(f"tokenizer.ggml.{key}")
            if isinstance(tid, int) and not isinstance(tid, bool) and tid not in _UNSET:
                if tid < 0 or tid >= v.length:
                    findings.append(finding(
                        "TOK001",
                        f"{key}={tid} is outside the vocabulary [0, {v.length}).",
                        location=f"tokenizer.ggml.{key}"))

    # TOK002 - vocabulary synchronized with embedding/output tensors. Skipped on
    # shards: the embedding/output tensor may live in a different shard file.
    split_count = meta.get("split.count")
    is_shard = isinstance(split_count, int) and split_count > 1
    if v is not None and not is_shard:
        for tname in ("token_embd.weight", "output.weight"):
            t = by_name.get(tname)
            if t and t.shape and v.length > max(t.shape):
                findings.append(finding(
                    "TOK002",
                    f"token count {v.length} exceeds every axis of {tname} {t.shape}; "
                    f"tokens would index outside the embedding table.",
                    location="tokenizer.ggml.tokens",
                    subject=f"tensor:{tname}"))

    # TOK003 - parallel arrays length + token_type enum. ``scores`` is a GGUF-only
    # parallel array, read directly; ``token_type`` comes from the vocab when it
    # was built from GGUF metadata.
    if v is not None:
        scores = meta.get("tokenizer.ggml.scores")
        if isinstance(scores, MetaArray) and scores.length != v.length:
            findings.append(finding(
                "TOK003",
                f"tokenizer.ggml.scores has {scores.length} entries but there are "
                f"{v.length} tokens.",
                location="tokenizer.ggml.scores",
                subject="length:scores"))
        if v.types_length is not None and v.types_length != v.length:
            findings.append(finding(
                "TOK003",
                f"tokenizer.ggml.token_type has {v.types_length} entries but there are "
                f"{v.length} tokens.",
                location="tokenizer.ggml.token_type",
                subject="length:token_type"))
    # A model with token_type but no token array still gets the enum scan: the
    # vocabulary abstraction does not exist then, so read the array directly.
    enum_values = v.types if v is not None else None
    if v is None:
        raw_types = meta.get("tokenizer.ggml.token_type")
        if isinstance(raw_types, MetaArray):
            enum_values = raw_types.preview
    if enum_values is not None:
        bad = sorted({value for value in enum_values
                      if isinstance(value, int) and value not in _TOKEN_TYPE_RANGE})
        if bad:
            findings.append(finding(
                "TOK003",
                f"token_type contains values outside the enum 1..6: {bad[:8]}.",
                location="tokenizer.ggml.token_type",
                subject="token_type_enum"))

    # TOK004 - implausibly large vocabulary token.
    if v is not None and v.max_surface_bytes > 4096:
        sev_note = " (exceeds INT32_MAX - signed-cast overflow risk)" \
            if v.max_surface_bytes > INT32_MAX else ""
        findings.append(finding(
            "TOK004",
            f"a vocabulary token is {v.max_surface_bytes} bytes{sev_note}.",
            location="tokenizer.ggml.tokens"))

    # TOK005 - add_bos/add_eos flags consistent with the ids they reference.
    for flag, id_key in (("add_bos_token", "bos_token_id"),
                         ("add_eos_token", "eos_token_id")):
        if meta.get(f"tokenizer.ggml.{flag}") is True:
            tid = meta.get(f"tokenizer.ggml.{id_key}")
            ok = (isinstance(tid, int) and not isinstance(tid, bool)
                  and tid not in _UNSET and tid >= 0
                  and (v is None or tid < v.length))
            if not ok:
                findings.append(finding(
                    "TOK005",
                    f"{flag} is set but {id_key} is missing or out of range.",
                    location=f"tokenizer.ggml.{flag}"))

    findings += _seam_findings(model, deep, v)
    return findings
