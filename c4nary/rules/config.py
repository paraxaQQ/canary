"""Decode-time behavioral levers in generation_config.json / config.json (CFG rules).

A config-level backdoor that touches neither weights nor template: ``suppress_tokens`` /
``bad_words_ids`` / ``forced_*`` steer or suppress output at DECODE time. Suppress the
stop token and the model can never end its turn; suppress the tokens a refusal is built
from and it silently cannot refuse. Deterministically auditable against the tokenizer
vocab + the declared stop set -- no weights, no execution. Opt-in bundle scan only.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..parser import MetaArray
from ..report import Finding
from .registry import finding

if TYPE_CHECKING:
    from ..parser import GGUFModel

# Vocab surfaces a refusal is built from (SentencePiece ▁ / GPT2 Ġ word-start stripped).
_REFUSAL_SURFACES = frozenset({
    "sorry", "cannot", "can't", "refuse", "unable", "apologize", "apologies",
    "won't", "decline", "cannot.",
})

_STOP_ID_KEYS = ("eos_token_id", "eot_token_id", "eom_token_id")

# refusal words in RECONSTRUCTED multi-token text -- catches split forms (can't, won't,
# "I cannot") that no single vocab token spells.
_REFUSAL_RE = re.compile(
    r"\b(sorry|cannot|can'?t|won'?t|refuse|unable|apolog(?:y|ize|ies)|decline)\b", re.I)


def _flatten_ids(v) -> set[int]:
    out: set[int] = set()

    def add(x) -> None:
        if isinstance(x, bool):
            return
        if isinstance(x, int):
            out.add(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                add(y)

    add(v)
    return out


def _banned_single_ids(cfg: dict) -> set[int]:
    """bad_words_ids is list[list[int]]; only a length-1 inner list bans a *single* token
    (a longer sequence bans a phrase, not the token, and must not count here)."""
    out: set[int] = set()
    bw = cfg.get("bad_words_ids")
    if isinstance(bw, (list, tuple)):
        for seq in bw:
            if (isinstance(seq, (list, tuple)) and len(seq) == 1
                    and isinstance(seq[0], int) and not isinstance(seq[0], bool)):
                out.add(seq[0])
    return out


def _bad_word_seqs(cfg: dict) -> list[list[int]]:
    """Multi-token (len>=2) bad_words_ids sequences -- a banned *phrase*."""
    out: list[list[int]] = []
    bw = cfg.get("bad_words_ids")
    if isinstance(bw, (list, tuple)):
        for seq in bw:
            if (isinstance(seq, (list, tuple)) and len(seq) >= 2
                    and all(isinstance(x, int) and not isinstance(x, bool) for x in seq)):
                out.append(list(seq))
    return out


def _reconstruct(ids: list[int], surfaces) -> str:
    """Join token surfaces into text, mapping SentencePiece ▁ / GPT2 Ġ word-start to a
    space. A crude decode -- enough to read a banned sequence as a phrase."""
    parts = []
    for tid in ids:
        if 0 <= tid < len(surfaces) and isinstance(surfaces[tid], str):
            s = surfaces[tid]
            parts.append(" " + s[1:] if s[:1] in ("▁", "Ġ") else s)
    return "".join(parts)


def _stop_ids(model: GGUFModel, cfg: dict) -> set[int]:
    ids: set[int] = set()
    for k in _STOP_ID_KEYS:
        v = model.metadata.get(f"tokenizer.ggml.{k}")
        if isinstance(v, int) and not isinstance(v, bool):
            ids.add(v)
    ids |= _flatten_ids(cfg.get("eos_token_id"))
    return ids


def analyze_config(model: GGUFModel, cfg: dict) -> list[Finding]:
    """Audit a parsed generation_config.json / config.json dict against the model."""
    if not isinstance(cfg, dict):
        return []
    findings: list[Finding] = []

    # S2 -- free-text config strings (a bundled system_prompt / default message): the
    # injection scanner + (threat-model §5) AST-routing of any Jinja-carrying value. Runs
    # regardless of the decode-lever checks below.
    import dataclasses

    from .template import analyze_embedded_template, scan_injection_text
    for k, v in cfg.items():
        if not (isinstance(v, str) and len(v) >= 16):
            continue
        concealed, hits = scan_injection_text(v)
        if concealed:
            findings.append(finding(
                "MET020",
                f"config {k!r} contains hidden codepoints "
                f"({', '.join(f'U+{cp:04X}' for cp in concealed)}).",
                location=k))
        if hits:
            findings.append(finding(
                "MET021",
                f"config {k!r} contains injection-idiom text (e.g. {hits[0]!r}).",
                location=k))
        for f in analyze_embedded_template(v):
            findings.append(dataclasses.replace(
                f, location=f"{k}:{f.location}" if f.location else k))

    # always-suppressed = suppress_tokens (every step) + single-token bad_words. NB:
    # begin_suppress_tokens is EXCLUDED -- suppressing eos only at the first step is a
    # legitimate anti-empty-output measure (Whisper does exactly this); always-suppress
    # is the hostile case.
    suppressed = _flatten_ids(cfg.get("suppress_tokens")) | _banned_single_ids(cfg)
    bad_seqs = _bad_word_seqs(cfg)
    if not suppressed and not bad_seqs:
        return findings

    # CFG001 -- stop-token suppression: the model can never emit end-of-turn.
    muzzled = sorted(suppressed & _stop_ids(model, cfg))
    if muzzled:
        findings.append(finding(
            "CFG001",
            f"generation config always-suppresses the stop-token id(s) {muzzled} "
            f"(suppress_tokens / single-token bad_words_ids): the model cannot emit "
            f"end-of-turn, so it cannot stop or cleanly end a refusal.",
            location="generation_config.suppress_tokens"))

    # CFG002 -- refusal suppression. (a) suppress_tokens is per-token, so a single token
    # whose surface spells a refusal is the direct lever (a shared sub-piece like ▁can
    # can't be targeted without breaking the model). (b) bad_words_ids can ban a
    # multi-token sequence, so we also RECONSTRUCT each banned phrase and match split
    # forms ("can't", "I cannot") that no single token spells.
    toks = model.metadata.get("tokenizer.ggml.tokens")
    # A valid vocab is an array (MetaArray); a crafted model can put a scalar here --
    # guard so the whole --bundle scan doesn't crash on toks.truncated / .preview.
    if isinstance(toks, MetaArray) and not toks.truncated:
        surfaces = toks.preview
        hits: list[str] = []
        for tid in sorted(suppressed):
            if (0 <= tid < len(surfaces) and isinstance(surfaces[tid], str)
                    and surfaces[tid].lstrip("▁Ġ ").lower() in _REFUSAL_SURFACES):
                hits.append(f"{tid}={surfaces[tid]!r}")
        for seq in bad_seqs:
            text = _reconstruct(seq, surfaces)
            if _REFUSAL_RE.search(text):
                hits.append(f"{seq}={text.strip()!r}")
        if hits:
            findings.append(finding(
                "CFG002",
                f"generation config suppresses/bans tokens that spell a refusal "
                f"({', '.join(hits[:6])}): decode-time refusal suppression, steering the "
                f"model away from declining. Manual review.",
                location="generation_config"))

    return findings
