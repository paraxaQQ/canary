"""Central rule registry.

Every finding c4nary emits maps to exactly one rule defined here (invariant
§7.5: no opaque scoring). A rule fixes its own ``id``, ``severity``, and
``title``; callers supply only the per-occurrence ``detail`` and ``location``.
The ``rules`` subcommand prints this table.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..report import FAIL, INFO, WARN, Finding


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: str
    title: str
    description: str


# Ordered for stable display in `canary rules`.
_RULES: tuple[Rule, ...] = (
    # ---- Template static analysis (TPL) ---------------------------------- #
    Rule("TPL000", WARN, "Template failed to parse as Jinja2",
         "The embedded chat template could not be parsed as Jinja2. It cannot "
         "be statically analyzed; treat with suspicion and review by hand."),
    Rule("TPL001", FAIL, "Dunder attribute access",
         "Access to a Python dunder attribute (e.g. __class__, __globals__, "
         "__subclasses__). These are the building blocks of Jinja2 SSTI "
         "sandbox escapes."),
    Rule("TPL002", FAIL, "SSTI gadget identifier",
         "Reference to a Jinja2 built-in global used to reach arbitrary Python "
         "(lipsum, cycler, joiner). These do not appear in legitimate chat "
         "templates."),
    Rule("TPL003", FAIL, "Dangerous callable or module name",
         "Reference to a name associated with code execution or OS access "
         "(os, subprocess, popen, system, eval, exec, getattr, __import__, ...)."),
    Rule("TPL004", FAIL, "Abusable attribute filter",
         "Use of the |attr filter or map('attr'), which bypasses Jinja2's "
         "attribute-access sandbox and is a known SSTI primitive."),
    Rule("TPL005", FAIL, "Reconstructed dangerous token via string ops",
         "Adjacent string literals combine (via ~, +, or |join) to spell a "
         "dangerous token such as 'popen' or '__globals__' - a common way to "
         "hide an SSTI payload from naive scanners."),
    Rule("TPL010", WARN, "Conditional keyed on a suspicious literal",
         "A branch or comparison tests a sensitive literal (password, secret, "
         "apikey, ...). Possible behavioral trigger - manual review; not proof "
         "of malice."),
    Rule("TPL011", WARN, "Embedded URL or IP in template",
         "The template text contains a URL or IP address. Templates are not "
         "rendered by this tool, but embedded endpoints warrant review."),
    Rule("TPL012", WARN, "Unusually large template",
         "The template exceeds the size threshold. Large templates can hide "
         "payloads and slow loaders."),
    Rule("TPL013", WARN, "Deeply nested template",
         "The template AST nests beyond the depth threshold, which can hide "
         "logic and indicates obfuscation."),
    # ---- Behavioral 'silent-hijack' analysis (render-faithful, no exec) --- #
    Rule("TPL020", WARN, "Branch keyed on message content",
         "A conditional test inspects message content/text instead of role or "
         "position. Vetted templates branch only on structure; testing what the "
         "user said is the trigger shape of a behavioral backdoor. Deviates from "
         "a vetted baseline - manual review, not proof of malice."),
    Rule("TPL021", FAIL, "Content-gated instruction injection",
         "A content-keyed branch also emits imperative instruction text not "
         "sourced from the conversation: a trigger condition stacked with an "
         "injected instruction."),
    Rule("TPL022", WARN, "Date/time-gated behavior (logic bomb)",
         "A conditional test depends on the current date/time, so behavior "
         "changes by date. Possible logic bomb - manual review."),
    Rule("TPL023", WARN, "Hidden instruction-like text",
         "The template emits imperative instruction idioms (ignore previous, "
         "always recommend, do not mention...) not derived from the conversation. "
         "Possible hidden instruction injection - manual review, not proof."),
    Rule("TPL024", FAIL, "Invisible / format-control codepoints",
         "The template contains zero-width, format-control, or Unicode-tag "
         "codepoints that can hide instructions from a human reader. Legitimate "
         "templates are plain printable text."),
    Rule("TPL025", FAIL, "Bidirectional-override codepoints (Trojan Source)",
         "The template contains bidi-override codepoints that can make the "
         "rendered text differ from what is actually tokenized."),
    Rule("TPL026", WARN, "Raw control characters",
         "The template contains C0/C1 control characters (other than tab/newline). "
         "Usually an artifact, occasionally smuggling - manual review."),
    Rule("TPL027", WARN, "Reconstructed instruction text",
         "String operations assemble instruction-like text from split literals, "
         "a way to evade literal scanning. Deviates from a vetted baseline."),
    Rule("TPL100", INFO, "Matches a known-good template",
         "The normalized template hash matches a vetted reference template; "
         "content-level template rules were skipped."),
    Rule("TPL101", INFO, "No embedded chat template",
         "The file declares no tokenizer.chat_template metadata key."),
    # ---- Metadata sanity (MET) ------------------------------------------- #
    Rule("MET001", WARN, "Embedded URL or IP in metadata",
         "A metadata value contains a URL or IP address."),
    Rule("MET002", WARN, "Oversized metadata field",
         "A string metadata value exceeds the size threshold (possible hidden "
         "payload or resource exhaustion)."),
    Rule("MET003", INFO, "Non-standard metadata key",
         "A metadata key falls outside the common GGUF namespaces; informational."),
    Rule("MET004", INFO, "String metadata field",
         "A scalar string metadata value, listed for review."),
    Rule("MET005", WARN, "Architecture / quantization inconsistency",
         "Declared architecture or file type appears inconsistent with the "
         "tensor map; review for tampering."),
    Rule("MET006", INFO, "Unrecognized tensor dtype",
         "A tensor uses a GGML type id this tool does not recognize."),
    # ---- Metadata vs tensor-map consistency (deterministic) -------------- #
    Rule("MET010", FAIL, "block_count vs layer-tensor mismatch",
         "Declared <arch>.block_count disagrees with the number of distinct "
         "blk.N layers in the tensor map (or indices are non-contiguous)."),
    Rule("MET011", FAIL, "embedding_length vs token_embd mismatch",
         "Declared <arch>.embedding_length matches neither axis of the "
         "token_embd.weight tensor."),
    Rule("MET012", FAIL, "Attention-head divisibility violation",
         "embedding_length is not divisible by head_count, or head_count is not "
         "divisible by head_count_kv: an impossible attention configuration."),
    Rule("MET013", FAIL, "feed_forward_length vs ffn tensor mismatch",
         "Declared <arch>.feed_forward_length disagrees with the intermediate "
         "dimension of the ffn_* tensors."),
    Rule("MET014", WARN, "Quantization label vs tensor dtype",
         "Declared general.file_type disagrees with the dominant weight-tensor "
         "dtype; possible mislabeling. Heuristic - manual review."),
    Rule("MET015", WARN, "Rope / context contradiction",
         "context_length, rope.freq_base, or rope.scaling values are internally "
         "contradictory (e.g. context below original, non-positive freq_base)."),
    Rule("MET016", FAIL, "Duplicate metadata key",
         "A metadata key appears more than once. Scanners read the first copy "
         "while some loaders read the last - a parser-differential evasion."),
    # ---- Tokenizer consistency (TOK) ------------------------------------- #
    Rule("TOK001", FAIL, "Special-token id out of range",
         "A tokenizer special-token id (bos/eos/unk/pad/...) is >= the vocabulary "
         "size, so it indexes outside the token table."),
    Rule("TOK002", FAIL, "Vocabulary desynchronized from tensors",
         "The tokenizer token count disagrees with the vocabulary axis of the "
         "token_embd / output tensor beyond padding tolerance."),
    Rule("TOK003", FAIL, "Parallel tokenizer arrays length mismatch",
         "tokenizer.ggml.scores / .token_type length differs from the token "
         "count, or a token_type value is outside the enum."),
    Rule("TOK004", WARN, "Oversized vocabulary token",
         "A vocabulary token's on-disk byte length is implausibly large "
         "(resource exhaustion; > INT32_MAX risks signed-cast overflow in loaders)."),
    Rule("TOK005", WARN, "BOS/EOS flag inconsistency",
         "add_bos_token / add_eos_token is set but the corresponding token id is "
         "missing or out of range."),
    # ---- Structural / parser-exploitation (STR) -------------------------- #
    Rule("STR001", FAIL, "Tensor element-count / size overflow",
         "A tensor's element count or byte size overflows a signed 64-bit value, "
         "which can wrap to a tiny allocation in C loaders."),
    Rule("STR003", FAIL, "Tensor data offset out of bounds",
         "A tensor's data offset (plus its computed size) falls outside the "
         "file, indicating a crafted header aimed at an out-of-bounds read."),
    Rule("STR004", WARN, "Overlapping tensor data regions",
         "Two tensors' computed byte ranges overlap (aliasing), which no "
         "legitimate GGUF does."),
    Rule("STR005", WARN, "Implausible general.alignment",
         "general.alignment is zero, not a power of two, or out of a sane range."),
    Rule("STR006", WARN, "Non-block-divisible tensor dimension",
         "A quantized tensor's innermost dimension is not a multiple of its block "
         "size, or a dimension is zero."),
    Rule("STR007", WARN, "Unknown ggml tensor type",
         "A tensor uses a ggml type id outside the known enum; its byte size is "
         "unknown and it exercises an untested loader path."),
    Rule("STR008", INFO, "Suspicious tensor name",
         "A tensor name is overlong, contains control/NUL bytes, or path-traversal "
         "tokens."),
    # ---- Integrity / provenance (INT) ------------------------------------ #
    Rule("INT001", FAIL, "Manifest mismatch: file hash",
         "The file SHA-256 differs from the manifest's expected value."),
    Rule("INT002", FAIL, "Manifest mismatch: template hash",
         "The chat-template hash differs from the manifest's expected value."),
    Rule("INT003", WARN, "Manifest mismatch: metadata",
         "A metadata key/value differs from the manifest (added, removed, or "
         "changed)."),
    Rule("INT004", WARN, "Manifest mismatch: tensor map",
         "The tensor map (names, shapes, or dtypes) differs from the manifest."),
    Rule("INT005", WARN, "Injected or anomalous tensor",
         "A tensor sits outside the declared layer range, duplicates a name, or "
         "matches adapter/LoRA naming - possible surgery on the model."),
    Rule("INT006", INFO, "Sharded / multi-file model",
         "This file is one shard of a multi-file model (split.count > 1); the "
         "verdict covers only the scanned shard."),
)

_BY_ID: dict[str, Rule] = {r.rule_id: r for r in _RULES}


def all_rules() -> tuple[Rule, ...]:
    return _RULES


def get_rule(rule_id: str) -> Rule:
    try:
        return _BY_ID[rule_id]
    except KeyError:  # pragma: no cover - programmer error
        raise KeyError(f"unregistered rule id: {rule_id!r}") from None


def finding(rule_id: str, detail: str, location: str | None = None) -> Finding:
    """Build a :class:`Finding` from a registered rule id.

    Severity and title come from the registry; callers supply only the
    occurrence-specific ``detail`` and ``location``. This guarantees every
    finding is explainable and that severities cannot drift per call site.
    """

    rule = get_rule(rule_id)
    return Finding(
        rule_id=rule.rule_id,
        severity=rule.severity,
        title=rule.title,
        detail=detail,
        location=location,
    )
