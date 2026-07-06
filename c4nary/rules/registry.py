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
    Rule("TPL030", WARN, "Repo chat template diverges from the GGUF's",
         "The repo bundles a chat template (tokenizer_config.json / chat_template.jinja) "
         "that differs from the GGUF's embedded template. A transformers loader reads the "
         "repo file, a GGUF loader reads the embedded one - a divergent template can hide a "
         "backdoor from a GGUF-only audit. The divergent template is scanned too."),
    Rule("TPL031", WARN, "Template pulls in external code (include/import/extends)",
         "The chat template uses include / import / extends to pull in external template "
         "code. A chat template should be self-contained; this hides logic outside the "
         "audited file. Anomaly - manual review."),
    Rule("TPL032", WARN, "Template uses a decode / deserialize filter",
         "The chat template uses a decode/deserialize filter (b64decode, from_json, "
         "urldecode, ...) - the machinery that turns an encoded blob into live content, a "
         "common obfuscation transport. Anomaly - manual review."),
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
    Rule("MET020", WARN, "Hidden codepoints in a metadata string",
         "A free-text metadata value contains invisible / zero-width / bidi codepoints that "
         "conceal text; metadata should be plain printable text."),
    Rule("MET021", WARN, "Injection-idiom text in a metadata string",
         "A free-text metadata value (e.g. general.description) contains imperative "
         "instruction idioms not tied to the conversation - a hidden instruction or second "
         "template stashed in metadata. Heuristic; manual review, not proof."),
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
    # ---- Tokenizer seam / reachability (deep pass, --deep-tokenizer) ------ #
    Rule("TOK012", INFO, "Confusable / legacy role-token forms present",
         "A role/turn delimiter has a confusable homoglyph twin also registered as a "
         "special token (e.g. ASCII <|User|> alongside fullwidth <｜User｜>). The "
         "legacy/twin form may still be honored by the model as a boundary; whether it "
         "actually is cannot be confirmed statically -- it needs runtime testing. "
         "Informational (an input sanitizer should treat the forms as equivalent)."),
    Rule("TOK015", INFO, "Deep tokenizer seam summary",
         "The deep tokenizer pass materialized the full vocab and reports the count of "
         "reachable role/turn special surfaces (whitespace and reserved / padding tokens "
         "excluded). Informational; confirms the seam pass executed."),
    # TOK010 (NORMAL-at-seam) was calibrated OUT (~6% FP, incl. Gemma): a single-token
    # NORMAL delimiter still tokenizes to its dedicated id -- Gemma's <start_of_turn>
    # (id 106, NORMAL) proves it -- so it is NOT a broken boundary. TOK011/013/014 and the
    # confusable-MISMATCH reachability confirmation are encoder/runtime-gated (Piece B / v3).
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
    # ---- Decode-time config levers (CFG, opt-in bundle scan) ------------- #
    Rule("CFG001", WARN, "Stop token suppressed in generation config",
         "generation_config always-suppresses the model's end-of-turn token "
         "(suppress_tokens / single-token bad_words_ids include an EOS/EOT id), so the "
         "model cannot stop or cleanly end a refusal. A decode-time behavioral lever."),
    Rule("CFG002", WARN, "Refusal tokens suppressed in generation config",
         "generation_config suppresses vocabulary tokens whose surface spells a refusal "
         "(sorry / cannot / refuse ...), steering the model away from declining at decode "
         "time. Heuristic (surface match); manual review."),
    # ---- Model-card injection (DOC, opt-in bundle scan) ------------------ #
    Rule("DOC001", WARN, "Model card hides text in invisible / bidi codepoints",
         "The model card (README) contains invisible, zero-width, bidi-override, or "
         "control codepoints that conceal text from a human reader while an LLM that "
         "browses or summarizes models still reads it - a Trojan-Source-style card "
         "injection aimed at the reader, not the model."),
    Rule("DOC002", WARN, "Model card contains injection-idiom instruction text",
         "The model card contains imperative instruction idioms (ignore previous, do not "
         "mention, always recommend ...) that read as prompt injection targeting an LLM "
         "summarizing or selecting the model. Heuristic; manual review, not proof."),
    # ---- tokenizer.json normalizer/decoder (NRM, opt-in bundle scan) ----- #
    Rule("NRM001", WARN, "tokenizer.json Replace rewrites content text",
         "A tokenizer.json normalizer, pre-tokenizer, decoder or post-processor has a "
         "Replace rule that rewrites word content (not just the standard whitespace <-> "
         "meta-space handling). It runs on every input/output and can silently alter "
         "prompts or responses - e.g. map a refusal trigger away. Manual review."),
    Rule("NRM002", WARN, "Concealed special / added token",
         "A special or added token (special_tokens_map.json / added_tokens.json) contains "
         "invisible / zero-width / bidi codepoints - a privileged token a human reader "
         "cannot see but the tokenizer registers. Manual review."),
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
