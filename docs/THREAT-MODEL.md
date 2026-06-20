<!-- Generated from an adversarially-verified 8-vector threat-enumeration. -->
<!-- Detection tags: DET = deterministic structural check; HEUR = heuristic review-prompt; OUT = outside static scope. -->

# GGUF Backdoor & Silent-Degradation Taxonomy (beyond Jinja SSTI)

Scope: every way to backdoor, silently degrade, or achieve code-exec from a GGUF
*other than* the well-covered chat_template SSTI/RCE class — tagged by what a
**static, read-only, no-render, weights-not-read** auditor can actually catch.
Tags: **DET** = precise deterministic structural check; **HEUR** = heuristic /
review-prompt (false-positive-prone, never "malicious"); **OUT** = fundamentally
outside static scope.

---

## 1. Behavioral "silent-hijack" chat templates — THE DIFFERENTIATOR (lead with this)

A template that **renders faithfully and executes no code** but conditionally
injects instructions, suppresses/replaces content, or branches on what the user
*said*. Every public source (Pillar, NeuralTrust, "Silent Hijack", the
inference-time-backdoor literature) recommends only **manual** inspection or
template-pinning; no shipping tool does render-free static AST detection of this
class — HF's `gguf-jinja-analysis` *renders* in a sandbox (which c4nary refuses).
This is c4nary's underserved niche. All checks use `Environment.parse()` over
`jinja2.nodes`, never `.render()`.

- **Content-conditioned branch (core signature)** — an `If.test` that inspects
  message **content/text** (Getitem `Const('content'/'text')` or Getattr
  `content`/`text` traced to the `messages` loop var) instead of role/loop/
  position. Vetted baselines branch ONLY on `role`, `loop.first/last/index0`,
  `add_generation_prompt`, `messages|length`. **DET** (shape; malice is HEUR).
- **Membership/`in` gate (TPL010 blind spot)** — `{% if 'TRIG' in
  messages[-1]['content'] %}` parses as `Compare op='in'` with the trigger
  `Const` on the **left**; current TPL010 only matches `Compare` *equality*
  against a credential wordlist, so arbitrary-literal `in`-gates evade it. **DET**.
- **Method-call gate** — `m.content.startswith('SUDO')` (`Call` over
  `Getattr('startswith'/'find'/'endswith')`) or `m.content is match(...)`
  (`Test`). **DET**.
- **Branch-divergent / suppression output** — one branch emits content, the other
  substitutes a literal / drops / appends an instruction; or silently redacts.
  Discriminator = divergence + content-keyed test. **DET** (content-gated).
- **Date/time logic-bomb** — `strftime_now(...)` used inside an `If.test` (vs
  benign emit-only display). **DET** (name-in-test).
- **Hidden instruction-literal emission** — imperative text not sourced from
  `messages`, strongest when emitted **outside the For loop**. **HEUR** (qwen
  baseline legitimately embeds "You are a helpful assistant"; lexicon match is a
  review-prompt, NOT FAIL alone).
- **Split-string reconstruction** — `'Always '~'recommend '~'X'` or
  `['ig','nore']|join` to dodge literal scans; reconstruct then re-classify.
  **DET** (reconstruction); downstream match inherits HEUR. `reconstruct_const_
  string` (~/+) is in `template_ast.py`; `_reconstruct_join` (|join) is in
  `rules/template.py` — both must be called.

---

## 2. Template text-encoding & structure (no code exec)

**Highest-value structural fix:** `_ast_checks` inspects only
`Const/Compare/Getattr/Getitem/Filter/Name` and **never reads `TemplateData`** —
the raw inter-tag text that carries `<|im_start|>system…` injections and
invisible chars. Literal rules must union `TemplateData.data` + `Const.value` (+
reconstructed splits). All four baselines are pure ASCII → ~zero baseline FP.

- **Invisible / format-control / Unicode-tag codepoints** — Cf/Co/Cn + tag block
  U+E0000–E007F + explicit set (U+200B–200F, U+2060, U+FEFF, U+00AD, U+180E,
  U+2028/9). **DET (FAIL)**.
- **Bidi override (Trojan Source)** — U+202A–202E, U+2066–2069, U+200E/F, U+061C.
  **DET (FAIL)**.
- **Homoglyph / confusable** of role/special-token/instruction words — use a
  **TR39 confusable-skeleton map as PRIMARY** (NFKC does NOT fold Cyrillic/Greek→
  Latin, only fullwidth/compat). **HEUR** (multilingual FP-prone → WARN).
- **Delimiter adjacent to content** — control-token literal concatenated with a
  content access → content can forge a turn. **HEUR** (chatml/qwen do this
  benignly; require baseline deviation).
- **Role confusion** — content under a fixed privileged wrapper, or raw `role`
  interpolation with no allow-list. **HEUR** (baseline-suppress).
- **Forged/synthetic turns** — a complete constant turn not from `messages`.
  **HEUR** — few-shot/example templates legitimately embed constant turns;
  escalate only on imperative-lexicon + baseline deviation.
- **BOS/EOS vs tokenizer flags** — double-BOS (`add_bos_token=true` AND template
  hard-codes BOS) is **DET**; mid-loop EOS is **HEUR**.

---

## 3. Tokenizer metadata (`tokenizer.ggml.*`)

Parser caveat: arrays preview to 64 but `MetaArray.length` is the true count →
length/shape checks work today; per-token *string* checks need an opt-in full
materialization of the specific arrays.

- **Special-token id out of range** — `0 <= id < vocab_size`. Ids are read
  **unsigned** ("negative-when-signed" is a downstream-loader artifact, not
  c4nary's view). **DET (FAIL)**.
- **Vocab desync** — `len(tokens)` vs token_embd/output vocab axis (pad band →
  WARN; tokens > axis → FAIL). **DET (FAIL)**.
- **Parallel-array length desync** — `len(scores)==len(token_type)==len(tokens)`,
  token_type enum range. **DET (FAIL)**. (CONTROL-on-ordinary-word relabeling is
  a separate **HEUR**, gated on materialization — not deterministic.)
- **EOS/BOS/unk/pad remap** — range = DET; "is this a stop-like string" = HEUR.
- **add_bos/add_eos flips** — DET only with a per-arch tokenizer baseline (none
  ships → currently INFO); `add_bos=true` + missing/out-of-range `bos_token_id` is
  the one DET sub-check today.
- **Control-string collision** / **homoglyph** / **exact-duplicate vocab** —
  gated on full materialization; collision/homoglyph **HEUR**, exact-dup **DET**.
- **Token length > INT32_MAX** (CVE-2025-49847 signed-cast) — capture element
  lengths during the existing array walk. **DET (FAIL)**.
- **Merge sanity** (halves exist in vocab; dup merges) — gated on full
  materialization (does NOT work under 64-preview); behavioral merge analysis OUT.

---

## 4. Metadata-vs-tensor-map consistency (silent quality/behavior)

Weights+template byte-faithful; numeric metadata fed into inference math is
edited. Cross-check scalars against tensor **shapes** (never weight data). Highest
DET value, **entirely unimplemented** today (MET005 only checks `<arch>.*` keys
exist).

- **`block_count` vs blk.* count** — distinct `^blk\.(\d+)\.`, max+1, contiguity.
  **DET (FAIL)**.
- **`embedding_length` vs token_embd axis** — accept either stored axis (GGUF
  stores dims reversed; do NOT hard-code shape[0]). **DET (FAIL)**.
- **`head_count | embedding_length`, `head_count_kv | head_count`** — integer
  invariants. **DET (FAIL)**. KV tensor-width check uses `key_length/value_length`
  if present, else INFO.
- **`feed_forward_length` vs ffn_* axis** — handle gated/MoE (`ffn_*_exps`).
  **DET (FAIL)**.
- **`file_type`/quant vs dominant dtype** — exclude norms/biases/embeddings; mixed
  *_K → WARN. **HEUR**.
- **context_length / rope.freq_base / rope.scaling** — DET only for internal
  contradictions (`context_length < original_context_length`; `scaling.type='none'`
  + non-identity factor; `freq_base<=0`/NaN); magnitude = **HEUR**.

---

## 5. Structural / parser-exploitation & provenance

Needs two read-only parser additions: (a) retain `file_size` + computed
data-section start; (b) preserve metadata key order + duplicate counts.
`TensorInfo.offset` is parsed but unused.

- **ggml_nbytes ne-product overflow** — recompute `product(dims)*type_size/
  blck_size` in **bignum** to see the C 64-bit wrap (CVE-2026-33298 /
  CVE-2024-21802). **DET (FAIL)**.
- **Cumulative ctx->size / mem_size overflow** — bignum sum of
  `align_up(true_bytes)`+overhead (CVE-2025-53630 + bypass CVE-2026-27940).
  **DET (FAIL)**.
- **Tensor offset OOB** — offsets are **relative to the data section**;
  `data_start + offset + true_bytes <= file_size` (no lower bound below 0).
  **DET (FAIL)**.
- **Overlapping tensor regions** — pairwise non-overlap (GGUF requires non-overlap,
  not tight contiguity). **DET (WARN/FAIL)**.
- **`general.alignment` sanity** — 0 / non-pow2 / <4 / >1048576. **DET**.
- **Zero / non-block-divisible dims** — `ne[0] % blck_size(type) == 0` (innermost
  dim) using exact per-type table (1 / 32 / 256). **DET**.
- **Duplicate / confusable metadata keys** — parser-differential (scanner reads
  first copy, loader reads last). **DET (FAIL once dup-preservation lands)**.
- **Template-like content under non-chat_template keys** — route metadata strings
  with Jinja delimiters through the AST rules. **HEUR**.
- **Unknown ggml_type / value_type id** — flag + treat size as unknown.
  **DET (WARN)**.
- **Tensor-name abuse** — >64 B, NUL/control, non-UTF8, traversal. **DET (INFO)**.
- **Injected/missing tensors** — index >= block_count, dup names, adapter
  patterns = DET; "no canonical match" = HEUR (profile-dependent).
- **Split/sharded external reference** — `split.count>1` bounds the scan to one
  shard. **DET (visibility note)**.

---

## 6. Weight-embedded backdoors — OUT OF SCOPE

Trigger-phrase→behavior, sleeper-agent, data-poisoning live in tensor-data float
values c4nary never reads; the effect is **not statically detectable** (identical
structure to a clean model). Only in-scope angle: opt-in per-tensor **streamed
SHA-256 manifest** detecting THAT weights changed vs a trusted reference, never
WHAT — requires a sanctioned opt-in exception to "never read weight bytes" + a new
GGML block-size table (neither exists today).