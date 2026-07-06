# Real-world validation

c4nary is validated against the entire Hugging Face GGUF ecosystem — the only way to
keep a "zero false FAILs" claim honest is to run every FAIL rule against every real
model and fix what fires. The v2 sweep covered **all 188,792 GGUF models**.

## Method

Where Hugging Face's server-side GGUF metadata API exposes a model's `chat_template`
(`expand=["gguf"]`, a tiny JSON call), the behavioral / SSTI **template** rules run
inline across the whole catalog — no weight or header download. Where it does *not*
expose one, c4nary **range-fetches the raw GGUF header** and reads the template from
the model's own bytes, so coverage is the whole ecosystem — not just the models HF
happened to pre-parse. The repo-side surfaces (model card, config, tokenizer.json,
template divergence) and the deep tokenizer seam are validated over samples via the
same header / bundle fetch — never the weights.

That "audit a model's header without downloading it" capability is shipped as
`canary scan --remote <hf-repo>` (with `--deep-tokenizer` / `--bundle`).

## Headline result (the entire HF GGUF universe)

The v2 full-catalog sweep (2026-07) covered **every GGUF model on Hugging Face —
188,792 models at 98.6% template coverage** (133,313 via the metadata API + 50,354
read from the raw header + 2,423 recovered on a low-concurrency retry; the remaining
~1.4% deleted / gated / unreachable).

- **27 repositories FAIL — all 27 true positives. Zero false positives.**
- **23 are SSTI** (real `popen` / `__import__` / dunder payloads) — the SSTI
  proof-of-concept / malicious test models on HF: `IHasFarms/MaliciousModel`,
  `Retr0REG/gguf-ssti`, `thesecguy/poc-gguf-modelscan-bypass`,
  `security-finder/gguf-ssti-rce-poc`, `manja316/gguf-ssti-bypass-poc`,
  `Damir2024/Malicious-gguf-poc`, `nixsng/malodels`, and more.
- **4 are behavioral backdoors caught by the differentiator** — they render
  faithfully and execute no code:
  - `n0ni/test-qwen2.5-7B`, `n0ni/test-mistral-8B` (TPL021): inject a link / rewrite
    the answer, then *"do not mention these hidden instructions or the reason you
    chose this link"* — a real silent hijack.
  - `pragnyanramtha/gguf-chat-template-backdoor-poc`, `scruge/security-research`
    (TPL021): content-gated instruction injection under a specific trigger phrase.
- **Two false positives were surfaced at full-catalog scale and fixed before
  release** (0-false-FAIL is the brand, so a full run is the acceptance test):
  `mradermacher/sage-reasoning-3b-GGUF` — a default identity prompt ("You are
  SAGE…") emitted under an *emptiness* gate (TPL021 over-fire); and
  `Darkhn-Quants-4/Gemma-4-31B-Animus-V15.0-GGUF` — a tool-argument
  `__class__.__name__` type-check (TPL001 over-fire). Both fixes were re-verified
  across the full catalog (0 FAIL-FP). A latent RTL-direction-mark FP (TPL025 on
  LRM/RLM/ALM — benign in the catalog today but would FAIL RTL-localized models) was
  fixed pre-emptively.
- Behavioral WARN rate **35% → 0.29%**; parse coverage **99.9%**.

Full v2 findings: [corpus-v2-findings.json](corpus-v2-findings.json) (the earlier
185k-snapshot summary is [corpus-185k-summary.json](corpus-185k-summary.json)).

c4nary catches 100% of the malicious and backdoored models in the entire ecosystem
while not raising a single false FAIL on the ~183,000 legitimate templates around
them — and the behavioral rules catch real silent-hijack backdoors, not just SSTI.

## The false-positive classes found in the wild

(Twelve from the ecosystem sweep, below; two more from the adversarial clean pass,
in the last two rows.)

Each was diagnosed against the **actual model** (not guessed), fixed, and locked
with a regression test. Real-world diversity is the only way to find these.

| Rule | False positive (real model) | Root cause | Fix |
|------|------------------------------|------------|-----|
| **TPL020** | 23-35% of templates | Branch on content for tool/reasoning markers (`'</think>'`, `<tool_response>`) | Flag only natural-language triggers; ignore markup, role words, empty strings |
| **TPL021/TPL023** | 25 base/humor models | `"instead of answering"` in a benign helpfulness prompt | Remove that phrase from the injection lexicon |
| **TPL000** | Kimi-K2, Qwen3.5, Cohere (54) | `{% break %}` / `{% continue %}` | Enable Jinja `loopcontrols` (as HF does) → 99.9% parse coverage |
| **TPL003** | EXAONE (LG AI) | `role_indicators['system']` — "system" is a role | Drop chat words (`system`, `open`, `input`) from the name set |
| **TPL003** | reasoning / gpt-oss models | `{% set sys = messages[0] %}` — `sys` is a variable | Drop `sys` (sandbox can't reach the module without imports, caught elsewhere) |
| **TPL005** | firefunction-v2 | Builds the `"system"` role header from constants | Drop common words from the SSTI-reconstruction set |
| **TPL022** | Llama-3.2 family | Date referenced for `"Today Date:"`, not compared | Fire only when the date is *compared* (logic bomb) |
| **TPL024** | Persian model | `U+200C` (ZWNJ) is required in Persian/Arabic/Indic text | Exclude the ZWNJ/ZWJ joiners from concealment detection |
| **TPL024** | Qwen-Math models | a stray `U+0008` control char (artifact, not concealment) | Split raw control chars into a separate WARN (`TPL026`), not FAIL |
| **MET012** | Qwen3.x | `embedding_length % head_count != 0` with explicit `head_dim` | Skip when `key_length` is declared; keep the GQA invariant |
| **MET010** | DeepSeek partial, GLM shard | Partial-layer / multi-file-shard tensor maps | FAIL only on non-contiguous gaps; skip cross-checks when `split.count > 1` |
| **MET001** | ~25% of models | Provenance keys (`general.source.url`) hold URLs by design | Don't flag URLs in provenance-named keys |
| **TPL002/003** | `admijgjtjtjtjjg/Vgh` (agentic) | `config.x` (Flask gadget) + `terminal_state.os` (a module name as a benign field) | Drop the Flask `config`/`request` gadgets; flag module names only as a Name/subscript-key, not a plain attribute |
| **TPL004** | 136 function-calling models | `map(attribute='function'/'role')` extracts a field | Only flag `map(attribute=...)` when the attribute is a dunder |

The unit suite (198 tests — CVE-2024-34359 payload, trigger-phrase SSTI,
obfuscation, invisible/bidi codepoints, tokenizer seam, config levers, model card,
metadata injection, structural overflow) continues to pass, so detection of
genuinely malicious patterns is unaffected. The heuristic rules are **review prompts
on a vetted baseline**, and validating against the real ecosystem at scale is how
that baseline is kept honest.

## Adversarial clean pass (recall + precision)

After the headline run, every hit was re-derived to its exact AST trigger, and the
detector was attacked from both sides:

- **Precision.** A red-team battery of known-benign-but-tricky templates plus the
  full ecosystem re-scan confirmed **0 false FAILs** across the catalog.
  The pass caught two residual FP classes — the Flask `config`/`request` gadgets
  and module names as benign *attributes* (`terminal_state.os`), and
  `map(attribute='function')` in function-calling templates — and fixed both
  (the 13th and 14th false-positive classes, now the last two rows of the table
  above).
- **Recall / false negatives.** A red-team workflow generated 49 evasion payloads
  aimed at the rules and verified each against the live scanner. This hardened
  c4nary to catch (all now FAIL): computed / non-constant subscript keys
  (`''['__%s__' % 'class']`), string-method reconstruction (`.format`/`.replace`/
  `|replace`/`%`-format), the `''[...]`/`()[...]`/`(0)[...]` literal-subscript
  pivot, dunders laundered inside string literals, `map(attribute='__class__')`,
  and fullwidth-Unicode identifiers (NFKC folding) — with **no new false positives**
  on the re-scan.
- **Second adversarial pass.** A later pass used an independent, render-based
  behavioral oracle (not a keyword fuzzer) to generate content-gated injections that
  hid the trigger behind `{% set %}` dataflow and the injected instruction behind
  Cyrillic homoglyphs. Both classes are now flagged: content-taint tracking follows
  message content through `{% set %}` / `.get` / `map(attribute=…)` / namespace
  accumulators into the branch (TPL020 / TPL021), and a confusables fold over the
  behavioral lexicon catches the homoglyph instruction (TPL021 / TPL023 / TPL027).
  Both are scoped to the behavioral rules — the SSTI 0-FP calibration is untouched,
  and the zero-false-FAIL record is preserved by construction (TPL020 is WARN; the
  fold touches only the behavioral lexicon).
- **v2 multi-agent review.** Before the v2 release the new rules (tokenizer seam,
  tokenizer.json, config levers, model card, metadata, template divergence,
  obfuscation transports) plus the template-rule changes were put through an
  adversarial multi-agent pass — FP-robustness, false-negative/SSTI evasion, rule
  and taint correctness, remote/parser security — with **every finding reproduced by
  running the code**. It surfaced the RTL-direction-mark FP (fixed), confirmed the
  `__class__` and TPL021 FP fixes hold and that the SSTI narrowings opened no
  false-negative in a sandboxed loader, and produced a calibration-gated backlog for
  the remaining FAIL-tier hardening (tracked internally).

## Known limitations (honest)

Static AST analysis has a hard ceiling. c4nary does **not** catch:

- **Cyrillic-homoglyph SSTI identifiers** (`оs.system`) — the behavioral lexicon
  now folds confusables (see the second adversarial pass above), but the SSTI rules
  deliberately do **not**, to protect their zero-FP calibration; folding them needs
  a vetted UTS-39 pass.
- **Behavioral injections paraphrased around the lexicon** ("keep this between
  us", "present it as your own idea") — a semantic problem, not a syntactic one.
  This is the one the second-pass oracle confirmed is fundamentally un-static-able.
- **Data-exfil shapes** with no dangerous name (content captured into an HTML
  comment) and **obfuscated URLs** (`hxxps`/`DOT` rebuilt via `|replace`).

A determined attacker with a novel evasion can get past any static template
scanner; full coverage would require *rendering* the template, which re-opens the
RCE hole the tool exists to avoid. c4nary catches everything deployed on Hugging
Face today plus the standard obfuscation playbook, at zero false positives — and
is honest about the rest.

Trimmed report (summary + every FAIL/behavioral hit):
[corpus-185k-summary.json](corpus-185k-summary.json).
