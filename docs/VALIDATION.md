# Real-world validation

c4nary's template FAIL rules are calibrated against a frozen inventory of the
Hugging Face GGUF ecosystem. This is false-positive hunting, not proof that a
model is safe and not a substitute for unit, adversarial, or runtime testing.

## Method

Where Hugging Face's server-side GGUF metadata API exposes a model's `chat_template`
(`expand=["gguf"]`, a tiny JSON call), the behavioral / SSTI **template** rules run
inline across the whole catalog — no weight or header download. Where it does *not*
expose one, c4nary **range-fetches a representative raw GGUF header** and reads the
template from the model's own bytes. The repo-side surfaces (model card, config, tokenizer.json,
template divergence) and the deep tokenizer seam are validated over samples via the
same header / bundle fetch — never the weights.

That "audit a model's header without downloading it" capability is shipped as
`canary scan --remote <hf-repo>` (with `--deep-tokenizer` / `--bundle`).

## v0.2.2 headline result

The 2026-07-22 release gate processed **192,032 / 192,032 unique repository
records tagged `gguf`** and derived its scope from all eight registered
FAIL-severity template rules.

- **137,698 repositories exposed an actual template that was analyzed.** This is
  71.7058% of the inventory: 136,996 inline templates, 678 from primary raw-header
  reads, and 24 recovered from bounded alternate-file passes.
- **52,378 representative GGUF headers parsed successfully with no template.**
- **839 tagged repositories advertised no GGUF file — one of those still supplied
  an inline template and was analyzed; the exclusive no-template/no-GGUF bucket is
  838. 1,118 remained explicit parser/access exclusions.** Parsed-repository
  coverage is 98.9814%, but that is not template coverage.
- **28 repositories FAIL — all 28 reviewed as true positives.** The gate recorded
  140 findings and zero reviewed false-positive FAILs in the analyzed templates.
- **24 are SSTI** (real `popen` / `__import__` / dunder payloads) — repositories
  whose content and names are consistent with proof-of-concept / test / research
  artifacts: `IHasFarms/MaliciousModel`,
  `Retr0REG/gguf-ssti`, `thesecguy/poc-gguf-modelscan-bypass`,
  `security-finder/gguf-ssti-rce-poc`, `manja316/gguf-ssti-bypass-poc`,
  `Damir2024/Malicious-gguf-poc`, `nixsng/malodels`, and more.
- **4 are content-triggered behavioral-backdoor test/research artifacts caught
  by the differentiator** — they render
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
- The earlier behavioral WARN calibration moved **35% → 0.29%**. That historical
  WARN metric is separate from the v0.2.2 all-template-FAIL gate.

Machine-readable v0.2.2 summary:
[corpus-v0.2.2-template-gate-summary.json](corpus-v0.2.2-template-gate-summary.json).
Historical v2 findings remain at [corpus-v2-findings.json](corpus-v2-findings.json).

The old v2 documentation called successful header handling "98.6% template
coverage." That label was wrong: the historical counter included successfully
parsed headers that contained no template, and its display formula double-counted
those no-template cases. v0.2.2 separates repository processing from actual
template analysis and does not repeat the overclaim.

## The false-positive classes found in the wild

(Twelve from the ecosystem sweep, below; two more from the adversarial clean pass,
in the last two rows.)

Each was diagnosed against the **actual model** (not guessed), fixed, and locked
with a regression test. Real-world diversity is the only way to find these.

| Rule | False positive (real model) | Root cause | Fix |
|------|------------------------------|------------|-----|
| **TPL020** | 23-35% of templates | Branch on content for tool/reasoning markers (`'</think>'`, `<tool_response>`) | Remove the noisy rule; TPL021 retains the instruction-bearing FAIL signal |
| **TPL021/TPL023** | 25 base/humor models | `"instead of answering"` in a benign helpfulness prompt | Remove that phrase from the injection lexicon |
| **TPL000** | Kimi-K2, Qwen3.5, Cohere (54) | `{% break %}` / `{% continue %}` | Enable Jinja `loopcontrols` (as HF does) → 99.9% parse coverage (historical calibration number, distinct from the current 98.9814% parsed-repository coverage) |
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

The unit suite (CVE-2024-34359 payload, trigger-phrase SSTI,
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
  aimed at the rules and verified each against the live scanner. The 19 retained as
  regressions ship in `tools/evasions.json`; the rest were duplicates or variants
  covered by the same fix. This hardened
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
  accumulators into the branch (TPL021), and a confusables fold over the
  behavioral lexicon catches the homoglyph instruction (TPL021 / TPL023 / TPL027).
  Both are scoped to the behavioral rules — the fold touches only the behavioral
  lexicon, so the SSTI 0-FP calibration is untouched. The zero-reviewed-false-FAIL
  record is empirical evidence from the frozen corpus re-scan, not a theorem:
  TPL021 is a behavioral FAIL rule, and its record holds only as far as the
  corpus it was measured on.
- **v2 multi-agent review.** Before the v2 release the new rules (tokenizer seam,
  tokenizer.json, config levers, model card, metadata, template divergence,
  obfuscation transports) plus the template-rule changes were put through an
  adversarial multi-agent pass — FP-robustness, false-negative/SSTI evasion, rule
  and taint correctness, remote/parser security — with **every finding reproduced by
  running the code**. It surfaced the RTL-direction-mark FP (fixed), confirmed the
  `__class__` and TPL021 FP fixes hold and that the SSTI narrowings opened no
  false-negative in a sandboxed loader, and produced a calibration-gated backlog for
  the remaining FAIL-tier hardening.
- **v2.1 FAIL-tier hardening.** That backlog shipped in **0.2.1** after three more
  adversarial rounds + a full-catalog re-sweep (**0 new false positives**): content-gated
  injection assembled via `{% set %}` alias / `|join` / `|replace` now FAILs; SSTI frame
  internals and dunders laundered in a computed subscript key are caught; a compound-test
  FP was fixed; and a crafted `.format()` spec that could crash a scan (fail-open) now fails
  **closed** (`TPL000`). Each round each fix was re-calibrated against the corpus before
  shipping — and the fixes that could not distinguish a real trigger from a benign default
  prompt were *dropped* rather than ship a false positive (see the changelog).

## v0.3 `trust_remote_code` boundary

The RMT family audits Python selected by `auto_map` only when `--bundle` is
enabled. It parses source with `ast.parse`; it never imports or executes the
repository. Import-time scope is module and class bodies, decorators, and
default arguments. Relative-import resolution stops after one hop, and names it
carries across that hop are resolved rather than dropped: a sink re-exported by a
sibling (`from .utils import system` beside `from os import system`) is reported
against the caller, in the plain, aliased, `import *`, and `from . import sibling`
forms. Where a name cannot be resolved — a second hop, or a sibling that could not
be read — the file's coverage row drops to `partial` rather than `examined`.

The production scanner still never renders a template. The sandbox render
checks live only in `tests/test_sandbox_baseline.py`, and an AST guard prevents
scanner modules from acquiring render or execute calls.

All RMT rules remain WARN or INFO. The benign and hostile fixture gates prove
matcher behavior and fail-closed handling, not a real-world false-positive
rate. The frozen v0.2.2 inventory contains only the earlier eight fixed bundle
filenames, not repository Python listings. Re-enumerating the widened ten-file
bundle and building the required `auto_map` Python corpus would be a new crawl;
that was deliberately not performed in this bounded implementation. No RMT
result inherits the 137,698-template calibration claim.

**Offline proxy measurement (2026-07-31).** RMT was run at import-time scope over
**3,200 real third-party Python modules** (the development environment's
`site-packages`). This is not the `auto_map` population and is explicitly not a
0-false-positive calibration; it is real-world benign Python, which is the surface
the matcher can get wrong. Result: **17 import-time findings across 15 files —
0.47% of modules** — 9 `__import__`, 4 `importlib.import_module`, 3 `ctypes.CDLL`, 1
`compile`. Every one was adjudicated by hand and every one is a correct WARN-tier
detection of a real import-time construct (compatibility shims, optional-dependency
probes, a platform library load).

The pass found and closed one false-positive class: bodies of
`if __name__ == "__main__":` were being treated as import-time scope. They are not
— `transformers` loads the module with `exec_module`, so `__name__` is the module's
own name and the guarded body is unreachable. Two findings were affected, both
`pty.spawn` (`rich/ansi.py`, under the compound guard
`sys.platform != "win32" and __name__ == "__main__"`). `pty.spawn` is a FAIL
promotion candidate, so this class would have produced a false FAIL had it survived
into a calibrated tier. Fixed in `c4nary/rules/python_code.py` with a regression
test covering the plain guard, the compound guard, the `else` branch (which does
run), and an unguarded conditional.

**Re-measured (2026-08-01) after adversarial review of that fix.** The review found
that the fix had itself introduced a bypass, and that the reachability premise it
rests on can be defeated outright. Both are closed:

- The guard was matched with a structural walk, so `if __name__ != "__main__":` —
  the branch that always runs under `exec_module` — also suppressed findings. A
  two-character bypass of the whole family, with the file still reported `examined`.
  Now matched by exact AST shape, recursing only through `and` chains.
- A module that *writes* `__name__` (directly, or through `globals()[...] = ` /
  `vars().update(...)`) makes the guard fire on import. The skip is now withdrawn
  for such a file and an RMT000 row records why. Matching any *mention* of
  `globals()` was tried first and rejected: it fired on 1.9% of site-packages
  modules that merely read it.
- Three AST walks that run after `ast.parse` were unguarded, so ~1 KB of `1+1+…`
  parsed cleanly and then exhausted the stack, aborting the entire scan — template
  FAILs included — at an exit code this tool documents as "WARN findings present".
  They now fail closed to RMT000, as the parse step already did.
- Shadowing a builtin sink name at module scope (`exec = exec`) still suppresses
  that sink, since rebinding can genuinely make it benign, but no longer does so
  silently: an RMT000 row names the rebound name.
- Findings are capped at 100 per Python file. A 1 MiB file with `os.system(...)` on
  every line produced ~69,800 findings and 68 MiB of SARIF — roughly 2 GiB across a
  full 32-file bundle, which no consumer can ingest (GitHub rejects SARIF above
  10 MB). The overflow is reported as an exact count and the file is marked
  `partial`, so a capped report is never mistaken for a complete one.

Re-running the same 3,200 modules after these changes gives **22 findings across 20
of the 3,200 modules**: the original 17, plus 5 RMT000 rows for modules that
really do write their own module dict or shadow a sink name. No finding changed tier
and no new RMT010/RMT011 appeared.

Denominators recorded for the future `auto_map` crawl, derived from the frozen
192,032-record inventory: **27,897** repositories carry `config.json`, **8,411**
`tokenizer_config.json`, **532** `processor_config.json`. `preprocessor_config.json`
and `README.md` were not recorded by the eight-file harness, so their populations
are unknown. `auto_map` values live in file *contents*, which the inventory never
stored, so the widened population cannot be derived offline at any confidence.

Repo-native GGUF-less ingest (`--repo`) is not shipped. Bundle and RMT analysis
remain anchored to an explicit local or remote GGUF scan, and the historical
192,032-repository population remains scoped to repositories tagged `gguf`.

Policy severity overrides and baselines are report-time controls. The registry
and default severities remain unchanged, so the frozen template-FAIL result
describes the default configuration only. Suppressed findings are retained with
their required justification and do not rewrite the historical corpus record.

## Known limitations (honest)

- SARIF findings without a source line are schema-valid but GitHub-invisible.
  Line-bearing findings include a physical artifact URI and start line plus a
  logical location. `INT004` has no binary region because the tensor manifest
  retains names, shapes, and dtypes rather than byte offsets.

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
RCE hole the tool exists to avoid. Across the 137,698 templates analyzed in the
frozen v0.2.2 gate, c4nary produced 140 FAIL findings and review found zero
false-positive FAILs. Unresolved, unexamined, runtime, weight, and novel-evasion
surfaces remain outside that claim.

Historical v1 trimmed report (summary + every FAIL/behavioral hit from that older
snapshot — not the current 192,032-repository result):
[corpus-185k-summary.json](corpus-185k-summary.json). The current machine-readable
summary is [corpus-v0.2.2-template-gate-summary.json](corpus-v0.2.2-template-gate-summary.json).
