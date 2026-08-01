# Changelog

## Unreleased

## 0.3.0 — 2026-08-01

### Changed (breaking)
- `canary rules --json` and the MCP `list_rules` tool now return an object
  `{tool_version, rules_bundle_sha256, rules: [...]}` instead of a bare array. A
  consumer doing `canary rules --json | jq '.[].rule_id'` must become `jq '.rules[].rule_id'`.

### Detection
- Audit `auto_map` across config, tokenizer, processor, and preprocessor config files
  without duplicate CFG004 warnings.
- Add WARN/INFO-only `RMT` rules for fail-closed custom-Python parsing, import-time
  execution and capability sinks, decode-then-execute composition, and unfetched
  cross-repository references.
- Resolve custom Python through actual import bindings and one bounded relative-import
  hop. Audited source is parsed with `ast.parse` only and is never imported or executed.
  Bindings are carried across the hop, so a sink split across two files -- neither
  dangerous alone -- is attributed to the caller in the plain, aliased, `import *`, and
  `from . import sibling` forms. A name that cannot be resolved leaves the caller's
  coverage row `partial` instead of `examined`.

### Reporting
- Add stable artifact and source-line fields while preserving legacy location strings.
- Add deterministic per-surface coverage manifests and a rules-bundle provenance digest.
  The manifest records `--policy` as its own surface, so a report whose severities were
  re-tiered is no longer indistinguishable from one with nothing to report.
- Add deterministic SARIF 2.1.0 output validated against the official OASIS schema.
- Add per-rule report policy, `--fail-on none`, and justified fingerprint baselines.
  Suppressed findings remain visible in human, JSON, SARIF, and coverage output.

### Safety and validation
- Refuse truncated, oversized, or unparseable custom Python as a clean audit, and validate
  attacker-derived Python filenames at every local, MCP, and remote reader sink. A name
  must be an importable module name, which closes a Windows path escape: `C:foo.py`
  carries no separator and passed every other check, but `os.path.join` drops the base
  directory for a drive-qualified name and read from the current directory of `C:`.
- Freeze all 140 v0.2.2 FAIL findings as offline fixtures and assert legacy identity
  remains unchanged.
- Codify the Jinja sandbox baseline for Jinja 3.1.6+ without rendering in production.
- Extend the never-render/never-execute source guard to the indirect spellings that
  matter -- `builtins.exec`, `pickle.loads`, `runpy.run_path`, `subprocess.*` and
  `getattr(obj, "render")` -- and assert each one is caught rather than trusting a guard
  nobody has watched fail.
- Make both corpus gates account positively for every registered FAIL rule, with a
  committed reason for each surface they cannot exercise.
- Qualify determinism by the fixed Python/Jinja2 version pair and scope the 0-FP record to
  the template FAIL family actually measured.
- Refuse to write a manifest over the artifact under audit. `hash m.gguf --manifest m.gguf`
  previously truncated the model, replaced it with JSON, and exited 0.
- Fail closed when the AST walkers that run after `ast.parse` are exhausted. A ~1 KB
  deeply-nested but valid Python file aborted the whole scan, discarding template findings.
- Bound the recursive `tokenizer.json` and special-token walkers by depth. JSON nesting is
  unbounded while the reader cap is 48 MiB, so a ~7 KB file nested 1000 deep aborted the
  scan.
- Report an unexpected Hugging Face API response as a clean error instead of a traceback.
- Bound retry sleeping: a hostile `Retry-After` could stall a fetch for hours.
- Cap import-time findings at 100 per Python file and report the overflow as an exact
  count. One 1 MiB file of `os.system(...)` per line produced ~69,800 findings and 68 MiB
  of SARIF, projecting to roughly 2 GiB across a full bundle -- past the point any tool
  can ingest it (GitHub rejects SARIF over 10 MB). A capped file is reported `partial`,
  never `examined`.
- Decode fetched bundle text as UTF-8 rather than the server-supplied charset, which chose
  what the rules saw and could raise an uncaught `LookupError`.

### False positives
- Stop treating `if __name__ == "__main__":` bodies as import-time scope. `transformers`
  loads custom Python with `exec_module`, so `__name__` is the module's own name and the
  guarded body is unreachable. The skip is withdrawn, with an RMT000 row, for a file that
  writes `__name__` and could make the guard fire.
- Give every FAIL rule that can fire twice at one location a distinguishing subject
  (TOK002, TOK003, TPL004, TPL021, STR001, STR003), and include the subject in
  deduplication so two distinct TPL021 triggers on one minified line both report.

### Packaging
- Cap the `mcp` extra at `mcp>=1.2,<2`: mcp 2.0 removed `mcp.server.fastmcp`, which the
  server imports.
- Add `jsonschema` and `tomli` (Python < 3.11) to the `dev` extra so the SARIF-conformance
  and version-parity tests are not silently skipped.

## 0.2.2 — 2026-07-23

### Security
- Scope `HF_TOKEN` authentication to the exact `huggingface.co` host so direct remote-scan
  URLs cannot receive the credential.
- Stream repo-bundle files under a hard byte cap and reject oversized responses instead of
  buffering the complete body before truncation.
- Close template trigger-camouflage and instruction-laundering paths involving string
  formatting, replacement, partition/search operations, scoped assignments, and Jinja tests.

### Detection
- Add WARN rules for forced output tokens (`CFG003`), custom-code mappings (`CFG004`),
  extreme decode-time steering (`CFG005`), and reachable instruction-bearing special tokens
  (`NRM003`).
- Remove noisy content-branch rule `TPL020`; retain FAIL-level content-gated instruction
  detection in `TPL021` and instruction laundering in `TPL027`.
- Retire the proposed `MET022` / `MET023` tokenizer warnings before release because their
  metadata did not establish malicious behavior.
- Audit `processor_config.json` chat templates in the bundle scan: multimodal models
  (LLaVA, Qwen-VL) can carry a divergent template there that a
  `tokenizer_config.json`-only audit never sees.

### False positives
- Narrow `TPL002` to Jinja gadget chains that actually reach Python globals. Bare capability
  checks such as `lipsum is defined` no longer FAIL; canonical
  `lipsum.__globals__` / `cycler.__init__.__globals__` payloads remain detected.

### Validation
- Add a resumable release-gate harness whose template FAIL scope is derived from the rule
  registry instead of a hand-maintained subset.
- Add metadata-only GGUF parsing for corpus template recovery without parsing tensor
  descriptor tables.
- Stream escalating GGUF header prefixes through one bounded request and parse headers in a
  rate-paced process pool, avoiding overlapping downloads and the thread-pool GIL bottleneck.
- Run all eight registered template FAIL rules over a frozen 192,032-repository inventory:
  137,698 actual templates analyzed, 140 findings across 28 repositories, and zero reviewed
  false-positive FAILs. Retain 1,118 unresolved parser/access cases as explicit exclusions.
- Correct the old coverage label: successfully parsed no-template headers count toward
  parsed-repository coverage, not template coverage.

### Packaging
- Add Python 3.14 to the CI matrix and trove classifiers.
- Add a CI package job: build, `twine check`, extracted-sdist test run, wheel smoke test.
- Ship `SECURITY.md`, the release-gate tools (`tools/template_fail_gate.py`,
  `tools/release_gate_scan.py`), and the machine-readable corpus summary
  (`docs/corpus-v0.2.2-template-gate-summary.json`) in the source distribution.
- Add PyPI project URLs and SPDX license metadata (`license = "MIT"`).

### Repository
- Add a security reporting policy and Dependabot update configuration.
- Pin GitHub Actions to immutable commit SHAs.
- Exercise the installed MCP extra and a real stdio protocol round-trip in CI.
- Harden local artifact, credential, coverage, and release-output ignore rules.
- Add complete PyPI project links and Python-version classifiers.
- Make the development extra complete for clean-environment gate-tool tests and add
  wheel/sdist build, metadata, and installed-command checks to CI.

## 0.2.1 — 2026-07-06

Behavioral / SSTI FAIL-tier hardening, validated against the full catalog (0 new false
positives) and three adversarial review rounds. Closes evasions that previously slipped to
WARN or escaped, without touching the 0-false-positive brand.

### Added / Changed
- **Content-gated injection** now FAILs when the injected instruction is assembled via a
  one-hop `{% set %}` alias, a `|join`, or a `|replace` filter (previously WARN-only).
- **SSTI**: generator/coroutine **frame internals** (`gi_frame`/`f_builtins`/`f_globals`/…)
  are flagged; a dunder **laundered in a computed subscript key** (`.replace`/`|replace`/`%`,
  e.g. `obj['gi_%srame' % 'f']`) is reconstructed and flagged — closing the getitem→getattr
  gadget.
- **Fewer false positives**: a trigger literal now counts only if message content is actually
  compared against it (a config-flag comparison beside a content truthiness check no longer
  fabricates a trigger).

### Fixed / Robustness
- **Fail-closed**: a crafted template that crashes a rule now surfaces `TPL000` (review)
  instead of silently returning zero findings (a `.format()` mini-language spec could
  previously abort the whole scan — a fail-open hole).

## 0.2.0 — 2026-07-06

The **v2 detection layer**: c4nary now audits *every controllable, renderable
surface* of a GGUF model, not just the chat template — all still without rendering
the template, reading weights, or (at the core) touching the network.

### Added
- **Tokenizer seam (`TOK`)** — confusable / duplicate role-token forms and
  special-token consistency; opt-in `--deep-tokenizer` materializes the full vocab.
- **tokenizer.json (`NRM`)** — normalizer / pre-tokenizer / decoder / post-processor
  `Replace` rules that rewrite text on every input/output, plus concealed special /
  added tokens.
- **Decode-time config levers (`CFG`)** — `suppress_tokens` / `bad_words_ids` that
  mute the stop token or the tokens a refusal is built from (multi-token
  reconstruction; Whisper `begin_suppress_tokens` guard).
- **Model card & metadata (`DOC`, `MET`)** — invisible / bidi payloads and
  prompt-injection idioms in the README and free-text metadata; Jinja carried in a
  metadata string is routed through the SSTI / behavioral AST rules.
- **Repo↔GGUF template divergence (`TPL030`)** and **obfuscation transports
  (`TPL031` / `TPL032`)** — `include` / `import` / `extends` and base64/url decode
  filters.
- **`--bundle`** — fetch the repo's card / config / tokenizer.json for the above.
- **Retrying remote session** — HF 429 backoff so bulk header scans self-throttle
  instead of failing; `stages_mb` fetch escalation for large-vocab headers.
- **MCP server** (`c4nary-mcp`, `pip install c4nary[mcp]`) exposing scan / diff /
  hash / rules.

### Changed
- Historical full-catalog validation reported **188,792 models at 98.6% template
  coverage**, **27 flagged repos, 0 false positives**. v0.2.2 corrects that coverage
  label because the old calculation included parsed headers with no template and
  double-counted those records. The v2 rules also went through an adversarial review.
- **TPL021** — a content-gated injection now FAILs only under a *specific trigger
  literal* (fixes a full-catalog false positive on default identity prompts emitted
  under an emptiness gate).
- **TPL001** — `__class__` is treated as a pivot, not an escape: a bare
  `x.__class__.__name__` type-check no longer FAILs (fixes a full-catalog false
  positive on tool-calling templates); a literal-rooted `''.__class__` or an
  escape-dunder chain still FAILs.
- **TPL025** — no longer flags the benign RTL direction marks (LRM/RLM/ALM); only the
  Trojan-Source override / isolate controls FAIL.

### Fixed
- `analyze_config` no longer crashes on a scalar (non-array) `tokenizer.ggml.tokens`
  from a crafted model.
- Remote header fetch escalates on a big-vocab "array cannot fit" parse error instead
  of aborting the scan.

## 0.1.0

Initial release — chat-template behavioral / SSTI static analysis, structural
consistency, provenance / integrity, and the offline read-only core.
