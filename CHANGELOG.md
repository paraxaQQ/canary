# Changelog

## Unreleased

### Architecture
- Add `c4nary/vocab.py`, a read-only vocabulary-source abstraction. Rules now ask a
  `Vocab` for its token count, surface strings, token types, and reconstructed text
  instead of reaching into `tokenizer.ggml.tokens` / `tokenizer.ggml.token_type`
  directly, so they no longer assume the vocabulary is GGUF-shaped.
- The GGUF constructor wraps the parsed metadata arrays verbatim -- preview, true
  length, truncation, and on-disk element size -- making it a no-op for existing scans.
  A `tokenizer.json` constructor (`model.vocab` plus `added_tokens`) is implemented and
  tested for the upcoming repo ingest, but is not wired into the CLI. Token ids are
  stored sparsely, so a declared id far above the entry count costs no allocation.
- GGUF output is byte-identical across the refactor: the baseline case diff is empty.
  Rule ids, severities, finding text and order, and coverage strings are unchanged.

### Naming
- Add `docs/NAMING.md` and standardize the public vocabulary: **c4nary** is the
  project/package, **canary** is the command, targets are **GGUF** / **repository**
  / **Ollama**, `--remote` means Hugging Face, and `--bundle` is a GGUF-only flag.
  Repository targets read those surfaces by default.
- `--gguf-file` is the canonical remote-GGUF filename flag; `--file` remains a
  hidden compatibility alias. `--manifest` is documented as local-GGUF-only, and
  `--deep-tokenizer` now states its repository behavior.
- The interactive menu gains local and Hugging Face repository scans; `--ollama`
  is documented as an alias for `--as ollama`.
- MCP gains the canonical `gguf_filename` parameter; `hf_filename` remains a
  compatibility alias. The `scan` docstring now describes GGUF and repository
  targets.
- Nightly builds now carry a distinct PEP 440 local version,
  `X.Y.Z+building.N`, matching the `vX.Y.Z-building.N` tag. The stable `main`
  version stays `X.Y.Z`; nightlies remain GitHub prereleases and are never
  uploaded to PyPI.
- Coverage IDs keep the historical `bundle.<file>` prefix for schema stability;
  the canonical term is “repository file,” and a rename is deferred to v1.0.0.
  No rule, finding, or coverage behavior changed; `tool_version` now carries the
  nightly local segment.

### Ingest
- Classify the scan target before any parser sees it, so a GGUF file becomes one ingest
  among several rather than the entry ticket. `scan` gains `--as {gguf,ollama,repo}` to
  force a type and `--ollama` to mark the target as an Ollama model reference. An Ollama
  reference and a Hugging Face repo id share a shape (`a/b`, `a/b:c`), so an Ollama
  target is never inferred from a bare string -- only declared.
- The GGUF path is unchanged by construction: GGUF magic is checked before any text
  heuristic, and an unrecognized target falls through to the GGUF ingest so the parser
  still raises its own error for a missing or non-model file. `ollama` targets still
  report that they are not implemented yet and exit 3.
- Add the **local repo target**, `canary scan path\to\repo --as repo` (a directory already
  classifies as `repo`). It reads the same allowlisted repo surfaces as `--bundle` --
  `config.json` / `generation_config.json`, `tokenizer.json`, special-token files,
  template sources, `README.md`, and `auto_map` Python -- and runs the CFG, NRM, DOC,
  TPL, TOK (vocab/special ids), and RMT rules without a GGUF anchor. GGUF-only checks
  (`MET*`, `STR*`, `TOK002`/`TOK003`, tensor-map and manifest comparison) are reported
  `absent`; when the deep pass is requested, `TOK012`/`TOK015` report `partial` (or
  `skipped` when no usable vocabulary exists) because `tokenizer.json` has no
  token-type split.
- Repo templates are scanned and source-tagged (`tokenizer_config.json`,
  `chat_template.jinja`, `processor_config.json`); `TPL030` needs a GGUF template to
  diverge from and does not run, and `TPL101` is never emitted -- coverage says
  `chat_templates=absent` when no source exists.
- Add `REP001` (WARN, local repo only): an allowlisted repo file is a symlink whose
  resolved target escapes the repository root. The target is not read; the file surface
  is `skipped` (never `absent`), a `repo.links` coverage row names the escape, and the
  row is repeated on stderr. In-root links are followed and scanned under the link name;
  broken/looping links and non-regular targets are `skipped` with a reason.
- Add the **remote repo target**, `canary scan org/model --remote --as repo` (also
  `hf://org/model`). It lists the Hugging Face repo once, then fetches the same fixed
  repo surface allowlist as the local target -- configs, `tokenizer.json`,
  special-token files, template sources, `README.md`, and `auto_map` Python -- with
  the same caps and renderers. No GGUF is selected, no weights are downloaded, and
  `sha256` / `template_sha256` stay null. `repo.links` reports `absent` (local link
  containment does not apply to server-served bytes) and no `REP001` is emitted. A
  listed-but-unreadable file is `skipped` (never `absent`), so the file surface and
  its summary (tokenizer, templates) say the content was not read; a 401/404/network
  failure on the repository itself exits 3. Repo ids are validated before any network
  call: URLs, local paths, backslashes, `..`, and malformed ids exit 3 with a clear
  message.
- The CLI and the MCP `scan` tool now share one `repo.scan_repo` pass, so repo
  findings, report-time policy/baseline handling, and coverage cannot drift. MCP gains
  an additive `target_type` (`None`/`"gguf"`/`"repo"`); a repo target returns the same
  dict shape as a GGUF scan and rejects `manifest_path` /
  `gguf_filename` (`hf_filename` alias).
- `--gguf-file` (`--file` alias) and `--manifest` remain single-GGUF concepts and
  are rejected for a repository target. A3b adds no rules and changes no GGUF
  output; it lands on `building` with no PyPI release.
- Repo scans now parse **safetensors headers only** and run the existing `TOK002`
  vocabulary-vs-embedding/output-axis check against the model's real tensor shapes.
  A local `scan <dir> --as repo` reads the 8-byte prefix and header JSON of each
  top-level `*.safetensors`; `scan org/model --remote --as repo` requests exactly
  those two byte ranges. Tensor data and weights are never read or fetched: a server
  that ignores a nonzero `Range` is refused rather than streamed. No new CLI flag,
  rule, version bump, or PyPI release. `TOK002` is unchanged for GGUF.
- Safetensors headers are bounded (32 MiB per header, 64 MiB total, 32 top-level
  files) and validated (`N > 0`, `8 + N` within the file/returned range, JSON object,
  integer shape list). Coverage gains a `safetensors.<name>` row per header
  (`examined` / `partial` / `unparseable` / `skipped`) and an honest
  `tensor_map.consistency` state: `absent` when no top-level file exists, `examined`
  only when headers parsed, a candidate embedding/output axis was found, and a
  usable `tokenizer.json` vocabulary was compared, otherwise `partial` / `skipped` /
  `unparseable`. Symlink escapes still emit `REP001` and are `skipped`, never
  `absent`; `repo.links` behavior is unchanged.

### Detection
- Extend the TPL023 structural shape set with two target-free forms, each requiring
  narrower evidence than the existing `target + force + persistence` shape: a persistent
  clause scoping a sensitive action (`leak`, `exfiltrate`, `steal`, `disclose`, `expose`,
  `transmit`) to the response, and a persistent clause emitting an opaque `[[marker]]`
  into the response. A conditional instruction carried by a non-default named template --
  `tokenizer.chat_template.tool_use` -- now reaches manual review without widening the
  TPL021 FAIL rule. The shapes feed TPL023 only; the model-card (DOC) and metadata (MET)
  lexicons are unchanged, and the existing shape is still matched first so findings
  already detected by 0.3.1 keep their detail text.

### Validation
- On the retained 136,996-template release-gate inventory, TPL023 is unchanged at 127
  records across 52 unique templates -- the new shapes add no corpus findings. The
  conditional artifact reports one TPL023 WARN tagged `chat_template[tool_use]` and no
  FAIL, and the benign control `If tools are available, always use the required JSON
  schema for every response.` stays clean.

### Repository
- Document the release-only `main` branch and the nightly `building` branch in
  `docs/BUILDING.md`. Nightly artifacts use the existing wheel/sdist release path;
  no standalone executable is produced.

## 0.3.1 — 2026-08-07

### Detection
- Broaden TPL023 WARN detection from exact imperative phrases to a statically emitted
  persistent-obligation shape (`target + force + persistence`). The action verb is no
  longer enumerated, so directives such as `you must always append ...` reach manual
  review without widening the TPL021 FAIL rule or the model-card/metadata lexicon.
- Preserve the existing detail text for lexicon-based TPL023 findings so deterministic
  reports and warning baselines do not churn for templates already detected by 0.3.0.

### Validation
- Confirm the previously missed `jeikei97/gguf-chat-template-metadata-backdoor` template
  now produces one TPL023 WARN and no FAIL. On the retained 136,996-template release-gate
  inventory, TPL023 moved from 115 findings (0.083944%) to 127 (0.092703%), adding six
  unique templates; all six additions were reviewed.

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
