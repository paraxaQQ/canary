# Changelog

## Unreleased

### Security
- Scope `HF_TOKEN` authentication to the exact `huggingface.co` host so direct remote-scan
  URLs cannot receive the credential.
- Stream repo-bundle files under a hard byte cap and reject oversized responses instead of
  buffering the complete body before truncation.

### Repository
- Add a security reporting policy and Dependabot update configuration.
- Pin GitHub Actions to immutable commit SHAs.
- Exercise the installed MCP extra and a real stdio protocol round-trip in CI.

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
- Full-catalog validation: **188,792 models at 98.6% template coverage** (raw-header
  fetch where HF's metadata API omits the template), **27 flagged repos, 0 false
  positives.** The v2 rules also went through an adversarial multi-agent review.
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
