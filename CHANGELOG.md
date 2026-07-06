# Changelog

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
