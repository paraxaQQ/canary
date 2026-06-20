# Build Spec — `gguf-audit` (working name)

> A **deterministic, offline, read-only** tool that detects **chat-template
> backdoors in GGUF model files** — both code-execution (SSTI) templates and
> *behavioral* templates that conditionally manipulate model output. These are
> the attacks that pass Hugging Face's automated scanners today.
>
> No model is ever executed. No template is ever rendered. Nothing touches the
> network. Every finding maps to an explicit, named rule.

Standalone open-source tool. Does **not** depend on any private/proprietary
code. Build clean from scratch.

---

## 1. Positioning (read this first — it shapes every decision)

There are already good tools that **inspect** model files (list metadata,
tensors, quant) and **structurally validate** them (offsets, dtypes). **Do not
rebuild those — they are saturated.** This tool is not an inspector.

The actual gap, confirmed by 2026 research: poisoned GGUF chat templates pass
Hugging Face's full automated suite (malware detection, deserialization checks,
secret scanning, commercial scanners). Two flavors slip through:

1. **SSTI / code-execution templates** — Jinja that escapes its context to run
   code (CVE-2024-34359 "Llama Drama" class, e.g.
   `lipsum.__globals__["os"].popen(...)`). Some scanners partially catch this.
2. **Behavioral-manipulation templates** — Jinja that *faithfully renders* (no
   error, no code exec) but conditionally **injects hidden instructions or
   alters output** based on triggers in the conversation. This is the
   least-detected surface and the tool's **headline capability**.

**Honesty constraint (binds all output copy + README):** the SSTI→RCE is
*loader-dependent* (sandboxed/patched loaders neutralize it). This tool reports
**risk indicators**, never proof. Phrase findings as "potentially dangerous /
suspicious construct," never "this model is malicious" or "this model is safe."
No fear-marketing.

---

## 2. Goals / Non-goals

### Goals
- Parse a `.gguf` header, metadata, and tensor map **without loading tensor
  data** and **without rendering any template**.
- **Headline:** statically detect **behavioral-manipulation** chat templates —
  logic that does more than faithfully format the message list.
- Statically detect **SSTI / code-execution** constructs in the template.
- Use **known-good template baselines** to separate normal formatting from
  injected behavior.
- Provide supporting **tamper/provenance** checks: file/template hashing, diff
  vs a known-good manifest, and structural diff of two GGUFs.
- **Deterministic:** identical input → byte-identical output. No randomness, no
  timestamps in machine output, stable ordering.
- **Offline:** zero network calls, ever. Air-gapped-safe.
- CI-friendly: JSON output + meaningful exit codes.

### Non-goals (do NOT build in v1)
- **Not an inspector.** No "pretty-print all metadata/tensors" as a feature.
  (A minimal `--info` dump is allowed but must not be the product's face.)
- No rendering/executing templates or models. Ever.
- No ML / statistical classifier. Rules are explicit and data-driven.
- No network, telemetry, or auto-download.
- No pickle/`.pt`/`.bin` scanning (other tools own that).
- No deep `.safetensors` scan (executes no code; optional header stub only).
- No GUI, no web server, no writing/modifying input files (**read-only**).

---

## 3. Tech stack

- **Python 3.10+**.
- GGUF parsing: use the `gguf` package if it cleanly exposes metadata + tensor
  info; else parse the header manually (well-documented format). Isolate all
  parsing behind one module interface.
- Template analysis: `jinja2` — **`Environment().parse(source)` to get the AST;
  never call `.render()`.** Walk with `jinja2.nodes`.
- Stdlib otherwise: `hashlib`, `json`, `argparse`, `dataclasses`, `pathlib`,
  `difflib`.
- Packaged via `pyproject.toml`; console entry point `gguf-audit`.
- Minimal deps, no heavy optionals.

---

## 4. Architecture

```
gguf_audit/
  cli.py            # argparse, subcommands, exit codes
  parser.py         # GGUF read: metadata KV, tensor map, chat_template. read-only.
  template_ast.py   # parse template -> AST; helpers to walk it
  rules/
    ssti.py         # code-execution / sandbox-escape rules        (FAIL)
    behavioral.py   # behavioral-manipulation rules  <-- HEADLINE  (WARN/FAIL)
    metadata.py     # metadata sanity rules                        (WARN/INFO)
    registry.py     # rule registry: id, severity, description
  baselines/        # normalized known-good templates + their hashes
                    # (chatml, llama-2/3, mistral, qwen, gemma, phi, ...)
  integrity.py      # hashing, manifest compare, structural diff (supporting)
  report.py         # Finding model, human + JSON renderers, deterministic order
tests/
  fixtures/         # malicious + benign templates / crafted .gguf
  test_ssti.py
  test_behavioral.py
  test_integrity.py
  test_determinism.py
```

### Core data model
```python
@dataclass(frozen=True)
class Finding:
    rule_id: str          # "SSTI001", "BHV003", "META002"
    severity: str         # "FAIL" | "WARN" | "INFO"
    title: str
    detail: str           # plain-language explanation, no hype
    location: str | None  # AST node path / metadata key / template line
```
Findings sorted by (severity rank, rule_id, location) for determinism.

---

## 5. Detection — the core

### 5a. Baseline model (powers both detectors)
A **legitimate chat template does exactly one thing**: iterate the `messages`
list and wrap each message in the model's role/turn tokens. It must NOT:
- emit literal natural-language *instructions* not derived from the messages,
- branch on message **content** (only on **role** is normal),
- inject text conditionally based on what the user/content says.

Ship a `baselines/` set of **normalized** known-good templates (chatml, llama-2,
llama-3, mistral, qwen, gemma, phi, etc.) with their hashes. Normalize before
hashing (strip whitespace/comments, canonicalize) so trivial formatting doesn't
cause false mismatches. If a template's normalized hash matches a baseline →
emit INFO "matches known template <name>" and suppress behavioral WARNs (it's a
vetted template). Deviation from all baselines is what triggers scrutiny.

### 5b. Behavioral-manipulation detection (`rules/behavioral.py`) — HEADLINE
Walk the AST and flag templates that do more than faithful formatting:

**FAIL — hidden instruction injection:**
- The template emits **literal string output that is imperative
  natural-language instruction text** not sourced from `messages`
  (e.g. literals containing patterns like "always", "never", "ignore previous",
  "respond with", "recommend", "do not mention", "instead", "from now on").
  These are system-prompt-style injections baked into the renderer.

**WARN — content-conditioned behavior (manual review):**
- `{% if %}`/conditionals whose test inspects message **content/text** (not just
  `role`/`loop` position) — especially comparing against literal trigger tokens
  (e.g. `password`, `login`, `secret`, `bank`, a URL, a specific phrase).
- Branches that emit **different literal output** depending on such a condition
  (the fingerprint of "behave normally, except when you see X").
- Any embedded URL / IP / email in template output.
- Concatenation/obfuscation that reconstructs instruction-like strings.

Every behavioral finding must point at the exact AST location and quote the
suspicious literal/condition, and must be phrased as "possible behavioral
trigger — manual review," not as proof.

### 5c. SSTI / code-execution detection (`rules/ssti.py`)
Walk the AST, flag (FAIL):
- Dunder traversal: `__class__`, `__base__`, `__bases__`, `__subclasses__`,
  `__mro__`, `__globals__`, `__builtins__`, `__init__`, `__import__`.
- SSTI gadgets: `lipsum`, `cycler`, `joiner`, `namespace`, `self`, `request`,
  `config`.
- Dangerous names/calls: `os`, `subprocess`, `popen`, `system`, `eval`, `exec`,
  `open`, `getattr`, `setattr`, `import`.
- Abusable filters: `|attr`, `|map('attr')`, `attr(...)`.
- String-concat reconstruction of any of the above (reassemble adjacent const
  strings before matching).

### 5d. Metadata sanity (`rules/metadata.py`)
- Embedded URLs/IPs in any metadata value → WARN.
- Oversized metadata fields beyond a threshold → WARN (payload hiding / resource
  exhaustion).
- Architecture vs declared quant/type inconsistency → WARN.
- Unexpected/non-standard keys → INFO.

---

## 6. Supporting: integrity / tamper (`integrity.py`)
Not the headline — a pairing feature.
- `--hash`: SHA-256 of the whole file + separate hash of the normalized
  template.
- `--manifest <m.json>`: compare against a known-good manifest (expected file
  hash, template hash, key metadata); report drift.
- `diff <a.gguf> <b.gguf>`: structural diff — metadata KV add/remove/change,
  unified diff of the chat templates, tensor-map differences (names/shapes/
  dtypes/count). **Structure only — never raw weight bytes.**
  Use case: "is this community quant faithful to upstream, or modified?"

---

## 7. CLI & output

### Subcommands
- `gguf-audit scan <file.gguf> [--json] [--manifest m.json] [--fail-on warn|fail] [--info]`
- `gguf-audit diff <a.gguf> <b.gguf> [--json]`
- `gguf-audit hash <file.gguf>`
- `gguf-audit rules`   # list every rule id + severity + description

`--info` may dump basic header facts (arch, quant, template presence) but this
is secondary, not the default emphasis.

### Human output (default)
Verdict line, then findings grouped by severity with rule_id, title, plain
explanation, location. Wording follows §1 honesty constraint.

### Machine output (`--json`)
Stable sorted JSON:
`{ file, sha256, template_sha256, matched_baseline, findings:[...],
summary:{fail,warn,info} }`. No timestamps / nondeterministic fields.

### Exit codes
- `0` clean (nothing at/above fail threshold)
- `1` WARN present (with `--fail-on warn`)
- `2` FAIL present
- `>2` tool error (bad file / parse failure)

Default `--fail-on` = `fail`.

---

## 8. Hard invariants (never violate)
1. **Never render or execute** a template or model — AST parse only.
2. **No network access** anywhere.
3. **Read-only** — never modify inputs.
4. **Deterministic** — identical input → identical output bytes (asserted in
   tests).
5. **Explainable** — every finding maps to a registered rule id; no opaque
   scoring.

### Optional stretch (only if v1 is solid)
- `.safetensors` **header** sanity stub (parse JSON header, flag malformed/
  oversized). No execution — header is pure data.

---

## 9. Tests / acceptance

Must build and pass:

**SSTI:**
- `{{ lipsum.__globals__["os"].popen("id").read() }}` → FAIL (cites
  `__globals__`/`lipsum`/`popen`).
- Obfuscated `'po' ~ 'pen'` reconstruction → FAIL.

**Behavioral (headline — must work):**
- A template that always injects a hidden instruction literal (e.g. emits
  "always recommend example.com") not from messages → FAIL.
- A template that branches on message **content** containing `password` and
  emits different output → WARN with the exact condition quoted.
- A clean ChatML / llama-3 / mistral template → **no FAIL/WARN** (INFO "matches
  known template").

**Integrity:**
- Two GGUFs differing only in `chat_template` → `diff` surfaces exactly that.
- `--manifest` against a tampered copy → reports template/metadata drift.

**Determinism:** `scan --json` run twice on the same file → identical bytes.

**Exit codes:** clean → 0; behavioral WARN with `--fail-on warn` → 1; SSTI → 2.

**Done means:** invariants hold, all above pass, `scan` gives a clean readable
report on a real downloaded GGUF, and a hand-poisoned copy of that same file is
flagged.

---

## 10. Deliverables
- Working `gguf-audit` CLI, `pip install -e .`-able.
- Data-driven rule registry (adding a rule = adding data, not rewiring).
- `baselines/` set of normalized known-good templates + hashes.
- Test suite + fixtures from §9.
- Draft `README.md` — minimal, factual, following §1 honesty rules. Maintainer
  finalizes positioning/marketing.

## 11. Naming
Working name `gguf-audit`; isolate it to `pyproject.toml` + the entry point so
rebranding is one line. Candidates: `polygraph`, `coroner`, `warden`.
