# A4 — Ollama Modelfile and model store

*Scoping document for A4 and the engine survey that sets the band after it. No A4 code
is in this document. Target: the v1.0.0 "global-on-surface" release, on `building`
first.*

## 1. Goal

Audit an Ollama model definition without executing Ollama, pulling a model, or reading
weights:

```powershell
canary scan .\Modelfile                  # classified as ollama
canary scan .\Modelfile --as ollama      # explicit
canary scan ollama://library/qwen3:0.6b  # A4c: resolve in the local store
```

Same renderers, exit codes, policy/baseline controls, and honesty rules as a GGUF or
repository scan. Every surface the Modelfile carries is either examined or reported
with a reason. A Modelfile that points at a model we did not read must say so.

### Non-goals

- No `ollama create`, no `ollama run`, no model pull, no registry write.
- No weights. A4a–A4c read the Modelfile, the manifest, and content-addressed
  instruction/config blobs. A model blob is passed to the existing GGUF scanner only if
  a later slice explicitly opts in.
- No Go-template execution. A4b parses; it never renders.
- No change to the GGUF, repository, or safetensors paths. The 19-case A2 CLI baseline
  must stay byte-identical.
- No behavior change to existing rule IDs in A4a. New Go-template rules are A4b.
- No nightly artifact on PyPI.

## 2. Why Ollama is A4

Ollama is the only widely used local runtime whose prompt format, sampling parameters,
system prompt, example messages, license, and base model are all declared in one
portable text file. That file is what a user shares when they share "a model" — and it
is a surface c4nary currently refuses with exit 3.

It is also the highest-value gap the template work leaves open. The Modelfile `TEMPLATE`
instruction is code that formats every conversation, and our `TPL` family cannot read it
at all because it is not Jinja2.

## 3. What the target actually is (verified)

### 3.1 Modelfile grammar

One instruction per line, `#` starts a comment, instructions are case-insensitive and
may appear in any order, and `FROM` is required. Verified against
`https://docs.ollama.com/modelfile` (2026-11-25).

| Instruction | Argument shape | Notes |
|---|---|---|
| `FROM` | `name:tag`, a directory, a `.gguf` path, or a glob | required; see 3.2 |
| `PARAMETER` | name + value | the reference table is a subset of what Ollama accepts; validation uses `api.Options`/`api.Runner` plus the parser's accepted-deprecated list (§7) |
| `TEMPLATE` | `"""..."""` | Go `text/template`; see 3.3 |
| `SYSTEM` | `"""..."""` or a line | system message passed to the template |
| `LICENSE` | `"""..."""` or a line | legal text; may be long |
| `MESSAGE` | role + text | roles `system`, `user`, `assistant` |
| `REQUIRES` | version | minimum Ollama version |
| `CAPABILITY` | token | additive capability, e.g. `decision` |
| `ADAPTER`, `DRAFT`, `RENDERER`, `PARSER` | one value | accepted by Ollama's parser but absent from the reference page; A4a records them and reports them as not resolved |

`stop` is repeatable; every other documented parameter is not. Triple-quoted values may
contain newlines, `#`, and instruction-looking text, so the parser cannot be line-based
after the first token.

### 3.2 `FROM` variants

All four are documented:

1. an existing model name and tag (`FROM llama3.2`);
2. a directory of Safetensors weights (`FROM .` or `FROM /path/to/dir`);
3. a single GGUF file, relative to the Modelfile or absolute (`FROM ./model.gguf`);
4. split GGUF shards, either as one wildcard (`FROM ./model-*.gguf`) or repeated `FROM`
   lines.

The path forms are exactly the A3 local-repository and A2 GGUF readers with a different
anchor. The model-name form needs the local store (3.4).

`ollama show --modelfile` emits a generated Modelfile whose `FROM` is an absolute path
into `~/.ollama/models/blobs/sha256-...`. That output is normal, not suspicious, and the
scanner must not flag a blob path as provenance risk.

### 3.3 `TEMPLATE` is Go `text/template`

Not Jinja2. Variables are `.System`, `.Prompt`, and `.Response`, and Ollama's own
template package registers exactly four functions: `json`, `currentDate`,
`yesterdayDate`, and `toTypeScriptType`, with `missingkey=zero`. Everything else is the
Go `text/template` builtin set (`if`, `range`, `with`, `template`, `define`, `block`,
`index`, `len`, `slice`, `printf`, `print`, `eq`, `ne`, `call`, and the like).

Consequences:

- Every `TPL000`–`TPL032` rule is blind to a Modelfile `TEMPLATE`. Running the Jinja
  parser on Go syntax produces `TPL000` on valid files, which is worse than silence.
- `{{ .Prompt }}` and `{{ .System }}` are data, not Python objects. The Jinja SSTI chain
  (`''.__class__.__mro__...`) has no direct analog, so `TPL001`–`TPL005` carry over
  neither mechanically nor in meaning.
- The behavioral risks do carry over: literal instruction text emitted outside the
  conversation, content-keyed branches, date-gated behavior (`currentDate` /
  `yesterdayDate` compared against a value), and hidden/zero-width characters.
- `call` invokes a function value from the data. In Ollama's sandbox the data has no
  funcs, but a Modelfile is a shareable artifact and the construct is unusual enough to
  review.

### 3.4 The local model store

Layout verified against `manifest/paths.go`, `manifest/manifest.go`, and
`server/images.go` (main, 2026-11-25):

- root: `OLLAMA_MODELS`, default `~/.ollama/models`;
- blobs: `models/blobs/sha256-<hex>`, content-addressed;
- manifests: two trees are live at once — legacy `models/manifests/` and
  `models/manifests-v2/`. v2 names are `<host>/<namespace>/<model>/<tag>` with `:`
  escaped as `%3A` and the public host canonicalized to `ollama.com`. Legacy defaults
  are `registry.ollama.ai`, namespace `library`, tag `latest`;
- a manifest is either a single manifest
  (`application/vnd.docker.distribution.manifest.v2+json`) or a manifest list
  (`application/vnd.ollama.manifest.list.v2+json`) whose children are selected by
  runner/format metadata;
- a manifest carries a `config` layer and `layers[]`. Instruction layers are
  `application/vnd.ollama.image.{model, embed, adapter, projector, prompt, template,
  system, params, messages, license}` plus a tensor media type for non-GGUF weights;
- a bare `sha256-<hex>` name resolves to a manifest blob directly, so a digest reference
  is a legitimate store entry, not a path.

Two facts matter for scope: the store is not "one JSON next to one GGUF", and a model
name can resolve to a manifest list before it resolves to layers.

## 4. The template problem

`c4nary/rules/template.py` is a Jinja2 AST analyzer built on `jinja2.nodes`. It cannot
parse Go. The options:

| option | what it buys | cost / risk |
|---|---|---|
| 1. Structural Go parser in Python (`c4nary/go_template.py`) | tokenizes `{{ }}`, `{{- -}}`, comments, `if/else/end`, `range`, `with`, `define`, `block`, `template`; extracts identifiers, literals, and calls; hard caps on size, actions, depth | ~400–600 lines plus tests; not Go's own parser, so it must fail closed on anything it cannot classify |
| 2. Text-only checks, template marked `partial` | URLs/IPs, size, invisible/bidi/control chars, instruction lexicon, all of which are engine-agnostic | the main surface stays unexamined; semantic rules absent |
| 3. Shell out to a Go helper | exact parser | new toolchain and binary across platforms; rejected |
| 4. Third-party Python Go-template parser | reuse | no maintained option; still not Go's parser; new dependency; rejected |

Recommendation: ship **2 in A4a** (honest, useful, no new rules) and **1 in A4b**
(semantic Go rules). Do not fake Jinja parsing of Go, and do not let A4a call the
template `examined` when only its literal text was read.

**Implemented.** A4a shipped text-only with `chat_templates=partial`; A4b now ships
option 1 (`c4nary/go_template.py` plus the `GOT` family) and the row moves to
`examined` / `unparseable`. See §16.

## 5. CLI contract

| invocation | behaviour |
|---|---|
| `scan <path>` named `Modelfile` / `*.Modelfile`, or whose first directive is `FROM` | ollama (existing classifier) |
| `scan <path> --as ollama` | explicit; same path |
| `scan ollama://library/qwen3:0.6b` | A4c; A4a reports "local model-store resolution is not implemented" with exit 3 |
| `--ollama` | alias for `--as ollama`; unchanged |
| `--remote` | error 3: `--remote` means Hugging Face and there is no remote Ollama registry target yet |
| `--gguf-file` (`--file` alias) | error 3: single-GGUF selector, no meaning for a Modelfile |
| `--manifest` | error 3 in A4a/A4b; A4c may compare a resolved GGUF blob |
| `--bundle` | error 3: bundle is the GGUF/repo surface; a Modelfile has its own files |
| `--deep-tokenizer` | accepted; no effect unless a slice scans a resolved GGUF. Emit a `note:` |
| `--json` / `--sarif` / `--policy` / `--baseline` / `--fail-on` | unchanged |
| exit codes | unchanged: 0 clean, 1 warn, 2 FAIL, 3 error |

JSON fields:

- `file` = the Modelfile path, or `ollama://<ref>`;
- `sha256` = SHA-256 of the Modelfile bytes (the artifact that was actually read);
- `template_sha256` = SHA-256 of the `TEMPLATE` instruction text, `null` when there is
  no `TEMPLATE`. This is the one target where the template is a first-class field rather
  than an embedded blob, so hashing it is not a change of meaning. To be confirmed in
  §12.

A hard parse failure is an error, not a finding: missing `FROM`, an unterminated
triple-quoted value, an unknown instruction, multiple `FROM` values that are not all
GGUF files, or a `MESSAGE` role outside the documented three. Soft oddities (unknown
`PARAMETER` name, a `FROM` that does not resolve, a duplicate single-valued instruction)
are coverage plus a `note:`; the scan continues. Repeated `FROM` is legal for split GGUF
shards and is never treated as a duplicate.

## 6. Context model

Follow the A3 shape: a read-only artifact, no fake GGUF metadata.

```python
@dataclass(frozen=True)
class OllamaModelFile:
    display: str                       # path or ollama://ref
    root: Path                         # directory used to resolve relative FROM
    instructions: tuple[Instruction, ...]
    from_spec: FromSpec                # kind, raw, resolved paths/shards, reason
    parameters: tuple[Parameter, ...]
    template: str | None
    system: str | None
    messages: tuple[Message, ...]
    license: str | None
    requires: str | None
    capabilities: tuple[str, ...]
```

`FromSpec` kinds: `model`, `gguf`, `gguf_glob`, `safetensors_dir`, `unresolved`. Each
carries the resolved absolute paths and, for a glob, the sorted shard list. Resolution
is read-only and containment-checked with the A3 `os.path.realpath` rule: a path that
escapes the Modelfile's directory is recorded, not read. Escaping `FROM` is not
automatically malicious (absolute paths are legal), so it is a coverage detail, not a
finding.

An `Instruction` keeps its line number so findings point at the file, not at a
synthesized blob.

## 7. Rule mapping

### A4a — Modelfile surface, Jinja rules not involved

| family | Modelfile mode | notes |
|---|---|---|
| `TPL011` | run on `TEMPLATE`, `SYSTEM`, `MESSAGE` | URLs/IPs in text the model sees |
| `TPL012` | run on `TEMPLATE` only | `LICENSE` is legitimately long |
| `TPL023` | run on `TEMPLATE` literal text, `SYSTEM`, `MESSAGE` | WARN: imperative text not sourced from the conversation |
| `TPL024` / `TPL025` / `TPL026` | run on every instruction value | text-only; invisible, bidi, raw control characters |
| `TPL000`–`TPL005`, `TPL010`, `TPL013`, `TPL021`, `TPL022`, `TPL027` | **not run** | Jinja2 AST semantics; Go analysis is A4b |
| `TPL030` | not run | needs a GGUF template to diverge from |
| `TPL031` / `TPL032` | not run | Jinja include/import and Jinja decode filters; Go analogs are A4b |
| `TPL100` | not run | the reference set is normalized Jinja2 |
| `CFG*` | absent | `PARAMETER` is a different surface; a Modelfile has no `generation_config.json` |
| `MET*` / `TOK*` / `STR*` / `INT*` | absent unless a later slice scans the `FROM` artifact | no GGUF is read by A4a |
| `DOC001` / `DOC002` | not run | `LICENSE` is not a model card; do not reuse card rules |
| `REP001` | not run | no repository allowlist; `FROM` containment is coverage only |

`PARAMETER` values are parsed against Ollama's own accepted names: the current
`api.Options`/`api.Runner` fields (`num_ctx`, `num_gpu`, `num_batch`, `num_thread`,
`main_gpu`, `use_mmap`, `presence_penalty`, `frequency_penalty`, `num_keep`, …) plus the
parser's accepted-deprecated set (`penalize_newline`, `low_vram`, `mirostat`, …). The
reference page documents only a subset; validating against it alone mislabels valid
files in the wild. `typical_p` is rejected by Ollama as a model parameter and is
reported as such. A known name with a well-formed value is `examined`; an unknown name
or an unparseable value is `partial` with the name listed, never silently ignored.
Whether the value is *sensible* (for example `temperature 0` or `repeat_penalty 100`)
is not a rule in A4a.

### A4b — Go template rules (implemented)

The Jinja family cannot see this surface at all, so a separate family is cleaner than
overloading `TPL` IDs with different semantics:

| id | severity | meaning |
|---|---|---|
| `GOT000` | WARN | Go template failed to parse; text checks still ran |
| `GOT001` | FAIL | a `.Prompt`/`.Messages`-keyed branch emits instruction text not sourced from the conversation |
| `GOT002` | WARN | `currentDate`/`yesterdayDate` is used in a test or comparison (date-gated behavior) |
| `GOT003` | WARN | `call` invokes a function value from data |
| `GOT004` | WARN | named template composition (`define`/`template`/`block`) |
| `GOT005` | WARN | reconstructed instruction text through `printf`/`print` concatenation |

Text-level `TPL011`, `TPL012`, `TPL023`, `TPL024`, `TPL025`, `TPL026` are reused
unchanged. No rule claims Go sandbox escape: there is no documented escape chain in a
Modelfile, and a FAIL there would be unfalsifiable.

`GOT001` requires both a specific content comparison (not truthiness, not a role word)
and instruction text in the branch body. `.System` is not treated as user content.
Variables assigned from `.Prompt`/`.Messages` inherit the taint, and `range .Messages`
taints the element variable. The corpus gate treats `GOT001` as excluded because that
gate inventories GGUF/repository Jinja2 templates, not Modelfiles.

## 8. Coverage contract

| id | state | reason |
|---|---|---|
| `ollama.source` | examined | path or `ollama://ref` |
| `ollama.from` | examined / partial / skipped | kind, resolved shards, or why resolution failed; `partial` when a glob matched some but not all expected shards |
| `ollama.parameters` | examined / partial / absent | known names and values vs unknown names |
| `chat_templates` | examined / unparseable / partial / absent | A4b: `examined` when parsed, `unparseable` with `GOT000` on failure, `partial` when empty or over the parser cap |
| `ollama.system` | examined / absent | SYSTEM and `MESSAGE system` text |
| `ollama.messages` | examined / absent | roles and text |
| `ollama.license` | examined / absent | text recorded; not treated as a card |
| `ollama.requires` | examined / absent | syntax only; installed-version compatibility not checked |
| `ollama.capability` | examined / partial / absent | known token vs unknown token |
| `metadata` | absent | a Modelfile carries no GGUF metadata |
| `tokenizer` | absent | no tokenizer surface |
| `structure` | absent | no container |
| `manifest` | absent in A4a/A4b; A4c owns it | store manifest / comparison manifest |
| `policy` / `baseline` | as today | report-time controls |

The `chat_templates` row is the load-bearing honesty lever: a non-examined state must
be visible in the human summary and the stderr notes, not only in `--json`. After
A4b, `examined` is a true statement and `unparseable` refuses to imply a clean
template.

## 9. Slices

- **A4a — local Modelfile (implemented; see §14).** Classifier already returns
  `ollama`; add the parser, `Modelfile`, `FROM` resolution for the four documented
  forms, text-level rule reuse, coverage, CLI/MCP wiring, fixtures. No new rules.
- **A4b — Go template analysis (implemented; see §16).** `c4nary/go_template.py`,
  `GOT000`–`GOT005`, wired into the Modelfile pass; `chat_templates` moves to
  `examined` / `unparseable`, and the rules digest changes because six FAIL/WARN
  rules are added. A bare Jinja `.jinja`/`.jinja2`/`.yaml` template target is the
  next band; see §11.
- **A4c — local model store.** `ollama://name[:tag]` and `FROM <name>` resolution:
  legacy + v2 manifest trees, manifest-list child selection, config/layer parse, read
  `template`/`params`/`system`/`messages`/`license` blobs and audit them through the same
  A4a/A4b path. No weights. Optional `--manifest` against a resolved GGUF blob.
- **A4d — remote registry.** Manifest + config fetch from `registry.ollama.ai` /
  `ollama.com`, digest pinning, auth, caps. Deferred past v1.0.0; it is a network
  protocol slice, not a parser slice.

## 10. Tests and acceptance

- Existing tests pass (A4a + A4b: 710); the 19-case GGUF CLI baseline keeps the
  same findings, coverage, human output, and SARIF results. The ruleset digest
  and the rule rows change because six `GOT` rules are registered.
- Parser fixtures for every instruction, in mixed case and any order; `#` comments
  before and inside values; CRLF; triple-quoted values containing newlines, `#`, and
  `FROM`-looking text; repeated `stop`; repeated `FROM`.
- `FROM` fixtures: model ref, relative GGUF, absolute GGUF, split wildcard with 3
  shards, wildcard matching 2 of 3, missing file, Safetensors directory, `.` with no
  weights, and a path escaping the Modelfile directory.
- Hard errors: missing `FROM`, unterminated `"""`, unknown instruction, mixed
  multiple `FROM`, bad `MESSAGE` role. Each exits 3 with one clear line, no
  traceback. Repeated `FROM` GGUF shards are accepted.
- Rule fixtures: instruction lexicon in `SYSTEM` → `TPL023`; bidi override in
  `MESSAGE` → `TPL025`; zero-width joiner kept clean; URL in `TEMPLATE` → `TPL011`;
  oversized `TEMPLATE` → `TPL012`; oversized `LICENSE` → no `TPL012`.
- A Modelfile with no `TEMPLATE` → `chat_templates=absent`, no `TPL101` (the GGUF
  verdict does not apply).
- A real `ollama show --modelfile`-shaped file scans clean apart from its template's
  literal text; a blob `FROM` path is not a finding.
- Same file scanned twice → byte-identical JSON.
- Local scans run with the network disabled.
- A4b adds a Go parse corpus: valid Llama/ChatML/Qwen-style Go templates, one
  unparseable, one date-gated, one content-keyed injection, one `template` include.
  Zero false-positive FAILs on the valid corpus.

## 11. Other engines (survey)

Confidence is what the linked documentation states, not what the engine does internally.
"Portable file" means a file a user can hand to someone else and have the definition
survive.

| engine | portable definition | template language | weights | audit path today | proposal |
|---|---|---|---|---|---|
| **Ollama** | `Modelfile` | Go `text/template` (+4 funcs) | GGUF, Safetensors dir, model ref | **none** (exit 3) | **A4** |
| **LocalAI** | per-model YAML (`name`, `backend`, `parameters.*`, `context_size`, `threads`, `template.chat/completion/...`, `options[]`, `download_files`, `artifacts`) | Go template + Sprig | GGUF (llama-cpp), HF repos, diffusers/whisper/other backends | none | A5 candidate: the only other portable prompt file |
| **LM Studio** | `model.yaml` (draft 1.0: `model`, `base[]`, `metadataOverrides`, `config.operation/load`, `customFields` with `setJinjaVariable`) | Jinja, from model metadata or the app; **model.yaml has no template field** | GGUF, MLX | none | watch the spec; no template body to audit |
| **Docker Model Runner** | no file; `docker model package --gguf/--safetensors-dir/--dduf/--from/--chat-template` | Jinja in a separate `--chat-template` file | GGUF, Safetensors, DDUF | template file unscannable | covered by a bare-template target (below) |
| **RamaLama** | no file; CLI/OCI transports | inherited from GGUF/HF | GGUF, Safetensors | n/a | not a file target |
| **Jan** | app state/UI, not a portable file | Jinja via llama.cpp | GGUF, MLX (imported separately) | n/a | not a file target |
| **GPT4All** | per-model settings in the app database | Jinja2; legacy `{# gpt4all v1 #}` templates | GGUF | n/a | not a file target |
| **llamafile** | `.args` (llama.cpp flags) + optional `--chat-template` / `--chat-template-file` / `-sys` / `-sysf` | Jinja | GGUF (often bundled into the executable) | template file unscannable | covered by a bare-template target |
| **textgen** (text-generation-webui) | `user_data/CMD_FLAGS.txt`, `--chat-template-file` (`.jinja`, `.jinja2`, `.yaml`), instruction-template YAML, `settings.yaml` | Jinja2 | GGUF, HF, EXL3 | template file unscannable | covered by a bare-template target |
| **vLLM** | no file; `--chat-template` inline Jinja or a file path | Jinja | HF/Safetensors | n/a | covered if a file is passed |
| **TGI** | no file; `--chat-template` file | Jinja | HF/Safetensors | n/a | covered if a file is passed |
| **SGLang** | no file; tokenizer config or `--chat-template` | Jinja | HF/Safetensors | already covered by a repo/GGUF scan | n/a |

What the survey implies:

1. **Only Ollama and LocalAI put prompt semantics in a portable file that is not already
   a HF repo or a GGUF.** Ollama is the larger surface and the one our vocabulary
   already names. LocalAI is the natural A5.
2. **Jinja is the common language everywhere else** — and c4nary already has the
   analyzer. What is missing is a target for a *standalone* `.jinja`/`.jinja2`/`.yaml`
   template file. Docker Model Runner, llamafile, textgen, vLLM, and TGI all expose
   exactly that file and nothing else auditable. A `--as template` target would cover
   five engines' only portable surface with zero new rules, and it is much smaller than
   A5. Proposed name: `template`; proposed behaviour: the file is the artifact, `TPL*`
   runs, `metadata`/`tokenizer`/`structure` are `absent`.
3. **Do not fake a parser for app state.** Jan, GPT4All, and RamaLama have no portable
   definition to audit; a synthetic one would invent provenance. Record them as out of
   scope.
4. **`model.yaml` is worth watching, not building.** The draft spec has no template
   body; a `customFields` entry can set a Jinja variable, but the template itself lives
   in the model or the app. Revisit if the spec gains a template field.

## 12. Decisions to settle

1. **Slice shape — settled.** A4a text-only; A4b adds the Go parser and `GOT` rules.
2. **Rule family — settled.** New `GOT` IDs for Go semantics; reusing `TPL000` for a
   Go parse error would make one rule mean two languages.
3. **Hashing — settled as implemented.** `sha256` = Modelfile bytes;
   `template_sha256` = the normalized `TEMPLATE` hash, `null` when absent.
4. **Store scope.** Is A4c in v1.0.0, or does v1.0.0 ship A4a + A4b and defer the store
   and registry?
5. **Next band after A4a — A4b done.** Go analysis is implemented; the bare Jinja
   template target is next. LocalAI YAML remains a later candidate.
6. **MCP — settled as implemented.** Reuse `path` with `target_type="ollama"` and
   reject `remote`/`gguf_filename`/`manifest_path`/`bundle`.

## 13. Sources

Fetched 2026-11-25:

- `https://docs.ollama.com/modelfile` — instructions, `FROM` variants, parameters.
- `https://docs.ollama.com/import` — Safetensors directory and GGUF import, split
  shards, no quantisation on import.
- `https://github.com/ollama/ollama/blob/main/template/template.go` — `FuncMap`,
  `missingkey=zero`, identifier walk.
- `https://github.com/ollama/ollama/blob/main/manifest/paths.go` — legacy and v2
  manifest trees, blob paths, digest references.
- `https://github.com/ollama/ollama/blob/main/manifest/manifest.go` — manifest and
  manifest-list media types.
- `https://github.com/ollama/ollama/blob/main/server/images.go` — instruction layer
  media types.
- `https://localai.io/docs/models/model-configuration/` — YAML fields, `template.*`,
  Go + Sprig, template variables.
- `https://lmstudio.ai/docs/app/modelyaml` and `https://modelyaml.org` — `model.yaml`
  draft 1.0 fields.
- `https://lmstudio.ai/docs/app/advanced/prompt-template` — Jinja template override.
- `https://docs.docker.com/reference/cli/docker/model/package/` — package flags,
  `--chat-template`.
- `https://docs.mozilla.ai/llamafile/reference/cli_arguments` — `.args`, chat-template
  flags.
- `https://github.com/oobabooga/textgen` README — `CMD_FLAGS.txt`,
  `--chat-template-file`.
- `https://docs.gpt4all.io/gpt4all_desktop/chat_templates.html` — Jinja, v1 templates.
- `https://docs.vllm.ai/en/latest/serving/online_serving/` — `--chat-template`.

## 14. A4a implementation notes

- **Modules.** `c4nary/ollama.py` holds the parser, `Modelfile`, `FROM`
  resolution, `analyze_modelfile`, and `scan_ollama`. The CLI and the MCP
  `scan` tool share `scan_ollama`, so findings, policy/baseline handling, and
  coverage cannot drift.
- **Parser.** A 4 MiB reader cap; a UTF-8 BOM is stripped; invalid UTF-8 is
  replaced with U+FFFD and stated in `ollama.source` rather than silently
  normalized. Values may be triple-quoted across lines, single-line, or
  double-quoted. A `#` is a comment only at the start of a line outside a value.
  A GGUF forced with `--as ollama` is rejected by magic; any NUL byte is
  rejected as non-text.
- **Duplicates.** `FROM` repeats only for GGUF shards. `PARAMETER stop` repeats;
  any other repeated parameter keeps the last value and is listed in
  `ollama.parameters`. `TEMPLATE` / `SYSTEM` / `REQUIRES` keep the last value and
  mark `ollama.instructions=partial` with a note. Every `LICENSE` is kept and
  joined, because Ollama keeps them all. `CAPABILITY` repeats additively.
- **Rules (A4a).** `analyze_template_text` in `rules/template.py` exposes the
  engine-agnostic half of `_text_checks`; `TPL011`, `TPL012`, `TPL023`, `TPL024`,
  `TPL025`, and `TPL026` run on the Modelfile values. `TPL012` is excluded for
  `LICENSE`, and `LICENSE`/example `MESSAGE` text are concealment-only. A4b adds
  the `GOT` family; see §16. A4a alone added no rule, so the GGUF findings were
  byte-identical to the post-naming capture; A4b changes only the ruleset digest
  and the rule rows, not any GGUF finding or coverage row.
- **Coverage.** The target emits `ollama.source`, `ollama.instructions`,
  `ollama.from`, `ollama.parameters`, `chat_templates`, `ollama.system`,
  `ollama.messages`, `ollama.license`, `ollama.requires`, `ollama.capability`,
  plus the structural `absent` rows (`metadata`, `tokenizer`, `structure`,
  `manifest`) and the report-time `policy` / `baseline` rows.
  `chat_templates` is `examined` for a parsed Go template, `unparseable` for a
  failed parse, and `partial` only when the template is empty or over the parser
  cap. Non-examined `ollama.*` rows and the template row are repeated on stderr;
  absent optional instructions are not.
- **JSON.** `sha256` is the Modelfile byte hash; `template_sha256` is the
  normalized `TEMPLATE` hash, or `null` when there is no `TEMPLATE`.
- **Limits.** The referenced model is not read; `FROM` a model reference is
  `skipped` and named in the notes. `FROM` uses filesystem glob semantics, not
  Ollama's own shard matcher. Go control flow is A4b (implemented); `FROM` store
  resolution and the registry remain A4c/A4d.

## 15. Real-world verification (A4a + A4b)

Run 2026-11-25 against 160 `Modelfile`/`*.modelfile` files downloaded from Hugging
Face (search terms: ollama, modelfile, unsloth ollama, gguf ollama, abliterated,
uncensored, jailbreak, roleplay, assistant). The corpus and scripts live under
`outputs/real-modelfiles/` (gitignored) and can be regenerated.

| check | result |
|---|---|
| our parser | 159 accepted, 1 rejected |
| Ollama's own `parser.ParseFile` on the same 160 | 159 accepted, 1 rejected |
| disagreement | none; both reject `NeuralQuantum/ollama` for `STOP` at line 42 |
| `TEMPLATE` blocks | 102 present; 101 parse with Go `text/template` using Ollama's `FuncMap` and `missingkey=zero` |
| our Go parser vs Go | 101 valid / 1 error on the same broken template, with the same message (`unexpected {{end}}`) and line (86) |
| coverage after A4b | `chat_templates=examined` 101, `unparseable` 1, `absent` 57 |
| findings after calibration | 1 WARN (`GOT000` on the genuinely broken template); 0 `GOT` findings on the 101 valid templates |
| Go parse failure | 1 upstream broken template (`mrlarp/.../Modelfile.gpt-oss-tuned`, an unmatched `{{end}}` at line 86); our extraction is byte-faithful |
| CLI loop | 159 exit 0–2 with valid JSON; 1 expected exit 3; coverage ids sorted/unique |
| determinism | 10/10 files byte-identical across two runs |
| mutation fuzz | 4,000 Modelfile mutations; 2,246 parsed, 1,754 `OllamaError`, 0 unexpected exceptions |
| Go differential fuzz | 600 template mutations vs Go's parser: 0 cases where c4nary rejected a template Go accepts; 15 invalid mutations over-accepted (undefined function/variable names) |
| GGUF regression | 4 real remote GGUF scans, 2 `--bundle`, 2 remote `--as repo` scans byte-identical to commit `bae3c87`; 2 gated repos returned the same 401 |

Calibration fixes driven by the corpus:

- `LICENSE` is concealment-only (`TPL024`/`TPL025`/`TPL026`). License text legitimately
  contains URLs and imperative legal prose; running `TPL011`/`TPL023` there produced 42
  false WARNs before the fix.
- Example `MESSAGE` text suppresses the URL check for the same reason; concealment and
  the instruction lexicon still run.
- Parameter validation uses Ollama's accepted names, not the reference table subset.
  `num_gpu`, `presence_penalty`, and `penalize_newline` were the common real-world names
  the page does not list.
- Multiple `LICENSE` instructions are joined, not treated as a duplicate.
- `REQUIRES v0.14.0` is accepted and normalized to `0.14.0`, matching the parser.
- A quote closes only when the accumulated value ends with the quote, so an internal
  `"""` stays template content and a single-quoted value may span lines.

## 16. A4b implementation notes

- **Module.** `c4nary/go_template.py` tokenizes actions (`{{-`/`-}}` trim markers,
  strings, raw strings, runes, numbers, fields, variables, comments), parses the
  block structure (`if`/`else if`/`else`/`end`, `range`, `with`, `define`, `block`,
  `template`, assignments), and parses pipelines, prefix calls, and parenthesized
  expressions. It never renders. Caps: 256 KiB source, 20,000 actions, 25 levels of
  block/expression nesting. Any unexpected exception becomes `GOT000`, never a
  silent clean.
- **Taint.** `.Prompt` and `.Messages` are content roots. Assignments from a tainted
  expression taint the target; `range` over a tainted collection taints the element
  variable and the dot inside the body; `with` over a tainted value taints the dot.
  `.System` is not user content. `GOT001` needs a specific string comparison (empty
  strings, role words, and structural markers are excluded) plus instruction text
  in the branch body.
- **Rules.** `GOT000` WARN parse failure; `GOT001` FAIL content-gated injection;
  `GOT002` WARN date test; `GOT003` WARN `call`; `GOT004` WARN named composition
  (collapsed to one finding per template); `GOT005` WARN reconstructed instruction
  text through `printf`/`print`. `GOT004` is composition, not external inclusion:
  `define`/`block` bodies live inside the audited `TEMPLATE` string.
- **Coverage and digest.** `chat_templates` is `examined` for a parsed template,
  `unparseable` (with the template-relative line) for a failed parse, and `partial`
  only for an empty or over-cap template. The six new rule rows change
  `rules_bundle_sha256`; no GGUF finding, coverage row, human output, or SARIF result
  changes. `tools/release_gate_scan.py` records `GOT001` as out of scope for the
  Hugging Face Jinja2 corpus gate, with a reason.
- **Limits.** This is a structural analyzer, not a Go compiler. It deliberately
  over-accepts undefined function and variable names: a template that Go would
  reject for `function "X" not defined` or `undefined variable "$x"` can still be
  `examined`. The direction is intentional -- a false `GOT000` on a valid template
  is worse than an over-accept on one that cannot render. No branch merging:
  assignments inside an `if` branch do not taint a sibling branch or the code after
  it. `range` taints both bindings, including the index. The real-world corpus
  produced zero `GOT` false positives.
