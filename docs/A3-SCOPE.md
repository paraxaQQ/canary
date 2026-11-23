# A3 — repo-native ingest (GGUF-less)

*Scoping document. No code is implemented by this document. Target: the v1.0.0
"global-on-surface" release, on `building` first.*

## 1. Goal

Audit a model **repository** that contains no GGUF file at all:

```powershell
canary scan path\to\repo --as repo
canary scan org/model --remote --as repo     # A3b
```

Same renderers, exit codes, policy/baseline controls, and honesty rules as a GGUF
scan. Rules that need GGUF metadata, tensors, or structure are reported as
**absent / not applicable**, never silently clean.

### Non-goals

- No weights. No safetensors tensor data in A3a/A3b. A safetensors *header* check
  (TOK002) is A3c, if we want it at all.
- No new *behavioral* detection rules. A3 adds one repo-integrity WARN,
  `REP001`, for a repo file whose symlink target escapes the repository root.
- No Ollama Modelfile ingest (A4).
- No change to the GGUF path. The 19-case A2 baseline must stay byte-identical.
- No nightly artifact on PyPI. A3 lands on `building`; nightly is
  `v0.3.1-building.2` (wheel + sdist prerelease).

## 2. Why this is the next slice

A1 already classifies a directory as a `repo` and exits 3. A2 already gives the
rule layer a source-neutral `Vocab`. `bundle_findings` already reads almost every
repo file for a GGUF-anchored `--bundle` scan. A3 is therefore mostly:

1. a repo context instead of a GGUF anchor;
2. a `SpecialIds` source, the one thing A2 deliberately left GGUF-only;
3. repo-native coverage and CLI wiring.

## 3. CLI contract

| invocation | behaviour |
|---|---|
| `scan <dir>` | repo (existing classification) |
| `scan <dir> --as repo` | explicit repo |
| `scan org/model --remote --as repo` | A3b; until then exit 3 with a clear message |
| `--bundle` in repo mode | implied by the target; accept and emit a `note:` that it is already on |
| `--deep-tokenizer` in repo mode | accepted; `tokenizer.deep` is `partial` (no token types); no seam findings |
| `--file` in repo mode | error 3: a specific GGUF filename has no meaning for a repo target |
| `--manifest` in repo mode | error 3 for now: manifest compare is tied to a single GGUF |
| `--json` / `--sarif` / `--policy` / `--baseline` / `--fail-on` | unchanged |
| exit codes | unchanged: 0 clean, 1 warn, 2 FAIL, 3 error |

`file`, `sha256`, and `template_sha256` in JSON:

- `file` = the directory path or `org/model`;
- `sha256` = `null` (no single artifact);
- `template_sha256` = `null` for now. It means "the GGUF's embedded template
  hash"; inventing one from a directory would change the field's meaning.

## 4. Context model

Do **not** synthesize a fake GGUF metadata dict. A repo has no
`tokenizer.ggml.*` keys, and invented keys would blur provenance.

Introduce `c4nary/repo.py` with a read-only context:

```python
@dataclass(frozen=True)
class RepoArtifact:
    display: str                       # local dir or org/model
    root: Path | None                  # local only
    remote: bool
    files: Mapping[str, ReadResult]    # allowlisted source -> text/truncation
    vocab: Vocab | None                # tokenizer.json, via A2
    special_ids: SpecialIds            # tokenizer_config + special_tokens_map
    templates: tuple[TemplateSource, ...]
```

`SpecialIds` is the A3 sibling of `Vocab`:

```python
@dataclass(frozen=True)
class SpecialIds:
    bos: int | None
    eos: int | None
    unk: int | None
    pad: int | None
    add_bos: bool
    add_eos: bool
    source: str
```

Constructors:

- `special_ids_from_gguf(model)` — wraps `tokenizer.ggml.*` metadata verbatim, so
  CFG001/TOK001/TOK005 keep their current GGUF behaviour byte-for-byte.
- `special_ids_from_tokenizer_config(tokenizer_config, special_tokens_map)` —
  resolves `bos_token`, `eos_token`, `unk_token`, `pad_token` surfaces and ids,
  `add_bos_token`, `add_eos_token`. Missing values stay `None`.

Consumers gain an optional `special_ids=` keyword, defaulting to the GGUF source:

- `analyze_config(..., special_ids=None)` — `_stop_ids` / CFG001;
- `analyze_tokenizer(..., special_ids=None)` — TOK001 and TOK005;
- `bundle_findings(..., special_ids=None)` — the bos/eos reachability block.

### Splitting `bundle_findings`

`bundle_findings` currently mixes two things: repo-file rules and the
GGUF↔repo template-divergence check (TPL030). Recommended split:

- `repo_findings(artifact_or_reader, *, vocab, special_ids)` — configs,
  tokenizer.json, special tokens, processor/preprocessor, auto_map Python,
  README, repo template sources. No GGUF comparison.
- `bundle_findings(model, reader, *, vocab=None, special_ids=None)` — resolves the
  GGUF vocab/special ids, calls `repo_findings`, then runs
  `analyze_repo_templates(..., gguf_templates=...)` for TPL030.

This keeps one implementation of the repo-file rules and avoids a fake model.
The GGUF path passes the same values it does today, so output must not move.

## 5. Surfaces read

Local and remote share one allowlist:

```
config.json
generation_config.json
tokenizer_config.json
special_tokens_map.json
added_tokens.json
tokenizer.json
processor_config.json
preprocessor_config.json
chat_template.jinja
README.md
<auto_map Python modules>.py
```

- Configs: `4 MiB` cap. `tokenizer.json`: `48 MiB` cap. Python: `1 MiB` per file,
  filename validated by `_safe_bundle_name`.
- No recursive directory walk. No weights, no `.safetensors`, no `.bin`.
- Local symlinks are **never** treated as absent. Resolve each fixed path with
  `os.path.realpath` before opening:
  - target inside the root: follow it and audit the resolved content; the file
    surface records that it resolved through a link;
  - target outside the root: do **not** read it — an audit must not pull
    arbitrary local files into its report. The file surface is `skipped`, a
    `repo.links` row is added, and `REP001` (WARN) names the link and target;
  - broken or looping link: `skipped` with the resolution failure as the reason;
  - only regular files are read. A link resolving to a device, FIFO, socket, or
    directory is `skipped`, never opened.
  Windows junctions and reparse points go through the same real-path containment
  check.
- Remote: same fixed names, `fetch_repo_text`, host-scoped auth, retries, caps.
  A 404 is `absent`; the repo itself 404/401 is an error exit 3. Git symlinks are
  served as their stored bytes or resolved by the server; the audit treats the
  served content as authoritative and notes that local containment does not apply.

## 6. Rule mapping

| family | repo mode | notes |
|---|---|---|
| TPL000–TPL027, TPL031, TPL032 | **run** on every template source | source-tagged: `tokenizer_config.json`, `chat_template.jinja`, `processor_config.json[...]` |
| TPL030 | **not applicable** | needs a GGUF template to diverge from |
| TPL100 | **run** | known-good hash |
| TPL101 | **not emitted** | "file declares no chat_template" is a GGUF verdict; coverage says `absent` instead |
| MET001/002/003/004/006, MET010–MET016 | **absent** | GGUF metadata / tensor-map only |
| MET005 | **absent** | GGUF architecture/quantisation |
| MET020/MET021 | **run** on config strings | via `analyze_config`'s text scan |
| TOK001 | **run** when vocab + special ids resolve | id range against `vocab.length` |
| TOK002 | **absent** | needs tensor axes; A3c safetensors header |
| TOK003 | **absent** | GGUF parallel arrays (`scores`, `token_type`) |
| TOK004 | **run** | `Vocab.max_surface_bytes`; repo value is UTF-8 byte length, documented |
| TOK005 | **run** when add flags + ids resolve | `add_bos_token` / `add_eos_token` |
| TOK012/TOK015 | **partial** | no CONTROL/USER_DEFINED split in tokenizer.json |
| STR001–STR008 | **absent** | no GGUF structure |
| INT001–INT006 | **absent** | no manifest / single file |
| CFG001–CFG005 | **run** | repo vocab + special ids |
| DOC001/DOC002 | **run** | `README.md` |
| NRM001–NRM003 | **run** | `tokenizer.json` |
| RMT000/001/010/011/020 | **run** | `auto_map` Python, `ast.parse` only, never imported |
| REP001 | **run** (local repo only) | WARN: an allowlisted repo file is a symlink whose target escapes the repository root; the target is not read |

## 7. Coverage contract

Every skipped check has a state and a reason. Silence never means clean.

Recommended rows:

| id | repo state | reason |
|---|---|---|
| `repo.source` | examined / error | local directory or remote repo |
| `repo.links` | examined / skipped | link containment; `skipped` lists each escape/broken/loop |
| `bundle.<file>` | examined / partial / unparseable / absent / skipped | reuse the existing per-file rows; `skipped` means a link was refused, not that the file was absent |
| `chat_templates` | examined / absent | repo template sources; never TPL101 |
| `tokenizer` | examined / absent | tokenizer.json present/usable |
| `tokenizer.deep` | partial | no token-type split |
| `metadata` | absent | no GGUF metadata |
| `structure` | absent | no GGUF |
| `tensor_map.consistency` | absent | no tensors |
| `manifest` | absent | not supported for a repo in A3 |
| `policy` / `baseline` | as today | report-time controls |

`partial` is load-bearing: a truncated `tokenizer.json` cannot claim a complete
vocab, and `Vocab.complete` must gate CFG002/CFG003 exactly as it does today.
`repo.links` joins the stderr note set, so a refused link is announced on every
human run, not only visible to someone reading the JSON coverage array.

## 8. Slices

- **A3a — local repo.** Context, `SpecialIds`, repo coverage, local reader,
  template/config/tokenizer/card/NRM/RMT routing, CLI wiring, fixtures.
  `--remote --as repo` still exits 3.
- **A3b — remote repo.** HF file listing, fixed-name fetch, auth/caps/errors,
  and an MCP decision (`scan_repo` tool vs extending `scan`).
- **A3c — optional.** Safetensors header → TOK002 and tensor-map consistency.
  Header only; no weight bytes.
- **A4 — Ollama Modelfile.** Separate target type.

## 9. Tests and acceptance

- Existing 478 tests pass.
- The 19-case A2 CLI baseline stays byte-identical.
- Synthetic repo fixtures:
  - backdoor in `tokenizer_config.json` `chat_template` → TPL021 FAIL;
  - backdoor in `chat_template.jinja` → TPL021 FAIL;
  - named template list → every variant scanned and source-tagged;
  - no template sources → no TPL101, coverage `absent`;
  - CFG001 stop id from `tokenizer_config.json`;
  - CFG002 refusal token from `tokenizer.json` vocab;
  - CFG003 forced ids reconstruct from the repo vocab;
  - NRM001 `Replace`; DOC002 README; RMT010 auto_map Python;
  - no `tokenizer.json` → vocab `absent`, CFG002/3 skipped honestly;
  - invalid/truncated JSON → `unparseable`/`partial`;
  - in-root symlink → target scanned under the link's source name, no finding;
  - out-of-root symlink → `REP001` WARN, file surface `skipped`, target bytes
    never read, `repo.links` names the escape;
  - broken/looping symlink → `skipped` with a reason, no REP001 unless it
    escapes;
  - symlink to a device/FIFO/directory → `skipped`, never opened;
  - same directory scanned twice → byte-identical JSON.
- Local repo scans run with the network disabled.
- Real-repo check: 3–5 safetensors HF repos (e.g. Qwen2.5-0.5B-Instruct,
  Phi-3-mini-4k-instruct, t5-small); review every FAIL, target 0 unexpected
  false-positive FAILs.

## 10. Decisions (settled)

1. **`--bundle` in repo mode:** implied by the target; accept it and emit a
   `note:` that bundle surfaces are already being read.
2. **`--file` / `--manifest` in repo mode:** reject with exit 3; both are
   single-GGUF concepts.
3. **`template_sha256`:** `null` for repo targets. Do not change the field's
   meaning.
4. **Coverage ids:** reuse `bundle.<file>` rows and add the repo summary rows
   (`repo.source`, `repo.links`). No rename churn.
5. **MCP:** extend `scan` with a repo target in A3b; A3a is CLI-only.
6. **Symlinks:** never `absent`. Follow in-root targets, refuse out-of-root
   targets without reading them, mark the file surface `skipped`, add a
   `repo.links` row, and emit `REP001` WARN. In-root links are scanned under the
   link's source name.
7. **TOK002:** deferred to A3c (safetensors header only, never weights).
