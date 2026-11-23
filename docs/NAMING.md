# Naming

Canonical vocabulary for c4nary. When docs, CLI help, MCP descriptions, and
release notes disagree, this file wins.

## Project and command

| Use | Meaning |
|---|---|
| **c4nary** | the project, PyPI package, Python module, and repository content |
| **canary** | the primary command (`c4nary` is an alias console script) |

Do not call the package “canary” or the command “c4nary” in new prose. The
GitHub repository is currently `paraxaQQ/canary`; that URL is not being renamed.

## Targets

| Use | Meaning | Avoid |
|---|---|---|
| **GGUF target** | one `.gguf` file, local or remote header | “model” when the distinction matters |
| **repository target** | a GGUF-less repository, local or on Hugging Face | “repo-native”, “GGUF-less” as nouns |
| **Ollama target** | an Ollama model reference (not implemented) | implying Ollama works |

`--as {gguf,repo,ollama}` selects the target type. A local directory defaults to
`repo`. A bare Hugging Face repo id defaults to `gguf`; add `--as repo` for a
repository target. There is no `--repo` flag.

## Remote and surfaces

- `--remote` means **the target is on Hugging Face**. It is not a target type.
  With a GGUF target it reads only the header; with `--as repo` it lists the
  repository and reads the fixed repository files and safetensors headers.
- `--bundle` is a **GGUF-only** flag meaning “also audit the repository surfaces
  beside this GGUF.” A repository target reads those surfaces by default; passing
  `--bundle` there is accepted and noted, never required.
- `--deep-tokenizer` is GGUF-specific: it materializes the vocabulary and runs
  the tokenizer seam. On a repository target it reports `tokenizer.deep` as
  `partial` (no token-type split) or `skipped` (no usable vocabulary).
- `--manifest` is **local GGUF only**.
- `--gguf-file` is the canonical filename flag for a remote GGUF. `--file` is a
  compatibility alias and is hidden from help.
- `--ollama` is an alias for `--as ollama`.

## Nightly channel

| Use | Meaning |
|---|---|
| **nightly** | the user-facing channel name |
| **`building`** | the branch that carries nightlies |
| **`vX.Y.Z-building.N`** | the nightly tag |
| **`X.Y.Z+building.N`** | the nightly package version (PEP 440 local version) |
| **Pre-release** | GitHub's release flag/label |

The nightly **tag** is a semver prerelease identifier, while the nightly
**package version** is a local build of the same base (`X.Y.Z+building.N`). That
keeps the wheel distinct from the stable `X.Y.Z` wheel and sorts after the stable
release, so the two can never be confused. GitHub marks the release as a
prerelease. Nightly artifacts are never uploaded to PyPI. `main` keeps the
stable `X.Y.Z` and official `vX.Y.Z` tags.

Do not call the channel “building” in user-facing prose, or the tag a “version”.

## MCP

| Use | Meaning | Legacy alias |
|---|---|---|
| `target_type` | `"gguf"`, `"repo"`, or `None` | — |
| `gguf_filename` | specific GGUF filename for a remote GGUF | `hf_filename` |
| `manifest_path` | local GGUF manifest | — |
| `bundle` | GGUF-only repository surfaces | — |

MCP `target_type` and CLI `--as` are the same concept. A repository target
rejects `gguf_filename`/`hf_filename` and `manifest_path`.

## Legacy names kept for compatibility

- `--file` → `--gguf-file`
- MCP `hf_filename` → `gguf_filename`
- Coverage IDs `bundle.<file>` → canonical term “repository file.” The prefix is
  retained for report-schema stability; a rename is a separate breaking change
  for v1.0.0.
- “GGUF-less” and “repo-native” are explanatory only; prefer “repository target.”

## Command overview

```text
canary                         interactive menu
canary --version               c4nary <version>
canary scan <target>           GGUF file, repository directory, or remote target
canary diff a.gguf b.gguf      structural GGUF diff
canary hash model.gguf         file + template SHA-256; optional manifest
canary rules                   list rule ids
```

`scan` controls: `--as`, `--remote`, `--gguf-file`, `--bundle`,
`--deep-tokenizer`, `--manifest`, `--json`, `--sarif`, `--policy`,
`--baseline`, `--fail-on`.

`diff` and `hash` are GGUF-only. Exit codes: `0` clean, `1` WARN (with
`--fail-on warn`), `2` FAIL, `3` error; `diff` returns `1` for a non-empty diff.
