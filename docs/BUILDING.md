# Branches, builds, and releases

`main` is the release branch. `building` is the nightly/integration branch.
Feature work, fixes, and refactors land on `building`; `main` only receives
release updates and official tags.

## Versioning and tags

- **`main` versions are independent.** A release version is chosen when `main`
  is promoted; it is never copied from `building`.
- **`building` has its own version sequence.** It starts at the package version
  `0.3.1` and increments its own prerelease build number from there.
- **Tags are semver prereleases for `building`:** `v0.3.1-building.1`,
  `v0.3.1-building.2`, then a new base version when `building` advances.
- **Official release tags are plain semver on `main`:** `v0.3.1`, `v0.4.0`,
  `v1.0.0`.
- `main` never inherits `building`'s version number.

A `building` tag points at the `building` commit that produced the artifact.
The standalone executable is attached to that tag's GitHub prerelease; it is
never committed to the repository.

## The building artifact

The artifact is a single-file, console-mode Windows executable named
`canary-building.exe`. It bundles:

- the c4nary scanner and its Jinja2 dependency;
- `c4nary/known_templates/*.jinja` package data (the rules provenance digest
  covers these files);
- `requests` for `scan --remote` (imported lazily by `c4nary.remote`, so it is an
  explicit hidden import);
- no MCP extra, no test dependencies.

Build requirements are pinned in `requirements-build.txt`. Install them into the
repository venv (which lives on `Z:`; nothing is installed system-wide or on
`C:`):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
```

Build:

```powershell
.\.venv\Scripts\python.exe tools\build_exe.py
```

This writes `outputs/building/canary-building.exe` and prints its SHA-256. The
PyInstaller cache, work tree, spec file, and temp directory are all forced under
`outputs/building/`, and the build uses the venv interpreter. `outputs/` and
`build/` are gitignored, so the artifact stays out of the tree.

## Verification

Run the smoke suite against the built executable:

```powershell
.\.venv\Scripts\python.exe tools\verify_exe.py
.\.venv\Scripts\python.exe tools\verify_exe.py --remote unsloth/Qwen3-0.6B-GGUF
```

The suite checks:

- `--version`;
- `rules --json`, including that the executable's
  `rules_bundle_sha256` matches the source tree (proves `known_templates` was
  bundled);
- a clean scan exits 0 with no FAIL;
- an SSTI fixture exits 2 with a `TPL001` FAIL;
- SARIF 2.1.0 output parses;
- a local `--bundle` scan reports `CFG001`;
- optionally, one remote scan over HTTPS.

Do not publish a `building` tag unless the suite passes without `--remote` and
the optional remote check passes when the network is available.

## Publishing a nightly

```powershell
git push -u origin building
git tag -a v0.3.1-building.1 -m "building: v0.3.1-building.1"
git push origin v0.3.1-building.1
```

Then attach the executable to a GitHub **prerelease** for that tag:

```powershell
gh release create v0.3.1-building.1 outputs\building\canary-building.exe `
  --prerelease --title "building v0.3.1-building.1" `
  --notes "Nightly build; see CHANGELOG.md. Not an official release."
```

## Promoting to main

Promotion is a deliberate merge (or PR) from `building` to `main`, never an
automatic fast-forward:

1. Confirm the full test suite and the executable smoke suite pass on
   `building`.
2. Merge `building` into `main`.
3. Choose `main`'s release version independently and update
   `pyproject.toml` and `c4nary/__init__.py` together.
4. Update `CHANGELOG.md` for the release, build and verify the release
   artifact, then tag `vX.Y.Z` on `main`.

`main` must remain the only branch that carries official release tags.
