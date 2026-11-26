# Branches and releases

`main` is the release branch. `building` is the branch that carries **nightly**
builds. Feature work, fixes, and refactors land on `building`; `main` only
receives release updates and official tags. Canonical names live in
[NAMING.md](NAMING.md).

## Versioning and tags

- **`main` versions are independent.** A release version is chosen when `main`
  is promoted; it is never copied from `building`.
- **`building` carries a nightly version.** The base is `X.Y.Z`; the local build
  segment carries the nightly number: `X.Y.Z+building.N`.
- **Tags are semver prereleases for `building`:** `vX.Y.Z-building.N`, e.g.
  `v0.3.1-building.5`.
- **Official release tags are plain semver on `main`:** `v0.3.1`, `v0.4.0`,
  `v1.0.0`.
- `main` never inherits `building`'s version number.
- The local `+building.N` segment keeps a nightly wheel distinct from the stable
  `X.Y.Z` wheel and sorts after the stable version. PyPI rejects local versions,
  so nightlies stay on GitHub prereleases.

## Nightly artifacts: the standard wheel/sdist path

Nightly builds use the same packaging path as a release. **No standalone
executable is produced.** The artifact is the normal wheel and source
distribution built from the `building` commit.

Build into a separate directory so nightly artifacts never mix with the
release `dist/` tree:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m build --outdir outputs\building-dist --no-isolation
.\.venv\Scripts\python.exe -m twine check outputs\building-dist\*
```

The nightly wheel carries `X.Y.Z+building.N` (for example
`c4nary-0.3.1+building.5-py3-none-any.whl`), so it is not confused with the
stable `0.3.1` artifact. **Do not upload a nightly build to PyPI.**

## Verification

Follow the installed-artifact checks in `RELEASE.md`, using a throwaway venv on
`Z:`:

- install the wheel into the fresh venv;
- run `canary --version` and `canary rules`;
- scan one clean fixture (exit 0) and one known-dangerous fixture (non-zero
  FAIL exit);
- extract the sdist and run its bundled tests, so `MANIFEST.in` omissions are
  caught.

A nightly tag should not be pushed unless these checks pass.

## Publishing a nightly

```powershell
git push -u origin building
git tag -a vX.Y.Z-building.N -m "nightly: vX.Y.Z-building.N"
git push origin vX.Y.Z-building.N
```

Write the release notes to a file and attach the wheel and sdist to a GitHub
**prerelease** for that tag. The body is not a one-liner; it carries:

- the status line ``Nightly `building` artifact. **Not an official release.**``;
- the band or feature highlights;
- wheel and sdist sizes with SHA-256 hashes;
- the commit and the rule count (and any rule changes);
- the install command;
- the verification results;
- the closing note that the build is not uploaded to PyPI.

```powershell
gh release create vX.Y.Z-building.N outputs\building-dist\* `
  --prerelease --title "nightly vX.Y.Z-building.N" `
  --notes-file outputs\vX.Y.Z-building.N-notes.md
```

GitHub reserves the "Latest" badge for non-prerelease releases, so the newest
stable release keeps it; the nightly is the newest release and sits at the top
of the Releases list, labeled Pre-release.

## Promoting to main

Promotion is a deliberate merge (or PR) from `building` to `main`, never an
automatic fast-forward:

1. Confirm the full test suite and the installed-wheel checks pass on
   `building`.
2. Merge `building` into `main`.
3. Choose `main`'s release version independently and update
   `pyproject.toml` and `c4nary/__init__.py` together.
4. Update `CHANGELOG.md`, then follow `RELEASE.md`: build, verify the wheel and
   sdist, tag `vX.Y.Z`, create the GitHub release, and upload the verified
   archives to PyPI.

`main` must remain the only branch that carries official release tags.
