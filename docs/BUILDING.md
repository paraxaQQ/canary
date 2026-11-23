# Branches and releases

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

The nightly wheel carries the same package version as the current base
(`0.3.1`); it is identified by its commit and tag, not its filename. **Do not
upload a nightly build to PyPI.**

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
git tag -a v0.3.1-building.1 -m "building: v0.3.1-building.1"
git push origin v0.3.1-building.1
```

Attach the wheel and sdist to a GitHub **prerelease** for that tag:

```powershell
gh release create v0.3.1-building.1 outputs\building-dist\* `
  --prerelease --title "building v0.3.1-building.1" `
  --notes "Nightly build; see CHANGELOG.md. Not an official release."
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
