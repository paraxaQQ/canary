"""Read-only repo target (GGUF-less) for ``canary scan ... --as repo``.

A repo target reads the same fixed allowlist the GGUF-anchored ``--bundle`` pass
already reads, but with no GGUF anchor: ``config.json`` / ``tokenizer.json`` /
templates / model card / ``auto_map`` Python, plus the repository's own
vocabulary and special-token ids.

``LocalRepo`` reads a directory. Symlinks are never reported as absent. Every
allowlisted path is resolved with ``os.path.realpath`` before it is opened:

* an in-root target is read and marked as resolved through a link;
* an out-of-root target is refused (never opened), marked ``skipped``, recorded
  in the ``repo.links`` coverage row, and reported as ``REP001``;
* a broken/looping link or a non-regular target is ``skipped`` with a reason.

Only regular files are opened. Windows junctions and reparse points go through
the same real-path containment check.

``RemoteRepo`` reads the same allowlist from a Hugging Face repository. It lists
the repo's files once, then fetches only listed allowlist names -- never a GGUF,
never weights. Server-served bytes are audited as-is, so local link containment
(``REP001`` / ``repo.links``) does not apply.

``scan_repo`` is the one rule pass both the CLI and the MCP ``scan`` tool call,
so the two surfaces cannot drift.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .bundle import BUNDLE_FILES, ReadResult, _safe_bundle_name
from .safetensors import (
    MAX_HEADER_FILES,
    MAX_TOTAL_HEADER_BYTES,
    PREFIX_BYTES,
    SafetensorsHeader,
    candidate_axes,
    header_length,
    is_safetensors_name,
    merge_shapes,
    read_header,
    read_header_from_stream,
)
from .vocab import (
    SpecialIds,
    Vocab,
    special_ids_from_tokenizer_config,
    vocab_from_tokenizer_json,
)

CONFIG_CAP = 4 << 20
TOKENIZER_CAP = 48 << 20
TEXT_CAP = 1 << 20

_ESCAPE = "escape"
_BROKEN = "broken"
_NOT_REGULAR = "not_regular"

# A Hugging Face repo id: ``org/name`` or a bare ``name``, each segment made of
# letters, digits, dots, dashes, and underscores, starting and ending alphanumeric.
_REPO_ID = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"(?:/[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)?"
)


class RepoError(ValueError):
    """A repo target that cannot be scanned (bad remote id, wrong target kind)."""


@dataclass(frozen=True)
class LinkRecord:
    """A refused or unresolved allowlisted repo path."""

    name: str
    target: str
    kind: str
    reason: str


@dataclass(frozen=True)
class RepoModel:
    """Duck-typed model for rule functions that only need a vocab and empty maps.

    ``metadata`` and ``tensors`` are deliberately empty -- a repo has no
    ``tokenizer.ggml.*`` keys and inventing them would blur provenance.
    """

    path: str
    vocab: Vocab | None
    metadata: Mapping[str, object] = field(default_factory=dict)
    metadata_types: Mapping[str, object] = field(default_factory=dict)
    version: int = 0
    tensor_count: int = 0
    tensors: tuple = ()


@dataclass(frozen=True)
class RepoArtifact:
    """Everything the repo pass read, plus the resolution records."""

    display: str
    root: Path | None
    files: Mapping[str, ReadResult]
    vocab: Vocab | None
    special_ids: SpecialIds
    links: tuple[LinkRecord, ...]
    model: RepoModel
    template_count: int
    tokenizer_truncated: bool = False
    tokenizer_unparseable: bool = False
    template_source_refused: bool = False
    template_source_unparseable: bool = False
    template_source_partial: bool = False
    remote: bool = False
    safetensors: tuple[SafetensorsHeader, ...] = ()
    safetensors_state: str = "absent"
    safetensors_reason: str = "no top-level .safetensors file was present."
    safetensors_capped: bool = False
    tensor_axes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)


def _within(root: str, path: str) -> bool:
    """True when ``path`` is ``root`` or below it, case-folded on Windows."""
    root_key = os.path.normcase(os.path.normpath(root))
    path_key = os.path.normcase(os.path.normpath(path))
    try:
        return os.path.commonpath([root_key, path_key]) == root_key
    except ValueError:  # different drives
        return False


def _safetensors_summary(
    headers: tuple[SafetensorsHeader, ...] | list[SafetensorsHeader],
    *,
    capped: bool,
) -> tuple[str, str]:
    """Aggregate per-header states into one honest repo-level state/reason."""
    count = len(headers)
    skipped = [header.name for header in headers if header.state == "skipped"]
    unparseable = [header.name for header in headers
                   if header.state == "unparseable"]
    partial = [header.name for header in headers if header.state == "partial"]
    examined = [header.name for header in headers if header.state == "examined"]

    if skipped:
        return "skipped", (
            f"{len(skipped)} of {count} top-level safetensors header(s) could not "
            f"be read: {', '.join(sorted(skipped))}."
        )
    if not examined and not partial:
        return "unparseable", (
            f"all {count} top-level safetensors header(s) were unparseable."
        )
    if partial or unparseable or capped:
        details: list[str] = []
        if partial:
            details.append(f"{len(partial)} were truncated or over a bound")
        if unparseable:
            details.append(f"{len(unparseable)} were unparseable")
        if capped:
            details.append(
                f"more than {MAX_HEADER_FILES} top-level .safetensors files were "
                f"present and only the first {MAX_HEADER_FILES} were examined"
            )
        return "partial", (
            f"{len(examined)} of {count} top-level safetensors header(s) were "
            f"parsed; " + "; ".join(details) + "."
        )
    return "examined", (
        f"all {count} top-level safetensors header(s) were parsed; only header "
        f"bytes were read."
    )


class _RepoReader:
    """Shared allowlist reading: vocab, special ids, template bookkeeping.

    Subclasses supply ``read_text`` and ``links``. ``_prepare`` is identical for
    local and remote, so the two readers cannot disagree about what a repo's
    vocabulary or template sources are.
    """

    remote: bool = False
    root: str | None = None

    def __init__(self, display: str) -> None:
        self.display = display
        self._cache: dict[tuple[str, int], ReadResult | None] = {}
        self._files: dict[str, ReadResult] = {}
        self._prepared = False
        self._vocab: Vocab | None = None
        self._tokenizer_truncated = False
        self._tokenizer_unparseable = False
        self._special_ids = SpecialIds()
        self._template_count = 0
        self._template_source_refused = False
        self._template_source_unparseable = False
        self._template_source_partial = False
        self._safetensors: tuple[SafetensorsHeader, ...] = ()
        self._safetensors_state = "absent"
        self._safetensors_reason = "no top-level .safetensors file was present."
        self._safetensors_capped = False
        self._tensor_axes: dict[str, tuple[int, ...]] = {}

    @property
    def files(self) -> Mapping[str, ReadResult]:
        return dict(self._files)

    @property
    def links(self) -> tuple[LinkRecord, ...]:
        return ()

    def read_text(
        self,
        name: str,
        max_bytes: int = TEXT_CAP,
        *,
        python_only: bool = False,
    ) -> ReadResult | None:
        raise NotImplementedError

    def _prepare(self) -> None:
        """Read the files that define the repo's vocab, special ids, and templates."""
        if self._prepared:
            return
        self._prepared = True
        tokenizer_result = self.read_text("tokenizer.json", TOKENIZER_CAP)
        self._tokenizer_truncated = bool(
            tokenizer_result is not None and tokenizer_result.truncated
        )
        tokenizer_data = None
        if (
            tokenizer_result is not None
            and not tokenizer_result.skipped
            and not tokenizer_result.truncated
        ):
            try:
                tokenizer_data = json.loads(tokenizer_result.text)
            except (ValueError, RecursionError):
                tokenizer_data = None
                self._tokenizer_unparseable = bool(tokenizer_result.text.strip())
        self._vocab = (
            vocab_from_tokenizer_json(tokenizer_data)
            if isinstance(tokenizer_data, dict)
            else None
        )
        tokenizer_config = _read_json(self, "tokenizer_config.json")
        special_tokens_map = _read_json(self, "special_tokens_map.json")
        processor_config = _read_json(self, "processor_config.json")
        jinja_result = self.read_text("chat_template.jinja", TEXT_CAP)
        chat_template_jinja = (
            jinja_result.text
            if jinja_result is not None and not jinja_result.skipped
            else None
        )
        self._special_ids = special_ids_from_tokenizer_config(
            tokenizer_config, special_tokens_map, self._vocab
        )
        self._template_count = _template_count(
            tokenizer_config, processor_config, chat_template_jinja
        )
        # A refused template source is unknown, not absent: the summary coverage
        # row must not claim "no template was present" when we never read it.
        self._template_source_refused = any(
            (result := self._files.get(name)) is not None and bool(result.skipped)
            for name in (
                "tokenizer_config.json",
                "processor_config.json",
                "chat_template.jinja",
            )
        )
        # A present-but-broken template source is unknown, not absent either.
        self._template_source_unparseable = any(
            parsed is None
            and (result := self._files.get(name)) is not None
            and not result.skipped
            and not result.truncated
            and bool(result.text.strip())
            for name, parsed in (
                ("tokenizer_config.json", tokenizer_config),
                ("processor_config.json", processor_config),
            )
        )
        self._template_source_partial = any(
            (result := self._files.get(name)) is not None
            and result.truncated
            and not result.skipped
            for name in (
                "tokenizer_config.json",
                "processor_config.json",
                "chat_template.jinja",
            )
        )
        self._prepare_safetensors()

    def safetensors_names(self) -> list[str]:
        """Top-level ``*.safetensors`` names, sorted and capped. No walk."""
        return []

    def read_safetensors_header(
        self,
        name: str,
        *,
        budget: int | None = None,
    ) -> SafetensorsHeader:
        raise NotImplementedError

    def _prepare_safetensors(self) -> None:
        """Read every top-level header once; never the tensor data."""
        names = self.safetensors_names()
        if not names:
            return
        headers: list[SafetensorsHeader] = []
        budget = MAX_TOTAL_HEADER_BYTES
        for name in names:
            header = self.read_safetensors_header(name, budget=budget)
            headers.append(header)
            # The total cap counts header JSON bytes, not the 8-byte prefixes.
            budget = max(0, budget - max(0, header.consumed - PREFIX_BYTES))
        self._safetensors = tuple(headers)
        self._tensor_axes = candidate_axes(merge_shapes(headers))
        self._safetensors_state, self._safetensors_reason = _safetensors_summary(
            headers, capped=self._safetensors_capped
        )

    @property
    def safetensors_headers(self) -> tuple[SafetensorsHeader, ...]:
        self._prepare()
        return self._safetensors

    @property
    def safetensors_state(self) -> str:
        self._prepare()
        return self._safetensors_state

    @property
    def safetensors_reason(self) -> str:
        self._prepare()
        return self._safetensors_reason

    @property
    def tensor_axes(self) -> Mapping[str, tuple[int, ...]]:
        self._prepare()
        return dict(self._tensor_axes)

    @property
    def vocab(self) -> Vocab | None:
        self._prepare()
        return self._vocab

    @property
    def special_ids(self) -> SpecialIds:
        self._prepare()
        return self._special_ids

    @property
    def template_count(self) -> int:
        self._prepare()
        return self._template_count

    @property
    def model(self) -> RepoModel:
        self._prepare()
        return RepoModel(path=self.display, vocab=self._vocab)

    def artifact(self) -> RepoArtifact:
        """Snapshot what the reader has seen so far (call after the rule pass)."""
        self._prepare()
        return RepoArtifact(
            display=self.display,
            root=Path(self.root) if self.root is not None else None,
            files=self.files,
            vocab=self._vocab,
            special_ids=self._special_ids,
            links=self.links,
            model=self.model,
            template_count=self._template_count,
            tokenizer_truncated=self._tokenizer_truncated,
            tokenizer_unparseable=self._tokenizer_unparseable,
            template_source_refused=self._template_source_refused,
            template_source_unparseable=self._template_source_unparseable,
            template_source_partial=self._template_source_partial,
            remote=self.remote,
            safetensors=self._safetensors,
            safetensors_state=self._safetensors_state,
            safetensors_reason=self._safetensors_reason,
            safetensors_capped=self._safetensors_capped,
            tensor_axes=dict(self._tensor_axes),
        )


class LocalRepo(_RepoReader):
    """On-demand reader over one local repository directory."""

    def __init__(self, directory: str) -> None:
        display = os.path.abspath(directory)
        if not os.path.isdir(display):
            raise NotADirectoryError(f"repo target is not a local directory: {display}")
        root = os.path.realpath(display)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"repo target is not a local directory: {display}")
        super().__init__(display)
        self.root = root
        self._links: list[LinkRecord] = []

    @property
    def links(self) -> tuple[LinkRecord, ...]:
        return tuple(self._links)

    def _record_link(self, record: LinkRecord) -> None:
        if all(
            (existing.name, existing.target, existing.kind)
            != (record.name, record.target, record.kind)
            for existing in self._links
        ):
            self._links.append(record)

    def read_text(
        self,
        name: str,
        max_bytes: int = TEXT_CAP,
        *,
        python_only: bool = False,
    ) -> ReadResult | None:
        """Read one allowlisted surface.

        ``None`` means absent (the historical reader contract). A refused path
        returns a :class:`ReadResult` whose ``skipped`` reason is set, so callers
        can never confuse "not looked at" with "not there".
        """
        if python_only:
            name = _safe_bundle_name(name)
        elif name not in BUNDLE_FILES:
            raise ValueError(f"unsupported repo file: {name!r}")
        elif "/" in name or "\\" in name or ".." in name or "\x00" in name:
            raise ValueError(f"unsafe repo filename: {name!r}")

        cache_key = (name, max_bytes)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._read_path(name, max_bytes)
        self._cache[cache_key] = result
        if result is not None:
            self._files[name] = result
        return result

    def _read_path(self, name: str, max_bytes: int) -> ReadResult | None:
        candidate = os.path.abspath(os.path.join(self.root, name))
        try:
            real = os.path.realpath(candidate)
        except OSError as exc:  # pragma: no cover - platform-specific guard
            reason = f"path resolution failed: {exc}"
            self._record_link(LinkRecord(name, candidate, _BROKEN, reason))
            return ReadResult("", skipped=reason)

        if not _within(self.root, real):
            reason = "resolved target is outside the repository root"
            self._record_link(LinkRecord(name, real, _ESCAPE, reason))
            return ReadResult(
                "",
                skipped=f"the symlink target {real!r} is outside the repository root",
            )

        try:
            info = os.stat(candidate)
        except OSError as exc:
            if os.path.islink(candidate) or os.path.normcase(
                os.path.normpath(real)
            ) != os.path.normcase(os.path.normpath(candidate)):
                reason = f"link could not be resolved: {exc.strerror or exc}"
                self._record_link(LinkRecord(name, real, _BROKEN, reason))
                return ReadResult("", skipped=reason)
            return None  # genuinely absent, not a link

        if not stat.S_ISREG(info.st_mode):
            reason = "resolved target is not a regular file"
            self._record_link(LinkRecord(name, real, _NOT_REGULAR, reason))
            return ReadResult("", skipped=reason)

        try:
            with open(real, "rb") as fh:
                data = fh.read(max_bytes + 1)
        except OSError as exc:
            reason = f"read failed: {exc.strerror or exc}"
            self._record_link(LinkRecord(name, real, _BROKEN, reason))
            return ReadResult("", skipped=reason)

        resolved_through_link = (
            os.path.normcase(os.path.normpath(real))
            != os.path.normcase(os.path.normpath(candidate))
        )
        return ReadResult(
            data[:max_bytes].decode("utf-8", errors="replace"),
            truncated=len(data) > max_bytes,
            link=real if resolved_through_link else None,
        )

    def safetensors_names(self) -> list[str]:
        """Sorted top-level ``*.safetensors`` names under the root, cap 32.

        One ``scandir`` of the root -- no recursive walk, and no ``.safetensors``
        buried in a subdirectory (a sharded model names each shard at top level).
        """
        names: list[str] = []
        try:
            with os.scandir(self.root) as entries:
                for entry in entries:
                    if is_safetensors_name(entry.name):
                        names.append(entry.name)
        except OSError:
            self._safetensors_capped = False
            return []
        names.sort()
        self._safetensors_capped = len(names) > MAX_HEADER_FILES
        return names[:MAX_HEADER_FILES]

    def read_safetensors_header(
        self,
        name: str,
        *,
        budget: int | None = None,
    ) -> SafetensorsHeader:
        """Read one local header under the same containment rule as ``_read_path``.

        An out-of-root, broken, or non-regular target records a :class:`LinkRecord`
        (``REP001``) and returns ``skipped``; it is never opened. The file is
        opened only after containment passes, and exactly ``8`` + ``N`` bytes are
        requested from it.
        """
        if not is_safetensors_name(name):
            raise ValueError(f"unsafe safetensors filename: {name!r}")

        candidate = os.path.abspath(os.path.join(self.root, name))
        try:
            real = os.path.realpath(candidate)
        except OSError as exc:  # pragma: no cover - platform-specific guard
            reason = f"path resolution failed: {exc}"
            self._record_link(LinkRecord(name, candidate, _BROKEN, reason))
            return SafetensorsHeader(name, {}, "skipped", reason)

        if not _within(self.root, real):
            reason = "resolved target is outside the repository root"
            self._record_link(LinkRecord(name, real, _ESCAPE, reason))
            return SafetensorsHeader(
                name,
                {},
                "skipped",
                f"the symlink target {real!r} is outside the repository root",
            )

        try:
            info = os.stat(candidate)
        except OSError as exc:
            if os.path.islink(candidate) or os.path.normcase(
                os.path.normpath(real)
            ) != os.path.normcase(os.path.normpath(candidate)):
                reason = f"link could not be resolved: {exc.strerror or exc}"
                self._record_link(LinkRecord(name, real, _BROKEN, reason))
                return SafetensorsHeader(name, {}, "skipped", reason)
            return SafetensorsHeader(
                name,
                {},
                "skipped",
                "the file disappeared before its header could be read",
            )

        if not stat.S_ISREG(info.st_mode):
            reason = "resolved target is not a regular file"
            self._record_link(LinkRecord(name, real, _NOT_REGULAR, reason))
            return SafetensorsHeader(name, {}, "skipped", reason)

        try:
            with open(real, "rb") as fh:
                info = os.fstat(fh.fileno())
                if not stat.S_ISREG(info.st_mode):
                    reason = "resolved target is not a regular file"
                    self._record_link(LinkRecord(name, real, _NOT_REGULAR, reason))
                    return SafetensorsHeader(name, {}, "skipped", reason)
                return read_header_from_stream(
                    name, fh, info.st_size, budget=budget
                )
        except OSError as exc:
            reason = f"read failed: {exc.strerror or exc}"
            self._record_link(LinkRecord(name, real, _BROKEN, reason))
            return SafetensorsHeader(name, {}, "skipped", reason)


def normalize_repo_id(target: str) -> str:
    """Validate a remote repo target, returning the bare ``org/name`` id.

    Remote repo ingest reads repo files, not a single artifact, so URLs, local
    paths, and anything that is not a Hugging Face repo id are refused here --
    before any network call.
    """
    if not isinstance(target, str):
        raise RepoError("remote repo target must be a Hugging Face repo id")
    value = target[len("hf://"):] if target.startswith("hf://") else target
    if target.startswith(("http://", "https://")) or value.startswith(
        ("http://", "https://")
    ):
        raise RepoError(
            "a remote repo target must be a Hugging Face repo id like "
            "'org/model', not a URL; repo ingest reads the repository's files, "
            "not a single .gguf URL"
        )
    if not value:
        raise RepoError(
            "remote repo target is empty; expected a Hugging Face repo id like "
            "'org/model'"
        )
    if "\\" in value or (os.name == "nt" and re.match(r"^[A-Za-z]:", value)):
        raise RepoError(
            f"remote repo target {value!r} looks like a local path; expected a "
            f"Hugging Face repo id like 'org/model'"
        )
    if ".." in value or "\x00" in value or value.startswith((".", "/")):
        raise RepoError(
            f"invalid Hugging Face repo id {value!r}; expected 'org/model' or "
            f"'model'"
        )
    if not _REPO_ID.fullmatch(value):
        raise RepoError(
            f"invalid Hugging Face repo id {value!r}; expected 'org/model' or "
            f"'model'"
        )
    return value


class RemoteRepo(_RepoReader):
    """On-demand reader over one Hugging Face repository.

    The repo is listed once at construction so an absent file costs no fetch and
    a missing/gated repo fails before any rule runs. ``list_files`` and
    ``fetch_text`` are injectable for tests; they default to the network helpers
    in :mod:`c4nary.remote`.
    """

    remote = True
    root = None

    def __init__(self, target: str, *, list_files=None, fetch_text=None,
                 fetch_range=None) -> None:
        repo = normalize_repo_id(target)
        super().__init__(repo)
        self._fetch_text = fetch_text
        self._fetch_range = fetch_range
        if list_files is None:
            list_files = _hf_list_files
        self._listed = frozenset(list_files(repo))

    def _fetch(
        self,
        name: str,
        max_bytes: int,
        *,
        python_only: bool,
    ) -> ReadResult | None:
        fetch = self._fetch_text
        if fetch is None:
            fetch = fetch_repo_text
        raw = fetch(
            self.display,
            name,
            max_bytes=max_bytes,
            python_only=python_only,
        )
        if raw is None:
            return None
        return raw if isinstance(raw, ReadResult) else ReadResult(raw)

    def read_text(
        self,
        name: str,
        max_bytes: int = TEXT_CAP,
        *,
        python_only: bool = False,
    ) -> ReadResult | None:
        if python_only:
            name = _safe_bundle_name(name)
        elif name not in BUNDLE_FILES:
            raise ValueError(f"unsupported repo file: {name!r}")
        elif "/" in name or "\\" in name or ".." in name or "\x00" in name:
            raise ValueError(f"unsafe repo filename: {name!r}")

        cache_key = (name, max_bytes)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # A name the repo never listed is absent: no fetch, no invented 404.
        result = None
        if name in self._listed:
            result = self._fetch(name, max_bytes, python_only=python_only)
            if result is None:
                # It was listed, so it is not absent; the server refused or could
                # not serve it. Keep the local reader's "not looked at" contract.
                result = ReadResult(
                    "",
                    skipped="the repository lists this file but it could not be "
                            "fetched (inaccessible or absent at the resolved path)",
                )
        self._cache[cache_key] = result
        if result is not None:
            self._files[name] = result
        return result

    def safetensors_names(self) -> list[str]:
        """Sorted top-level ``*.safetensors`` names from the fetched listing."""
        names = sorted(
            name for name in self._listed if is_safetensors_name(name)
        )
        self._safetensors_capped = len(names) > MAX_HEADER_FILES
        return names[:MAX_HEADER_FILES]

    def _range(self, name: str, start: int, length: int) -> bytes | None:
        fetch = self._fetch_range
        if fetch is None:
            fetch = fetch_remote_range
        return fetch(self.display, name, start, length)

    def read_safetensors_header(
        self,
        name: str,
        *,
        budget: int | None = None,
    ) -> SafetensorsHeader:
        """Fetch one remote header prefix + header JSON, never the tensor data."""
        if not is_safetensors_name(name):
            raise ValueError(f"unsafe safetensors filename: {name!r}")
        if name not in self._listed:
            return SafetensorsHeader(
                name,
                {},
                "skipped",
                "the repository does not list this file",
            )

        prefix = self._range(name, 0, PREFIX_BYTES)
        if prefix is None:
            return SafetensorsHeader(
                name,
                {},
                "skipped",
                "the repository lists this file but its header prefix could not "
                "be fetched",
            )
        if len(prefix) < PREFIX_BYTES:
            return SafetensorsHeader(
                name,
                {},
                "partial",
                f"only {len(prefix)} of {PREFIX_BYTES} header prefix byte(s) were "
                f"returned; the tensor map is incomplete",
                consumed=len(prefix),
            )

        length, problem = header_length(prefix)
        if problem is not None:
            return read_header(name, prefix, b"")
        assert length is not None
        if budget is not None and length > budget:
            return SafetensorsHeader(
                name,
                {},
                "partial",
                f"header length {length} would exceed the remaining total "
                f"safetensors header budget ({budget} bytes); it was not read",
                consumed=len(prefix),
            )

        header = self._range(name, PREFIX_BYTES, length)
        if header is None:
            return SafetensorsHeader(
                name,
                {},
                "skipped",
                "the header JSON range could not be fetched",
                consumed=len(prefix),
            )
        return read_header(
            name,
            prefix,
            header,
            truncated=len(header) < length,
        )


def scan_repo(
    repo: _RepoReader,
    *,
    deep: bool = False,
    policy=None,
    baseline=None,
    policy_requested: bool = False,
    baseline_requested: bool = False,
):
    """Run the repo rule pass shared by the CLI and the MCP ``scan`` tool.

    ``repo`` is a :class:`LocalRepo` or :class:`RemoteRepo`; both expose
    ``read_text``, ``vocab``, ``special_ids``, ``links``, ``model``,
    ``artifact()``, and ``display``. Report-time policy/baseline controls behave
    exactly as they did in the A3a CLI path.
    """
    from .bundle import repo_findings
    from .coverage import build_repo_coverage
    from .policy import apply_report_policy, load_baseline, load_policy
    from .rules.repo import analyze_repo_links
    from .rules.tokenizer import analyze_tokenizer

    if policy is None:
        policy = load_policy(None)
    if baseline is None:
        baseline = load_baseline(None)

    vocab = repo.vocab
    special_ids = repo.special_ids

    findings, bundle_surfaces = repo_findings(
        repo.read_text,
        vocab=vocab,
        special_ids=special_ids,
    )
    findings = list(findings)
    findings += analyze_tokenizer(
        repo.model,
        deep=deep,
        vocab=vocab,
        special_ids=special_ids,
        tensor_axes=repo.tensor_axes,
    )
    findings += analyze_repo_links(repo.links)

    registered = [finding.severity for finding in findings]
    findings = apply_report_policy(findings, policy, baseline)
    suppressed_count = sum(finding.suppressed for finding in findings)
    retiered_count = sum(
        before != after.severity for before, after in zip(registered, findings)
    )
    artifact = repo.artifact()
    coverage = build_repo_coverage(
        artifact,
        template_findings=findings,
        deep_tokenizer=deep,
        bundle_surfaces=bundle_surfaces,
        baseline_requested=baseline_requested,
        suppressed_count=suppressed_count,
        policy_requested=policy_requested,
        retiered_count=retiered_count,
    )
    return findings, coverage


def _read_json(repo: _RepoReader, name: str, max_bytes: int = CONFIG_CAP) -> object:
    result = repo.read_text(name, max_bytes)
    if result is None or result.skipped:
        return None
    try:
        return json.loads(result.text)
    except (ValueError, RecursionError):
        return None


def _template_count(tokenizer_config: object, processor_config: object,
                    chat_template_jinja: str | None) -> int:
    count = 0
    if isinstance(tokenizer_config, dict):
        value = tokenizer_config.get("chat_template")
        if isinstance(value, str) and value.strip():
            count += 1
        elif isinstance(value, list):
            count += sum(
                1 for item in value
                if isinstance(item, dict)
                and isinstance(item.get("template"), str)
                and item["template"].strip()
            )
    if isinstance(processor_config, dict):
        value = processor_config.get("chat_template")
        if isinstance(value, str) and value.strip():
            count += 1
    if isinstance(chat_template_jinja, str) and chat_template_jinja.strip():
        count += 1
    return count


def load_local_repo(directory: str) -> LocalRepo:
    """Open a local repository directory for a repo-native scan."""
    return LocalRepo(directory)


def _hf_list_files(repo: str) -> list[str]:
    """Lazy indirection over ``c4nary.remote._hf_list_files``.

    Kept as a module-level name so tests (and callers) can patch the listing
    without importing the optional network module.
    """
    from .remote import _hf_list_files as impl

    return impl(repo)


def fetch_repo_text(
    repo: str,
    name: str,
    *,
    max_bytes: int = TEXT_CAP,
    python_only: bool = False,
) -> ReadResult | None:
    """Lazy indirection over ``c4nary.remote.fetch_repo_text``."""
    from .remote import fetch_repo_text as impl

    return impl(repo, name, max_bytes=max_bytes, python_only=python_only)


def fetch_remote_range(
    repo: str,
    name: str,
    start: int,
    length: int,
) -> bytes | None:
    """Lazy indirection over ``c4nary.remote.fetch_remote_range``."""
    from .remote import fetch_remote_range as impl

    return impl(repo, name, start, length)


def load_remote_repo(target: str, **kwargs) -> RemoteRepo:
    """Open a Hugging Face repository for a repo-native scan."""
    return RemoteRepo(target, **kwargs)
