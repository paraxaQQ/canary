"""Local, read-only repo target (GGUF-less) for ``canary scan <dir> --as repo``.

A repo target reads the same fixed allowlist the GGUF-anchored ``--bundle`` pass
already reads, but with no GGUF anchor: ``config.json`` / ``tokenizer.json`` /
templates / model card / ``auto_map`` Python, plus the repository's own
vocabulary and special-token ids.

Symlinks are never reported as absent. Every allowlisted path is resolved with
``os.path.realpath`` before it is opened:

* an in-root target is read and marked as resolved through a link;
* an out-of-root target is refused (never opened), marked ``skipped``, recorded
  in the ``repo.links`` coverage row, and reported as ``REP001``;
* a broken/looping link or a non-regular target is ``skipped`` with a reason.

Only regular files are opened. Windows junctions and reparse points go through
the same real-path containment check.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .bundle import BUNDLE_FILES, ReadResult, _safe_bundle_name
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
    root: Path
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


def _within(root: str, path: str) -> bool:
    """True when ``path`` is ``root`` or below it, case-folded on Windows."""
    root_key = os.path.normcase(os.path.normpath(root))
    path_key = os.path.normcase(os.path.normpath(path))
    try:
        return os.path.commonpath([root_key, path_key]) == root_key
    except ValueError:  # different drives
        return False


class LocalRepo:
    """On-demand reader over one local repository directory."""

    def __init__(self, directory: str) -> None:
        display = os.path.abspath(directory)
        if not os.path.isdir(display):
            raise NotADirectoryError(f"repo target is not a local directory: {display}")
        self.display = display
        root = os.path.realpath(display)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"repo target is not a local directory: {display}")
        self.root = root
        self._cache: dict[tuple[str, int], ReadResult | None] = {}
        self._files: dict[str, ReadResult] = {}
        self._links: list[LinkRecord] = []
        self._prepared = False
        self._vocab: Vocab | None = None
        self._tokenizer_truncated = False
        self._tokenizer_unparseable = False
        self._special_ids = SpecialIds()
        self._template_count = 0
        self._template_source_refused = False
        self._template_source_unparseable = False
        self._template_source_partial = False

    @property
    def files(self) -> Mapping[str, ReadResult]:
        return dict(self._files)

    @property
    def links(self) -> tuple[LinkRecord, ...]:
        return tuple(self._links)

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
            root=Path(self.root),
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
        )

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


def _read_json(repo: LocalRepo, name: str, max_bytes: int = CONFIG_CAP) -> object:
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
