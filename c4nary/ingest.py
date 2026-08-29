"""Scan-target classification.

From v1.0.0 a GGUF file is one ingest among several, so ``canary scan`` decides
*what kind of thing* it was handed before any parser sees it. Classification is
cheap and read-only: a four-byte magic check, a filename, or a one-line text
sniff.

Two properties keep the calibrated GGUF path from moving:

* GGUF magic is checked before any text heuristic, and ``--as`` lets a caller
  bypass sniffing entirely.
* An unrecognized target falls through to ``gguf``, so the GGUF parser still
  raises its own historical error for a missing file or a non-model file. This
  classifier never invents an error message for a case that already had one.
"""

from __future__ import annotations

import os
from typing import Literal

Target = Literal["gguf", "ollama", "repo"]
TARGETS: tuple[Target, ...] = ("gguf", "ollama", "repo")

GGUF_MAGIC = b"GGUF"
OLLAMA_SCHEME = "ollama://"

# Enough to clear a licence header or a run of comments before the first directive.
_SNIFF_BYTES = 8192


class IngestError(Exception):
    """An explicitly requested target type that does not exist."""


def _has_gguf_magic(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == GGUF_MAGIC
    except OSError:
        return False


def _looks_like_modelfile(path: str) -> bool:
    """True when the first meaningful line is a ``FROM`` directive.

    Deliberately narrow: ``FROM`` is the one directive an Ollama Modelfile must
    carry, and requiring it *first* keeps arbitrary text files out.
    """
    name = os.path.basename(path)
    if name == "Modelfile" or name.endswith(".Modelfile"):
        return True
    try:
        with open(path, "rb") as fh:
            head = fh.read(_SNIFF_BYTES)
    except OSError:
        return False
    for raw in head.decode("utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        return line.split(None, 1)[0].upper() == "FROM"
    return False


def classify_target(
    target: str,
    *,
    remote: bool = False,
    ollama: bool = False,
    override: str | None = None,
) -> Target:
    """Classify a scan target, defaulting to ``gguf``."""
    if override is not None:
        if override not in TARGETS:
            raise IngestError(
                f"unknown target type {override!r}; expected one of {', '.join(TARGETS)}")
        return override  # type: ignore[return-value]

    # An Ollama ref and a Hugging Face repo id are the same shape ('a/b', 'a/b:c'),
    # so an ollama target is never inferred from a bare string -- only declared.
    if ollama or target.startswith(OLLAMA_SCHEME):
        return "ollama"

    if remote:
        return "gguf"

    if os.path.isdir(target):
        return "repo"

    if os.path.isfile(target) and not _has_gguf_magic(target) \
            and _looks_like_modelfile(target):
        return "ollama"

    return "gguf"


def strip_ollama_scheme(target: str) -> str:
    return target[len(OLLAMA_SCHEME):] if target.startswith(OLLAMA_SCHEME) else target
