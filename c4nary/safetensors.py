"""Safetensors **header** reading -- never tensor data and never weights.

A safetensors file starts with an 8-byte little-endian unsigned header length
``N``, followed by ``N`` bytes of UTF-8 JSON:

    {"tensor.name": {"dtype": "F16", "shape": [10, 4], "data_offsets": [0, 80]},
     "__metadata__": {"format": "pt"}}

Only that JSON is read here. The tensor payload that follows the header is never
opened for reading: the local reader asks the opened file for exactly 8 bytes
and then exactly ``N`` bytes, so a multi-GB ``.safetensors`` costs at most its
header. The remote reader requests the same two byte ranges, and a server that
ignores the range is refused rather than streamed.

The parsed shapes are enough for the existing ``TOK002`` vocabulary-vs-embedding
axis check. This module is deliberately stdlib-only and imports nothing from
``c4nary.repo`` so the repo readers can use it without a cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Mapping

PREFIX_BYTES = 8
# A single header, the sum across every header of one repo, and the number of
# top-level shard files examined. Over any bound is ``partial``, never a full
# read: a hostile file must not turn a header scan into a weights download.
MAX_HEADER_BYTES = 32 << 20
MAX_TOTAL_HEADER_BYTES = 64 << 20
MAX_HEADER_FILES = 32

SAFETENSORS_SUFFIX = ".safetensors"

# Embedding tables (input) and output projections (lm_head / tied output) across
# the common architectures: Llama/Qwen/Mistral (`model.embed_tokens.weight`,
# `lm_head.weight`), GPT-2 (`wte.weight`), GPT-NeoX (`embed_in` is not a
# token-axis table, so it is not a candidate), T5/BART (`shared` is
# `shared.weight`, handled below), GGUF-derived exports (`token_embd.weight`),
# and `output.weight` for an untied GGUF-style output.
EMBEDDING_SUFFIXES = (
    "embed_tokens.weight",
    "wte.weight",
    "tok_embeddings.weight",
    "token_embd.weight",
    "shared.weight",
    "word_embeddings.weight",
)
OUTPUT_SUFFIXES = ("lm_head.weight",)
OUTPUT_EXACT = frozenset({"output.weight"})


@dataclass(frozen=True)
class SafetensorsHeader:
    """One parsed header: tensor shapes plus an honest read state.

    ``state`` is the coverage vocabulary: ``examined`` (header JSON parsed),
    ``partial`` (a bound was hit or the header bytes were short), ``unparseable``
    (malformed length/JSON/shape) or ``skipped`` (never read: refused link,
    unfetchable file). ``consumed`` counts the bytes actually requested from the
    file/range, for the repo-wide header budget.
    """

    name: str
    shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    state: str = "examined"
    reason: str = "header parsed"
    consumed: int = 0

    @property
    def examined(self) -> bool:
        return self.state == "examined"

    @property
    def tensor_count(self) -> int:
        return len(self.shapes)


def is_safetensors_name(name: object) -> bool:
    """True for a top-level, non-hidden ``*.safetensors`` filename.

    No separators, no parent traversal, no NUL, no alternate-data-stream colon:
    a name that could address anything other than one file directly under the
    repo root is not a safetensors surface.
    """
    return (
        isinstance(name, str)
        and bool(name)
        and name.endswith(SAFETENSORS_SUFFIX)
        and len(name) > len(SAFETENSORS_SUFFIX)
        and not name.startswith(".")
        and "/" not in name
        and "\\" not in name
        and ".." not in name
        and "\x00" not in name
        and ":" not in name
        # URL-reserved characters would change the remote path or its encoding.
        and "?" not in name
        and "#" not in name
        and "%" not in name
    )


def header_length(prefix: bytes) -> tuple[int | None, str | None]:
    """Validate the 8-byte prefix. Returns ``(length, problem)``.

    ``length`` is returned even when the only problem is the size cap, so the
    caller can report ``partial`` (a bound) rather than ``unparseable`` (a lie).
    """
    if len(prefix) < PREFIX_BYTES:
        return None, (
            f"file is shorter than the {PREFIX_BYTES}-byte header length prefix"
        )
    length = int.from_bytes(prefix, "little")
    if length <= 0:
        return None, f"header length must be positive, got {length}"
    if length > MAX_HEADER_BYTES:
        return length, (
            f"header length {length} exceeds the {MAX_HEADER_BYTES}-byte cap; "
            f"the header was not read"
        )
    return length, None


def _parse_shapes(header: bytes) -> tuple[dict[str, tuple[int, ...]] | None, str | None]:
    try:
        text = header.decode("utf-8")
    except UnicodeDecodeError:
        return None, "header is not valid UTF-8"
    try:
        payload = json.loads(text)
    except (ValueError, RecursionError):
        return None, "header is not valid JSON"
    if not isinstance(payload, dict):
        return None, "header JSON is not an object"

    shapes: dict[str, tuple[int, ...]] = {}
    for tensor_name, entry in payload.items():
        if tensor_name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            return None, f"tensor {tensor_name!r} is not an object"
        shape = entry.get("shape")
        if not isinstance(shape, list):
            return None, f"tensor {tensor_name!r} has no shape list"
        dims: list[int] = []
        for dim in shape:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                return None, f"tensor {tensor_name!r} has a non-integer shape"
            dims.append(dim)
        shapes[str(tensor_name)] = tuple(dims)
    return shapes, None


def read_header(
    name: str,
    prefix: bytes,
    header: bytes,
    *,
    file_size: int | None = None,
    truncated: bool = False,
) -> SafetensorsHeader:
    """Validate and parse header bytes that were already read.

    This function never performs I/O. Callers pass the 8-byte prefix and the
    header bytes they obtained (locally or over a range request), plus the file
    size when known. ``truncated`` marks a short remote range explicitly; a
    short local read is detected from ``len(header) < length``.
    """
    length, problem = header_length(prefix)
    if problem is not None:
        state = "partial" if length is not None else "unparseable"
        return SafetensorsHeader(name, {}, state, problem, consumed=len(prefix))

    assert length is not None
    if file_size is not None and PREFIX_BYTES + length > file_size:
        return SafetensorsHeader(
            name,
            {},
            "unparseable",
            f"header length {length} extends past the end of the file "
            f"({file_size} bytes)",
            consumed=len(prefix),
        )
    if truncated or len(header) < length:
        return SafetensorsHeader(
            name,
            {},
            "partial",
            f"only {len(header)} of {length} header byte(s) were available; "
            f"the tensor map is incomplete",
            consumed=PREFIX_BYTES + len(header),
        )

    shapes, error = _parse_shapes(header)
    if error is not None:
        return SafetensorsHeader(name, {}, "unparseable", error,
                                 consumed=PREFIX_BYTES + length)
    return SafetensorsHeader(
        name,
        shapes,
        "examined",
        f"header parsed: {len(shapes)} tensor(s)",
        consumed=PREFIX_BYTES + length,
    )


def read_header_from_stream(
    name: str,
    stream,
    file_size: int | None,
    *,
    budget: int | None = None,
) -> SafetensorsHeader:
    """Read a header from an open binary stream -- and only the header.

    Exactly two reads are issued: ``PREFIX_BYTES`` and then ``N`` (and the
    second is skipped when a bound already decides the outcome). This is the
    structure that makes "no tensor bytes" testable with a spy stream.
    """
    prefix = stream.read(PREFIX_BYTES)
    length, problem = header_length(prefix)
    if problem is not None:
        state = "partial" if length is not None else "unparseable"
        return SafetensorsHeader(name, {}, state, problem, consumed=len(prefix))
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
    if file_size is not None and PREFIX_BYTES + length > file_size:
        return read_header(name, prefix, b"", file_size=file_size)
    return read_header(name, prefix, stream.read(length), file_size=file_size)


def is_candidate_tensor(name: str) -> bool:
    """True when ``name`` is a token-axis embedding/output tensor."""
    if name in OUTPUT_EXACT:
        return True
    return name.endswith(EMBEDDING_SUFFIXES) or name.endswith(OUTPUT_SUFFIXES)


def candidate_axes(
    shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    """The sorted ``{tensor_name: shape}`` mapping TOK002 can compare."""
    return {
        name: tuple(shapes[name])
        for name in sorted(shapes)
        # A scalar/empty shape cannot be a token axis; do not claim it was compared.
        if is_candidate_tensor(name) and shapes[name]
    }


def merge_shapes(
    headers: Iterable[SafetensorsHeader],
) -> dict[str, tuple[int, ...]]:
    """Merge every parsed header's shapes deterministically (by tensor name)."""
    merged: dict[str, tuple[int, ...]] = {}
    for header in headers:
        for name in sorted(header.shapes):
            merged[name] = tuple(header.shapes[name])
    return merged
