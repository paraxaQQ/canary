"""Read-only GGUF parser.

Parses a ``.gguf`` file's header, metadata key/value store, and tensor *map*
(names, shapes, dtypes, offsets) **without ever reading tensor data** and
**without rendering anything**. The parser is deliberately defensive: every
length and count read from the (untrusted) file is bounds-checked against the
remaining file size before it is acted on, so a hostile or corrupt file cannot
drive the parser into a huge allocation or unbounded loop.

It also exposes the structural facts the rules need to detect files crafted to
exploit naive loaders (``file_size``, the computed ``data_start`` of the tensor
data section, ``alignment``, raw ggml type ids, duplicate metadata keys) plus
GGML block-size tables so a tensor's on-disk size can be computed and bounded.
Computing those sizes never reads the bytes themselves.

Format reference: GGUF v2/v3, little-endian.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"

# GGUF metadata value type enum.
_T_UINT8 = 0
_T_INT8 = 1
_T_UINT16 = 2
_T_INT16 = 3
_T_UINT32 = 4
_T_INT32 = 5
_T_FLOAT32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64 = 10
_T_INT64 = 11
_T_FLOAT64 = 12

_VALUE_TYPE_NAMES = {
    _T_UINT8: "uint8", _T_INT8: "int8", _T_UINT16: "uint16", _T_INT16: "int16",
    _T_UINT32: "uint32", _T_INT32: "int32", _T_FLOAT32: "float32",
    _T_BOOL: "bool", _T_STRING: "string", _T_ARRAY: "array",
    _T_UINT64: "uint64", _T_INT64: "int64", _T_FLOAT64: "float64",
}

# Smallest possible on-disk size of one element of each scalar type. Used to
# reject arrays whose declared length could not possibly fit in the file.
_MIN_ELEM_SIZE = {
    _T_UINT8: 1, _T_INT8: 1, _T_UINT16: 2, _T_INT16: 2, _T_UINT32: 4,
    _T_INT32: 4, _T_FLOAT32: 4, _T_BOOL: 1, _T_STRING: 8, _T_ARRAY: 12,
    _T_UINT64: 8, _T_INT64: 8, _T_FLOAT64: 8,
}

# GGML tensor type enum -> human name.
_GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2", 5: "Q4_3",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
    28: "F64", 29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4", 32: "Q4_0_4_8",
    33: "Q4_0_8_8", 34: "TQ1_0", 35: "TQ2_0",
}

# Elements per quantization block. Confident for all listed types.
_GGML_BLOCK_SIZE = {
    0: 1, 1: 1, 28: 8, 30: 1, 24: 1, 25: 1, 26: 1, 27: 1,
    2: 32, 3: 32, 6: 32, 7: 32, 8: 32, 9: 32, 20: 32,
    10: 256, 11: 256, 12: 256, 13: 256, 14: 256, 15: 256, 16: 256, 17: 256,
    18: 256, 19: 256, 21: 256, 22: 256, 23: 256, 29: 256, 34: 256, 35: 256,
}

# On-disk bytes per block. Intentionally only the types whose layout is certain;
# unknown types yield ``None`` from ``ggml_type_size`` so size-dependent checks
# are *skipped* (never false-positive) rather than guessed.
_GGML_TYPE_SIZE = {
    0: 4, 1: 2, 28: 8, 30: 2,                      # F32, F16, F64, BF16
    24: 1, 25: 2, 26: 4, 27: 8,                    # I8, I16, I32, I64
    2: 18, 3: 20, 6: 22, 7: 24, 8: 34,             # Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
}

# When a metadata array is longer than this we keep only a preview of its
# elements (plus the true length). Token/merge vocabularies routinely have 10^5
# entries; we never need them all to do length/shape checks.
_ARRAY_PREVIEW = 64

CHAT_TEMPLATE_KEY = "tokenizer.chat_template"

# Maximum int63 (the C signed 64-bit ceiling a malicious size would wrap past).
INT63_MAX = (1 << 63) - 1
INT32_MAX = (1 << 31) - 1
DEFAULT_ALIGNMENT = 32


class GGUFParseError(Exception):
    """Raised when a file is not a parseable GGUF (bad magic, truncation, etc.)."""


def ggml_block_size(type_id: int) -> int | None:
    return _GGML_BLOCK_SIZE.get(type_id)


def ggml_type_size(type_id: int) -> int | None:
    return _GGML_TYPE_SIZE.get(type_id)


def ggml_nbytes(type_id: int, shape: tuple[int, ...]) -> int | None:
    """On-disk byte size of a tensor, in Python bignums (no C wrap).

    Returns ``None`` when the type's layout is unknown (so callers skip the
    size-dependent check instead of guessing). GGUF stores ``shape[0]`` as the
    innermost dim ``ne[0]``.
    """

    bs = _GGML_BLOCK_SIZE.get(type_id)
    ts = _GGML_TYPE_SIZE.get(type_id)
    if bs is None or ts is None:
        return None
    if not shape:
        return ts
    ne0 = shape[0]
    rest = 1
    for d in shape[1:]:
        rest *= d
    row = ts * ne0 // bs
    return row * rest


def ne_product(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class TensorInfo:
    """Structural description of a tensor. No weight data is included."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    type_id: int


@dataclass(frozen=True)
class MetaArray:
    """Placeholder for an array-valued metadata entry.

    ``preview`` holds up to ``_ARRAY_PREVIEW`` leading elements; ``length`` is
    the true element count; ``max_elem_bytes`` is the largest on-disk element
    size seen (meaningful for string arrays).
    """

    elem_type: str
    length: int
    preview: tuple[Any, ...]
    truncated: bool
    max_elem_bytes: int = 0


@dataclass(frozen=True)
class GGUFModel:
    """In-memory view of a GGUF file's structure (no tensor data)."""

    path: str
    version: int
    tensor_count: int
    metadata: dict[str, Any]
    metadata_types: dict[str, str]
    tensors: tuple[TensorInfo, ...]
    file_size: int = 0
    data_start: int = 0
    alignment: int = DEFAULT_ALIGNMENT
    duplicate_keys: tuple[str, ...] = field(default_factory=tuple)

    @property
    def chat_template(self) -> str | None:
        value = self.metadata.get(CHAT_TEMPLATE_KEY)
        return value if isinstance(value, str) else None

    @property
    def architecture(self) -> str | None:
        value = self.metadata.get("general.architecture")
        return value if isinstance(value, str) else None


class _Reader:
    """Bounds-checked little-endian reader over an open binary file."""

    def __init__(self, fh, size: int) -> None:
        self._fh = fh
        self._size = size

    @property
    def pos(self) -> int:
        return self._fh.tell()

    def read(self, n: int) -> bytes:
        if n < 0:
            raise GGUFParseError("negative read length")
        if self.pos + n > self._size:
            raise GGUFParseError(
                f"read of {n} bytes at offset {self.pos} exceeds file size {self._size}"
            )
        data = self._fh.read(n)
        if len(data) != n:
            raise GGUFParseError("unexpected end of file")
        return data

    def _unpack(self, fmt: str, n: int):
        return struct.unpack(fmt, self.read(n))[0]

    def u8(self) -> int: return self._unpack("<B", 1)
    def i8(self) -> int: return self._unpack("<b", 1)
    def u16(self) -> int: return self._unpack("<H", 2)
    def i16(self) -> int: return self._unpack("<h", 2)
    def u32(self) -> int: return self._unpack("<I", 4)
    def i32(self) -> int: return self._unpack("<i", 4)
    def u64(self) -> int: return self._unpack("<Q", 8)
    def i64(self) -> int: return self._unpack("<q", 8)
    def f32(self) -> float: return self._unpack("<f", 4)
    def f64(self) -> float: return self._unpack("<d", 8)
    def boolean(self) -> bool: return self.read(1) != b"\x00"

    def remaining(self) -> int:
        return self._size - self.pos

    def skip(self, n: int) -> None:
        if n < 0 or n > self.remaining():
            raise GGUFParseError(
                f"skip of {n} bytes at offset {self.pos} exceeds file size {self._size}"
            )
        self._fh.seek(n, io.SEEK_CUR)

    def gguf_string_bytes(self) -> bytes:
        length = self.u64()
        if length > self.remaining():
            raise GGUFParseError(
                f"string length {length} exceeds remaining {self.remaining()} bytes"
            )
        return self.read(length)

    def gguf_string(self) -> str:
        # Robustness over strictness: an auditor must not crash on odd bytes.
        return self.gguf_string_bytes().decode("utf-8", errors="replace")


def _read_scalar(r: _Reader, vtype: int) -> Any:
    if vtype == _T_UINT8: return r.u8()
    if vtype == _T_INT8: return r.i8()
    if vtype == _T_UINT16: return r.u16()
    if vtype == _T_INT16: return r.i16()
    if vtype == _T_UINT32: return r.u32()
    if vtype == _T_INT32: return r.i32()
    if vtype == _T_FLOAT32: return r.f32()
    if vtype == _T_BOOL: return r.boolean()
    if vtype == _T_STRING: return r.gguf_string()
    if vtype == _T_UINT64: return r.u64()
    if vtype == _T_INT64: return r.i64()
    if vtype == _T_FLOAT64: return r.f64()
    raise GGUFParseError(f"unknown metadata value type {vtype}")


def _read_array(r: _Reader, *, full: bool = False) -> MetaArray:
    elem_type = r.u32()
    length = r.u64()
    if elem_type == _T_ARRAY:
        raise GGUFParseError("nested metadata arrays are not supported")
    if elem_type not in _MIN_ELEM_SIZE:
        raise GGUFParseError(f"unknown array element type {elem_type}")
    # Reject impossible lengths before looping (resource-exhaustion guard).
    if length * _MIN_ELEM_SIZE[elem_type] > r.remaining():
        raise GGUFParseError(
            f"array of {length} elements cannot fit in remaining "
            f"{r.remaining()} bytes"
        )
    # Default: keep only a 64-element preview (memory bound). Opt-in ``full`` keeps
    # every element -- needed for the per-token-string checks (special-token
    # reachability, control-string collision) that the seam rules require.
    cap = length if full else _ARRAY_PREVIEW
    preview: list[Any] = []
    max_elem_bytes = 0
    if elem_type == _T_STRING:
        for i in range(length):
            raw = r.gguf_string_bytes()
            if len(raw) > max_elem_bytes:
                max_elem_bytes = len(raw)
            if i < cap:
                preview.append(raw.decode("utf-8", errors="replace"))
    else:
        max_elem_bytes = _MIN_ELEM_SIZE[elem_type]
        for i in range(length):
            value = _read_scalar(r, elem_type)
            if i < cap:
                preview.append(value)
    return MetaArray(
        elem_type=_VALUE_TYPE_NAMES.get(elem_type, f"type{elem_type}"),
        length=length,
        preview=tuple(preview),
        truncated=(not full) and length > _ARRAY_PREVIEW,
        max_elem_bytes=max_elem_bytes,
    )


def _read_value(r: _Reader, vtype: int, *, full: bool = False) -> Any:
    if vtype == _T_ARRAY:
        return _read_array(r, full=full)
    return _read_scalar(r, vtype)


def _skip_value(r: _Reader, vtype: int) -> None:
    if vtype == _T_STRING:
        length = r.u64()
        r.skip(length)
        return
    if vtype == _T_ARRAY:
        elem_type = r.u32()
        length = r.u64()
        if elem_type == _T_ARRAY:
            raise GGUFParseError("nested metadata arrays are not supported")
        elem_size = _MIN_ELEM_SIZE.get(elem_type)
        if elem_size is None:
            raise GGUFParseError(f"unknown array element type {elem_type}")
        if length * elem_size > r.remaining():
            raise GGUFParseError(
                f"array of {length} elements cannot fit in remaining "
                f"{r.remaining()} bytes"
            )
        if elem_type == _T_STRING:
            for _ in range(length):
                item_length = r.u64()
                r.skip(item_length)
        else:
            r.skip(length * elem_size)
        return
    elem_size = _MIN_ELEM_SIZE.get(vtype)
    if elem_size is None or vtype == _T_ARRAY:
        raise GGUFParseError(f"unknown metadata value type {vtype}")
    r.skip(elem_size)


def parse_gguf(path: str | Path,
               materialize: frozenset[str] | set[str] | None = None) -> GGUFModel:
    """Parse ``path`` as a GGUF file and return its structure. Read-only.

    ``materialize`` is an opt-in set of metadata keys (e.g. ``tokenizer.ggml.tokens``)
    whose arrays are kept in FULL rather than 64-element preview -- the per-token
    string data the special-token-reachability / collision checks need. Costs the
    array's full size in memory (a vocab is a few MB); off by default.
    """

    p = Path(path)
    size = p.stat().st_size
    with p.open("rb") as fh:
        return _parse_stream(fh, size, str(p), materialize)


def parse_gguf_bytes(data: bytes,
                     materialize: frozenset[str] | set[str] | None = None,
                     label: str = "<bytes>") -> GGUFModel:
    """Parse GGUF header bytes already held in memory -- no temp file. The remote
    fetcher uses this so concurrent scans never race on a temp file (on Windows, AV
    can briefly lock a just-written .gguf and fail the reopen/unlink)."""

    return _parse_stream(io.BytesIO(data), len(data), label, materialize)


def parse_gguf_metadata_bytes(data: bytes, label: str = "<bytes>") -> GGUFModel:
    """Parse only GGUF metadata from bounded header bytes.

    The returned model deliberately has no tensor descriptors or usable data offset.
    It is valid for metadata/template analysis, not STR checks or whole-file hashing.
    """

    return _parse_stream(io.BytesIO(data), len(data), label, None, metadata_only=True)


def extract_gguf_chat_template_bytes(data: bytes, label: str = "<bytes>") -> str | None:
    """Extract only the chat template from a bounded GGUF metadata prefix.

    Non-template arrays are skipped without decoding or retaining their elements.
    This keeps full-catalog validation fast while preserving the parser's bounds
    checks and duplicate-key last-value behavior.
    """

    r = _Reader(io.BytesIO(data), len(data))
    magic = r.read(4)
    if magic != GGUF_MAGIC:
        raise GGUFParseError(
            f"not a GGUF file (magic={magic!r}, expected {GGUF_MAGIC!r})"
        )
    version = r.u32()
    if version not in (2, 3):
        raise GGUFParseError(f"unsupported GGUF version {version} (need 2 or 3)")

    r.u64()  # tensor count; tensor descriptors are outside this extractor's scope
    metadata_count = r.u64()
    if metadata_count * 13 > r.remaining():
        raise GGUFParseError("metadata count exceeds what the file can hold")

    template: str | None = None
    for _ in range(metadata_count):
        key = r.gguf_string()
        vtype = r.u32()
        if key == CHAT_TEMPLATE_KEY and vtype == _T_STRING:
            template = r.gguf_string()
        else:
            _skip_value(r, vtype)
            if key == CHAT_TEMPLATE_KEY:
                template = None
    return template


def _parse_stream(fh, size: int, path_str: str,
                  materialize: frozenset[str] | set[str] | None,
                  *, metadata_only: bool = False) -> GGUFModel:
    r = _Reader(fh, size)
    magic = r.read(4)
    if magic != GGUF_MAGIC:
        raise GGUFParseError(
            f"not a GGUF file (magic={magic!r}, expected {GGUF_MAGIC!r})"
        )
    version = r.u32()
    if version not in (2, 3):
        raise GGUFParseError(f"unsupported GGUF version {version} (need 2 or 3)")

    tensor_count = r.u64()
    metadata_count = r.u64()
    if metadata_count * 13 > r.remaining():
        raise GGUFParseError("metadata count exceeds what the file can hold")
    if not metadata_only and tensor_count * 17 > r.remaining():
        raise GGUFParseError("tensor count exceeds what the file can hold")

    metadata: dict[str, Any] = {}
    metadata_types: dict[str, str] = {}
    duplicate_keys: list[str] = []
    for _ in range(metadata_count):
        key = r.gguf_string()
        vtype = r.u32()
        full = materialize is not None and key in materialize
        value = _read_value(r, vtype, full=full)
        if key in metadata:
            # Parser-differential risk: scanners read the first copy, some
            # loaders the last. Record rather than silently collapse.
            duplicate_keys.append(key)
        metadata[key] = value
        metadata_types[key] = _VALUE_TYPE_NAMES.get(vtype, f"type{vtype}")

    raw_align = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
    alignment = raw_align if isinstance(raw_align, int) and raw_align > 0 else DEFAULT_ALIGNMENT
    if metadata_only:
        return GGUFModel(
            path=path_str,
            version=version,
            tensor_count=tensor_count,
            metadata=metadata,
            metadata_types=metadata_types,
            tensors=(),
            file_size=size,
            data_start=0,
            alignment=alignment,
            duplicate_keys=tuple(duplicate_keys),
        )

    tensors: list[TensorInfo] = []
    for _ in range(tensor_count):
        name = r.gguf_string()
        n_dims = r.u32()
        if n_dims > 8:  # GGML_MAX_DIMS is 4; allow slack, reject absurd.
            raise GGUFParseError(f"implausible tensor dimension count {n_dims}")
        shape = tuple(r.u64() for _ in range(n_dims))
        ggml_type = r.u32()
        offset = r.u64()
        tensors.append(
            TensorInfo(
                name=name,
                shape=shape,
                dtype=_GGML_TYPE_NAMES.get(ggml_type, f"GGML_TYPE_{ggml_type}"),
                offset=offset,
                type_id=ggml_type,
            )
        )

    # Tensor data begins after the info table, padded up to ``alignment``.
    data_start = align_up(r.pos, alignment)

    return GGUFModel(
        path=path_str,
        version=version,
        tensor_count=tensor_count,
        metadata=metadata,
        metadata_types=metadata_types,
        tensors=tuple(tensors),
        file_size=size,
        data_start=data_start,
        alignment=alignment,  # effective alignment used; STR005 checks the raw value
        duplicate_keys=tuple(duplicate_keys),
    )
