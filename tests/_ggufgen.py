"""Minimal GGUF *writer* — for synthesizing test fixtures only.

The audited tool is strictly read-only; this helper lives in the test suite so
we can build small, well-formed GGUF files to parse/scan/diff. It mirrors the
little-endian v3 layout the parser reads. Supports the scalar types the tests
need (str / int / bool / float) plus string arrays.
"""

from __future__ import annotations

import struct
from pathlib import Path

from c4nary.parser import align_up, ggml_nbytes, ne_product

GGUF_MAGIC = b"GGUF"

_T_INT32 = 5
_T_UINT32 = 4
_T_FLOAT32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64 = 10
_T_INT64 = 11


def _gguf_str(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(value) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I", _T_BOOL) + (b"\x01" if value else b"\x00")
    if isinstance(value, str):
        return struct.pack("<I", _T_STRING) + _gguf_str(value)
    if isinstance(value, int):
        if 0 <= value <= 0xFFFFFFFF:
            return struct.pack("<I", _T_UINT32) + struct.pack("<I", value)
        if -(2 ** 31) <= value < 0:
            return struct.pack("<I", _T_INT32) + struct.pack("<i", value)
        return struct.pack("<I", _T_INT64) + struct.pack("<q", value)
    if isinstance(value, float):
        return struct.pack("<I", _T_FLOAT32) + struct.pack("<f", value)
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
            body = struct.pack("<I", _T_ARRAY) + struct.pack("<I", _T_INT32)
            body += struct.pack("<Q", len(value))
            for item in value:
                body += struct.pack("<i", item)
            return body
        if value and all(isinstance(x, float) for x in value):
            body = struct.pack("<I", _T_ARRAY) + struct.pack("<I", _T_FLOAT32)
            body += struct.pack("<Q", len(value))
            for item in value:
                body += struct.pack("<f", item)
            return body
        # string array (tokens etc.)
        body = struct.pack("<I", _T_ARRAY) + struct.pack("<I", _T_STRING)
        body += struct.pack("<Q", len(value))
        for item in value:
            body += _gguf_str(str(item))
        return body
    raise TypeError(f"unsupported metadata value type: {type(value)!r}")


def write_gguf(
    path: str | Path,
    metadata: dict,
    tensors: list[tuple[str, tuple[int, ...], int]] | None = None,
    *,
    version: int = 3,
    tail: bytes = b"",
    offsets: list[int] | None = None,
    data_len: int | None = None,
) -> Path:
    """Write a structurally valid GGUF file.

    ``metadata``: ordered key -> value. ``tensors``: list of
    (name, shape, ggml_type_id). By default each tensor is given a correct,
    aligned offset and a real (zero-filled) data section is written, so the file
    passes the structural checks. ``offsets``/``data_len`` override the computed
    layout to synthesize *malformed* files for STR tests. ``tail`` appends extra
    bytes (e.g. to make two structurally identical files differ byte-wise).
    """

    tensors = tensors or []
    align = metadata.get("general.alignment")
    if not isinstance(align, int) or align <= 0:
        align = 32

    # Lay out a valid data section (aligned, non-overlapping) unless overridden.
    computed_offsets: list[int] = []
    cursor = 0
    for _name, shape, tid in tensors:
        nb = ggml_nbytes(tid, shape)
        if nb is None:
            nb = max(ne_product(shape), 1)
        off = align_up(cursor, align)
        computed_offsets.append(off)
        cursor = off + nb
    if offsets is None:
        offsets = computed_offsets
    if data_len is None:
        data_len = cursor

    out = bytearray()
    out += GGUF_MAGIC
    out += struct.pack("<I", version)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(metadata))

    for key, value in metadata.items():
        out += _gguf_str(key)
        out += _value(value)

    for (name, shape, ggml_type), off in zip(tensors, offsets):
        out += _gguf_str(name)
        out += struct.pack("<I", len(shape))
        for dim in shape:
            out += struct.pack("<Q", dim)
        out += struct.pack("<I", ggml_type)
        out += struct.pack("<Q", off)

    # Pad up to the alignment to mark the start of the tensor-data section.
    while len(out) % align != 0:
        out += b"\x00"

    # Write header+infos, then extend the file to include the (zero) data
    # section via seek instead of materializing it -- a realistic-vocab tensor
    # is hundreds of MB and we never need the actual bytes.
    p = Path(path)
    end = len(out) + max(data_len, 0)
    with p.open("wb") as fh:
        fh.write(bytes(out))
        if tail:
            fh.seek(end)
            fh.write(tail)
        elif data_len > 0:
            fh.seek(end - 1)
            fh.write(b"\x00")
    return p
