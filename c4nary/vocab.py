"""Read-only vocabulary source shared by every rule that reads token surfaces.

Rules ask the vocabulary three questions: how many tokens there are, what the
surface string at an id is, and what text a sequence of ids reconstructs to.
Today those facts live in GGUF metadata (``tokenizer.ggml.tokens`` /
``tokenizer.ggml.token_type``). A Hugging Face repo's ``tokenizer.json`` carries
the same facts in a different shape (``model.vocab`` plus ``added_tokens``).

``Vocab`` is the single immutable value both sources can produce, so rule code
never has to assume GGUF arrays. Its GGUF constructor wraps the parsed
``MetaArray`` values verbatim -- including the 64-element preview and the true
length -- so it is a no-op for every existing scan. The ``tokenizer.json``
constructor backs the local repo target (``canary scan <dir> --as repo``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parser import GGUFModel

GGUF_TOKENS_KEY = "tokenizer.ggml.tokens"
GGUF_TOKEN_TYPE_KEY = "tokenizer.ggml.token_type"


class SparseSurfaces(Sequence[object]):
    """Id-indexed token surfaces that stores only the ids actually present.

    ``tokenizer.json`` can name an id far above its entry count. A list sized to
    the maximum id would turn a small file into a large allocation, so holes read
    as ``None`` and nothing is materialised for the empty span.

    Rules index by id (``surface`` / ``reconstruct``); iterating the sequence
    walks the whole declared id space, so do not use it as a sparse-dict
    replacement when the id space is large.
    """

    __slots__ = ("_values", "_length")

    def __init__(self, values: dict[int, object], length: int) -> None:
        self._values = values
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(self._length)))
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("token ids must be integers")
        i = index + self._length if index < 0 else index
        if i < 0 or i >= self._length:
            raise IndexError(index)
        return self._values.get(i)

    def __iter__(self):
        for i in range(self._length):
            yield self._values.get(i)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SparseSurfaces):
            return self._length == other._length and self._values == other._values
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return self._length == len(other) and all(
                self[i] == other[i] for i in range(self._length)
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._length, tuple(sorted(self._values.items()))))

    def __repr__(self) -> str:
        return f"SparseSurfaces(length={self._length}, present={len(self._values)})"


@dataclass(frozen=True)
class Vocab:
    """A vocabulary's surfaces, true size, and completeness.

    ``surfaces`` is indexed by token id. It may hold only a preview, in which
    case ``complete`` is false; ``length`` is always the true token count.
    Elements that are not ``str`` are holes (a missing id or a non-string
    metadata entry) and are never returned by :meth:`surface`.
    """

    surfaces: Sequence[object]
    length: int
    complete: bool
    types: Sequence[object] | None = None
    types_length: int | None = None
    types_complete: bool = False
    max_surface_bytes: int = 0
    source: str = "gguf:tokenizer.ggml.tokens"

    def surface(self, token_id: int) -> str | None:
        """The string surface at ``token_id``, or ``None``. Never raises."""
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            return None
        if not 0 <= token_id < len(self.surfaces):
            return None
        value = self.surfaces[token_id]
        return value if isinstance(value, str) else None

    def reconstruct(self, ids: Sequence[int]) -> str:
        """Join token surfaces into text, mapping SentencePiece ▁ / GPT2 Ġ to a
        space. Byte-for-byte the behaviour rules/config.py has always used."""
        parts: list[str] = []
        for tid in ids:
            if 0 <= tid < len(self.surfaces) and isinstance(self.surfaces[tid], str):
                surface = self.surfaces[tid]
                parts.append(
                    " " + surface[1:] if surface[:1] in ("▁", "Ġ") else surface
                )
        return "".join(parts)


def vocab_from_metadata(metadata: dict) -> Vocab | None:
    """Wrap GGUF metadata arrays verbatim; ``None`` when tokens are not an array."""
    # Local import: parser imports this module lazily from GGUFModel.vocab, so a
    # module-level import here would be a cycle. The isinstance gate is kept
    # exactly (rather than duck-typed) so a scalar at the key stays rejected.
    from .parser import MetaArray

    tokens = metadata.get(GGUF_TOKENS_KEY)
    if not isinstance(tokens, MetaArray):
        return None

    token_type = metadata.get(GGUF_TOKEN_TYPE_KEY)
    if isinstance(token_type, MetaArray):
        types: Sequence[object] | None = token_type.preview
        types_length: int | None = token_type.length
        types_complete = not token_type.truncated
    else:
        types = None
        types_length = None
        types_complete = False

    return Vocab(
        surfaces=tokens.preview,
        length=tokens.length,
        complete=not tokens.truncated,
        types=types,
        types_length=types_length,
        types_complete=types_complete,
        max_surface_bytes=tokens.max_elem_bytes,
    )


def vocab_from_gguf(model: "GGUFModel") -> Vocab | None:
    """Thin wrapper over :func:`vocab_from_metadata`."""
    return vocab_from_metadata(model.metadata)


def vocab_from_tokenizer_json(data: dict) -> Vocab | None:
    """Build a vocabulary from a parsed ``tokenizer.json`` (repo targets).

    ``model.vocab`` is either a ``{surface: id}`` mapping (BPE / WordPiece) or a
    list of ``[token, score]`` pairs (Unigram), where the index is the id. Ids
    are materialized into a list, leaving holes as ``None``. ``added_tokens``
    entries with an int id and a string content overlay the result and extend it
    when their id is past the current end.

    ``added_tokens[].special`` is deliberately not mapped onto ``types``:
    tokenizer.json cannot distinguish GGUF's CONTROL / USER_DEFINED enum, so a
    repo vocabulary reports no token-type information.
    """
    if not isinstance(data, dict):
        return None
    model = data.get("model")
    if not isinstance(model, dict):
        return None

    raw = model.get("vocab")
    values: dict[int, object] = {}

    def put(token_id: object, surface: object) -> None:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            return
        if not isinstance(surface, str):
            return
        values[token_id] = surface

    if isinstance(raw, dict):
        for surface, token_id in raw.items():
            put(token_id, surface)
    elif isinstance(raw, (list, tuple)):
        for token_id, item in enumerate(raw):
            if isinstance(item, (list, tuple)) and item:
                put(token_id, item[0])
    else:
        return None

    added_tokens = data.get("added_tokens")
    if isinstance(added_tokens, (list, tuple)):
        for entry in added_tokens:
            if isinstance(entry, dict):
                put(entry.get("id"), entry.get("content"))

    if not values:
        return None

    length = max(values) + 1
    # A JSON string may carry a lone surrogate (``"\ud800"``); strict UTF-8
    # encoding would raise and turn a hostile tokenizer.json into a scan crash.
    max_surface_bytes = max(
        (len(surface.encode("utf-8", "surrogatepass")) for surface in values.values()
         if isinstance(surface, str)),
        default=0,
    )
    return Vocab(
        surfaces=SparseSurfaces(values, length),
        length=length,
        complete=True,
        types=None,
        types_length=None,
        types_complete=False,
        max_surface_bytes=max_surface_bytes,
        source="tokenizer.json:model.vocab",
    )


def vocab_surface_id(vocab: Vocab | None, surface: str) -> int | None:
    """The smallest id whose surface is exactly ``surface``, or ``None``.

    ``tokenizer.json`` declares special-token surfaces but not their ids; the
    repo special-id constructor resolves a surface through the vocabulary. A
    sparse vocabulary is walked over its present entries only, so a hostile
    declared id cannot turn this into a long loop.
    """
    if vocab is None or not isinstance(surface, str):
        return None
    surfaces = vocab.surfaces
    if isinstance(surfaces, SparseSurfaces):
        matches = [tid for tid, value in surfaces._values.items() if value == surface]
        return min(matches) if matches else None
    for tid in range(min(len(surfaces), vocab.length)):
        if surfaces[tid] == surface:
            return tid
    return None


@dataclass(frozen=True)
class SpecialIds:
    """Read-only special-token ids from a GGUF file or a repo's tokenizer config.

    ``eos`` is the single end-of-sequence id (TOK005), while ``eos_ids`` is the
    stop set the decode-lever rules use. The GGUF constructor keeps all three of
    eos/eot/eom there, exactly as CFG001 has always read them.
    """

    bos: int | None = None
    eos: int | None = None
    unk: int | None = None
    pad: int | None = None
    add_bos: bool = False
    add_eos: bool = False
    source: str = "gguf:tokenizer.ggml"
    eos_ids: frozenset[int] = frozenset()

    def token(self, name: str) -> int | None:
        """The id for ``bos`` / ``eos`` / ``unk`` / ``pad``; unknown names are ``None``."""
        if name not in {"bos", "eos", "unk", "pad"}:
            return None
        return getattr(self, name)

    def add_flag(self, name: str) -> bool:
        """The ``add_*`` flag for ``bos`` / ``eos``; everything else is ``False``."""
        return getattr(self, f"add_{name}", False) is True


def _special_id(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def special_ids_from_gguf(model) -> SpecialIds:
    """Wrap ``tokenizer.ggml.*`` metadata verbatim, preserving current behaviour."""
    metadata = getattr(model, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}

    def get(key: str) -> int | None:
        return _special_id(metadata.get(f"tokenizer.ggml.{key}"))

    # GGUF spells the unknown token key "unknown_token_id"; accept the short
    # spelling too so a hand-built metadata dict is not silently ignored.
    unk = get("unknown_token_id")
    if unk is None:
        unk = get("unk_token_id")
    pad = get("padding_token_id")
    if pad is None:
        pad = get("pad_token_id")

    eos_ids = frozenset(
        value for value in (
            get("eos_token_id"), get("eot_token_id"), get("eom_token_id"),
        )
        if value is not None
    )
    return SpecialIds(
        bos=get("bos_token_id"),
        eos=get("eos_token_id"),
        unk=unk,
        pad=pad,
        add_bos=metadata.get("tokenizer.ggml.add_bos_token") is True,
        add_eos=metadata.get("tokenizer.ggml.add_eos_token") is True,
        source="gguf:tokenizer.ggml",
        eos_ids=eos_ids,
    )


def _resolve_special_value(value: object, vocab: Vocab | None) -> int | None:
    """Resolve a tokenizer-config token value: an int, ``{content, id}``, or a surface."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        explicit = _special_id(value.get("id"))
        if explicit is not None:
            return explicit
        value = value.get("content")
    if isinstance(value, str):
        return vocab_surface_id(vocab, value)
    return None


def special_ids_from_tokenizer_config(
    tokenizer_config: object,
    special_tokens_map: object,
    vocab: Vocab | None,
) -> SpecialIds:
    """Resolve BOS/EOS/UNK/PAD from ``tokenizer_config.json`` then the special map.

    Each value may be an int, a ``{content, id}`` object, or a surface string;
    a surface resolves through the repo vocabulary. Anything missing or
    unresolvable stays ``None`` -- the rule layer must not guess an id.
    """
    config = tokenizer_config if isinstance(tokenizer_config, dict) else {}
    specials = special_tokens_map if isinstance(special_tokens_map, dict) else {}

    def pick(name: str) -> int | None:
        for source in (config, specials):
            value = source.get(name)
            if value is None:
                continue
            resolved = _resolve_special_value(value, vocab)
            if resolved is not None:
                return resolved
        return None

    eos = pick("eos_token")
    return SpecialIds(
        bos=pick("bos_token"),
        eos=eos,
        unk=pick("unk_token"),
        pad=pick("pad_token"),
        add_bos=config.get("add_bos_token") is True,
        add_eos=config.get("add_eos_token") is True,
        source="tokenizer_config.json" if config else "special_tokens_map.json",
        eos_ids=frozenset({eos}) if eos is not None else frozenset(),
    )


def resolve_vocab(model, vocab: Vocab | None = None) -> Vocab | None:
    """The effective vocabulary: an injected one, else the model's own.

    Rules accept an optional ``vocab`` so a future non-GGUF source can be
    threaded through unchanged. Existing callers pass nothing. Test doubles that
    expose only ``metadata`` still resolve through the GGUF constructor.
    """
    if vocab is not None:
        return vocab
    found = getattr(model, "vocab", None)
    if found is not None:
        return found
    metadata = getattr(model, "metadata", None)
    return vocab_from_metadata(metadata) if isinstance(metadata, dict) else None
