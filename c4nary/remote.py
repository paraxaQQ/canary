"""Optional remote header fetch -- the ONLY network-touching component.

The core parser and analysis engine never touch the network (the air-gapped
guarantee). This module is imported lazily, only when ``scan --remote`` is used.
It range-fetches a model's GGUF **header** (metadata + chat template + tensor
map) -- never the weights -- so the template / metadata / tokenizer rules can
audit a model without downloading it. Structural (STR*) and whole-file integrity
checks need the complete file and are skipped for remote scans.

``requests`` is an optional dependency (``pip install c4nary[remote]``); it is
imported lazily so the offline core has no network dependency at all.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from .bundle import ReadResult, _safe_bundle_name
from .parser import GGUFModel, parse_gguf_bytes

HF_BASE = "https://huggingface.co"
# Escalating header sizes: most model headers fit in 16MB; big-vocab models need
# more. We stream and stop at the cap even if the server ignores the Range header.
_FETCH_STAGES_MB = (16, 48, 120)


class RemoteError(Exception):
    """A remote fetch/resolve problem (network, 404, unparseable header)."""


def _requests():
    try:
        import requests  # noqa: PLC0415 - optional, lazily imported
    except ImportError as exc:  # pragma: no cover
        raise RemoteError(
            "remote scanning needs the 'requests' package "
            "(pip install c4nary[remote])"
        ) from exc
    return requests


def _auth_headers(url: str) -> dict[str, str]:
    """Bearer auth from ``HF_TOKEN`` if set -- authenticated fetches get a much higher
    rate limit (bulk header scans throttle hard unauthenticated). Opt-in via env; when
    unset the headers are empty and behavior is unchanged.

    Scheme is part of the scope, not just the host: an ``http://huggingface.co/...`` URL
    would otherwise put the bearer on the wire in cleartext."""
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != "huggingface.co":
        return {}
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


_SESSION = None


def _session():
    """A shared, retrying session. HF rate-limits per-IP (429); a bulk header scan MUST
    back off and retry rather than treat a throttle as a permanent failure. Exponential
    backoff honoring ``Retry-After``. Thread-safe (share across a ThreadPoolExecutor);
    NOT fork-safe -- use threads, not processes, for concurrent scans."""
    global _SESSION
    if _SESSION is None:
        requests = _requests()
        from requests.adapters import HTTPAdapter  # noqa: PLC0415
        try:
            from urllib3.util.retry import Retry  # noqa: PLC0415
        except ImportError:  # pragma: no cover - very old urllib3
            from requests.packages.urllib3.util.retry import Retry  # noqa: PLC0415
        # The per-request timeouts bound socket activity, not sleeping, so an honoured
        # Retry-After is otherwise unbounded stall. urllib3 1.26 lacks both kwargs.
        bounds = {"backoff_max": 30, "retry_after_max": 60}
        common = dict(total=6, connect=3, read=3, backoff_factor=1.5,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"GET"}),
                      respect_retry_after_header=True, raise_on_status=False)
        try:
            retry = Retry(**common, **bounds)
        except TypeError:  # pragma: no cover - urllib3 < 2
            retry = Retry(**common)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
        s = requests.Session()
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _SESSION = s
    return _SESSION


def pick_gguf(files: list[str]) -> str | None:
    """Choose one .gguf from a repo's file list (smallest single-file quant)."""

    ggufs = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
    if not ggufs:
        return None
    singles = [f for f in ggufs if "-of-" not in f]
    if singles:
        return sorted(singles, key=len)[0]
    firsts = [f for f in ggufs if "00001-of" in f]
    return firsts[0] if firsts else ggufs[0]


def resolve_target(target: str, filename: str | None = None, *, list_files=None):
    """Resolve a target to ``(url, display)``.

    ``target`` may be a full ``http(s)://`` URL or a Hugging Face repo id
    (optionally ``hf://org/name``). ``list_files`` is injectable for testing;
    by default it queries the HF API.
    """

    if target.startswith(("http://", "https://")):
        return target, target
    repo = target[len("hf://"):] if target.startswith("hf://") else target
    if filename:
        return f"{HF_BASE}/{repo}/resolve/main/{filename}", f"{repo}/{filename}"
    files = (list_files or _hf_list_files)(repo)
    chosen = pick_gguf(files)
    if not chosen:
        raise RemoteError(f"no .gguf file found in repo {repo!r}")
    return f"{HF_BASE}/{repo}/resolve/main/{chosen}", f"{repo}/{chosen}"


def _hf_list_files(repo: str) -> list[str]:
    url = f"{HF_BASE}/api/models/{repo}"
    try:
        r = _session().get(url, timeout=30, headers=_auth_headers(url))
    except Exception as exc:  # noqa: BLE001
        raise RemoteError(f"could not reach Hugging Face: {exc}") from exc
    if r.status_code != 200:
        raise RemoteError(f"Hugging Face API returned {r.status_code} for {repo!r}")
    # `scan hf:// --remote` yields an empty repo id, which hits the model-LIST endpoint
    # and returns a JSON array. Indexing straight in raises AttributeError, which is not
    # a RemoteError and escapes the CLI handler as a traceback.
    payload = r.json()
    siblings = payload.get("siblings") if isinstance(payload, dict) else None
    if not isinstance(siblings, list):
        raise RemoteError(f"unexpected Hugging Face API response for repo {repo!r}")
    return [s.get("rfilename", "") for s in siblings if isinstance(s, dict)]


def _read_capped_response(response, n_bytes: int) -> bytes:
    buf = bytearray()
    chunk_size = max(1, min(1 << 20, n_bytes))
    try:
        for chunk in response.iter_content(chunk_size):
            remaining = n_bytes - len(buf)
            if remaining <= 0:
                break
            buf += chunk[:remaining]
            if len(buf) >= n_bytes:
                break
    finally:
        response.close()
    return bytes(buf)


def _fetch_capped(url: str, n_bytes: int) -> bytes:
    """Fetch at most ``n_bytes`` from ``url`` (Range + streamed hard cap)."""

    try:
        r = _session().get(url, headers={"Range": f"bytes=0-{n_bytes - 1}",
                                         **_auth_headers(url)},
                           stream=True, timeout=90, allow_redirects=True)
    except Exception as exc:  # noqa: BLE001
        raise RemoteError(f"network error: {exc}") from exc
    if r.status_code not in (200, 206):
        r.close()
        raise RemoteError(f"HTTP {r.status_code} fetching header")
    return _read_capped_response(r, n_bytes)


def fetch_remote_model(target: str, filename: str | None = None,
                       materialize: frozenset[str] | set[str] | None = None,
                       stages_mb: tuple[int, ...] = _FETCH_STAGES_MB,
                       ) -> tuple[GGUFModel, str]:
    """Fetch and parse a remote model's header. Returns ``(model, display)``.

    The returned model has a truncated ``file_size`` (header only), so callers
    must not run structural (STR*) or whole-file hashing checks on it.

    ``materialize`` is passed through to :func:`parse_gguf` -- opt-in full arrays
    for the tokenizer-seam checks. A fully-materialized vocab can push the header
    past the first fetch stage; the escalating stages handle that.

    ``stages_mb`` is the escalating fetch-size ladder. A template-only bulk scan can
    pass a smaller first stage (most headers are tiny; the ladder still escalates for
    big-vocab models), cutting download volume dramatically.
    """

    url, display = resolve_target(target, filename)
    last = "unknown"
    for mb in stages_mb:
        data = _fetch_capped(url, mb * 1024 * 1024)
        try:
            return parse_gguf_bytes(data, materialize=materialize, label=display), display
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            if ("exceeds" in last or "end of file" in last or "cannot fit" in last):
                continue  # header (metadata / a big vocab array) bigger than this chunk -> fetch more
            raise RemoteError(f"could not parse header: {last}") from exc
    raise RemoteError(
        f"header did not fit in {_FETCH_STAGES_MB[-1]}MB (last error: {last})")


def fetch_repo_text(
    target: str,
    filename: str,
    *,
    max_bytes: int = 1 << 20,
    python_only: bool = False,
) -> ReadResult | None:
    """Fetch a small repo *bundle* file (generation_config.json, config.json, README.md)
    for an opt-in bundle scan. Returns its text plus a truncation bit, or ``None`` if
    absent/inaccessible or if ``target`` is a direct file URL (no repo bundle).
    Capped -- these are small config/doc/Python files, never weights."""
    if python_only:
        filename = _safe_bundle_name(filename)
    if target.startswith(("http://", "https://")):
        return None
    repo = target[len("hf://"):] if target.startswith("hf://") else target
    url = f"{HF_BASE}/{repo}/resolve/main/{filename}"
    try:
        r = _session().get(
            url,
            headers={"Range": f"bytes=0-{max_bytes}", **_auth_headers(url)},
            stream=True,
            timeout=30,
            allow_redirects=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RemoteError(f"network error fetching {filename}: {exc}") from exc
    if r.status_code not in (200, 206):
        r.close()
        return None
    data = _read_capped_response(r, max_bytes + 1)
    # utf-8, never the server's Content-Type charset: that would let the server choose
    # what text the rules see, and an unknown codec raises LookupError. Also keeps the
    # remote decode byte-identical to the local sibling reader.
    return ReadResult(
        data[:max_bytes].decode("utf-8", errors="replace"),
        truncated=len(data) > max_bytes,
    )


def _safe_safetensors_name(filename: str) -> str:
    """A top-level, non-hidden ``*.safetensors`` filename, or ``ValueError``."""
    if (
        not isinstance(filename, str)
        or not filename
        or not filename.endswith(".safetensors")
        or len(filename) <= len(".safetensors")
        or filename.startswith(".")
        or "/" in filename
        or "\\" in filename
        or ".." in filename
        or "\x00" in filename
        or ":" in filename
        or "?" in filename
        or "#" in filename
        or "%" in filename
    ):
        raise ValueError(f"unsafe safetensors filename: {filename!r}")
    return filename


def fetch_remote_range(
    target: str,
    filename: str,
    start: int,
    length: int,
) -> bytes | None:
    """Fetch ``length`` bytes at ``start`` from one repository file.

    Strictly a range request: this is how a safetensors header is read without
    touching the tensor payload. ``start == 0`` may accept a ``200`` (a server
    that ignores ranges), but a nonzero start **requires** ``206`` -- otherwise
    the server would send the whole file and the range read is refused before a
    single body byte is consumed. The body is streamed with a hard cap of
    ``length + 1`` bytes, so an ignored range on the prefix cannot turn into a
    weights download.

    Returns the bytes, or ``None`` when the file/range is unavailable. Network
    failures raise :class:`RemoteError` as everywhere else in this module.
    """
    filename = _safe_safetensors_name(filename)
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(length, int)
        or isinstance(length, bool)
        or start < 0
        or length <= 0
    ):
        raise ValueError(f"invalid safetensors range: start={start!r} length={length!r}")
    if length > (32 << 20):
        raise ValueError(
            f"safetensors range length {length} exceeds the 32 MiB header cap"
        )
    if target.startswith(("http://", "https://")):
        return None
    repo = target[len("hf://"):] if target.startswith("hf://") else target
    url = f"{HF_BASE}/{repo}/resolve/main/{filename}"
    try:
        response = _session().get(
            url,
            headers={
                "Range": f"bytes={start}-{start + length - 1}",
                # Byte offsets only make sense over the identity-coded body.
                "Accept-Encoding": "identity",
                **_auth_headers(url),
            },
            stream=True,
            timeout=30,
            allow_redirects=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RemoteError(f"network error fetching {filename}: {exc}") from exc

    if response.status_code == 206 or (start == 0 and response.status_code == 200):
        # Stream with a hard ``length + 1`` cap, then keep only the requested
        # bytes: a 200 for a prefix can send the whole file, and a header JSON
        # parse must never see a tensor-data byte.
        return _read_capped_response(response, length + 1)[:length]
    # A nonzero start with 200 means the server ignored the Range header. Do not
    # read the body: that would be a full-file (weights) fetch.
    response.close()
    return None
