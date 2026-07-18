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
    unset the headers are empty and behavior is unchanged."""
    if urlsplit(url).hostname != "huggingface.co":
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
        retry = Retry(total=6, connect=3, read=3, backoff_factor=1.5,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"GET"}),
                      respect_retry_after_header=True, raise_on_status=False)
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
    return [s.get("rfilename", "") for s in r.json().get("siblings", [])]


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


def fetch_repo_text(target: str, filename: str, *, max_bytes: int = 1 << 20) -> str | None:
    """Fetch a small repo *bundle* file (generation_config.json, config.json, README.md)
    for an opt-in bundle scan. Returns its text, or ``None`` if absent/inaccessible/too
    large or if ``target`` is a direct file URL (no repo bundle). Capped -- these are
    small config/doc files, never weights."""
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
    encoding = r.encoding or "utf-8"
    data = _read_capped_response(r, max_bytes + 1)
    if len(data) > max_bytes:
        return None
    return data.decode(encoding, errors="replace")
