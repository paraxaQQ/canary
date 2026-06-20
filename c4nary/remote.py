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
import tempfile

from .parser import GGUFModel, parse_gguf

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
    requests = _requests()
    try:
        r = requests.get(f"{HF_BASE}/api/models/{repo}", timeout=30)
    except Exception as exc:  # noqa: BLE001
        raise RemoteError(f"could not reach Hugging Face: {exc}") from exc
    if r.status_code != 200:
        raise RemoteError(f"Hugging Face API returned {r.status_code} for {repo!r}")
    return [s.get("rfilename", "") for s in r.json().get("siblings", [])]


def _fetch_capped(url: str, n_bytes: int) -> bytes:
    """Fetch at most ``n_bytes`` from ``url`` (Range + streamed hard cap)."""

    requests = _requests()
    try:
        r = requests.get(url, headers={"Range": f"bytes=0-{n_bytes - 1}"},
                         stream=True, timeout=90, allow_redirects=True)
    except Exception as exc:  # noqa: BLE001
        raise RemoteError(f"network error: {exc}") from exc
    if r.status_code not in (200, 206):
        r.close()
        raise RemoteError(f"HTTP {r.status_code} fetching header")
    buf = bytearray()
    try:
        for chunk in r.iter_content(1 << 20):
            buf += chunk
            if len(buf) >= n_bytes:  # stop even if the server ignored Range
                break
    finally:
        r.close()
    return bytes(buf[:n_bytes])


def fetch_remote_model(target: str, filename: str | None = None) -> tuple[GGUFModel, str]:
    """Fetch and parse a remote model's header. Returns ``(model, display)``.

    The returned model has a truncated ``file_size`` (header only), so callers
    must not run structural (STR*) or whole-file hashing checks on it.
    """

    url, display = resolve_target(target, filename)
    last = "unknown"
    for mb in _FETCH_STAGES_MB:
        data = _fetch_capped(url, mb * 1024 * 1024)
        tf = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        try:
            tf.write(data)
            tf.close()
            return parse_gguf(tf.name), display
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            if "exceeds" in last or "end of file" in last:
                continue  # header bigger than this chunk -> fetch more
            raise RemoteError(f"could not parse header: {last}") from exc
        finally:
            os.unlink(tf.name)
    raise RemoteError(
        f"header did not fit in {_FETCH_STAGES_MB[-1]}MB (last error: {last})")
