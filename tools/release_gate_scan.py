#!/usr/bin/env python3
"""Full-corpus release gate for changed c4nary bundle and template rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

SOURCE = Path(os.environ.get("C4NARY_SOURCE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SOURCE))

from c4nary.bundle import bundle_findings
from c4nary.parser import GGUFModel, MetaArray
from c4nary.remote import fetch_remote_model, pick_gguf
from c4nary.rules.template import analyze_template

HF_API = "https://huggingface.co/api/models"
TARGET_RULES = frozenset({"TPL021", "TPL027", "CFG003", "CFG004", "CFG005", "NRM003"})
BUNDLE_FILES = frozenset({
    "generation_config.json",
    "config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
})
FORCED_KEYS = frozenset({
    "forced_decoder_ids",
    "forced_bos_token_id",
    "forced_eos_token_id",
})
LOG_LOCK = threading.Lock()


class RequestRateLimiter:
    def __init__(self, rate_per_second: float) -> None:
        self._lock = threading.Lock()
        self.configure(rate_per_second)

    def configure(self, rate_per_second: float) -> None:
        if rate_per_second <= 0:
            raise ValueError("request rate must be positive")
        with getattr(self, "_lock", threading.Lock()):
            self._interval = 1.0 / rate_per_second
            self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_at)
            self._next_at = slot + self._interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


RESOLVER_LIMITER = RequestRateLimiter(14.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(root.joinpath("c4nary").rglob("*.py"))
    paths.extend([root / "pyproject.toml", Path(__file__).resolve()])
    for path in paths:
        relative = path.relative_to(root) if path.is_relative_to(root) else path.name
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def logger(log_path: Path) -> Callable[[str], None]:
    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with LOG_LOCK:
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    return log


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def rate_limit_delay(headers: dict[str, str]) -> float:
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    match = re.search(r"(?:^|;)t=(\d+)(?:;|$)", headers.get("RateLimit", ""))
    return float(match.group(1)) + 1.0 if match else 60.0


def fetch_gate_text(repo: str, filename: str, max_bytes: int) -> str | None:
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    network_failures = 0
    while True:
        RESOLVER_LIMITER.wait()
        try:
            response = requests.get(
                url,
                headers={
                    "Range": f"bytes=0-{max_bytes}",
                    **auth_headers(os.environ.get("HF_TOKEN", "")),
                },
                stream=True,
                timeout=30,
                allow_redirects=True,
            )
        except requests.RequestException:
            network_failures += 1
            if network_failures >= 5:
                raise
            time.sleep(2 ** network_failures)
            continue

        if response.status_code == 429:
            delay = rate_limit_delay(dict(response.headers))
            response.close()
            time.sleep(delay)
            continue
        if response.status_code == 404:
            response.close()
            return None
        if response.status_code not in (200, 206):
            status = response.status_code
            response.close()
            raise RuntimeError(f"HTTP {status} fetching {filename}")

        encoding = response.encoding or "utf-8"
        data = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) > max_bytes:
                    return None
        finally:
            response.close()
        return bytes(data).decode(encoding, errors="replace")


def fetch_inventory_page(
    token: str,
    url: str | None = None,
    limit: int = 1000,
) -> tuple[list[dict], str | None]:
    params = [
        ("filter", "gguf"),
        ("expand[]", "gguf"),
        ("expand[]", "siblings"),
        ("limit", str(limit)),
    ]
    for attempt in range(5):
        try:
            response = requests.get(
                url or HF_API,
                params=None if url else params,
                headers=auth_headers(token),
                timeout=60,
            )
            if response.status_code == 429:
                time.sleep(int(response.headers.get("Retry-After", 30)))
                continue
            response.raise_for_status()
            data = response.json()
            page = data if isinstance(data, list) else []
            next_url = response.links.get("next", {}).get("url")
            return page, next_url if isinstance(next_url, str) else None
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(10)
    return [], None


def inline_template(model: dict) -> str | None:
    gguf = model.get("gguf")
    if not isinstance(gguf, dict):
        return None
    template = gguf.get("chat_template")
    return template if isinstance(template, str) else None


def build_inventory(path: Path, token: str, log: Callable[[str], None]) -> list[dict]:
    inventory: list[dict] = []
    next_url: str | None = None
    visited_urls: set[str] = set()
    while True:
        page, following_url = fetch_inventory_page(token, next_url)
        if not page:
            break
        for model in page:
            repo = model.get("id")
            if not isinstance(repo, str):
                continue
            siblings = model.get("siblings")
            files = []
            if isinstance(siblings, list):
                files = [
                    item["rfilename"]
                    for item in siblings
                    if isinstance(item, dict) and isinstance(item.get("rfilename"), str)
                ]
            inventory.append({
                "repo": repo,
                "bundle_files": sorted(BUNDLE_FILES.intersection(files)),
                "gguf_files": sorted(
                    name for name in files
                    if name.lower().endswith(".gguf") and "mmproj" not in name.lower()
                ),
                "chat_template": inline_template(model),
            })
        log(f"inventory repos={len(inventory)}")
        if following_url is None:
            break
        if following_url in visited_urls:
            raise RuntimeError("inventory cursor loop detected")
        visited_urls.add(following_url)
        next_url = following_url

    repos = [item["repo"] for item in inventory]
    if len(repos) != len(set(repos)):
        raise RuntimeError("inventory contains duplicate repository ids")
    atomic_json(path, inventory)
    log(f"inventory complete repos={len(inventory)} sha256={sha256_file(path)}")
    return inventory


def parse_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def tokenizer_surfaces(data: dict | None) -> tuple[str, ...] | None:
    if not isinstance(data, dict):
        return None
    model = data.get("model")
    if not isinstance(model, dict):
        return None
    vocab = model.get("vocab")
    by_id: dict[int, str] = {}
    if isinstance(vocab, dict):
        for token, token_id in vocab.items():
            if (isinstance(token, str) and isinstance(token_id, int)
                    and not isinstance(token_id, bool) and token_id >= 0):
                by_id[token_id] = token
    elif isinstance(vocab, list):
        for token_id, item in enumerate(vocab):
            token = item[0] if isinstance(item, list) and item else item
            if isinstance(token, str):
                by_id[token_id] = token

    added = data.get("added_tokens")
    if isinstance(added, list):
        for item in added:
            if not isinstance(item, dict):
                continue
            token_id = item.get("id")
            content = item.get("content")
            if (isinstance(token_id, int) and not isinstance(token_id, bool)
                    and token_id >= 0 and isinstance(content, str)):
                by_id[token_id] = content

    if not by_id:
        return None
    highest = max(by_id)
    if highest > 2_000_000:
        return None
    surfaces = [""] * (highest + 1)
    for token_id, token in by_id.items():
        surfaces[token_id] = token
    return tuple(surfaces)


def empty_model(repo: str, surfaces: tuple[str, ...] | None = None) -> GGUFModel:
    metadata: dict[str, Any] = {}
    metadata_types: dict[str, str] = {}
    if surfaces is not None:
        metadata["tokenizer.ggml.tokens"] = MetaArray(
            elem_type="string",
            length=len(surfaces),
            preview=surfaces,
            truncated=False,
        )
        metadata_types["tokenizer.ggml.tokens"] = "array[string]"
    return GGUFModel(
        path=repo,
        version=3,
        tensor_count=0,
        metadata=metadata,
        metadata_types=metadata_types,
        tensors=(),
    )


def has_forced_config(cache: dict[str, str | None]) -> bool:
    for name in ("generation_config.json", "config.json"):
        config = parse_json(cache.get(name))
        if config and FORCED_KEYS.intersection(config):
            return True
    return False


def scan_repo(item: dict) -> dict:
    repo = item["repo"]
    available = set(item.get("bundle_files", []))
    cache: dict[str, str | None] = {}
    errors: list[str] = []

    def read_text(name: str, max_bytes: int = 1 << 20) -> str | None:
        if name not in available:
            return None
        if name not in cache:
            try:
                cache[name] = fetch_gate_text(repo, name, max_bytes)
            except Exception as exc:  # noqa: BLE001
                cache[name] = None
                errors.append(f"{name}: {exc}")
            if cache[name] is None and not any(e.startswith(f"{name}:") for e in errors):
                errors.append(f"{name}: present in inventory but unreadable or over cap")
        return cache[name]

    for name in ("generation_config.json", "config.json", "tokenizer.json"):
        read_text(name, 48 << 20 if name == "tokenizer.json" else 1 << 20)

    tokenizer_data = parse_json(cache.get("tokenizer.json"))
    surfaces = tokenizer_surfaces(tokenizer_data)
    model = empty_model(repo, surfaces)
    header_fallback = False
    if has_forced_config(cache) and surfaces is None:
        chosen = pick_gguf(item.get("gguf_files", []))
        if chosen:
            try:
                RESOLVER_LIMITER.wait()
                model, _ = fetch_remote_model(
                    repo,
                    chosen,
                    materialize={"tokenizer.ggml.tokens"},
                    stages_mb=(16, 48, 120),
                )
                header_fallback = True
            except Exception as exc:  # noqa: BLE001
                errors.append(f"CFG003 header fallback {chosen}: {exc}")
        else:
            errors.append("CFG003 requires token surfaces but no tokenizer.json vocab or GGUF file")

    findings = []
    template = item.get("chat_template")
    if isinstance(template, str):
        findings.extend(analyze_template(template))
    try:
        findings.extend(bundle_findings(model, read_text))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"bundle analysis: {exc}")

    hits = [
        {
            "rule": finding.rule_id,
            "severity": finding.severity,
            "title": finding.title,
            "detail": finding.detail,
            "location": finding.location,
        }
        for finding in findings
        if finding.rule_id in TARGET_RULES
    ]
    return {
        "repo": repo,
        "hits": hits,
        "errors": errors,
        "header_fallback": header_fallback,
    }


def initial_state(
    inventory_path: Path,
    inventory: list[dict],
    workers: int,
    requests_per_second: float,
) -> dict:
    return {
        "schema_version": 1,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_at": None,
        "complete": False,
        "source_sha256": source_sha256(SOURCE),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "inventory_repos": len(inventory),
        "workers": workers,
        "requests_per_second": requests_per_second,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "source": str(SOURCE),
        },
        "processed_repos": [],
        "successful_repos": 0,
        "repos_with_errors": 0,
        "header_fallbacks": 0,
        "hits": 0,
        "by_rule": {rule: 0 for rule in sorted(TARGET_RULES)},
        "hit_models": [],
        "error_models": [],
    }


def save_state(path: Path, state: dict, processed: set[str]) -> None:
    state["processed_repos"] = sorted(processed)
    state["updated_at"] = utc_now()
    atomic_json(path, state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--requests-per-second", type=float, default=14.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--token-file", type=Path, default=Path("/root/.hf_token"))
    args = parser.parse_args()
    RESOLVER_LIMITER.configure(args.requests_per_second)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_dir / "inventory.json"
    state_path = args.output_dir / "results.json"
    log = logger(args.output_dir / "scan.log")

    token = os.environ.get("HF_TOKEN", "")
    if not token and args.token_file.is_file():
        token = args.token_file.read_text(encoding="utf-8").strip()
    if token:
        os.environ["HF_TOKEN"] = token

    if inventory_path.is_file():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        log(f"using inventory repos={len(inventory)} sha256={sha256_file(inventory_path)}")
    else:
        inventory = build_inventory(inventory_path, token, log)
    if args.limit:
        inventory = inventory[:args.limit]

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("source_sha256") != source_sha256(SOURCE):
            raise RuntimeError("source changed since this scan started")
        if state.get("inventory_sha256") != sha256_file(inventory_path):
            raise RuntimeError("inventory changed since this scan started")
    else:
        state = initial_state(
            inventory_path,
            inventory,
            args.workers,
            args.requests_per_second,
        )
    processed = set(state.get("processed_repos", []))
    remaining = [item for item in inventory if item["repo"] not in processed]
    log(f"scan start inventory={len(inventory)} processed={len(processed)} remaining={len(remaining)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_repo, item): item["repo"] for item in remaining}
        for future in as_completed(futures):
            result = future.result()
            repo = result["repo"]
            processed.add(repo)
            if result["errors"]:
                state["repos_with_errors"] += 1
                state["error_models"].append({"repo": repo, "errors": result["errors"]})
            else:
                state["successful_repos"] += 1
            if result["header_fallback"]:
                state["header_fallbacks"] += 1
            if result["hits"]:
                state["hits"] += len(result["hits"])
                state["hit_models"].append({"repo": repo, "hits": result["hits"]})
                for hit in result["hits"]:
                    rule = hit["rule"]
                    state["by_rule"][rule] = state["by_rule"].get(rule, 0) + 1
            if len(processed) % 100 == 0:
                save_state(state_path, state, processed)
                log(
                    f"checkpoint processed={len(processed)}/{len(inventory)} "
                    f"success={state['successful_repos']} errors={state['repos_with_errors']} "
                    f"hits={state['hits']} by_rule={state['by_rule']}"
                )

    state["complete"] = len(processed) == len(inventory)
    state["completed_at"] = utc_now() if state["complete"] else None
    save_state(state_path, state, processed)
    log(
        f"FINAL complete={state['complete']} processed={len(processed)}/{len(inventory)} "
        f"success={state['successful_repos']} errors={state['repos_with_errors']} "
        f"hits={state['hits']} by_rule={state['by_rule']}"
    )
    return 0 if state["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
