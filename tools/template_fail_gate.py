#!/usr/bin/env python3
"""Run every registered FAIL-severity template rule over a frozen GGUF corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests

SOURCE = Path(os.environ.get("C4NARY_SOURCE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(SOURCE))

from c4nary.parser import extract_gguf_chat_template_bytes
from c4nary.remote import pick_gguf
from c4nary.report import FAIL
from c4nary.rules.registry import all_rules
from c4nary.rules.template import analyze_template
from tools.release_gate_scan import (
    RequestRateLimiter,
    atomic_json,
    auth_headers,
    logger,
    rate_limit_delay,
    sha256_file,
    utc_now,
)

TEMPLATE_FAIL_RULES = frozenset(
    rule.rule_id
    for rule in all_rules()
    if rule.severity == FAIL and rule.rule_id.startswith("TPL")
)
INCOMPLETE_HEADER_ERRORS = ("exceeds", "end of file", "cannot fit")
RESOLVER_LIMITER = RequestRateLimiter(14.0)


def source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(root.joinpath("c4nary").rglob("*.py"))
    paths.extend([
        root / "pyproject.toml",
        root / "tools" / "release_gate_scan.py",
        Path(__file__).resolve(),
    ])
    for path in paths:
        relative = path.relative_to(root) if path.is_relative_to(root) else path.name
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fetch_prefixes(
    url: str,
    stages: tuple[int, ...],
    token: str,
) -> Iterator[bytes]:
    """Yield staged prefixes from one bounded range request.

    A repo may need a large metadata prefix because its tokenizer vocabulary
    precedes the chat template. Re-fetching 0..N for every stage wastes both
    resolver quota and bandwidth, so one response is consumed progressively.
    """
    network_failures = 0
    rate_limits = 0
    while True:
        RESOLVER_LIMITER.wait()
        try:
            response = requests.get(
                url,
                headers={
                    "Range": f"bytes=0-{stages[-1] - 1}",
                    **auth_headers(token),
                },
                stream=True,
                timeout=90,
                allow_redirects=True,
            )
        except requests.RequestException:
            network_failures += 1
            if network_failures >= 5:
                raise
            time.sleep(2 ** network_failures)
            continue

        if response.status_code == 429:
            rate_limits += 1
            delay = rate_limit_delay(dict(response.headers))
            response.close()
            if rate_limits >= 12:
                raise RuntimeError("HTTP 429 persisted across 12 retries")
            time.sleep(delay)
            continue
        if response.status_code not in (200, 206):
            status = response.status_code
            response.close()
            raise RuntimeError(f"HTTP {status} fetching GGUF header")

        data = bytearray()
        stage_index = 0
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                remaining = stages[-1] - len(data)
                data.extend(chunk[:remaining])
                while stage_index < len(stages) and len(data) >= stages[stage_index]:
                    yield bytes(data[:stages[stage_index]])
                    stage_index += 1
                if len(data) == stages[-1]:
                    break
        except requests.RequestException:
            network_failures += 1
            if network_failures >= 5:
                raise
            time.sleep(2 ** network_failures)
            continue
        finally:
            response.close()
        last_yielded_size = stages[stage_index - 1] if stage_index else -1
        if len(data) != last_yielded_size:
            yield bytes(data)
        return


def fetch_header_template(
    repo: str,
    filename: str,
    stages_mb: tuple[int, ...],
    token: str,
) -> str | None:
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    last = "unknown"
    stages = tuple(mb << 20 for mb in stages_mb)
    for data in _fetch_prefixes(url, stages, token):
        try:
            return extract_gguf_chat_template_bytes(data, label=f"{repo}/{filename}")
        except Exception as exc:  # noqa: BLE001 - malformed corpus input is recorded
            last = str(exc)
            if any(marker in last.lower() for marker in INCOMPLETE_HEADER_ERRORS):
                continue
            raise RuntimeError(f"could not parse GGUF header: {last}") from exc
    raise RuntimeError(
        f"GGUF header did not fit in {stages_mb[-1]}MB (last error: {last})"
    )


def finding_dict(finding: Any) -> dict[str, Any]:
    return {
        "rule": finding.rule_id,
        "severity": finding.severity,
        "title": finding.title,
        "detail": finding.detail,
        "location": finding.location,
    }


def scan_repo(item: dict[str, Any], stages_mb: tuple[int, ...], token: str) -> dict[str, Any]:
    repo = item["repo"]
    template = item.get("chat_template")
    source = "inline"
    filename = None

    if not isinstance(template, str):
        filename = pick_gguf(item.get("gguf_files", []))
        if filename is None:
            return {"repo": repo, "status": "no_gguf", "source": None, "hits": []}
        source = "header"
        try:
            template = fetch_header_template(repo, filename, stages_mb, token)
        except Exception as exc:  # noqa: BLE001 - exclusion must be retained, not abort corpus
            return {
                "repo": repo,
                "status": "error",
                "source": source,
                "file": filename,
                "error": str(exc),
                "hits": [],
            }
        if template is None:
            return {
                "repo": repo,
                "status": "no_template",
                "source": source,
                "file": filename,
                "hits": [],
            }

    hits = [
        finding_dict(finding)
        for finding in analyze_template(template)
        if finding.rule_id in TEMPLATE_FAIL_RULES and finding.severity == FAIL
    ]
    return {
        "repo": repo,
        "status": "analyzed",
        "source": source,
        "file": filename,
        "hits": hits,
    }


def scan_inline_repo(item: dict[str, Any]) -> dict[str, Any]:
    return scan_repo(item, (1,), "")


def init_header_worker() -> None:
    # Request starts are paced by the parent before submission. Child-local
    # retries retain their exponential/429 backoff without adding another
    # 1/rate delay to every successful request.
    RESOLVER_LIMITER.configure(1000.0)


def initial_state(
    inventory_path: Path,
    inventory: list[dict[str, Any]],
    workers: int,
    cpu_workers: int,
    requests_per_second: float,
    stages_mb: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gate": "all registered FAIL-severity template rules",
        "template_fail_rules": sorted(TEMPLATE_FAIL_RULES),
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_at": None,
        "complete": False,
        "source_sha256": source_sha256(SOURCE),
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "inventory_repos": len(inventory),
        "workers": workers,
        "cpu_workers": cpu_workers,
        "requests_per_second": requests_per_second,
        "stages_mb": list(stages_mb),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "source": str(SOURCE),
        },
        "processed_repos": [],
        "analyzed_templates": 0,
        "inline_templates": 0,
        "header_templates": 0,
        "repos_without_template": 0,
        "repos_without_gguf": 0,
        "repos_with_errors": 0,
        "hits": 0,
        "repos_with_hits": 0,
        "by_rule": {rule: 0 for rule in sorted(TEMPLATE_FAIL_RULES)},
        "hit_models": [],
        "exclusions": [],
    }


def save_state(path: Path, state: dict[str, Any], processed: set[str]) -> None:
    state["processed_repos"] = sorted(processed)
    state["updated_at"] = utc_now()
    atomic_json(path, state)


def record_result(state: dict[str, Any], result: dict[str, Any]) -> None:
    status = result["status"]
    source = result.get("source")
    if status == "analyzed":
        state["analyzed_templates"] += 1
        state[f"{source}_templates"] += 1
    elif status == "no_template":
        state["repos_without_template"] += 1
    elif status == "no_gguf":
        state["repos_without_gguf"] += 1
        state["exclusions"].append(result)
    elif status == "error":
        state["repos_with_errors"] += 1
        state["exclusions"].append(result)
    else:
        raise RuntimeError(f"unknown scan status {status!r}")

    hits = result["hits"]
    if hits:
        state["hits"] += len(hits)
        state["repos_with_hits"] += 1
        state["hit_models"].append({
            "repo": result["repo"],
            "source": source,
            "file": result.get("file"),
            "hits": hits,
        })
        for hit in hits:
            state["by_rule"][hit["rule"]] += 1


def parse_stages(value: str) -> tuple[int, ...]:
    stages = tuple(int(part) for part in value.split(","))
    if not stages or any(stage <= 0 for stage in stages) or tuple(sorted(set(stages))) != stages:
        raise argparse.ArgumentTypeError("stages must be unique ascending positive integers")
    return stages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--cpu-workers", type=int, default=max(1, min(os.cpu_count() or 1, 12)))
    parser.add_argument("--requests-per-second", type=float, default=14.0)
    parser.add_argument("--stages-mb", type=parse_stages, default=(1, 4, 16, 48, 120))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--token-file", type=Path, default=Path("/root/.hf_token"))
    args = parser.parse_args()
    RESOLVER_LIMITER.configure(args.requests_per_second)

    if not args.inventory.is_file():
        raise FileNotFoundError(args.inventory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "results.json"
    log = logger(args.output_dir / "scan.log")

    token = os.environ.get("HF_TOKEN", "")
    if not token and args.token_file.is_file():
        token = args.token_file.read_text(encoding="utf-8").strip()

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if args.limit:
        inventory = inventory[:args.limit]

    current_source_sha = source_sha256(SOURCE)
    current_inventory_sha = sha256_file(args.inventory)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("source_sha256") != current_source_sha:
            raise RuntimeError("source changed since this gate started")
        if state.get("inventory_sha256") != current_inventory_sha:
            raise RuntimeError("inventory changed since this gate started")
        if state.get("template_fail_rules") != sorted(TEMPLATE_FAIL_RULES):
            raise RuntimeError("registered template FAIL rules changed since this gate started")
    else:
        state = initial_state(
            args.inventory,
            inventory,
            args.workers,
            args.cpu_workers,
            args.requests_per_second,
            args.stages_mb,
        )

    processed = set(state.get("processed_repos", []))
    remaining = [item for item in inventory if item["repo"] not in processed]
    inline_remaining = [item for item in remaining if isinstance(item.get("chat_template"), str)]
    header_remaining = [item for item in remaining if not isinstance(item.get("chat_template"), str)]
    log(
        f"gate start rules={sorted(TEMPLATE_FAIL_RULES)} inventory={len(inventory)} "
        f"processed={len(processed)} remaining={len(remaining)} "
        f"inline={len(inline_remaining)} header={len(header_remaining)}"
    )

    def handle(result: dict[str, Any]) -> None:
        processed.add(result["repo"])
        record_result(state, result)
        if len(processed) % 100 == 0:
            save_state(state_path, state, processed)
            log(
                f"checkpoint processed={len(processed)}/{len(inventory)} "
                f"templates={state['analyzed_templates']} errors={state['repos_with_errors']} "
                f"hits={state['hits']} repos_with_hits={state['repos_with_hits']}"
            )

    if inline_remaining:
        log(f"inline phase start repos={len(inline_remaining)} workers={args.cpu_workers}")
        with ProcessPoolExecutor(max_workers=args.cpu_workers) as pool:
            for result in pool.map(scan_inline_repo, inline_remaining, chunksize=100):
                handle(result)
        log(f"inline phase complete repos={len(inline_remaining)}")

    if header_remaining:
        log(
            f"header phase start repos={len(header_remaining)} "
            f"processes={args.workers}"
        )
        remaining_iter = iter(header_remaining)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_header_worker,
        ) as pool:
            futures = set()

            def submit_one() -> bool:
                try:
                    item = next(remaining_iter)
                except StopIteration:
                    return False
                RESOLVER_LIMITER.wait()
                futures.add(pool.submit(scan_repo, item, args.stages_mb, token))
                return True

            for _ in range(args.workers):
                if not submit_one():
                    break
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    handle(future.result())
                for _ in done:
                    if not submit_one():
                        break
        log(f"header phase complete repos={len(header_remaining)}")

    state["complete"] = len(processed) == len(inventory)
    state["completed_at"] = utc_now() if state["complete"] else None
    state["hit_models"].sort(key=lambda item: item["repo"])
    state["exclusions"].sort(key=lambda item: item["repo"])
    save_state(state_path, state, processed)
    log(
        f"FINAL complete={state['complete']} processed={len(processed)}/{len(inventory)} "
        f"templates={state['analyzed_templates']} no_template={state['repos_without_template']} "
        f"errors={state['repos_with_errors']} hits={state['hits']} "
        f"repos_with_hits={state['repos_with_hits']} by_rule={state['by_rule']}"
    )
    return 0 if state["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
