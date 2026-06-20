"""Validate c4nary against a large corpus of real GGUF chat templates.

Range-fetches only each model's header (metadata + chat template + tensor map)
from Hugging Face -- never the weights -- and runs the template / metadata /
tokenizer rules. Measures the false-positive rate of the heuristic behavioral
rules on genuine production templates and surfaces real FAIL-level hits.

Parallel (I/O-bound): a thread pool fetches+scans many models at once, and the
gguf filename comes from the model listing's siblings so there is no per-repo
API call. Structural (STR*) and hashing checks need the whole file and are
skipped in header-only mode.

Usage (on the pod):  LIMIT=5000 WORKERS=48 python tools/validate_corpus.py
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from huggingface_hub import HfApi

from c4nary.parser import GGUFParseError, parse_gguf
from c4nary.report import FAIL, WARN, summarize
from c4nary.rules.metadata import analyze_metadata
from c4nary.rules.template import analyze_template
from c4nary.rules.tokenizer import analyze_tokenizer

LIMIT = int(os.environ.get("LIMIT", "5000"))
WORKERS = int(os.environ.get("WORKERS", "48"))
OUT = os.environ.get("OUT", "/workspace/corpus5k.json")
FETCH_STAGES_MB = (16, 48)

api = HfApi()
_session = requests.Session()
_session.mount("https://", requests.adapters.HTTPAdapter(
    pool_connections=WORKERS, pool_maxsize=WORKERS))

_lock = threading.Lock()
_done = 0
_fail = 0


def pick_gguf(files: list[str]) -> str | None:
    ggufs = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
    if not ggufs:
        return None
    singles = [f for f in ggufs if "-of-" not in f]
    if singles:
        return sorted(singles, key=len)[0]
    firsts = [f for f in ggufs if "00001-of" in f]
    return firsts[0] if firsts else ggufs[0]


def fetch_header(repo: str, filename: str):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    last = "unknown"
    for mb in FETCH_STAGES_MB:
        try:
            r = _session.get(url, headers={"Range": f"bytes=0-{mb*1024*1024-1}"},
                             timeout=60, allow_redirects=True)
        except requests.RequestException as exc:
            last = f"net:{str(exc)[:60]}"
            continue
        if r.status_code not in (200, 206):
            last = f"http {r.status_code}"
            break
        tf = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        try:
            tf.write(r.content)
            tf.close()
            return parse_gguf(tf.name)
        except GGUFParseError as exc:
            last = str(exc)
            if "exceeds" in last or "end of file" in last:
                continue
            break
        finally:
            os.unlink(tf.name)
    raise GGUFParseError(last)


def scan_one(repo: str, filename: str) -> dict:
    model = fetch_header(repo, filename)
    findings = (analyze_template(model.chat_template)
                + analyze_metadata(model)
                + analyze_tokenizer(model))
    counts = summarize(findings)
    return {
        "repo": repo, "file": filename, "arch": model.architecture,
        "has_template": model.chat_template is not None,
        "fail": counts[FAIL], "warn": counts[WARN],
        "rules": sorted({f.rule_id for f in findings}),
        "fail_detail": [{"rule": f.rule_id, "loc": f.location, "detail": f.detail}
                        for f in findings if f.severity == FAIL],
        "behavioral": sorted({f.rule_id for f in findings
                              if f.rule_id.startswith("TPL02")}),
    }


def work(item):
    global _done, _fail
    repo, fn = item
    try:
        out = ("ok", scan_one(repo, fn))
    except Exception as exc:  # noqa: BLE001 - survey, keep going
        out = ("skip", {"repo": repo, "why": str(exc)[:160]})
    with _lock:
        _done += 1
        if out[0] == "ok" and out[1]["fail"]:
            _fail += 1
        if _done % 100 == 0:
            print(f"  ...{_done} scanned, {_fail} with FAIL", flush=True)
    return out


def list_models():
    try:
        models = list(api.list_models(filter="gguf", sort="downloads",
                                      limit=LIMIT, expand=["siblings"]))
        if models and getattr(models[0], "siblings", None):
            return models
    except (TypeError, ValueError):
        pass
    print("  (falling back to full=True listing)", flush=True)
    return list(api.list_models(filter="gguf", sort="downloads",
                                limit=LIMIT, full=True))


def main() -> None:
    print(f"listing up to {LIMIT} trending GGUF models...", flush=True)
    models = list_models()
    work_items = []
    for m in models:
        sibs = getattr(m, "siblings", None) or []
        fn = pick_gguf([s.rfilename for s in sibs])
        if fn:
            work_items.append((m.id, fn))
    print(f"{len(models)} models, {len(work_items)} with a gguf file; "
          f"scanning with {WORKERS} workers...", flush=True)

    results, skips = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(work, it) for it in work_items]):
            kind, payload = fut.result()
            (results if kind == "ok" else skips).append(payload)

    rule_hist: dict[str, int] = {}
    for r in results:
        for rid in r["rules"]:
            rule_hist[rid] = rule_hist.get(rid, 0) + 1
    with_tpl = [r for r in results if r["has_template"]]
    fails = [r for r in results if r["fail"]]
    behavioral = [r for r in results if r["behavioral"]]
    summary = {
        "scanned": len(results), "with_template": len(with_tpl),
        "skipped": len(skips), "models_with_FAIL": len(fails),
        "models_with_behavioral_flag": len(behavioral),
        "behavioral_flag_rate_pct": round(100 * len(behavioral) / max(len(with_tpl), 1), 3),
        "architectures": len({r["arch"] for r in results if r["arch"]}),
        "rule_histogram": dict(sorted(rule_hist.items(), key=lambda kv: -kv[1])),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "fails": fails, "behavioral_hits": behavioral,
                   "results": results, "skips": skips}, fh, indent=2)

    print("\n================= SUMMARY =================")
    print(json.dumps(summary, indent=2))
    print(f"\nmodels with FAIL findings: {len(fails)}")
    for r in fails[:60]:
        print(f"  {r['repo']}/{r['file']} -> {[d['rule'] for d in r['fail_detail']]}")
    print(f"\nfull report: {OUT}")


if __name__ == "__main__":
    main()
