"""Validate c4nary's TEMPLATE rules against thousands of real GGUF chat templates.

Hugging Face parses each GGUF header server-side and exposes the full
``chat_template`` via the model API (``expand=["gguf"]``) -- a small JSON call,
no weight or header download. So we can run the behavioral / SSTI template rules
across thousands of real production templates cheaply and measure the
false-positive rate at scale, plus surface any genuine FAIL-level hits.

(Metadata / tokenizer / structural rules need the full header and are validated
separately by validate_corpus.py on a smaller header-fetch sample.)

Usage (on the pod):  LIMIT=5000 WORKERS=24 python tools/validate_templates.py
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi

from c4nary.report import FAIL, WARN, summarize
from c4nary.rules.template import analyze_template

LIMIT = int(os.environ.get("LIMIT", "5000"))
WORKERS = int(os.environ.get("WORKERS", "24"))
OUT = os.environ.get("OUT", "/workspace/templates5k.json")

api = HfApi()
_lock = threading.Lock()
_done = 0


def list_items():
    """Return [(repo, gguf_dict_or_None)] and whether templates came inline."""
    try:
        models = list(api.list_models(filter="gguf", sort="downloads",
                                      limit=LIMIT, expand=["gguf"]))
        if models and getattr(models[0], "gguf", None):
            return [(m.id, m.gguf) for m in models], True
    except (TypeError, ValueError):
        pass
    ids = [m.id for m in api.list_models(filter="gguf", sort="downloads", limit=LIMIT)]
    return [(i, None) for i in ids], False


def fetch_gguf(repo):
    for attempt in range(2):
        try:
            return getattr(api.model_info(repo, expand=["gguf"]), "gguf", None)
        except Exception:
            if attempt == 0:
                time.sleep(0.5)
            else:
                raise


def work(item):
    global _done
    repo, g = item
    try:
        if g is None:
            g = fetch_gguf(repo)
        g = g or {}
        tpl = g.get("chat_template")
        arch = g.get("architecture")
        if not tpl:
            out = ("notpl", {"repo": repo, "arch": arch})
        else:
            findings = analyze_template(tpl)
            c = summarize(findings)
            out = ("ok", {
                "repo": repo, "arch": arch, "fail": c[FAIL], "warn": c[WARN],
                "rules": sorted({f.rule_id for f in findings}),
                "fail_detail": [{"rule": f.rule_id, "detail": f.detail}
                                for f in findings if f.severity == FAIL],
                "behavioral": sorted({f.rule_id for f in findings
                                      if f.rule_id.startswith("TPL02")}),
            })
    except Exception as exc:  # noqa: BLE001
        out = ("skip", {"repo": repo, "why": str(exc)[:140]})
    with _lock:
        _done += 1
        if _done % 250 == 0:
            print(f"  ...{_done}/{LIMIT}", flush=True)
    return out


def main() -> None:
    print(f"listing up to {LIMIT} trending GGUF models (+ templates)...", flush=True)
    items, inline = list_items()
    print(f"{len(items)} models (templates inline={inline}); "
          f"analyzing with {WORKERS} workers...", flush=True)

    res, notpl, skips = [], [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(work, it) for it in items]):
            kind, payload = fut.result()
            (res if kind == "ok" else notpl if kind == "notpl" else skips).append(payload)

    hist: dict[str, int] = {}
    for r in res:
        for rid in r["rules"]:
            hist[rid] = hist.get(rid, 0) + 1
    fails = [r for r in res if r["fail"]]
    beh = [r for r in res if r["behavioral"]]
    summary = {
        "models_listed": len(items), "with_template": len(res),
        "no_template": len(notpl), "skipped": len(skips),
        "models_with_FAIL": len(fails), "models_with_behavioral": len(beh),
        "fail_rate_pct": round(100 * len(fails) / max(len(res), 1), 3),
        "behavioral_rate_pct": round(100 * len(beh) / max(len(res), 1), 3),
        "architectures": len({r["arch"] for r in res if r["arch"]}),
        "rule_histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
    }
    json.dump({"summary": summary, "fails": fails, "behavioral": beh,
               "results": res, "skips": skips[:300]}, open(OUT, "w"), indent=2)

    print("\n================= SUMMARY =================")
    print(json.dumps(summary, indent=2))
    print(f"\nFAIL-level template hits ({len(fails)}):")
    for r in fails[:80]:
        print(f"  {r['repo']} -> {sorted({d['rule'] for d in r['fail_detail']})}")
    print(f"\nreport: {OUT}")


if __name__ == "__main__":
    main()
