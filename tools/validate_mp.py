"""Validate c4nary's template rules across the ENTIRE Hugging Face GGUF universe.

Same approach as validate_templates.py (HF serves each chat_template server-side
via expand=["gguf"], no downloads) but analysis runs on a process pool to use all
cores -- the rule analysis is pure-Python/CPU-bound, so threads (GIL) don't help
but processes do.

Usage (on the pod):  LIMIT=200000 WORKERS=80 python tools/validate_mp.py
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from huggingface_hub import HfApi

from c4nary.report import FAIL, WARN, summarize
from c4nary.rules.template import analyze_template

LIMIT = int(os.environ.get("LIMIT", "200000"))
WORKERS = int(os.environ.get("WORKERS", "80"))
OUT = os.environ.get("OUT", "/workspace/universe.json")


def analyze_one(item):
    repo, arch, tpl = item
    findings = analyze_template(tpl)
    c = summarize(findings)
    return {
        "repo": repo, "arch": arch, "fail": c[FAIL], "warn": c[WARN],
        "rules": sorted({f.rule_id for f in findings}),
        "fail_detail": [{"rule": f.rule_id, "detail": f.detail}
                        for f in findings if f.severity == FAIL],
        "behavioral": sorted({f.rule_id for f in findings
                              if f.rule_id.startswith("TPL02")}),
    }


def main() -> None:
    api = HfApi()
    print(f"listing up to {LIMIT} GGUF models (+ templates)...", flush=True)
    items, no_template = [], 0
    t = time.time()
    for m in api.list_models(filter="gguf", sort="downloads", limit=LIMIT,
                             expand=["gguf"]):
        g = getattr(m, "gguf", None)
        if g and g.get("chat_template"):
            items.append((m.id, g.get("architecture"), g["chat_template"]))
        else:
            no_template += 1
    listed = len(items) + no_template
    print(f"listed {listed} models, {len(items)} with templates in "
          f"{time.time()-t:.0f}s; analyzing with {WORKERS} processes...", flush=True)

    res = []
    t = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(analyze_one, items, chunksize=200), 1):
            res.append(r)
            if i % 20000 == 0:
                print(f"  ...{i}/{len(items)} in {time.time()-t:.0f}s", flush=True)
    print(f"analysis done in {time.time()-t:.0f}s", flush=True)

    hist: dict[str, int] = {}
    for r in res:
        for rid in r["rules"]:
            hist[rid] = hist.get(rid, 0) + 1
    fails = [r for r in res if r["fail"]]
    beh = [r for r in res if r["behavioral"]]
    summary = {
        "models_listed": listed, "with_template": len(res),
        "no_template": no_template, "models_with_FAIL": len(fails),
        "models_with_behavioral": len(beh),
        "fail_rate_pct": round(100 * len(fails) / max(len(res), 1), 4),
        "behavioral_rate_pct": round(100 * len(beh) / max(len(res), 1), 4),
        "architectures": len({r["arch"] for r in res if r["arch"]}),
        "rule_histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
    }
    json.dump({"summary": summary, "fails": fails, "behavioral": beh},
              open(OUT, "w"), indent=2)

    print("\n================= SUMMARY =================")
    print(json.dumps(summary, indent=2))
    print(f"\nFAIL-level hits ({len(fails)}):")
    for r in fails:
        print(f"  {r['repo']} -> {sorted({d['rule'] for d in r['fail_detail']})}")
    print(f"\nreport: {OUT}")


if __name__ == "__main__":
    main()
