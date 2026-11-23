"""Compare two refactor-baseline captures, normalizing only the A3 REP001 delta.

A3 adds one rule (``REP001``), which legitimately changes:

* ``rules --json``: a new rule row plus ``rules_bundle_sha256``;
* every JSON report: ``rules_bundle_sha256``;
* SARIF: ``rulesBundleSha256`` and the new rule entry in the driver rules.

Everything else must be byte-identical. This tool parses each captured stdout
as JSON, replaces the rule-bundle digest with ``<hash>``, drops the ``REP001``
rule entry, re-serializes with the renderer's own formatting, and compares the
whole normalized case.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HASH_KEYS = frozenset({"rules_bundle_sha256", "rulesBundleSha256"})
_NEW_RULE = "REP001"


def _strip_rule(rows: list) -> list:
    return [
        row for row in rows
        if not (isinstance(row, dict) and row.get("rule_id") == _NEW_RULE)
    ]


def _strip_sarif_rule(rows: list) -> list:
    return [
        row for row in rows
        if not (isinstance(row, dict) and row.get("id") == _NEW_RULE)
    ]


def _normalize_json(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in _HASH_KEYS:
                out[key] = "<hash>"
            else:
                out[key] = _normalize_json(item)
        if isinstance(out.get("rules"), list):
            out["rules"] = _strip_rule(out["rules"])
        runs = out.get("runs")
        if isinstance(runs, list):
            for run in runs:
                driver = ((run or {}).get("tool") or {}).get("driver") or {}
                if isinstance(driver.get("rules"), list):
                    driver["rules"] = _strip_sarif_rule(driver["rules"])
        return out
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    return value


def _normalize_stdout(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return text
    try:
        payload = json.loads(stripped)
    except ValueError:
        return text
    return json.dumps(_normalize_json(payload), indent=2, ensure_ascii=True)


def _split_case(text: str) -> tuple[str, str, str]:
    head, rest = text.split("--- stdout ---\n", 1)
    stdout, stderr = rest.split("--- stderr ---\n", 1)
    return head, stdout, stderr


def _normalized(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    head, stdout, stderr = _split_case(text)
    return (
        head
        + "--- stdout ---\n"
        + _normalize_stdout(stdout)
        + "--- stderr ---\n"
        + stderr
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before", help="pre-A3 capture directory containing cases/")
    ap.add_argument("after", help="post-A3 capture directory containing cases/")
    args = ap.parse_args(argv)

    before_dir = Path(args.before) / "cases"
    after_dir = Path(args.after) / "cases"
    before_names = sorted(path.name for path in before_dir.glob("*.txt"))
    after_names = sorted(path.name for path in after_dir.glob("*.txt"))

    failed = False
    if before_names != after_names:
        failed = True
        print(f"case set differs: before={before_names} after={after_names}")

    for name in before_names:
        if name not in after_names:
            continue
        expected = _normalized(before_dir / name)
        actual = _normalized(after_dir / name)
        if expected == actual:
            print(f"OK   {name}")
        else:
            failed = True
            print(f"DIFF {name}")

    print("normalized baseline: " + ("DIFFERENCES" if failed else "identical"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
