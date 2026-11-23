"""Smoke-test the standalone building artifact.

Builds deterministic fixtures, then checks version, rules provenance, clean and
FAIL scans, SARIF, a bundle scan, and optionally one remote scan. Use:

    .venv\\Scripts\\python.exe tools\\verify_exe.py
    .venv\\Scripts\\python.exe tools\\verify_exe.py --remote unsloth/Qwen3-0.6B-GGUF
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "outputs" / "building" / "canary-building.exe"
FIXTURES = ROOT / "outputs" / "building" / "verify"

sys.path.insert(0, str(ROOT / "tests"))
from _ggufgen import write_gguf  # noqa: E402

SSTI = ("{{ ''.__class__.__mro__[1].__subclasses__() }}"
        "{% for m in messages %}{{ m }}{% endfor %}")
TENSOR = [("token_embd.weight", (2, 2), 0)]


def fixture() -> tuple[Path, Path, Path]:
    if FIXTURES.exists():
        import shutil
        shutil.rmtree(FIXTURES)
    FIXTURES.mkdir(parents=True)
    clean = FIXTURES / "clean.gguf"
    write_gguf(clean, {
        "general.architecture": "llama",
        "tokenizer.ggml.tokens": ["<s>", "a"],
        "tokenizer.ggml.token_type": [3, 1],
    }, tensors=TENSOR)
    ssti = FIXTURES / "ssti.gguf"
    write_gguf(ssti, {
        "general.architecture": "llama",
        "tokenizer.chat_template": SSTI,
        "tokenizer.ggml.tokens": ["<s>", "a"],
        "tokenizer.ggml.token_type": [3, 1],
    }, tensors=TENSOR)
    bundle_dir = FIXTURES / "bundle"
    bundle_dir.mkdir()
    bundle = bundle_dir / "model.gguf"
    write_gguf(bundle, {
        "general.architecture": "llama",
        "tokenizer.ggml.tokens": ["<pad>", "<eos>"],
        "tokenizer.ggml.token_type": [1, 3],
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.add_eos_token": True,
    }, tensors=TENSOR)
    (bundle_dir / "config.json").write_text(json.dumps({
        "eos_token_id": 1,
        "suppress_tokens": [1],
    }), encoding="utf-8")
    return clean, ssti, bundle


def run(exe: Path, args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [str(exe), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, env=env,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", default=str(DEFAULT_EXE))
    ap.add_argument("--remote", metavar="REPO",
                    help="also scan one Hugging Face repo (uses HF_TOKEN if set)")
    args = ap.parse_args()
    exe = Path(args.exe)
    if not exe.is_file():
        print(f"FAIL: {exe} does not exist; run tools/build_exe.py first")
        return 2

    clean, ssti, bundle = fixture()
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""))
        if not ok:
            failures.append(name)

    p = run(exe, ["--version"])
    check("--version", p.returncode == 0 and p.stdout.startswith("c4nary "), p.stderr)

    p = run(exe, ["rules", "--json"])
    rules = json.loads(p.stdout) if p.returncode == 0 else {}
    from c4nary.provenance import rules_bundle_sha256
    check("rules --json", p.returncode == 0 and bool(rules.get("rules")), p.stderr)
    check("known_templates bundled",
          rules.get("rules_bundle_sha256") == rules_bundle_sha256(),
          f"{rules.get('rules_bundle_sha256')} != {rules_bundle_sha256()}")

    p = run(exe, ["scan", str(clean), "--json"])
    data = json.loads(p.stdout) if p.stdout else {}
    check("clean scan exit", p.returncode == 0, f"exit={p.returncode} {p.stderr}")
    check("clean scan has no FAIL",
          not any(f["severity"] == "FAIL" for f in data.get("findings", [])))

    p = run(exe, ["scan", str(ssti), "--json"])
    data = json.loads(p.stdout) if p.stdout else {}
    check("SSTI scan exit", p.returncode == 2, f"exit={p.returncode} {p.stderr}")
    check("SSTI TPL001 FAIL",
          any(f["rule_id"] == "TPL001" and f["severity"] == "FAIL"
              for f in data.get("findings", [])))

    p = run(exe, ["scan", str(ssti), "--sarif"])
    sarif = json.loads(p.stdout) if p.stdout else {}
    check("SARIF exit", p.returncode == 2, f"exit={p.returncode} {p.stderr}")
    check("SARIF 2.1.0", sarif.get("version") == "2.1.0" and bool(sarif.get("runs")))

    p = run(exe, ["scan", str(bundle), "--bundle", "--json"])
    data = json.loads(p.stdout) if p.stdout else {}
    check("bundle exit", p.returncode == 0, f"exit={p.returncode} {p.stderr}")
    check("bundle CFG001",
          any(f["rule_id"] == "CFG001" for f in data.get("findings", [])))

    if args.remote:
        if not os.environ.get("HF_TOKEN"):
            env_file = ROOT / ".env"
            if env_file.is_file():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("HF_TOKEN="):
                        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
        p = run(exe, ["scan", args.remote, "--remote", "--json"], timeout=420)
        ok = p.returncode in (0, 1, 2) and p.stdout.startswith("{")
        check(f"remote scan {args.remote}", ok, f"exit={p.returncode} {p.stderr[:300]}")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
