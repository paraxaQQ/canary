"""CLI behaviour: help fallback and the interactive menu."""

import builtins
from pathlib import Path

from _ggufgen import write_gguf

from c4nary import cli

CHATML = (
    Path(__file__).parents[1] / "c4nary" / "known_templates" / "chatml.jinja"
).read_text(encoding="utf-8")


def _feed(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda *a: next(it))


def test_no_args_non_tty_prints_help_not_hang(capsys):
    # Piped/CI (no TTY) must print help and exit, never block on input().
    rc = cli.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "scan" in out and "diff" in out


def test_interactive_rules_then_quit(monkeypatch, capsys):
    _feed(monkeypatch, ["5", "q"])
    rc = cli.interactive()
    out = capsys.readouterr().out
    assert rc == 0
    assert "TPL001" in out


def test_interactive_scan_local_file(monkeypatch, capsys, tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "llama", "llama.context_length": 4096,
        "tokenizer.chat_template": CHATML,
    }, tensors=[("a", (2, 2), 0)], tail=b"d")
    _feed(monkeypatch, ["1", str(p), "q"])
    rc = cli.interactive()
    out = capsys.readouterr().out
    assert rc == 0
    assert "c4nary scan" in out


def test_interactive_bad_choice_reprompts(monkeypatch, capsys):
    _feed(monkeypatch, ["x", "q"])
    assert cli.interactive() == 0
    assert "choose 1-5" in capsys.readouterr().out


def test_interactive_eof_exits_clean(monkeypatch):
    def boom(*a):
        raise EOFError
    monkeypatch.setattr(builtins, "input", boom)
    assert cli.interactive() == 0
