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
    _feed(monkeypatch, ["7", "q"])
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


def test_interactive_local_repository(monkeypatch, capsys, tmp_path):
    (tmp_path / "tokenizer.json").write_text(
        '{"model":{"vocab":{"a":0}}}', encoding="utf-8")
    _feed(monkeypatch, ["3", str(tmp_path), "q"])

    assert cli.interactive() == 0
    out = capsys.readouterr().out
    assert "c4nary scan:" in out
    assert "repo scan" in out


def test_interactive_local_modelfile(monkeypatch, capsys, tmp_path):
    p = tmp_path / "Modelfile"
    p.write_text("FROM llama3.2\n", encoding="utf-8")
    _feed(monkeypatch, ["8", str(p), "q"])

    assert cli.interactive() == 0
    out = capsys.readouterr().out
    assert "c4nary scan:" in out


def test_interactive_modelfile_error_does_not_crash(monkeypatch, capsys, tmp_path):
    p = tmp_path / "Modelfile"
    p.write_text("SYSTEM hi\n", encoding="utf-8")
    _feed(monkeypatch, ["8", str(p), "q"])

    assert cli.interactive() == 0
    captured = capsys.readouterr()
    assert "error: the Modelfile has no FROM" in captured.out


def test_interactive_bad_choice_reprompts(monkeypatch, capsys):
    _feed(monkeypatch, ["x", "q"])
    assert cli.interactive() == 0
    assert "choose 1-8" in capsys.readouterr().out


def test_interactive_eof_exits_clean(monkeypatch):
    def boom(*a):
        raise EOFError
    monkeypatch.setattr(builtins, "input", boom)
    assert cli.interactive() == 0
