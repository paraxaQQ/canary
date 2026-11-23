"""Scan-target dispatch. The load-bearing property is that nothing which
classified as GGUF before classifies as anything else now."""

import pytest
from _ggufgen import write_gguf

from c4nary import cli
from c4nary.ingest import IngestError, classify_target, strip_ollama_scheme

MODELFILE = 'FROM llama3.2\nSYSTEM """be helpful"""\n'


def _gguf(tmp_path, name="m.gguf"):
    return str(write_gguf(tmp_path / name, {
        "general.architecture": "llama",
    }, tensors=[("a", (2, 2), 0)]))


# --- the regression that matters ---------------------------------------------- #

def test_gguf_file_still_classifies_as_gguf(tmp_path):
    assert classify_target(_gguf(tmp_path)) == "gguf"


def test_gguf_magic_beats_the_modelfile_name(tmp_path):
    # A GGUF that happens to be named Modelfile is still a GGUF: magic wins, so the
    # calibrated path can never be diverted by a filename.
    assert classify_target(_gguf(tmp_path, "Modelfile")) == "gguf"


def test_unrecognized_file_falls_through_to_gguf(tmp_path):
    # Preserves the historical parser error rather than inventing a new one.
    p = tmp_path / "random.bin"
    p.write_bytes(b"\x00\x01\x02\x03not a model")
    assert classify_target(str(p)) == "gguf"


def test_missing_path_falls_through_to_gguf(tmp_path):
    assert classify_target(str(tmp_path / "nope.gguf")) == "gguf"


def test_remote_is_always_gguf():
    assert classify_target("org/model", remote=True) == "gguf"


# --- the new targets ----------------------------------------------------------- #

def test_directory_is_a_repo(tmp_path):
    assert classify_target(str(tmp_path)) == "repo"


def test_modelfile_by_name(tmp_path):
    p = tmp_path / "Modelfile"
    p.write_text(MODELFILE, encoding="utf-8")
    assert classify_target(str(p)) == "ollama"


def test_modelfile_by_leading_from_directive(tmp_path):
    p = tmp_path / "qwen.txt"
    p.write_text("# a comment\n\n" + MODELFILE, encoding="utf-8")
    assert classify_target(str(p)) == "ollama"


def test_text_file_without_from_is_not_a_modelfile(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("SYSTEM be helpful\nFROM llama3.2\n", encoding="utf-8")
    assert classify_target(str(p)) == "gguf"


def test_ollama_flag_and_scheme():
    assert classify_target("library/qwen3:0.6b", ollama=True) == "ollama"
    assert classify_target("ollama://library/qwen3:0.6b") == "ollama"


def test_bare_ref_is_never_sniffed_as_ollama():
    # 'a/b' is equally a Hugging Face repo id; only an explicit flag or scheme decides.
    assert classify_target("library/qwen3:0.6b", remote=True) == "gguf"


def test_strip_ollama_scheme():
    assert strip_ollama_scheme("ollama://library/qwen3") == "library/qwen3"
    assert strip_ollama_scheme("library/qwen3") == "library/qwen3"


# --- override ------------------------------------------------------------------ #

def test_override_beats_every_heuristic(tmp_path):
    path = _gguf(tmp_path)
    assert classify_target(path, override="ollama") == "ollama"
    assert classify_target(path, override="repo") == "repo"
    assert classify_target(str(tmp_path), override="gguf") == "gguf"


def test_unknown_override_raises():
    with pytest.raises(IngestError, match="unknown target type"):
        classify_target("x", override="safetensors")


# --- CLI wiring ---------------------------------------------------------------- #

def test_cli_scans_a_gguf_unchanged(tmp_path, capsys):
    assert cli.main(["scan", _gguf(tmp_path), "--json"]) == cli.EXIT_OK
    assert '"findings"' in capsys.readouterr().out


def test_cli_reports_ollama_unimplemented(tmp_path, capsys):
    rc = cli.main(["scan", _gguf(tmp_path), "--as", "ollama", "--json"])
    assert rc == cli.EXIT_ERROR
    captured = capsys.readouterr()
    assert "ollama targets are not implemented" in captured.err
    assert captured.out == ""


def test_cli_repo_as_non_directory_exits_3(tmp_path, capsys):
    rc = cli.main(["scan", _gguf(tmp_path), "--as", "repo", "--json"])
    assert rc == cli.EXIT_ERROR
    captured = capsys.readouterr()
    assert "repo target must be a local directory" in captured.err
    assert captured.out == ""


def test_cli_rejects_an_unknown_as_value(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["scan", _gguf(tmp_path), "--as", "safetensors"])
