"""Template static-analysis acceptance tests (spec §8)."""

from pathlib import Path

import pytest

from c4nary.report import FAIL, WARN, summarize
from c4nary.rules.template import analyze_template

FIX = Path(__file__).parent / "fixtures"
KNOWN = Path(__file__).parents[1] / "c4nary" / "known_templates"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_cve_llama_drama_fails():
    findings = analyze_template(_read(FIX / "cve_llama_drama.jinja"))
    ids = {f.rule_id for f in findings}
    assert summarize(findings)[FAIL] >= 1
    # Must cite the __globals__ / lipsum / popen rules.
    assert "TPL001" in ids  # __globals__ dunder
    assert "TPL002" in ids  # lipsum gadget
    assert "TPL003" in ids  # os / popen dangerous name


def test_trigger_phrase_ssti_fails_and_warns():
    findings = analyze_template(_read(FIX / "trigger_ssti.jinja"))
    counts = summarize(findings)
    ids = {f.rule_id for f in findings}
    assert counts[FAIL] >= 1
    assert counts[WARN] >= 1
    assert "TPL010" in ids  # 'password' behavioral trigger
    assert "TPL002" in ids  # cycler gadget


def test_obfuscated_reconstruction_fails():
    findings = analyze_template(_read(FIX / "obfuscated.jinja"))
    ids = {f.rule_id for f in findings}
    assert "TPL005" in ids
    assert summarize(findings)[FAIL] >= 1


def test_behavioral_literal_is_warn_not_fail():
    findings = analyze_template(_read(FIX / "behavioral_warn.jinja"))
    counts = summarize(findings)
    ids = {f.rule_id for f in findings}
    assert counts[FAIL] == 0
    assert counts[WARN] >= 1
    assert "TPL010" in ids


@pytest.mark.parametrize("name", ["chatml", "llama-3", "mistral", "qwen"])
def test_known_good_templates_pass(name):
    findings = analyze_template(_read(KNOWN / f"{name}.jinja"))
    ids = {f.rule_id for f in findings}
    assert summarize(findings)[FAIL] == 0
    assert "TPL100" in ids  # matched known-good reference


def test_no_template_is_info():
    findings = analyze_template(None)
    assert {f.rule_id for f in findings} == {"TPL101"}


def test_unparseable_template_warns_not_crashes():
    findings = analyze_template("{% if %}{{ broken ")
    ids = {f.rule_id for f in findings}
    assert "TPL000" in ids
    assert summarize(findings)[FAIL] == 0


def test_hf_generation_block_parses_cleanly():
    # Modern HF templates use {% generation %}; must parse (no TPL000) and not FAIL.
    findings = analyze_template(_read(FIX / "modern_generation.jinja"))
    ids = {f.rule_id for f in findings}
    assert "TPL000" not in ids
    assert summarize(findings)[FAIL] == 0


def test_loopcontrols_break_continue_parse():
    # HF enables jinja2.ext.loopcontrols; {% break %}/{% continue %} must parse,
    # not fall to TPL000 (this was 54/4065 real templates).
    src = ("{% for m in messages %}{% if m['role'] == 'x' %}{% break %}"
           "{% else %}{% continue %}{% endif %}{% endfor %}")
    assert "TPL000" not in {f.rule_id for f in analyze_template(src)}


def test_payload_inside_generation_block_still_detected():
    # The fix must preserve the block body for analysis, not discard it.
    findings = analyze_template(
        "{% generation %}{{ cycler.__init__.__globals__ }}{% endgeneration %}"
    )
    ids = {f.rule_id for f in findings}
    assert "TPL000" not in ids
    assert summarize(findings)[FAIL] >= 1
    assert "TPL001" in ids  # __globals__ dunder inside the block


def test_role_dict_lookup_not_flagged():
    # role_indicators['system'] (EXAONE) is a benign role lookup, not os.system.
    findings = analyze_template("{{ role_indicators['system'] }}")
    assert "TPL003" not in {f.rule_id for f in findings}


def test_os_system_still_flagged():
    # os.system is still caught via the bare 'os' name reference.
    findings = analyze_template("{{ os.system('id') }}")
    assert "TPL003" in {f.rule_id for f in findings}


def test_module_as_benign_attribute_not_flagged():
    # terminal_state.os (a benign field, e.g. agentic templates) must NOT FAIL.
    findings = analyze_template("{{ terminal_state.os }} {{ device.platform }}")
    assert "TPL003" not in {f.rule_id for f in findings}
    assert summarize(findings)[FAIL] == 0


def test_globals_subscript_module_still_flagged():
    # __globals__['os'] keeps 'os' as a flagged subscript key, plus the dunder.
    findings = analyze_template("{{ x.__globals__['os'].popen('id') }}")
    ids = {f.rule_id for f in findings}
    assert "TPL001" in ids and "TPL003" in ids


def test_config_variable_not_flagged():
    # config.x is a benign passed-in variable, not the Flask gadget.
    findings = analyze_template("{{ config.temperature }}{% if config.enable %}x{% endif %}")
    assert "TPL002" not in {f.rule_id for f in findings}
    assert summarize(findings)[FAIL] == 0


def test_config_exploit_still_flagged():
    # config.__class__... is still caught via the dunder.
    findings = analyze_template("{{ config.__class__.__init__.__globals__ }}")
    assert "TPL001" in {f.rule_id for f in findings}
    assert summarize(findings)[FAIL] >= 1


def test_system_role_concat_not_flagged():
    # Building the 'system' role header from constants is not SSTI reconstruction.
    findings = analyze_template(
        "{{ '<|start_header_id|>' ~ 'system' ~ '<|end_header_id|>' }}")
    assert "TPL005" not in {f.rule_id for f in findings}


def test_literal_subscript_pivot_flagged():
    # ''[...] / (0)[...] with a COMPUTED key (evades const-key detection).
    for src in ("{{ ''[k] }}", "{{ (0)[x] }}", "{{ ()[d % 'class'] }}"):
        assert "TPL001" in {f.rule_id for f in analyze_template(src)}, src


def test_role_dict_literal_lookup_not_flagged():
    # {'user': ...}[role] is a benign role lookup, NOT a literal-subscript pivot.
    findings = analyze_template(
        "{{ {'user':'<|user|>','assistant':'<|a|>'}[message['role']] }}")
    assert summarize(findings)[FAIL] == 0


def test_introspection_dunder_in_literal_flagged():
    # var-keyed chains hide the dunder in a string literal.
    findings = analyze_template("{% set g = '__globals__' %}{{ x[g] }}")
    assert "TPL001" in {f.rule_id for f in findings}


def test_code_example_dunder_not_flagged():
    # __init__/__class__/__name__ appear in benign code-example templates.
    findings = analyze_template("{{ 'class Foo:\\n    def __init__(self): ...' }}")
    assert summarize(findings)[FAIL] == 0


def test_nfkc_fullwidth_name_flagged():
    findings = analyze_template("{{ ｏｓ.system('id') }}")  # fullwidth 'os'
    assert "TPL003" in {f.rule_id for f in findings}


def test_map_attribute_dunder_kwarg_flagged():
    findings = analyze_template("{{ x | map(attribute='__class__') | list }}")
    assert "TPL001" in {f.rule_id for f in findings}


def test_map_attribute_benign_field_not_flagged():
    # Function-calling templates use map(attribute='function'/'role') benignly.
    findings = analyze_template(
        "{{ tools | map(attribute='function') | list }}"
        "{{ messages | map(attribute='role') | list }}")
    assert summarize(findings)[FAIL] == 0


def test_class_typecheck_on_variable_not_flagged():
    # foo.__class__.__name__ is an inert type-check idiom real tool-calling templates use
    # (Darkhn Gemma-Animus type-checks tool args); a bare content.__class__ escapes nothing.
    # Neither is a sandbox escape -> no FAIL.
    for tpl in ("{{ function['arguments'].__class__.__name__ }}",
                "{{ messages[0].content.__class__ }}",
                "{% if x.__class__ is defined %}{{ 'ok' }}{% endif %}"):
        assert summarize(analyze_template(tpl))[FAIL] == 0, tpl


def test_class_pivot_on_literal_flagged():
    # ''.__class__ -- __class__ on a bare literal is the canonical SSTI pivot start.
    assert "TPL001" in {f.rule_id for f in analyze_template("{{ ''.__class__ }}")}
    assert "TPL001" in {f.rule_id for f in analyze_template("{{ ().__class__.__mro__ }}")}


def test_class_escape_chain_still_flagged():
    # A genuine escape via __class__ always continues into an escape dunder (kept flagged).
    assert "TPL001" in {f.rule_id for f in analyze_template(
        "{{ x.__class__.__base__.__subclasses__() }}")}


def test_get_content_gate_flagged():
    # message.get('content') laundering still engages the content-branch detector.
    assert "TPL020" in {f.rule_id for f in analyze_template(
        "{% if m.get('content') == 'trigger' %}{{ 'x' }}{% endif %}")}


def test_map_attr_filter_flagged():
    findings = analyze_template("{{ data | map('attr', '__globals__') | list }}")
    ids = {f.rule_id for f in findings}
    assert "TPL004" in ids
    assert summarize(findings)[FAIL] >= 1
