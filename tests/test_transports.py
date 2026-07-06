"""Obfuscation transports: external template code (TPL031) + decode filters (TPL032).

A self-contained chat template needs neither an include/import/extends nor a base64/url
decoder. from_json is deliberately NOT a transport -- tool-calling templates use it to
parse tool arguments (calibrated: 15/1665 templates, all legit).
"""

from c4nary.rules.template import analyze_template


def _ids(t):
    return {f.rule_id for f in analyze_template(t)}


def test_tpl031_include():
    # the real wild case: a template that is *only* an include, hiding its logic in a file
    assert "TPL031" in _ids("{% include 'chat_template.jinja' %}")


def test_tpl031_extends():
    assert "TPL031" in _ids("{% extends 'base' %}")


def test_tpl032_b64decode_filter():
    assert "TPL032" in _ids("{{ messages[0]['content']|b64decode }}")


def test_tpl032_b64decode_call():
    assert "TPL032" in _ids("{{ b64decode('payload') }}")


def test_no_fp_from_json_tool_calling():
    # from_json is legit tool-argument parsing -> must NOT flag as a transport
    assert "TPL032" not in _ids("{% for t in tools %}{{ t.arguments|from_json }}{% endfor %}")


def test_no_fp_benign_chatml():
    ids = _ids("{% for m in messages %}{{ m['role'] + m['content'] }}{% endfor %}")
    assert "TPL031" not in ids and "TPL032" not in ids
