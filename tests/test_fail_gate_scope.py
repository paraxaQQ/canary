from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from tools import release_gate_scan, template_fail_gate


def test_both_corpus_gates_account_for_every_registered_fail_rule() -> None:
    release_gate_scan.assert_fail_gate_scope(
        release_gate_scan.TEMPLATE_FAIL_RULES,
        release_gate_scan.CORPUS_FAIL_EXCLUSIONS,
        gate="release corpus gate",
    )
    release_gate_scan.assert_fail_gate_scope(
        template_fail_gate.TEMPLATE_FAIL_RULES,
        release_gate_scan.CORPUS_FAIL_EXCLUSIONS,
        gate="template FAIL gate",
    )


def test_new_unaccounted_fail_rule_stops_gate_startup() -> None:
    registered = release_gate_scan.REGISTERED_FAIL_RULES | {"RMT999"}

    with pytest.raises(RuntimeError, match=r"missing=\['RMT999'\]"):
        release_gate_scan.assert_fail_gate_scope(
            release_gate_scan.TEMPLATE_FAIL_RULES,
            release_gate_scan.CORPUS_FAIL_EXCLUSIONS,
            gate="test gate",
            registered=registered,
        )


def test_fail_exclusions_are_disjoint_and_justified() -> None:
    exclusions = release_gate_scan.CORPUS_FAIL_EXCLUSIONS

    assert not release_gate_scan.TEMPLATE_FAIL_RULES.intersection(exclusions)
    assert all(reason.strip() for reason in exclusions.values())
