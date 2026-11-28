"""Tests of the inference helpers and of the audit's comparison rules.   python -X utf8 -m pytest tests -q"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "f1"))
import audit
from inference import signflip_p


def test_signflip_exact_for_few_units():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert signflip_p(x) == 2 / 32                         # only the all-positive and all-negative patterns


def test_signflip_monte_carlo_is_never_zero():
    x = np.full(52, 3.0)                                   # every random pattern is less extreme than the observed one
    p = signflip_p(x, n_mc=1000)
    assert p == 1 / 1001


def test_signflip_symmetric_null():
    x = np.r_[np.ones(10), -np.ones(10)]
    assert signflip_p(x) == 1.0


def test_check_tolerance_is_rounding_only():
    audit.checks.clear()
    assert audit.check("a", "4.0", 4.04) == "OK"
    assert audit.check("b", "4.0", 4.06) == "MISMATCH"
    assert audit.check("c", "0.11", 0.1026) == "MISMATCH"   # printed precision decides the tolerance
    assert audit.check("d", "<0.001", 0.0004) == "OK"
    assert audit.check("e", "<0.001", 0.002) == "MISMATCH"
    assert audit.check("f", "+61", 61.18) == "OK"
    assert audit.check("g", "0.0059", 0.0061, mc_se=0.0001) == "OK"   # Monte Carlo allowance
    assert audit.check("h", "12", None) == "SKIPPED"


def test_md_table_repeats_grouping_cells():
    text = "**Table X.** caption\n\n| Survey | Comparator | Value |\n|---|---|---|\n| A | one | 1 |\n|  | two | 2 |\n| B | one | 3 |\n"
    head, body = audit.md_table(text, "**Table X.**")
    assert head == ["Survey", "Comparator", "Value"]
    assert [r[0] for r in body] == ["A", "A", "B"]


def test_ci_parts():
    assert audit.ci_parts("4.5 (-0.1, 9.0)") == ("4.5", "-0.1", "9.0")
    assert audit.ci_parts("7.8 (2.0 to 13.6)") == ("7.8", "2.0", "13.6")
