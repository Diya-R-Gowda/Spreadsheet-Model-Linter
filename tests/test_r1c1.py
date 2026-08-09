"""Tests for r1c1.normalize — every case asserts an actual string equality
or inequality against another normalized string, never just "didn't throw".
"""

from __future__ import annotations

import pytest

from ssmlint.errors import UnsupportedFormulaError
from ssmlint.formula_parser import parse_formula
from ssmlint.r1c1 import normalize


def _n(formula: str, origin: str, sheet: str = "Sheet1") -> str:
    return normalize(parse_formula(formula), origin, sheet)


# ---------------------------------------------------------------------------
# Edge case 1: identically-shaped relative formulas normalize IDENTICALLY
# ---------------------------------------------------------------------------


def test_identical_relative_shape_normalizes_identically() -> None:
    a = _n("=B4*C4", "D4")
    b = _n("=B5*C5", "D5")
    assert a == b
    assert a == "(R[0]C[-2]*R[0]C[-1])"


def test_identical_relative_shape_at_a_third_far_away_origin_still_matches() -> None:
    # Same relative relationship, three wildly different rows — this is
    # the actual point of the whole module.
    at_row_4 = _n("=B4*C4", "D4")
    at_row_10 = _n("=B10*C10", "D10")
    at_row_100 = _n("=B100*C100", "D100")
    assert at_row_4 == at_row_10 == at_row_100


# ---------------------------------------------------------------------------
# Edge case 2: one absolute ref must NOT collapse to the same string as
# the fully-relative version
# ---------------------------------------------------------------------------


def test_absolute_ref_normalizes_differently_from_relative() -> None:
    relative = _n("=B4*C4", "D4")
    mixed = _n("=$B$4*C4", "D4")
    assert relative != mixed
    assert relative == "(R[0]C[-2]*R[0]C[-1])"
    assert mixed == "(R4C2*R[0]C[-1])"


# ---------------------------------------------------------------------------
# Edge case 3: mixed anchoring, both directions, at multiple origins —
# proves the offsets are computed, not just "$ stripped"
# ---------------------------------------------------------------------------


def test_mixed_anchoring_col_absolute_row_relative() -> None:
    # $B14: column pinned to col 2, row relative to the origin's row.
    assert _n("=$B14", "C14") == "R[0]C2"
    assert _n("=$B14", "E20") == "R[-6]C2"


def test_mixed_anchoring_row_absolute_col_relative() -> None:
    # B$14: row pinned to row 14, column relative to the origin's column.
    assert _n("=B$14", "C14") == "R14C[-1]"
    assert _n("=B$14", "E20") == "R14C[-3]"


def test_mixed_anchoring_variants_differ_from_each_other() -> None:
    # $B14 and B$14 anchor opposite axes — must never coincide.
    at_c14 = _n("=$B14", "C14"), _n("=B$14", "C14")
    assert at_c14[0] != at_c14[1]


# ---------------------------------------------------------------------------
# Edge case 4: cross-sheet refs vs. self-sheet refs
# ---------------------------------------------------------------------------


def test_cross_sheet_reference_is_independent_of_which_sheet_hosts_the_formula() -> None:
    # As long as the origin's own sheet is neither Sheet1 nor Sheet3 (the
    # two hosts tried here), a reference to Sheet2!B2 keeps its prefix and
    # normalizes identically regardless of which of those two sheets
    # actually hosts the formula.
    node = parse_formula("=Sheet2!B2")
    assert normalize(node, "A1", "Sheet1") == normalize(node, "A1", "Sheet3")
    assert normalize(node, "A1", "Sheet1") == "Sheet2!R[1]C[1]"


def test_cross_sheet_reference_offset_still_responds_to_origin_position() -> None:
    # Same target cell, different origins -> different offsets. Proves
    # the cross-sheet case is genuinely offset-based, not a fixed string.
    from_a1 = _n("=Sheet2!B2", "A1", "Sheet1")
    from_b1 = _n("=Sheet2!B2", "B1", "Sheet1")
    assert from_a1 != from_b1
    assert from_a1 == "Sheet2!R[1]C[1]"
    assert from_b1 == "Sheet2!R[1]C[0]"


def test_same_sheet_reference_has_no_sheet_prefix() -> None:
    assert _n("=B4", "D4") == "R[0]C[-2]"
    assert "!" not in _n("=B4", "D4")


def test_explicit_self_sheet_reference_normalizes_same_as_implicit() -> None:
    # The bug this fix addresses: a formula living on Sheet1 that spells
    # out "=Sheet1!B14*C14" means exactly the same thing in Excel as the
    # implicit "=B14*C14" — they must normalize to the identical string.
    explicit = _n("=Sheet1!B14*C14", "D14", "Sheet1")
    implicit = _n("=B14*C14", "D14", "Sheet1")
    assert explicit == implicit
    assert "!" not in explicit


def test_explicit_reference_to_a_different_sheet_is_not_collapsed() -> None:
    # Guard against overcorrecting: only self-sheet references collapse.
    # A formula on Sheet1 explicitly referencing Sheet2 must still keep
    # its prefix — this is a genuine cross-sheet reference.
    result = _n("=Sheet2!B14*C14", "D14", "Sheet1")
    assert result == "(Sheet2!R[0]C[-2]*R[0]C[-1])"
    assert "Sheet2!" in result


def test_explicit_self_sheet_range_reference_also_collapses() -> None:
    explicit = _n("=SUM(Sheet1!B4:B10)", "C4", "Sheet1")
    implicit = _n("=SUM(B4:B10)", "C4", "Sheet1")
    assert explicit == implicit
    assert "!" not in explicit


# ---------------------------------------------------------------------------
# Edge case 5: ranges normalize both endpoints independently
# ---------------------------------------------------------------------------


def test_range_normalizes_both_endpoints() -> None:
    assert _n("=SUM(B4:B10)", "C4") == "SUM(R[0]C[-1]:R[6]C[-1])"


def test_cross_sheet_range_sheet_prefix_applies_once() -> None:
    result = _n("=SUM(Sheet2!B4:B10)", "C4", "Sheet1")
    assert result == "SUM(Sheet2!R[0]C[-1]:R[6]C[-1])"
    assert result.count("Sheet2!") == 1


# ---------------------------------------------------------------------------
# Edge case 6: nested function calls preserve structure
# ---------------------------------------------------------------------------


def test_nested_function_calls_normalize_every_embedded_ref() -> None:
    result = _n("=SUM(IF(A1>0,B1,0))", "D1")
    assert result == "SUM(IF((R[0]C[-3]>0),R[0]C[-2],0))"


def test_nested_function_calls_differ_when_inner_ref_differs() -> None:
    a = _n("=SUM(IF(A1>0,B1,0))", "D1")
    b = _n("=SUM(IF(A1>0,B2,0))", "D1")
    assert a != b


# ---------------------------------------------------------------------------
# Realistic formulas
# ---------------------------------------------------------------------------


def test_driver_style_formula_matches_module_docstring_worked_example() -> None:
    assert _n("=B14*(1+B15)", "C14") == "(R[0]C[-1]*(1+R[1]C[-1]))"


def test_named_range_passes_through_unresolved() -> None:
    # Unlike function names, named-range names ARE case-sensitive in
    # Excel, so — unlike SUM/sum — this must NOT be case-canonicalized.
    assert _n("=TaxRate*B1", "C1") == "(TaxRate*R[0]C[-1])"


def test_sheet_scoped_named_range_is_prefixed() -> None:
    # NamedRangeRef is deliberately untouched by the self-sheet fix (see
    # r1c1.py's docstring) — this stays prefixed even when origin_sheet
    # is something else entirely, since named-range sheet-prefixing was
    # never reported as a bug and isn't in scope for it.
    assert _n("=Assumptions!TaxRate", "A1", "Model") == "Assumptions!TaxRate"


def test_function_name_case_is_canonicalized() -> None:
    assert _n("=sum(B1)", "C1") == _n("=SUM(B1)", "C1")


def test_string_and_boolean_and_number_literals() -> None:
    assert _n('=A1&"!"', "B1") == '(R[0]C[-1]&"!")'
    assert _n("=IF(A1,TRUE,FALSE)", "B1") == "IF(R[0]C[-1],TRUE,FALSE)"
    assert _n("=B1*1.05", "C1") == "(R[0]C[-1]*1.05)"


# ---------------------------------------------------------------------------
# Pipeline boundary: formulas the parser rejects never produce an AST,
# so they can never reach normalize() in the first place.
# ---------------------------------------------------------------------------


def test_rejected_formula_never_produces_an_ast_for_this_module() -> None:
    # A SUMIFS-shaped formula using $-anchored full-column ranges — a very
    # realistic shape — is rejected at parse_formula(), stage [2], well
    # before r1c1.normalize() (stage [3]) would ever see it.
    rejected_formula = '=SUMIFS(Data!$C:$C,Data!$A:$A,">="&A1,Data!$B:$B,B1)'
    with pytest.raises(UnsupportedFormulaError):
        parse_formula(rejected_formula)
    # There is no AST to pass to normalize() here — this test's own
    # existence, and its pytest.raises around parse_formula rather than
    # normalize, is the confirmation the prompt asked for.
