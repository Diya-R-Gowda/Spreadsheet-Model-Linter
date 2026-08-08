from __future__ import annotations

import pytest

from ssmlint.ast_nodes import (
    BinaryOp,
    CellRef,
    FunctionCall,
    Literal,
    NamedRangeRef,
    RangeRef,
    UnaryOp,
)
from ssmlint.errors import FormulaSyntaxError, UnsupportedFormulaError
from ssmlint.formula_parser import parse_formula


def _eval_literal_expr(node):
    """Evaluate an AST built from literals only — used to pin down precedence."""
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, UnaryOp):
        value = _eval_literal_expr(node.operand)
        return -value if node.op == "-" else value
    if isinstance(node, BinaryOp):
        left = _eval_literal_expr(node.left)
        right = _eval_literal_expr(node.right)
        return {
            "+": lambda: left + right,
            "-": lambda: left - right,
            "*": lambda: left * right,
            "/": lambda: left / right,
            "^": lambda: left**right,
        }[node.op]()
    raise TypeError(f"cannot evaluate {node!r}")


# ---------------------------------------------------------------------------
# Precedence and associativity
# ---------------------------------------------------------------------------


def test_mixed_precedence() -> None:
    # * before +, ^ before *: 2 + 3*16 = 50
    assert _eval_literal_expr(parse_formula("2+3*4^2")) == 50


def test_unary_minus_binds_tighter_than_power_excel_quirk() -> None:
    # Unlike Python/most languages, Excel evaluates -2^2 as (-2)^2 = 4,
    # not -(2^2) = -4. This is the case that would silently mis-parse if
    # unary and power were layered the "normal" way.
    assert _eval_literal_expr(parse_formula("-2^2")) == 4


def test_unary_minus_as_exponent() -> None:
    assert _eval_literal_expr(parse_formula("2^-2")) == 0.25


def test_power_is_left_associative() -> None:
    # Excel evaluates 2^3^2 as (2^3)^2 = 64, not 2^(3^2) = 512.
    assert _eval_literal_expr(parse_formula("2^3^2")) == 64


def test_parentheses_override_precedence() -> None:
    assert _eval_literal_expr(parse_formula("((2+3)*(4-1))/2")) == 7.5


def test_division_and_multiplication_are_left_associative() -> None:
    # 100 / 10 * 2 = 20, not 100 / (10*2) = 5
    assert _eval_literal_expr(parse_formula("100/10*2")) == 20


# ---------------------------------------------------------------------------
# Structural cases involving cell references (can't be numerically evaluated)
# ---------------------------------------------------------------------------


def test_unary_minus_on_cell_ref_binds_before_multiply() -> None:
    # -A1*B1 == (-A1)*B1
    node = parse_formula("-A1*B1")
    assert node == BinaryOp(
        op="*",
        left=UnaryOp(op="-", operand=CellRef(col="A", row=1)),
        right=CellRef(col="B", row=1),
    )


def test_string_concat_is_left_associative() -> None:
    node = parse_formula('A1&" "&B1')
    assert node == BinaryOp(
        op="&",
        left=BinaryOp(op="&", left=CellRef(col="A", row=1), right=Literal(value=" ")),
        right=CellRef(col="B", row=1),
    )


def test_nested_function_calls() -> None:
    node = parse_formula("SUM(IF(A1>0,B1,0))")
    assert node == FunctionCall(
        name="SUM",
        args=(
            FunctionCall(
                name="IF",
                args=(
                    BinaryOp(op=">", left=CellRef(col="A", row=1), right=Literal(value=0.0)),
                    CellRef(col="B", row=1),
                    Literal(value=0.0),
                ),
            ),
        ),
    )


def test_comparison_operators() -> None:
    assert parse_formula("A1<=B1") == BinaryOp(
        op="<=", left=CellRef(col="A", row=1), right=CellRef(col="B", row=1)
    )
    assert parse_formula("A1<>B1") == BinaryOp(
        op="<>", left=CellRef(col="A", row=1), right=CellRef(col="B", row=1)
    )


# ---------------------------------------------------------------------------
# References: anchoring, cross-sheet, ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("formula", "col_abs", "row_abs"),
    [
        ("A1", False, False),
        ("$A1", True, False),
        ("A$1", False, True),
        ("$A$1", True, True),
    ],
)
def test_mixed_anchoring(formula: str, col_abs: bool, row_abs: bool) -> None:
    node = parse_formula(formula)
    assert node == CellRef(col="A", row=1, col_abs=col_abs, row_abs=row_abs)


def test_quoted_sheet_name_with_spaces() -> None:
    node = parse_formula("'My Sheet'!A1")
    assert node == CellRef(col="A", row=1, sheet="My Sheet")
    assert node.is_cross_sheet is True


def test_bare_cross_sheet_reference() -> None:
    node = parse_formula("Sheet2!A1")
    assert node == CellRef(col="A", row=1, sheet="Sheet2")


def test_same_sheet_reference_has_no_sheet_set() -> None:
    node = parse_formula("A1")
    assert node.sheet is None
    assert node.is_cross_sheet is False


def test_range_reference() -> None:
    node = parse_formula("SUM(Sheet2!A1:A10)")
    assert node == FunctionCall(
        name="SUM",
        args=(
            RangeRef(
                start=CellRef(col="A", row=1, sheet="Sheet2"),
                end=CellRef(col="A", row=10, sheet="Sheet2"),
            ),
        ),
    )


def test_named_range_reference() -> None:
    node = parse_formula("TaxRate*B1")
    assert node == BinaryOp(
        op="*", left=NamedRangeRef(name="TaxRate"), right=CellRef(col="B", row=1)
    )


def test_sheet_scoped_named_range_reference() -> None:
    node = parse_formula("Assumptions!TaxRate")
    assert node == NamedRangeRef(name="TaxRate", sheet="Assumptions")


# ---------------------------------------------------------------------------
# Whitelist rejections — must raise cleanly, never crash or silently mis-parse
# ---------------------------------------------------------------------------


def test_rejects_let() -> None:
    with pytest.raises(UnsupportedFormulaError):
        parse_formula("=LET(x,5,x*2)")


def test_rejects_lambda() -> None:
    with pytest.raises(UnsupportedFormulaError):
        parse_formula("=LAMBDA(x,x*2)(5)")


def test_rejects_structured_table_reference() -> None:
    with pytest.raises(UnsupportedFormulaError):
        parse_formula("=SUM(Table1[Column1])")


def test_rejects_array_formula_braces() -> None:
    with pytest.raises(UnsupportedFormulaError):
        parse_formula("{=SUM(A1:A10)}")


def test_rejects_full_column_range() -> None:
    with pytest.raises(UnsupportedFormulaError):
        parse_formula("SUM(A:A)")


def test_rejects_full_row_range() -> None:
    with pytest.raises(UnsupportedFormulaError):
        parse_formula("SUM(1:1)")


def test_rejects_anchored_full_column_range() -> None:
    # Common in real VLOOKUP/SUMIFS formulas, e.g. Sheet2!$A:$B — must be a
    # clean rejection, not a raw "unexpected character '$'" crash.
    with pytest.raises(UnsupportedFormulaError):
        parse_formula("VLOOKUP(A1,Sheet2!$A:$B,2,FALSE)")


def test_rejects_anchored_full_row_range() -> None:
    with pytest.raises(UnsupportedFormulaError):
        parse_formula("SUM($1:$1)")


def test_rejects_3d_sheet_range() -> None:
    with pytest.raises(UnsupportedFormulaError):
        parse_formula("SUM(Sheet1!A1:Sheet3!A1)")


def test_rejects_percent_operator() -> None:
    with pytest.raises(UnsupportedFormulaError):
        parse_formula("A1%")


# ---------------------------------------------------------------------------
# Genuine syntax errors — malformed input, not just unsupported features
# ---------------------------------------------------------------------------


def test_unbalanced_parens_raises_syntax_error() -> None:
    with pytest.raises(FormulaSyntaxError):
        parse_formula("SUM(A1,B1")


def test_trailing_operator_raises_syntax_error() -> None:
    with pytest.raises(FormulaSyntaxError):
        parse_formula("A1+")


def test_empty_formula_raises_syntax_error() -> None:
    with pytest.raises(FormulaSyntaxError):
        parse_formula("=")
