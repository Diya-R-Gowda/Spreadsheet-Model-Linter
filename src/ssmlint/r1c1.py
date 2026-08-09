"""R1C1 normalization — stage [3] in the architecture, "the heart of the project".

Converts a formula's AST (from formula_parser.parse_formula) plus the
address of the cell the formula lives in ("the origin") into a
position-independent string: two structurally identical formulas at
different addresses normalize to the identical string, and two formulas
that differ in any way that actually changes their meaning normalize to
different strings.

THE OFFSET CONVENTION
----------------------
Every CellRef's row and column are converted independently (per axis) into
either a relative offset from the origin, or an absolute position, based
on that axis's $-anchoring:

  - No $ on an axis  -> relative, bracketed: the offset from the origin,
    e.g. a formula at row 14 referencing row 14 itself -> R[0], row 15 ->
    R[1], row 13 -> R[-1]. Zero is still written as R[0], not R (there is
    no bare "R" or "C" in this module's output).
  - $ on an axis      -> absolute, unbracketed: the literal 1-based
    row number / numeric column index, completely independent of the
    origin, e.g. $B$14 -> row 14 always R14, column B (index 2) always C2.
  - Mixed anchoring combines both rules per axis independently. $B14 (col
    absolute, row relative) at origin row 14 is R[0]C2, not "R[0]C[-1]"
    and not "R14C2" — the row is still relative, only the column is
    pinned.

WORKED EXAMPLE
---------------
Formula "=B14*(1+B15)" living in cell C14 (origin row=14, col=C=3):

  B14 -> row offset 14-14=0, col offset col_index(B)=2 - 3 = -1
      -> R[0]C[-1]
  B15 -> row offset 15-14=1, col offset -1
      -> R[1]C[-1]
  literal 1 -> "1"

  normalize(...) == "(R[0]C[-1]*(1+R[1]C[-1]))"

The same formula shifted to any other cell in the same column — D20,
Z2000, whatever — normalizes to the exact same string, because every
reference is relative and the offsets never change. Move it to a
*different column* and the col offset (and therefore the string) changes,
correctly: same-shape-but-different-neighbor is not the same pattern.

CROSS-SHEET REFERENCES
------------------------
A CellRef/RangeRef with sheet=None means "same sheet as the formula's own
cell" and contributes no sheet marker to the output at all. A CellRef with
an explicit sheet name is compared against `origin_sheet` (a required
parameter — see FIXED below): if it names the origin's own sheet, it is
collapsed to exactly the same output as the implicit form, since Excel
treats "=Sheet1!B14" and "=B14" as identical when both physically live on
Sheet1. Only a reference to a genuinely *different* sheet keeps a
`Sheet!` prefix — Excel has no concept of "the sheet one to the right"
for a plain formula reference (3-D ranges, which would need that, are
already rejected upstream in stage [2]) — emitted verbatim, once, kept
structurally separate from the R/C part (`Sheet2!R[1]C[1]`, never
interleaved into the offsets).

  The offset math itself still depends only on the *origin cell's*
  row/column, regardless of sheet: a formula on Sheet1!A1 referencing
  Sheet2!B2 and the "same" formula on Sheet3!A1 referencing Sheet2!B2
  still normalize identically to each other — neither origin sheet is
  Sheet2, so both keep the `Sheet2!` prefix and compute the same offsets.

  FIXED: earlier versions of `normalize()` took only the origin's plain
  address, never its sheet, so an explicit self-sheet reference
  (`=Sheet1!B14*C14` while living on Sheet1) had no way to be recognized
  as "actually the same sheet" and was always treated as cross-sheet —
  it would NOT normalize identically to the equivalent implicit form
  `=B14*C14`, even though Excel treats them as identical. This caused a
  real false split in the block detector's row clustering (blocks.py):
  a cell using the explicit self-sheet form was excluded from an
  otherwise-identical block purely because of that spelling difference.
  Fixed by adding `origin_sheet` as a required parameter to `normalize()`.
  This does NOT extend to `NamedRangeRef` — a sheet-scoped named range's
  `Sheet!Name` prefix is unaffected by this fix and still always emitted
  verbatim when present (see OTHER NODE TYPES below); collapsing that
  too was never reported as a bug and is out of scope for this fix.

OTHER NODE TYPES
------------------
  - RangeRef: both endpoints normalized independently with the same rule,
    joined with ":"; the sheet prefix (if any) is applied once to the
    whole range, not to each endpoint (the parser already guarantees both
    endpoints share one sheet).
  - NamedRangeRef: passes through as the name itself — named ranges are
    already position-independent by definition, so there is nothing to
    resolve to R1C1. A sheet-scoped name is prefixed the same way a
    cross-sheet cell ref is (`Sheet1!MyName`), since referencing a name
    local to a different sheet is a genuinely different thing to
    reference than a workbook-global name of the same spelling.
  - Literal: passed through as its Excel-literal spelling (numbers
    without a redundant ".0", strings re-quoted with doubled internal
    quotes, booleans as TRUE/FALSE) — never treated as position data.
  - FunctionCall: name is upper-cased (Excel function names are
    case-insensitive, so "sum(...)" and "SUM(...)" are the same pattern)
    and its arguments normalized recursively.
  - BinaryOp / UnaryOp: normalized recursively and always explicitly
    parenthesized in the output, even where the original formula didn't
    need parens for precedence. This isn't about producing valid Excel
    syntax — the output is a comparison key, not a formula — it's what
    guarantees two different ASTs can never coincidentally flatten to the
    same string.
"""

from __future__ import annotations

from openpyxl.utils import column_index_from_string

from .ast_nodes import (
    ASTNode,
    BinaryOp,
    CellRef,
    FunctionCall,
    Literal,
    NamedRangeRef,
    RangeRef,
    UnaryOp,
)
from .tokenizer import parse_cell_addr_parts


def _parse_origin(origin_cell: str) -> tuple[int, int]:
    """Parse a plain address like "C14" into (row, col_index).

    Reuses the tokenizer's cell-address regex, so a malformed origin_cell
    (e.g. one that still carries a sheet prefix, or isn't a valid address
    at all) raises the same FormulaSyntaxError the tokenizer would.
    """
    _col_abs, col, _row_abs, row = parse_cell_addr_parts(origin_cell)
    return row, column_index_from_string(col)


def _cell_ref_rc(ref: CellRef, origin_row: int, origin_col: int) -> str:
    """The bare R..C.. part of a CellRef, with no sheet prefix."""
    row_part = str(ref.row) if ref.row_abs else f"[{ref.row - origin_row}]"
    col_part = str(ref.col_index) if ref.col_abs else f"[{ref.col_index - origin_col}]"
    return f"R{row_part}C{col_part}"


def _cell_ref_full(ref: CellRef, origin_row: int, origin_col: int, origin_sheet: str) -> str:
    bare = _cell_ref_rc(ref, origin_row, origin_col)
    if ref.sheet is not None and ref.sheet != origin_sheet:
        return f"{ref.sheet}!{bare}"
    return bare


def _range_ref_full(ref: RangeRef, origin_row: int, origin_col: int, origin_sheet: str) -> str:
    start_bare = _cell_ref_rc(ref.start, origin_row, origin_col)
    end_bare = _cell_ref_rc(ref.end, origin_row, origin_col)
    combined = f"{start_bare}:{end_bare}"
    if ref.sheet is not None and ref.sheet != origin_sheet:
        return f"{ref.sheet}!{combined}"
    return combined


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _format_string(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _format_literal(value: float | str | bool) -> str:
    # bool must be checked before float/int-like checks: in Python,
    # `isinstance(True, int)` is True, and our AST's Literal.value can
    # genuinely be a bool (not just something that behaves like one).
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        return _format_number(value)
    if isinstance(value, str):
        return _format_string(value)
    raise TypeError(f"unexpected literal type: {type(value)!r}")  # pragma: no cover


def _normalize(node: ASTNode, origin_row: int, origin_col: int, origin_sheet: str) -> str:
    if isinstance(node, Literal):
        return _format_literal(node.value)
    if isinstance(node, CellRef):
        return _cell_ref_full(node, origin_row, origin_col, origin_sheet)
    if isinstance(node, RangeRef):
        return _range_ref_full(node, origin_row, origin_col, origin_sheet)
    if isinstance(node, NamedRangeRef):
        return f"{node.sheet}!{node.name}" if node.sheet is not None else node.name
    if isinstance(node, FunctionCall):
        args = ",".join(_normalize(arg, origin_row, origin_col, origin_sheet) for arg in node.args)
        return f"{node.name.upper()}({args})"
    if isinstance(node, BinaryOp):
        left = _normalize(node.left, origin_row, origin_col, origin_sheet)
        right = _normalize(node.right, origin_row, origin_col, origin_sheet)
        return f"({left}{node.op}{right})"
    if isinstance(node, UnaryOp):
        operand = _normalize(node.operand, origin_row, origin_col, origin_sheet)
        return f"({node.op}{operand})"
    raise TypeError(f"unrecognized AST node type: {type(node)!r}")  # pragma: no cover


def normalize(ast_node: ASTNode, origin_cell: str, origin_sheet: str) -> str:
    """Normalize a formula AST into a position-independent R1C1-style string.

    `origin_cell` is the plain address the formula lives at, e.g. "C14"
    (no sheet prefix — it's a separate parameter, `origin_sheet`, so a
    reference's own sheet can be compared against it; see the
    CROSS-SHEET REFERENCES section in this module's docstring).
    """
    origin_row, origin_col = _parse_origin(origin_cell)
    return _normalize(ast_node, origin_row, origin_col, origin_sheet)
