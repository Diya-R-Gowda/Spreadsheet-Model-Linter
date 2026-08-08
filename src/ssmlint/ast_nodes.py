"""AST node types produced by formula_parser.parse_formula.

These are the structures the future R1C1 normalizer and rule engine will
consume directly, so field shapes here are deliberately stable and carry
more raw detail (e.g. independent column/row $-anchoring) than a plain
evaluator would need.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

from openpyxl.utils import column_index_from_string

_BARE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _quote_sheet_if_needed(sheet: str) -> str:
    if _BARE_NAME_RE.match(sheet):
        return sheet
    return "'" + sheet.replace("'", "''") + "'"


@dataclass(frozen=True)
class CellRef:
    """A single cell reference, e.g. `$B$4`, `B4`, or `Sheet2!A1`.

    `sheet` is None when the formula did not write a sheet prefix, meaning
    "same sheet as whatever cell hosts this formula" — that context lives
    in parser.py's CellRecord, not here, so it must be resolved by the
    caller. col/row $-anchoring is tracked per axis since R1C1
    normalization treats $B4 and B$4 differently.
    """

    col: str  # column letters, always uppercase, e.g. "B"
    row: int
    col_abs: bool = False
    row_abs: bool = False
    sheet: str | None = None

    @property
    def col_index(self) -> int:
        return column_index_from_string(self.col)

    @property
    def is_cross_sheet(self) -> bool:
        return self.sheet is not None

    def to_a1(self) -> str:
        col_part = f"${self.col}" if self.col_abs else self.col
        row_part = f"${self.row}" if self.row_abs else str(self.row)
        addr = f"{col_part}{row_part}"
        if self.sheet is not None:
            return f"{_quote_sheet_if_needed(self.sheet)}!{addr}"
        return addr


@dataclass(frozen=True)
class RangeRef:
    """A rectangular one-dimensional-or-2D range, e.g. `A1:B10` or `Sheet2!A1:A10`.

    Both endpoints share a sheet by construction (3-D and cross-sheet
    range endpoints are rejected by the parser, not represented here).
    """

    start: CellRef
    end: CellRef

    @property
    def sheet(self) -> str | None:
        return self.start.sheet

    def to_a1(self) -> str:
        start_addr = self.start.to_a1()
        col_part = f"${self.end.col}" if self.end.col_abs else self.end.col
        row_part = f"${self.end.row}" if self.end.row_abs else str(self.end.row)
        return f"{start_addr}:{col_part}{row_part}"


@dataclass(frozen=True)
class NamedRangeRef:
    """A reference to a defined name, e.g. `TaxRate` or `Sheet1!LocalName`."""

    name: str
    sheet: str | None = None


@dataclass(frozen=True)
class Literal:
    """A number, string, or boolean literal. Numbers are always float."""

    value: Union[float, str, bool]


@dataclass(frozen=True)
class FunctionCall:
    name: str
    args: tuple["ASTNode", ...]


@dataclass(frozen=True)
class BinaryOp:
    op: str  # one of + - * / ^ & = <> <= >= < >
    left: "ASTNode"
    right: "ASTNode"


@dataclass(frozen=True)
class UnaryOp:
    op: str  # "+" or "-"
    operand: "ASTNode"


ASTNode = Union[CellRef, RangeRef, NamedRangeRef, Literal, FunctionCall, BinaryOp, UnaryOp]
