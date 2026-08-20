"""Block detector — stage [5] in the architecture.

Clusters contiguous cells within a sheet that share an identical R1C1
pattern into "blocks", so later stages (the rule engine, the semantic
layer) can compare a cell against *its block's* pattern instead of every
other cell in the workbook pairwise.

AXIS: ROWS ONLY
-----------------
A block is a maximal run of *horizontally* adjacent cells in the same row
sharing one normalized R1C1 string. Column-wise clustering (grouping a
vertical run instead) is deliberately not implemented in this stage: the
README's own worked example (`C14:N14`, a revenue row with a month per
column) is row-wise, and it explicitly frames "row-labeled, time periods
across columns" as the primary shape of the financial models this project
targets. Column-wise runs (e.g. a driver repeated down a column) are a
real pattern too, but doubling the clustering logic for a shape that isn't
the documented primary case is out of scope here; if a future stage needs
it, `_row_groups` below is the only function that would need a column-wise
sibling — the run-detection, deviation-reporting, and near-miss logic are
all already axis-agnostic given a list of cells in address order.

CONTIGUITY
------------
"Adjacent" means strictly consecutive column indices in the same row. Any
break — a different pattern, a literal, a blank cell, or a cell whose
formula fails to parse — ends the run immediately. Two same-pattern runs
separated by so much as one non-conforming cell are reported as two
separate blocks, never silently merged back into one span with the
breaking cell absorbed inside it. (The README's illustrative JSON shows a
single `C14:N14` block with `H14` as an embedded deviation; that shape is
a later report-builder's presentation choice, built by merging this
stage's raw two-block output with a semantic judgment about whether the
gap matters — not what this stage itself produces.)

MINIMUM BLOCK SIZE: 3
------------------------
A run of exactly 2 same-pattern adjacent cells is weak evidence of an
intentional pattern — two formulas can easily coincide by chance (e.g. two
adjacent flat `=B2*1` unit-conversion cells) without implying the rest of
a row was ever meant to follow the same shape. A run of 3+ is a much
stronger signal that whoever built the model intended a repeating
structure. This is a judgment call, not a derived fact — flagged as such
in this stage's delivery summary rather than picked silently.

NON-CONFORMING NEIGHBORS
---------------------------
For each block, the cell immediately to its left and immediately to its
right (if either exists in that row) is classified into exactly one of
two buckets:
  - `non_conforming`: a blank cell, a pure literal, a cell whose formula
    doesn't parse, or a cell with a different R1C1 pattern that scored
    below the near-miss threshold against rapidfuzz.
  - `near_miss`: a cell with a different-but-similar R1C1 pattern (score
    >= the threshold via rapidfuzz.fuzz.ratio — character-level, order-
    sensitive, unlike token_sort_ratio, which would wrongly call
    `R[0]C[-1]-R[0]C[-2]` and `R[0]C[-2]-R[0]C[-1]` identical when they
    are not).
Only the immediate neighbor on each side is inspected, not the whole row:
by construction a block's boundary is exactly where non-conformance
starts, so anything beyond the immediate neighbor either belongs to a
different block already, or is itself that block's own immediate
neighbor — no cell needs to be inspected twice from two different blocks'
"whole row" scans.

BLANK/EMPTY, VIA THE DEPENDENCY GRAPH
-----------------------------------------
Blank-cell detection defers to `DependencyGraph.is_empty()` when the
neighbor address is a graph node, reusing depgraph.py's already-documented
single definition of "empty" instead of re-deriving it a third time in
this module. Not every cell is a graph node, though (depgraph.py only adds
nodes for cells with formulas, or cells some formula references — see its
own NODE SCOPE docstring section) — a stray literal nobody ever references
never becomes one. For those, emptiness is derived directly from the
CellRecord (`formula is None and value is None`), which is exactly what
depgraph.py itself would have computed had the cell been a node.

KNOWN LIMITATION: CATASTROPHIC PARSE FAILURES ARE INVISIBLE HERE
----------------------------------------------------------------------
parser.py's per-cell catch-all can fail on a cell so completely that it
never produces a CellRecord at all — only an entry in `ParsedWorkbook.
skipped`, with a bare node added later in depgraph.py so references to it
still resolve. This module walks `ParsedWorkbook.sheets[*].cells` to
build each row, so such a cell — having no CellRecord — never appears in
any row grouping and can never be reported as a block's non-conforming
neighbor, even though the dependency graph knows about it. This is judged
acceptable: it requires parser.py's catch-all to fail outright (in
practice, essentially never — openpyxl per-cell reads don't normally
raise), and splicing graph-only ghost nodes into row grouping by
re-parsing their address strings would add real complexity for a case no
fixture or test currently exercises.

FIXED: SELF-SHEET-QUALIFIED REFERENCES NO LONGER CAUSE FALSE SPLITS
------------------------------------------------------------------------
r1c1.normalize() used to have no way to know which sheet a formula's own
cell lived on, so an explicit self-sheet reference (`=Sheet1!B14*C14`
while physically living on Sheet1) normalized differently from the
equivalent implicit form (`=B14*C14`) even though Excel treats them
identically — this module inherited that as a real bug: two structurally
identical cells in the same row, one spelling out its own sheet name,
would be seen as a pattern break and could split what should be one block
into two. `r1c1.normalize()` now takes the origin's sheet as a required
parameter and collapses that case (see r1c1.py's own docstring); this
module passes each cell's own sheet (split from its address) as that
parameter, so the false split no longer occurs. Regression-tested in
test_blocks.py.

ROW_LABEL / COL_LABELS: A HEURISTIC, NOT A GENERAL SOLUTION (added 2026-08-20)
------------------------------------------------------------------------
Closes the confirmed gap that `Block`/`BlockExample` never carried the
README's `row_label`/`col_labels` fields (they were always `None`).
`parser.py` already captures every literal cell's text -- the gap was
only ever that nothing linked a header cell to a `Block`. Two fixed-
offset heuristics (`_capture_row_label`/`_capture_col_labels`, backed by
a per-sheet `(row, col) -> CellRecord` grid built once in `detect_blocks`
via `_full_grid`), honestly limited, not general: `row_label` only ever
looks at column A of the block's own row (a common financial-model
convention, but never any other column, never a multi-row-tall merged
label); `col_labels` only ever looks exactly one row above each spanned
column (never further up, never a multi-row header stack). Either can
legitimately be `None`/all-`None` when a real sheet doesn't follow this
layout -- that's the honest, expected result of a heuristic, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openpyxl.utils import column_index_from_string, get_column_letter
from rapidfuzz import fuzz

from .depgraph import DependencyGraph
from .errors import FormulaError
from .formula_parser import parse_formula
from .parser import CellRecord, ParsedWorkbook
from .r1c1 import normalize
from .tokenizer import parse_cell_addr_parts

DEFAULT_MIN_BLOCK_SIZE = 3
DEFAULT_NEAR_MISS_THRESHOLD = 85.0


@dataclass
class NonConformingCell:
    """A neighbor cell that breaks a block's pattern outright (not a near-miss)."""

    cell: str
    kind: str  # "literal" | "blank" | "different_formula" | "unparseable"
    value: Any = None
    formula_r1c1: str | None = None
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "kind": self.kind,
            "value": self.value,
            "formula_r1c1": self.formula_r1c1,
            "parse_error": self.parse_error,
        }


@dataclass
class NearMissCell:
    """A neighbor cell whose pattern is similar-but-not-identical to the block's."""

    cell: str
    formula_r1c1: str
    similarity: float  # rapidfuzz fuzz.ratio score, 0-100

    def to_dict(self) -> dict[str, Any]:
        return {"cell": self.cell, "formula_r1c1": self.formula_r1c1, "similarity": self.similarity}


@dataclass
class Block:
    sheet: str
    row: int
    span: str  # bare, e.g. "C14:N14" (no sheet prefix)
    pattern: str  # the shared R1C1 string, exactly as r1c1.normalize() returns it
    cells: list[str] = field(default_factory=list)  # sheet-qualified, in column order
    non_conforming: list[NonConformingCell] = field(default_factory=list)
    near_misses: list[NearMissCell] = field(default_factory=list)
    row_label: str | None = None  # heuristic: literal text at column A of this block's row, if any
    col_labels: list[str | None] = field(default_factory=list)  # heuristic: literal text one row above, per column

    def to_dict(self) -> dict[str, Any]:
        return {
            "sheet": self.sheet,
            "row": self.row,
            "span": self.span,
            "pattern": self.pattern,
            "cells": self.cells,
            "non_conforming": [c.to_dict() for c in self.non_conforming],
            "near_misses": [c.to_dict() for c in self.near_misses],
            "row_label": self.row_label,
            "col_labels": self.col_labels,
        }


@dataclass
class SheetBlocks:
    sheet: str
    blocks: list[Block] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"sheet": self.sheet, "blocks": [b.to_dict() for b in self.blocks]}


@dataclass
class _RowCell:
    """One cell in a row, with its already-attempted R1C1 pattern (if any)."""

    address: str  # sheet-qualified
    col_index: int
    record: CellRecord
    pattern: str | None  # None if blank, pure literal, or unparseable
    parse_error: str | None  # set only when it had a formula that failed to parse


def _split_address(address: str) -> tuple[str, str]:
    sheet, plain = address.split("!", 1)
    return sheet, plain


def _plain_to_col_row(plain_addr: str) -> tuple[int, int]:
    _col_abs, col, _row_abs, row = parse_cell_addr_parts(plain_addr)
    return column_index_from_string(col), row


def _row_groups(sheet_name: str, cells: list[CellRecord]) -> dict[int, list[_RowCell]]:
    """Group a sheet's cells by row number, each row sorted by column index."""
    rows: dict[int, list[_RowCell]] = {}
    for record in cells:
        record_sheet, plain = _split_address(record.address)
        col_index, row = _plain_to_col_row(plain)

        pattern: str | None = None
        parse_error: str | None = None
        if record.formula is not None:
            try:
                ast_node = parse_formula(record.formula)
                pattern = normalize(ast_node, plain, record_sheet)
            except FormulaError as exc:
                parse_error = str(exc)

        rows.setdefault(row, []).append(
            _RowCell(
                address=record.address,
                col_index=col_index,
                record=record,
                pattern=pattern,
                parse_error=parse_error,
            )
        )

    for row_cells in rows.values():
        row_cells.sort(key=lambda rc: rc.col_index)
    return rows


def _full_grid(cells: list[CellRecord]) -> dict[tuple[int, int], CellRecord]:
    """(row, col_index) -> CellRecord for every cell parser.py captured on this
    sheet -- built once per sheet, reused by every block on it for
    row_label/col_labels capture below.
    """
    grid: dict[tuple[int, int], CellRecord] = {}
    for record in cells:
        _sheet, plain = _split_address(record.address)
        col_index, row = _plain_to_col_row(plain)
        grid[(row, col_index)] = record
    return grid


def _literal_text(record: CellRecord | None) -> str | None:
    if record is not None and isinstance(record.value, str) and record.value.strip():
        return record.value
    return None


def _capture_row_label(grid: dict[tuple[int, int], CellRecord], row: int) -> str | None:
    """HEURISTIC, not a general solution: the literal text at column A
    (col_index == 1) of the block's own row, if any -- a common financial-
    model convention (line-item names in column A). Real, stated
    limitation: never looks in any other column, never handles a multi-
    row-tall merged label cell.
    """
    return _literal_text(grid.get((row, 1)))


def _capture_col_labels(
    grid: dict[tuple[int, int], CellRecord], row: int, start_col: int, end_col: int
) -> list[str | None]:
    """HEURISTIC, not a general solution: for each column the block spans,
    the literal text exactly one row above, if any -- `None` per-column
    where nothing is found (never collapsed to a single all-or-nothing
    result). Real, stated limitation: never searches further than one row
    up, never handles a multi-row header stack.
    """
    return [_literal_text(grid.get((row - 1, col))) for col in range(start_col, end_col + 1)]


def _find_runs(row_cells: list[_RowCell]) -> list[list[_RowCell]]:
    """Maximal runs of column-consecutive cells sharing one non-None pattern."""
    runs: list[list[_RowCell]] = []
    current: list[_RowCell] = []

    for rc in row_cells:
        if (
            rc.pattern is not None
            and current
            and current[-1].pattern == rc.pattern
            and rc.col_index == current[-1].col_index + 1
        ):
            current.append(rc)
        else:
            if current:
                runs.append(current)
            current = [rc] if rc.pattern is not None else []
    if current:
        runs.append(current)
    return runs


def _classify_neighbor(
    neighbor: _RowCell,
    block_pattern: str,
    graph: DependencyGraph,
    near_miss_threshold: float,
) -> tuple[NonConformingCell | None, NearMissCell | None]:
    record = neighbor.record

    if neighbor.parse_error is not None:
        return (
            NonConformingCell(cell=neighbor.address, kind="unparseable", parse_error=neighbor.parse_error),
            None,
        )

    if neighbor.pattern is not None:
        score = fuzz.ratio(block_pattern, neighbor.pattern)
        if score >= near_miss_threshold:
            return None, NearMissCell(cell=neighbor.address, formula_r1c1=neighbor.pattern, similarity=score)
        return (
            NonConformingCell(cell=neighbor.address, kind="different_formula", formula_r1c1=neighbor.pattern),
            None,
        )

    is_blank = (
        graph.is_empty(neighbor.address)
        if neighbor.address in graph
        else (record.formula is None and record.value is None)
    )
    if is_blank:
        return NonConformingCell(cell=neighbor.address, kind="blank"), None
    return NonConformingCell(cell=neighbor.address, kind="literal", value=record.value), None


def _build_block(
    sheet_name: str,
    row: int,
    run: list[_RowCell],
    row_index: dict[int, _RowCell],
    graph: DependencyGraph,
    near_miss_threshold: float,
    grid: dict[tuple[int, int], CellRecord],
) -> Block:
    pattern = run[0].pattern
    assert pattern is not None  # runs are only ever built from non-None patterns
    start_col, end_col = run[0].col_index, run[-1].col_index

    block = Block(
        sheet=sheet_name,
        row=row,
        span=f"{get_column_letter(start_col)}{row}:{get_column_letter(end_col)}{row}",
        pattern=pattern,
        cells=[rc.address for rc in run],
        row_label=_capture_row_label(grid, row),
        col_labels=_capture_col_labels(grid, row, start_col, end_col),
    )

    for neighbor_col in (start_col - 1, end_col + 1):
        neighbor = row_index.get(neighbor_col)
        if neighbor is None:
            continue
        non_conforming, near_miss = _classify_neighbor(neighbor, pattern, graph, near_miss_threshold)
        if non_conforming is not None:
            block.non_conforming.append(non_conforming)
        if near_miss is not None:
            block.near_misses.append(near_miss)

    return block


def detect_blocks(
    parsed: ParsedWorkbook,
    graph: DependencyGraph,
    *,
    min_block_size: int = DEFAULT_MIN_BLOCK_SIZE,
    near_miss_threshold: float = DEFAULT_NEAR_MISS_THRESHOLD,
) -> list[SheetBlocks]:
    """Cluster each sheet's rows into blocks of identically-patterned formulas.

    See this module's docstring for the axis choice (rows only), the
    minimum-block-size judgment call, and how non-conforming/near-miss
    neighbors are classified.
    """
    result: list[SheetBlocks] = []

    for sheet in parsed.sheets:
        rows = _row_groups(sheet.name, sheet.cells)
        grid = _full_grid(sheet.cells)
        sheet_blocks = SheetBlocks(sheet=sheet.name)

        for row, row_cells in sorted(rows.items()):
            row_index = {rc.col_index: rc for rc in row_cells}
            for run in _find_runs(row_cells):
                if len(run) < min_block_size:
                    continue
                sheet_blocks.blocks.append(
                    _build_block(sheet.name, row, run, row_index, graph, near_miss_threshold, grid)
                )

        result.append(sheet_blocks)

    return result
