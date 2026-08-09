"""Tests for blocks.detect_blocks.

Most cases build a ParsedWorkbook directly (rather than a real .xlsx) for
precise control over row layout — gaps, exact hardcode placement, and
near-miss formulas are awkward to coax out of a real workbook but trivial
to construct by hand. The fixture-based test at the bottom covers the
real integration path with the README's own C14:N14-style example.
"""

from __future__ import annotations

from pathlib import Path

from ssmlint.blocks import DEFAULT_NEAR_MISS_THRESHOLD, detect_blocks
from ssmlint.depgraph import build_graph
from ssmlint.parser import CellRecord, ParsedWorkbook, SheetRecord, parse_workbook


def _cell(address: str, formula: str | None = None, value=None) -> CellRecord:
    return CellRecord(
        address=address, formula=formula, value=value, number_format="General", merged=False
    )


def _workbook(sheets: list[SheetRecord]) -> ParsedWorkbook:
    return ParsedWorkbook(workbook="test.xlsx", sheets=sheets, named_ranges=[], skipped=[])


def _detect(sheets: list[SheetRecord], **kwargs):
    parsed = _workbook(sheets)
    graph = build_graph(parsed)
    return detect_blocks(parsed, graph, **kwargs)


# ---------------------------------------------------------------------------
# Clean multi-cell row block
# ---------------------------------------------------------------------------


def test_clean_row_block() -> None:
    # B14 seed, C14:F14 all "=<prev>*(1+<prev-row-15>)" — 4 identical-shape cells.
    cells = [
        _cell("S1!B14", value=100000),
        _cell("S1!C14", formula="=B14*(1+B15)"),
        _cell("S1!D14", formula="=C14*(1+C15)"),
        _cell("S1!E14", formula="=D14*(1+D15)"),
        _cell("S1!F14", formula="=E14*(1+E15)"),
        _cell("S1!B15", value=0.05),
        _cell("S1!C15", value=0.05),
        _cell("S1!D15", value=0.05),
        _cell("S1!E15", value=0.05),
    ]
    sheet_blocks = _detect([SheetRecord(name="S1", cells=cells)])
    assert len(sheet_blocks) == 1
    blocks = sheet_blocks[0].blocks

    assert len(blocks) == 1
    block = blocks[0]
    assert block.sheet == "S1"
    assert block.row == 14
    assert block.span == "C14:F14"
    assert block.pattern == "(R[0]C[-1]*(1+R[1]C[-1]))"
    assert block.cells == ["S1!C14", "S1!D14", "S1!E14", "S1!F14"]
    # Left neighbor B14 is a literal -> non-conforming, not silently ignored.
    assert len(block.non_conforming) == 1
    assert block.non_conforming[0].cell == "S1!B14"
    assert block.non_conforming[0].kind == "literal"
    assert block.non_conforming[0].value == 100000
    assert block.near_misses == []


# ---------------------------------------------------------------------------
# Block broken by a gap: must yield two separate blocks, not one merged span
# ---------------------------------------------------------------------------


def test_gap_produces_two_separate_blocks_not_one() -> None:
    cells = [
        _cell("S1!B14", value=100000),
        _cell("S1!C14", formula="=B14*(1+B15)"),
        _cell("S1!D14", formula="=C14*(1+C15)"),
        _cell("S1!E14", formula="=D14*(1+D15)"),
        # Gap: F14 is a totally different, unrelated formula.
        _cell("S1!F14", formula="=SUM(B14:E14)"),
        _cell("S1!G14", formula="=F14*(1+F15)"),
        _cell("S1!H14", formula="=G14*(1+G15)"),
        _cell("S1!I14", formula="=H14*(1+H15)"),
        _cell("S1!B15", value=0.05),
        _cell("S1!C15", value=0.05),
        _cell("S1!D15", value=0.05),
        _cell("S1!F15", value=0.05),
        _cell("S1!G15", value=0.05),
        _cell("S1!H15", value=0.05),
    ]
    sheet_blocks = _detect([SheetRecord(name="S1", cells=cells)])
    blocks = sheet_blocks[0].blocks

    assert len(blocks) == 2
    spans = {b.span for b in blocks}
    assert spans == {"C14:E14", "G14:I14"}

    # F14 (the gap cell) is not absorbed into either block's `cells`.
    for block in blocks:
        assert "S1!F14" not in block.cells

    left_block = next(b for b in blocks if b.span == "C14:E14")
    right_block = next(b for b in blocks if b.span == "G14:I14")

    # F14 is reported as the non-conforming right-neighbor of the left block...
    assert any(nc.cell == "S1!F14" for nc in left_block.non_conforming)
    # ...and as the non-conforming left-neighbor of the right block.
    assert any(nc.cell == "S1!F14" for nc in right_block.non_conforming)


# ---------------------------------------------------------------------------
# One hardcoded literal breaking an otherwise-uniform row, in the middle
# ---------------------------------------------------------------------------


def test_literal_in_middle_is_reported_not_absorbed_or_ignored() -> None:
    cells = [
        _cell("S1!B14", value=100000),
        _cell("S1!C14", formula="=B14*(1+B15)"),
        _cell("S1!D14", formula="=C14*(1+C15)"),
        _cell("S1!E14", formula="=D14*(1+D15)"),
        _cell("S1!F14", value=4500000),  # hardcoded — breaks the pattern
        _cell("S1!G14", formula="=F14*(1+F15)"),
        _cell("S1!H14", formula="=G14*(1+G15)"),
        _cell("S1!I14", formula="=H14*(1+H15)"),
        _cell("S1!B15", value=0.05),
        _cell("S1!C15", value=0.05),
        _cell("S1!D15", value=0.05),
        _cell("S1!F15", value=0.05),
        _cell("S1!G15", value=0.05),
        _cell("S1!H15", value=0.05),
    ]
    sheet_blocks = _detect([SheetRecord(name="S1", cells=cells)])
    blocks = sheet_blocks[0].blocks

    # Not silently absorbed into one big C14:I14 block.
    assert len(blocks) == 2
    spans = {b.span for b in blocks}
    assert spans == {"C14:E14", "G14:I14"}
    for block in blocks:
        assert "S1!F14" not in block.cells

    # Not silently ignored either — F14 shows up as a literal deviation on both sides.
    left_block = next(b for b in blocks if b.span == "C14:E14")
    right_block = next(b for b in blocks if b.span == "G14:I14")

    left_hit = next(nc for nc in left_block.non_conforming if nc.cell == "S1!F14")
    assert left_hit.kind == "literal"
    assert left_hit.value == 4500000

    right_hit = next(nc for nc in right_block.non_conforming if nc.cell == "S1!F14")
    assert right_hit.kind == "literal"
    assert right_hit.value == 4500000


# ---------------------------------------------------------------------------
# Near-miss: a genuinely similar-but-different formula, with a real score
# ---------------------------------------------------------------------------


def test_near_miss_reports_real_rapidfuzz_score() -> None:
    cells = [
        # No B14 seed cell here on purpose: the block's left edge (C14) has
        # no left neighbor at all, so the only neighbor in play is F14 on
        # the right, keeping this test isolated to the near-miss behavior.
        _cell("S1!C14", formula="=B14*(1+B15)"),
        _cell("S1!D14", formula="=C14*(1+C15)"),
        _cell("S1!E14", formula="=D14*(1+D15)"),
        # Near miss: same shape, one extra "+1" term tacked on.
        _cell("S1!F14", formula="=E14*(1+E15)+1"),
        _cell("S1!B15", value=0.05),
        _cell("S1!C15", value=0.05),
        _cell("S1!D15", value=0.05),
    ]
    sheet_blocks = _detect([SheetRecord(name="S1", cells=cells)])
    block = sheet_blocks[0].blocks[0]

    assert block.non_conforming == []
    assert len(block.near_misses) == 1
    near_miss = block.near_misses[0]
    assert near_miss.cell == "S1!F14"
    assert near_miss.formula_r1c1 == "((R[0]C[-1]*(1+R[1]C[-1]))+1)"

    # The score is a real rapidfuzz computation, not a hardcoded stand-in:
    # independently recompute it here and check it matches exactly.
    from rapidfuzz import fuzz

    expected_score = fuzz.ratio(block.pattern, near_miss.formula_r1c1)
    assert near_miss.similarity == expected_score
    assert near_miss.similarity >= DEFAULT_NEAR_MISS_THRESHOLD
    assert near_miss.similarity < 100


def test_dissimilar_formula_is_non_conforming_not_near_miss() -> None:
    cells = [
        # No B14 seed cell: keeps the only neighbor in play F14, on the right.
        _cell("S1!C14", formula="=B14*(1+B15)"),
        _cell("S1!D14", formula="=C14*(1+C15)"),
        _cell("S1!E14", formula="=D14*(1+D15)"),
        # Totally unrelated shape, not a small tweak.
        _cell("S1!F14", formula="=SUM(A1:A20)/COUNT(A1:A20)"),
    ]
    sheet_blocks = _detect([SheetRecord(name="S1", cells=cells)])
    block = sheet_blocks[0].blocks[0]

    assert block.near_misses == []
    assert len(block.non_conforming) == 1
    assert block.non_conforming[0].cell == "S1!F14"
    assert block.non_conforming[0].kind == "different_formula"


# ---------------------------------------------------------------------------
# Minimum block size boundary: N-1 cells does not form a block, N does
# ---------------------------------------------------------------------------


def test_minimum_block_size_boundary() -> None:
    # 2 identical-pattern cells: below the default minimum of 3.
    two_cells = [
        _cell("S1!B14", value=100000),
        _cell("S1!C14", formula="=B14*(1+B15)"),
        _cell("S1!D14", formula="=C14*(1+C15)"),
        _cell("S1!B15", value=0.05),
        _cell("S1!C15", value=0.05),
    ]
    sheet_blocks = _detect([SheetRecord(name="S1", cells=two_cells)])
    assert sheet_blocks[0].blocks == []

    # 3 identical-pattern cells: meets the default minimum.
    three_cells = two_cells + [
        _cell("S1!E14", formula="=D14*(1+D15)"),
        _cell("S1!D15", value=0.05),
    ]
    sheet_blocks = _detect([SheetRecord(name="S1", cells=three_cells)])
    assert len(sheet_blocks[0].blocks) == 1
    assert sheet_blocks[0].blocks[0].span == "C14:E14"


def test_minimum_block_size_is_configurable() -> None:
    two_cells = [
        _cell("S1!B14", value=100000),
        _cell("S1!C14", formula="=B14*(1+B15)"),
        _cell("S1!D14", formula="=C14*(1+C15)"),
        _cell("S1!B15", value=0.05),
        _cell("S1!C15", value=0.05),
    ]
    sheet_blocks = _detect([SheetRecord(name="S1", cells=two_cells)], min_block_size=2)
    assert len(sheet_blocks[0].blocks) == 1
    assert sheet_blocks[0].blocks[0].span == "C14:D14"


# ---------------------------------------------------------------------------
# Blank and unparseable neighbors
# ---------------------------------------------------------------------------


def test_blank_neighbor_is_reported_as_blank() -> None:
    cells = [
        _cell("S1!C14", formula="=B14*(1+B15)"),
        _cell("S1!D14", formula="=C14*(1+C15)"),
        _cell("S1!E14", formula="=D14*(1+D15)"),
        _cell("S1!B14"),  # present but blank: formula=None, value=None
        _cell("S1!B15", value=0.05),
        _cell("S1!C15", value=0.05),
        _cell("S1!D15", value=0.05),
    ]
    sheet_blocks = _detect([SheetRecord(name="S1", cells=cells)])
    block = sheet_blocks[0].blocks[0]

    left_hit = next(nc for nc in block.non_conforming if nc.cell == "S1!B14")
    assert left_hit.kind == "blank"


def test_unparseable_neighbor_is_reported_with_parse_error() -> None:
    cells = [
        _cell("S1!C14", formula="=B14*(1+B15)"),
        _cell("S1!D14", formula="=C14*(1+C15)"),
        _cell("S1!E14", formula="=D14*(1+D15)"),
        _cell("S1!F14", formula="=LET(x, 5, x*2)"),  # rejected by formula_parser
        _cell("S1!B14", value=100000),
        _cell("S1!B15", value=0.05),
        _cell("S1!C15", value=0.05),
        _cell("S1!D15", value=0.05),
    ]
    sheet_blocks = _detect([SheetRecord(name="S1", cells=cells)])
    block = sheet_blocks[0].blocks[0]

    right_hit = next(nc for nc in block.non_conforming if nc.cell == "S1!F14")
    assert right_hit.kind == "unparseable"
    assert right_hit.parse_error is not None


# ---------------------------------------------------------------------------
# Regression: explicit self-sheet references no longer cause a false split
# (r1c1.py bugfix — normalize() now takes origin_sheet and collapses
# "=Model!E14..." on Model to the same pattern as the implicit form)
# ---------------------------------------------------------------------------


def test_explicit_self_sheet_reference_does_not_cause_a_false_split() -> None:
    # C14:E14 and G14:I14 are structurally identical to F14 in every way
    # except F14 spells out its own host sheet ("Model!") on both refs.
    # Before the r1c1.py fix this used to split into two 3-cell blocks
    # with F14 excluded from both (reproduced during the stage-5 audit).
    cells = [
        _cell("Model!C14", formula="=B14*(1+B15)"),
        _cell("Model!D14", formula="=C14*(1+C15)"),
        _cell("Model!E14", formula="=D14*(1+D15)"),
        _cell("Model!F14", formula="=Model!E14*(1+Model!E15)"),
        _cell("Model!G14", formula="=F14*(1+F15)"),
        _cell("Model!H14", formula="=G14*(1+G15)"),
        _cell("Model!I14", formula="=H14*(1+H15)"),
    ]
    sheet_blocks = _detect([SheetRecord(name="Model", cells=cells)])
    blocks = sheet_blocks[0].blocks

    assert len(blocks) == 1
    block = blocks[0]
    assert block.span == "C14:I14"
    assert block.cells == [
        "Model!C14", "Model!D14", "Model!E14", "Model!F14", "Model!G14", "Model!H14", "Model!I14",
    ]
    assert block.non_conforming == []
    assert block.near_misses == []


# ---------------------------------------------------------------------------
# Real-fixture integration: README's own C14:N14-style example
# ---------------------------------------------------------------------------


def test_revenue_row_fixture(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "revenue_row_with_hardcode.xlsx")
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)

    assert len(sheet_blocks) == 1
    blocks = sheet_blocks[0].blocks

    # H14 is hardcoded, splitting the C14:M14 run into two blocks: C14:G14, I14:M14.
    assert len(blocks) == 2
    spans = {b.span for b in blocks}
    assert spans == {"C14:G14", "I14:M14"}

    for block in blocks:
        assert "Model!H14" not in block.cells
        assert block.pattern == "(R[0]C[-1]*(1+R[1]C[-1]))"

    left_block = next(b for b in blocks if b.span == "C14:G14")
    right_block = next(b for b in blocks if b.span == "I14:M14")

    h14_from_left = next(nc for nc in left_block.non_conforming if nc.cell == "Model!H14")
    assert h14_from_left.kind == "literal"
    assert h14_from_left.value == 4500000

    h14_from_right = next(nc for nc in right_block.non_conforming if nc.cell == "Model!H14")
    assert h14_from_right.kind == "literal"
    assert h14_from_right.value == 4500000
