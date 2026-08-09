"""Tests for rules.range_boundary.RangeBoundaryRule — the second Tier 0 rule.

Covers the constructed off-by-one case with an accurate boundary-delta
explanation, the precision baseline (a clean block with matching ranges
fires nothing), and a genuinely different formula that must NOT be
claimed by this rule (that's literal_in_block's or a future rule's
territory). Also covers the real range_boundary_mismatch.xlsx fixture
end-to-end through the full pipeline.
"""

from __future__ import annotations

from pathlib import Path

from ssmlint.blocks import detect_blocks
from ssmlint.depgraph import build_graph
from ssmlint.parser import CellRecord, ParsedWorkbook, SheetRecord, parse_workbook
from ssmlint.rules import Issue
from ssmlint.rules.literal_in_block import LiteralInBlockRule
from ssmlint.rules.range_boundary import RangeBoundaryRule


def _cell(address: str, formula: str | None = None, value=None) -> CellRecord:
    return CellRecord(
        address=address, formula=formula, value=value, number_format="General", merged=False
    )


def _evaluate(sheets: list[SheetRecord]) -> list[Issue]:
    parsed = ParsedWorkbook(workbook="test.xlsx", sheets=sheets, named_ranges=[], skipped=[])
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    return RangeBoundaryRule().evaluate(sheet_blocks)


# ---------------------------------------------------------------------------
# Constructed off-by-one: correctly flagged, with an accurate boundary delta
# ---------------------------------------------------------------------------


def test_off_by_one_at_range_end_is_flagged_high_with_accurate_delta() -> None:
    cells = [
        _cell("S1!C14", formula="=SUM(B4:B10)"),
        _cell("S1!D14", formula="=SUM(C4:C10)"),
        _cell("S1!E14", formula="=SUM(D4:D10)"),
        _cell("S1!F14", formula="=SUM(E4:E9)"),  # off-by-one: ends one row short
        _cell("S1!G14", formula="=SUM(F4:F10)"),
        _cell("S1!H14", formula="=SUM(G4:G10)"),
        _cell("S1!I14", formula="=SUM(H4:H10)"),
    ]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])

    assert len(issues) == 1
    issue = issues[0]
    assert issue.cell == "S1!F14"
    assert issue.rule_id == "range-boundary-mismatch"
    assert issue.severity == "high"

    # The explanation must state the real cell counts, not a vague message.
    assert "covers 6 cells" in issue.explanation
    assert "cover 7 cells each" in issue.explanation
    assert "S1!C14:E14" in issue.explanation
    assert "S1!G14:I14" in issue.explanation
    assert "ends one row short" in issue.explanation

    # Suggested fix names the expected pattern, never ready-to-paste formula text.
    assert "SUM(R[-10]C[-1]:R[-4]C[-1])" in issue.suggested_fix
    assert not issue.suggested_fix.startswith("=")


def test_off_by_one_at_range_start_is_flagged_with_accurate_delta() -> None:
    cells = [
        _cell("S1!C14", formula="=SUM(B4:B10)"),
        _cell("S1!D14", formula="=SUM(C4:C10)"),
        _cell("S1!E14", formula="=SUM(D4:D10)"),
        _cell("S1!F14", formula="=SUM(E5:E10)"),  # off-by-one: starts one row late
        _cell("S1!G14", formula="=SUM(F4:F10)"),
        _cell("S1!H14", formula="=SUM(G4:G10)"),
        _cell("S1!I14", formula="=SUM(H4:H10)"),
    ]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])

    assert len(issues) == 1
    issue = issues[0]
    assert issue.cell == "S1!F14"
    assert "covers 6 cells" in issue.explanation
    assert "cover 7 cells each" in issue.explanation
    assert "starts one row short" in issue.explanation


def test_off_by_one_extending_the_range_is_flagged_as_too_long() -> None:
    cells = [
        _cell("S1!C14", formula="=SUM(B4:B10)"),
        _cell("S1!D14", formula="=SUM(C4:C10)"),
        _cell("S1!E14", formula="=SUM(D4:D10)"),
        _cell("S1!F14", formula="=SUM(E4:E11)"),  # off-by-one: one row too many
        _cell("S1!G14", formula="=SUM(F4:F10)"),
        _cell("S1!H14", formula="=SUM(G4:G10)"),
        _cell("S1!I14", formula="=SUM(H4:H10)"),
    ]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])

    assert len(issues) == 1
    issue = issues[0]
    assert "covers 8 cells" in issue.explanation
    assert "ends one row past" in issue.explanation


# ---------------------------------------------------------------------------
# Precision baseline: a clean block with matching ranges fires nothing
# ---------------------------------------------------------------------------


def test_clean_block_with_matching_ranges_produces_zero_issues() -> None:
    cells = [
        _cell("S1!C14", formula="=SUM(B4:B10)"),
        _cell("S1!D14", formula="=SUM(C4:C10)"),
        _cell("S1!E14", formula="=SUM(D4:D10)"),
    ]
    assert _evaluate([SheetRecord(name="S1", cells=cells)]) == []


# ---------------------------------------------------------------------------
# Scope guard: a genuinely different formula is NOT this rule's territory
# ---------------------------------------------------------------------------


def test_genuinely_different_formula_is_not_claimed_by_this_rule() -> None:
    cells = [
        _cell("S1!C14", formula="=SUM(B4:B10)"),
        _cell("S1!D14", formula="=SUM(C4:C10)"),
        _cell("S1!E14", formula="=SUM(D4:D10)"),
        _cell("S1!F14", formula="=AVERAGE(A1:A20)*2+COUNT(B1:B5)"),
        _cell("S1!G14", formula="=SUM(F4:F10)"),
        _cell("S1!H14", formula="=SUM(G4:G10)"),
        _cell("S1!I14", formula="=SUM(H4:H10)"),
    ]
    assert _evaluate([SheetRecord(name="S1", cells=cells)]) == []


def test_literal_neighbor_is_not_claimed_by_this_rule() -> None:
    # A hardcoded literal touching a SUM block is literal_in_block's
    # territory, not this rule's — confirm the two don't cross-fire.
    cells = [
        _cell("S1!C14", formula="=SUM(B4:B10)"),
        _cell("S1!D14", formula="=SUM(C4:C10)"),
        _cell("S1!E14", formula="=SUM(D4:D10)"),
        _cell("S1!F14", value=4500000),
        _cell("S1!G14", formula="=SUM(F4:F10)"),
        _cell("S1!H14", formula="=SUM(G4:G10)"),
        _cell("S1!I14", formula="=SUM(H4:H10)"),
    ]
    sheets = [SheetRecord(name="S1", cells=cells)]
    parsed = ParsedWorkbook(workbook="t.xlsx", sheets=sheets, named_ranges=[], skipped=[])
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)

    assert RangeBoundaryRule().evaluate(sheet_blocks) == []
    literal_issues = LiteralInBlockRule().evaluate(sheet_blocks)
    assert len(literal_issues) == 1
    assert literal_issues[0].cell == "S1!F14"


# ---------------------------------------------------------------------------
# Real-fixture integration: a rolling trailing-sum row with one off-by-one
# ---------------------------------------------------------------------------


def test_range_boundary_mismatch_fixture(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "range_boundary_mismatch.xlsx")
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    issues = RangeBoundaryRule().evaluate(sheet_blocks)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.cell == "Rolling!H5"
    assert issue.severity == "high"
    assert "covers 2 cells" in issue.explanation
    assert "cover 3 cells each" in issue.explanation
    assert "Rolling!D5:G5" in issue.explanation
    assert "Rolling!I5:M5" in issue.explanation
