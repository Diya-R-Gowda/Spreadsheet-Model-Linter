"""Tests for rules.inconsistent_anchoring.InconsistentAnchoringRule — the
fourth and final named Tier 0 rule.

Most cases use hand-built ParsedWorkbook constructions for precise
control over which axis drifts. Covers all three anchoring-drift shapes
found during investigation (full drift, row-axis-only, col-axis-only),
the precision baseline (a clean block with consistent anchoring fires
nothing), an explicit proof that this rule and range_boundary never
double-fire on the same deviation, and a real-fixture integration test.
"""

from __future__ import annotations

from pathlib import Path

from ssmlint.blocks import detect_blocks
from ssmlint.depgraph import build_graph
from ssmlint.parser import CellRecord, ParsedWorkbook, SheetRecord, parse_workbook
from ssmlint.rules import Issue
from ssmlint.rules.inconsistent_anchoring import InconsistentAnchoringRule
from ssmlint.rules.range_boundary import RangeBoundaryRule


def _cell(address: str, formula: str | None = None, value=None) -> CellRecord:
    return CellRecord(
        address=address, formula=formula, value=value, number_format="General", merged=False
    )


def _evaluate(sheets: list[SheetRecord]) -> list[Issue]:
    parsed = ParsedWorkbook(workbook="test.xlsx", sheets=sheets, named_ranges=[], skipped=[])
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    return InconsistentAnchoringRule().evaluate(sheet_blocks)


_GROWTH_CHAIN_BASE = [
    _cell("S1!B1", value=0.05),
    _cell("S1!C14", formula="=B14*(1+$B$1)"),
    _cell("S1!D14", formula="=C14*(1+$B$1)"),
    _cell("S1!E14", formula="=D14*(1+$B$1)"),
]
_GROWTH_CHAIN_TAIL = [
    _cell("S1!G14", formula="=F14*(1+$B$1)"),
    _cell("S1!H14", formula="=G14*(1+$B$1)"),
    _cell("S1!I14", formula="=H14*(1+$B$1)"),
]


# ---------------------------------------------------------------------------
# All three anchoring-drift shapes found during investigation
# ---------------------------------------------------------------------------


def test_full_anchoring_drift_flags_both_axes() -> None:
    cells = [*_GROWTH_CHAIN_BASE, _cell("S1!F14", formula="=E14*(1+B1)"), *_GROWTH_CHAIN_TAIL]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])

    assert len(issues) == 1
    issue = issues[0]
    assert issue.cell == "S1!F14"
    assert issue.rule_id == "inconsistent-anchoring"
    assert issue.severity == "high"
    assert "row axis" in issue.explanation
    assert "col axis" in issue.explanation
    assert "reference to B1" in issue.explanation
    assert "$B$1" in issue.explanation
    assert not issue.suggested_fix.startswith("=")


def test_row_axis_drift_names_row_axis_specifically() -> None:
    cells = [*_GROWTH_CHAIN_BASE, _cell("S1!F14", formula="=E14*(1+$B1)"), *_GROWTH_CHAIN_TAIL]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])

    assert len(issues) == 1
    issue = issues[0]
    assert issue.cell == "S1!F14"
    assert "the row axis of its reference to B1 is relative ($B1) instead of absolute ($B$1)" in issue.explanation
    assert "col axis" not in issue.explanation  # only the row axis actually drifted


def test_col_axis_drift_names_col_axis_specifically() -> None:
    cells = [*_GROWTH_CHAIN_BASE, _cell("S1!F14", formula="=E14*(1+B$1)"), *_GROWTH_CHAIN_TAIL]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])

    assert len(issues) == 1
    issue = issues[0]
    assert issue.cell == "S1!F14"
    assert "the col axis of its reference to B1 is relative (B$1) instead of absolute ($B$1)" in issue.explanation
    assert "row axis" not in issue.explanation  # only the col axis actually drifted


# ---------------------------------------------------------------------------
# Precision baseline: consistent anchoring throughout -> zero issues
# ---------------------------------------------------------------------------


def test_clean_block_with_consistent_anchoring_produces_zero_issues() -> None:
    cells = [*_GROWTH_CHAIN_BASE]
    assert _evaluate([SheetRecord(name="S1", cells=cells)]) == []


def test_fully_relative_block_with_no_anchoring_produces_zero_issues() -> None:
    # No $ anywhere, but everything is internally consistent -- nothing to flag.
    cells = [
        _cell("S1!C14", formula="=B14*(1+B15)"),
        _cell("S1!D14", formula="=C14*(1+C15)"),
        _cell("S1!E14", formula="=D14*(1+D15)"),
        _cell("S1!B15", value=0.05),
        _cell("S1!C15", value=0.05),
        _cell("S1!D15", value=0.05),
    ]
    assert _evaluate([SheetRecord(name="S1", cells=cells)]) == []


# ---------------------------------------------------------------------------
# Explicit proof: this rule and range_boundary never double-fire on the
# same deviation
# ---------------------------------------------------------------------------


def test_pure_anchoring_deviation_does_not_fire_range_boundary() -> None:
    cells = [*_GROWTH_CHAIN_BASE, _cell("S1!F14", formula="=E14*(1+B$1)"), *_GROWTH_CHAIN_TAIL]
    sheets = [SheetRecord(name="S1", cells=cells)]
    parsed = ParsedWorkbook(workbook="t.xlsx", sheets=sheets, named_ranges=[], skipped=[])
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)

    anchoring_issues = InconsistentAnchoringRule().evaluate(sheet_blocks)
    range_issues = RangeBoundaryRule().evaluate(sheet_blocks)

    assert len(anchoring_issues) == 1
    assert anchoring_issues[0].cell == "S1!F14"
    assert range_issues == []


def test_pure_range_boundary_deviation_does_not_fire_anchoring() -> None:
    cells = [
        _cell("S1!C14", formula="=SUM(B4:B10)"),
        _cell("S1!D14", formula="=SUM(C4:C10)"),
        _cell("S1!E14", formula="=SUM(D4:D10)"),
        _cell("S1!F14", formula="=SUM(E4:E9)"),  # off-by-one, not an anchoring issue
        _cell("S1!G14", formula="=SUM(F4:F10)"),
        _cell("S1!H14", formula="=SUM(G4:G10)"),
        _cell("S1!I14", formula="=SUM(H4:H10)"),
    ]
    sheets = [SheetRecord(name="S1", cells=cells)]
    parsed = ParsedWorkbook(workbook="t.xlsx", sheets=sheets, named_ranges=[], skipped=[])
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)

    anchoring_issues = InconsistentAnchoringRule().evaluate(sheet_blocks)
    range_issues = RangeBoundaryRule().evaluate(sheet_blocks)

    assert anchoring_issues == []
    assert len(range_issues) == 1
    assert range_issues[0].cell == "S1!F14"


# ---------------------------------------------------------------------------
# Real-fixture integration: growth-rate row with a dropped column anchor
# ---------------------------------------------------------------------------


def test_inconsistent_anchoring_fixture(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "inconsistent_anchoring.xlsx")
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    issues = InconsistentAnchoringRule().evaluate(sheet_blocks)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.cell == "Model!H14"
    assert issue.severity == "high"
    assert "col axis" in issue.explanation
    assert "reference to B1" in issue.explanation
    assert "Model!C14:G14" in issue.explanation
    assert "Model!I14:M14" in issue.explanation

    # And confirm range_boundary doesn't also fire on this fixture.
    assert RangeBoundaryRule().evaluate(sheet_blocks) == []
