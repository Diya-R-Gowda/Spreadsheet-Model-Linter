"""Tests for rules.reference_to_blank.ReferenceToBlankRule — the third Tier 0 rule.

Most cases use hand-built ParsedWorkbook constructions for precise
control over block shape (the three investigation scenarios: genuine gap
flagged, isolated reference never flagged, uniform block reference never
flagged), plus the severity split, a member with more than one blank
precedent, and the required graph argument. The fixture-based test at
the bottom covers the real integration path end-to-end (parser ->
tokenizer -> r1c1 -> depgraph -> blocks -> rule) via
reference_to_blank_gap.xlsx.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ssmlint.blocks import SheetBlocks
from ssmlint.depgraph import build_graph
from ssmlint.parser import CellRecord, ParsedWorkbook, SheetRecord, parse_workbook
from ssmlint.blocks import detect_blocks
from ssmlint.rules import Issue
from ssmlint.rules.reference_to_blank import ReferenceToBlankRule


def _cell(address: str, formula: str | None = None, value=None) -> CellRecord:
    return CellRecord(
        address=address, formula=formula, value=value, number_format="General", merged=False
    )


def _evaluate(sheets: list[SheetRecord]) -> list[Issue]:
    parsed = ParsedWorkbook(workbook="test.xlsx", sheets=sheets, named_ranges=[], skipped=[])
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    return ReferenceToBlankRule().evaluate(sheet_blocks, graph)


# ---------------------------------------------------------------------------
# Scenario A: a genuine gap in a data block -- must be flagged
# ---------------------------------------------------------------------------


def test_genuine_gap_in_block_data_is_flagged() -> None:
    cells = [
        _cell("S1!B4", value=100), _cell("S1!C4", value=110), _cell("S1!D4", value=120),
        _cell("S1!E4", value=130), _cell("S1!F4", value=140),
        _cell("S1!G4"),  # blank -- a genuine gap in otherwise-populated monthly data
        _cell("S1!H4", value=160), _cell("S1!I4", value=170), _cell("S1!J4", value=180),
        _cell("S1!D5", formula="=SUM(B4:D4)"),
        _cell("S1!E5", formula="=SUM(C4:E4)"),
        _cell("S1!F5", formula="=SUM(D4:F4)"),
        _cell("S1!G5", formula="=SUM(E4:G4)"),  # references the blank G4
        _cell("S1!H5", formula="=SUM(F4:H4)"),  # also references the blank G4
        _cell("S1!I5", formula="=SUM(G4:I4)"),  # also references the blank G4
        _cell("S1!J5", formula="=SUM(H4:J4)"),
    ]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])

    assert {i.cell for i in issues} == {"S1!G5", "S1!H5", "S1!I5"}
    for issue in issues:
        assert issue.rule_id == "reference-to-blank"
        assert issue.severity == "high"  # 4 clean vs 3 affected -> clean is the majority
        assert "S1!G4" in issue.explanation
        assert "4 other cells" in issue.explanation
        assert "3 of 7 members" in issue.explanation
        assert not issue.suggested_fix.startswith("=")


# ---------------------------------------------------------------------------
# Scenario B: an isolated one-off reference to a blank spacer -- never flagged
# ---------------------------------------------------------------------------


def test_isolated_non_block_member_reference_is_never_flagged() -> None:
    cells = [
        _cell("S1!A1", value="Notes:"),
        _cell("S1!B1"),  # deliberately blank spacer
        _cell("S1!C1", formula="=B1"),  # a one-off formula, not part of any block
    ]
    assert _evaluate([SheetRecord(name="S1", cells=cells)]) == []


# ---------------------------------------------------------------------------
# Scenario C: every block member uniformly references the same blank cell
# -- no minority/majority split, never flagged
# ---------------------------------------------------------------------------


def test_uniform_blank_reference_across_whole_block_is_never_flagged() -> None:
    cells = [
        _cell("S1!B14", value=100),
        _cell("S1!H15"),  # blank shared driver, e.g. an unfilled discount rate
        _cell("S1!C14", formula="=B14*$H$15"),
        _cell("S1!D14", formula="=C14*$H$15"),
        _cell("S1!E14", formula="=D14*$H$15"),
    ]
    assert _evaluate([SheetRecord(name="S1", cells=cells)]) == []


# ---------------------------------------------------------------------------
# Severity: medium when affected members are not outnumbered by clean ones
# ---------------------------------------------------------------------------


def test_severity_is_medium_when_affected_ties_or_outnumbers_clean() -> None:
    # 4-member block: only 1 clean (C5), 3 affected (D5, E5, F5) -- affected
    # outnumbers clean, so this must NOT get the "high" (clean-majority) severity.
    cells = [
        _cell("S1!B4", value=100), _cell("S1!C4", value=110),
        _cell("S1!D4"),  # blank
        _cell("S1!E4"),  # blank
        _cell("S1!F4", value=140),
        _cell("S1!C5", formula="=SUM(B4:C4)"),
        _cell("S1!D5", formula="=SUM(C4:D4)"),  # references blank D4
        _cell("S1!E5", formula="=SUM(D4:E4)"),  # references blank D4 and E4
        _cell("S1!F5", formula="=SUM(E4:F4)"),  # references blank E4
    ]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])
    assert {i.cell for i in issues} == {"S1!D5", "S1!E5", "S1!F5"}
    for issue in issues:
        assert issue.severity == "medium"


# ---------------------------------------------------------------------------
# A single formula referencing more than one blank precedent -> one Issue
# ---------------------------------------------------------------------------


def test_member_with_multiple_blank_precedents_produces_one_issue() -> None:
    cells = [
        _cell("S1!B4", value=100), _cell("S1!C4", value=110),
        _cell("S1!D4"),  # blank
        _cell("S1!E4"),  # blank
        _cell("S1!F4", value=140), _cell("S1!G4", value=150), _cell("S1!H4", value=160),
        _cell("S1!C5", formula="=SUM(B4:C4)"),
        _cell("S1!D5", formula="=SUM(C4:D4)"),
        _cell("S1!E5", formula="=SUM(D4:E4)"),  # references BOTH blank D4 and blank E4
        _cell("S1!F5", formula="=SUM(E4:F4)"),
        _cell("S1!G5", formula="=SUM(F4:G4)"),
        _cell("S1!H5", formula="=SUM(G4:H4)"),
    ]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])
    e5_issues = [i for i in issues if i.cell == "S1!E5"]
    assert len(e5_issues) == 1
    assert "S1!D4" in e5_issues[0].explanation
    assert "S1!E4" in e5_issues[0].explanation


# ---------------------------------------------------------------------------
# Missing graph: fail loudly, not silently
# ---------------------------------------------------------------------------


def test_missing_graph_raises_value_error() -> None:
    with pytest.raises(ValueError, match="requires the dependency graph"):
        ReferenceToBlankRule().evaluate([SheetBlocks(sheet="S1", blocks=[])])


# ---------------------------------------------------------------------------
# Real-fixture integration: a rolling trailing-sum row with one blank month
# ---------------------------------------------------------------------------


def test_reference_to_blank_gap_fixture(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "reference_to_blank_gap.xlsx")
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    issues = ReferenceToBlankRule().evaluate(sheet_blocks, graph)

    assert {i.cell for i in issues} == {"Rolling!F5", "Rolling!G5", "Rolling!H5"}
    for issue in issues:
        assert issue.rule_id == "reference-to-blank"
        assert issue.severity == "high"  # 7 clean vs 3 affected
        assert "Rolling!F4" in issue.explanation
        assert "7 other cells" in issue.explanation
        assert "3 of 10 members" in issue.explanation
