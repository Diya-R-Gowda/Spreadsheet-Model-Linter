"""Tests for depgraph.build_graph / DependencyGraph.

Most cases build a ParsedWorkbook directly (rather than a real .xlsx) for
precise control over the scenario — circular references, dangling named
ranges, and cells that never made it into ParsedWorkbook.sheets at all are
awkward to coax out of a real workbook but trivial to construct by hand.
The fixture-based tests at the bottom cover the real integration path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ssmlint.depgraph import CyclicDependencyError, DependencyGraph, build_graph
from ssmlint.parser import (
    CellRecord,
    NamedRangeRecord,
    ParsedWorkbook,
    SheetRecord,
    SkippedItem,
    parse_workbook,
)


def _cell(address: str, formula: str | None = None, value=None) -> CellRecord:
    return CellRecord(
        address=address, formula=formula, value=value, number_format="General", merged=False
    )


def _workbook(sheets: list[SheetRecord], named_ranges=None, skipped=None) -> ParsedWorkbook:
    return ParsedWorkbook(
        workbook="test.xlsx",
        sheets=sheets,
        named_ranges=named_ranges or [],
        skipped=skipped or [],
    )


# ---------------------------------------------------------------------------
# Simple same-sheet precedent/dependent chain
# ---------------------------------------------------------------------------


def test_simple_chain() -> None:
    sheet = SheetRecord(
        name="S1",
        cells=[
            _cell("S1!A1", value=1),
            _cell("S1!B1", formula="=A1+1"),
            _cell("S1!C1", formula="=B1+1"),
        ],
    )
    graph = build_graph(_workbook([sheet]))

    assert graph.precedents("S1!B1") == ["S1!A1"]
    assert graph.precedents("S1!C1") == ["S1!B1"]
    assert graph.dependents("S1!A1") == ["S1!B1"]
    assert graph.dependents("S1!B1") == ["S1!C1"]

    order = graph.topological_order()
    assert order.index("S1!A1") < order.index("S1!B1") < order.index("S1!C1")


# ---------------------------------------------------------------------------
# Cross-sheet edges
# ---------------------------------------------------------------------------


def test_cross_sheet_edge() -> None:
    assumptions = SheetRecord(name="Assumptions", cells=[_cell("Assumptions!A1", value=0.21)])
    model = SheetRecord(
        name="Model", cells=[_cell("Model!B1", formula="=Assumptions!A1*2")]
    )
    graph = build_graph(_workbook([assumptions, model]))

    assert graph.precedents("Model!B1") == ["Assumptions!A1"]
    assert graph.dependents("Assumptions!A1") == ["Model!B1"]


# ---------------------------------------------------------------------------
# Range expansion: SUM(B4:B10) must produce 7 edges, not 1
# ---------------------------------------------------------------------------


def test_range_expands_to_individual_cell_edges() -> None:
    cells = [_cell("S1!C4", formula="=SUM(B4:B10)")]
    graph = build_graph(_workbook([SheetRecord(name="S1", cells=cells)]))

    precedents = graph.precedents("S1!C4")
    assert len(precedents) == 7
    assert set(precedents) == {f"S1!B{row}" for row in range(4, 11)}


# ---------------------------------------------------------------------------
# Named range resolution
# ---------------------------------------------------------------------------


def test_named_range_resolves_to_underlying_cell() -> None:
    assumptions = SheetRecord(name="Assumptions", cells=[_cell("Assumptions!B1", value=0.21)])
    model = SheetRecord(name="Model", cells=[_cell("Model!C1", formula="=TaxRate*2")])
    named_ranges = [NamedRangeRecord(name="TaxRate", refers_to="Assumptions!$B$1", scope="workbook")]

    graph = build_graph(_workbook([assumptions, model], named_ranges=named_ranges))

    assert graph.precedents("Model!C1") == ["Assumptions!B1"]
    assert graph.unresolved_named_ranges == ()


def test_sheet_scoped_named_range_shadows_workbook_scoped() -> None:
    sheet = SheetRecord(
        name="Model",
        cells=[_cell("Model!A1", value=1), _cell("Model!B1", value=2), _cell("Model!C1", formula="=Local")],
    )
    named_ranges = [
        NamedRangeRecord(name="Local", refers_to="Model!$A$1", scope="workbook"),
        NamedRangeRecord(name="Local", refers_to="Model!$B$1", scope="Model"),
    ]
    graph = build_graph(_workbook([sheet], named_ranges=named_ranges))

    # The sheet-scoped definition (-> B1) must win over the workbook one (-> A1).
    assert graph.precedents("Model!C1") == ["Model!B1"]


def test_dangling_named_range_is_logged_not_crashed() -> None:
    sheet = SheetRecord(name="S1", cells=[_cell("S1!A1", formula="=Ghost*2")])
    graph = build_graph(_workbook([sheet]))

    assert graph.precedents("S1!A1") == []
    assert len(graph.unresolved_named_ranges) == 1
    assert "Ghost" in graph.unresolved_named_ranges[0]


def test_dynamic_named_range_is_logged_not_crashed_and_siblings_still_resolve() -> None:
    sheet = SheetRecord(
        name="S1",
        cells=[_cell("S1!A1", value=1), _cell("S1!B1", formula="=Dynamic+A1")],
    )
    named_ranges = [NamedRangeRecord(name="Dynamic", refers_to="OFFSET(A1,0,0)", scope="workbook")]
    graph = build_graph(_workbook([sheet], named_ranges=named_ranges))

    # Dynamic can't be resolved to a plain ref, but the ordinary A1 ref in
    # the same formula must still produce its edge.
    assert graph.precedents("S1!B1") == ["S1!A1"]
    assert len(graph.unresolved_named_ranges) == 1


# ---------------------------------------------------------------------------
# Circular references
# ---------------------------------------------------------------------------


def test_circular_reference_detected() -> None:
    sheet = SheetRecord(
        name="S1",
        cells=[_cell("S1!A1", formula="=B1+1"), _cell("S1!B1", formula="=A1+1")],
    )
    graph = build_graph(_workbook([sheet]))

    cycles = graph.find_cycles()
    assert len(cycles) == 1
    assert set(cycles[0]) == {"S1!A1", "S1!B1"}


def test_topological_order_raises_cyclic_dependency_error() -> None:
    sheet = SheetRecord(
        name="S1",
        cells=[_cell("S1!A1", formula="=B1+1"), _cell("S1!B1", formula="=A1+1")],
    )
    graph = build_graph(_workbook([sheet]))

    with pytest.raises(CyclicDependencyError) as exc_info:
        graph.topological_order()
    assert len(exc_info.value.cycles) == 1


# ---------------------------------------------------------------------------
# References to genuinely empty cells
# ---------------------------------------------------------------------------


def test_reference_to_cell_missing_entirely_is_empty() -> None:
    # C1 is referenced but has no CellRecord at all (outside the used range).
    sheet = SheetRecord(name="S1", cells=[_cell("S1!A1", value=1), _cell("S1!B1", formula="=A1+C1")])
    graph = build_graph(_workbook([sheet]))

    assert "S1!C1" in graph
    assert graph.is_empty("S1!C1") is True
    assert graph.is_empty("S1!A1") is False


def test_reference_to_cell_present_but_blank_is_empty() -> None:
    # C1 has a CellRecord (it's within the used range) but no formula/value.
    sheet = SheetRecord(
        name="S1",
        cells=[
            _cell("S1!A1", value=1),
            _cell("S1!B1", formula="=A1+C1"),
            _cell("S1!C1"),  # formula=None, value=None
        ],
    )
    graph = build_graph(_workbook([sheet]))

    assert graph.is_empty("S1!C1") is True


# ---------------------------------------------------------------------------
# Skipped/unparseable cells still appear as nodes, with no outgoing edges
# ---------------------------------------------------------------------------


def test_formula_rejected_by_formula_parser_still_becomes_a_node() -> None:
    # $-anchored full-column range: parser.py wouldn't flag this (it's not
    # an array formula/LET/structured-ref), but formula_parser rejects it.
    sheet = SheetRecord(name="S1", cells=[_cell("S1!A1", formula="=SUM($A:$A)")])
    graph = build_graph(_workbook([sheet]))

    assert "S1!A1" in graph
    assert graph.precedents("S1!A1") == []
    assert graph.parse_error("S1!A1") is not None


def test_catastrophic_parser_failure_cell_still_becomes_a_node() -> None:
    # Simulates parser.py's per-cell catch-all: the address is in
    # `skipped` but there is NO matching CellRecord anywhere in `sheets`.
    sheet = SheetRecord(name="S1", cells=[_cell("S1!B1", formula="=A1+1")])
    skipped = [SkippedItem(address="S1!A1", reason="unparseable cell: ValueError('boom')")]
    graph = build_graph(_workbook([sheet], skipped=skipped))

    assert "S1!A1" in graph
    assert graph.precedents("S1!A1") == []
    assert graph.parse_error("S1!A1") == "unparseable cell: ValueError('boom')"
    # We have no reliable value info for it either, by the same signal.
    assert graph.is_empty("S1!A1") is True


# ---------------------------------------------------------------------------
# Real-fixture integration checks
# ---------------------------------------------------------------------------


def test_clean_model_fixture_chain(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "clean_model.xlsx")
    graph = build_graph(parsed)

    assert graph.precedents("Model!C2") == ["Model!B2"]
    assert graph.precedents("Model!D2") == ["Model!C2"]
    assert graph.precedents("Model!E2") == ["Model!D2"]


def test_merged_cells_fixture(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "merged_cells.xlsx")
    graph = build_graph(parsed)

    assert set(graph.precedents("Budget!D3")) == {"Budget!B3", "Budget!C3"}


def test_named_range_cross_sheet_fixture(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "named_range_cross_sheet.xlsx")
    graph = build_graph(parsed)

    assert set(graph.precedents("Model!B2")) == {"Model!B1", "Assumptions!B1"}
    assert set(graph.precedents("Model!B3")) == {"Model!B1", "Model!B2"}


def test_skip_cases_fixture(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "skip_cases.xlsx")
    graph = build_graph(parsed)

    # B1's array formula is syntactically fine (SUM of two 3-cell ranges
    # multiplied together) even though parser.py flagged it for an
    # unrelated reason (the openpyxl ArrayFormula wrapper) — 3+3 edges.
    assert len(graph.precedents("Edge Cases!B1")) == 6

    # B2 (LET) and B3 (structured table ref) are genuinely rejected by
    # formula_parser: nodes exist, no outgoing edges, error recorded.
    assert graph.precedents("Edge Cases!B2") == []
    assert graph.parse_error("Edge Cases!B2") is not None
    assert graph.precedents("Edge Cases!B3") == []
    assert graph.parse_error("Edge Cases!B3") is not None

    # B4 is a normal formula and resolves normally.
    assert graph.precedents("Edge Cases!B4") == ["Edge Cases!A4"]


def test_circular_reference_fixture(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "circular_reference.xlsx")
    graph = build_graph(parsed)

    cycles = graph.find_cycles()
    assert len(cycles) == 1
    assert set(cycles[0]) == {"Circular!A1", "Circular!B1"}
    with pytest.raises(CyclicDependencyError):
        graph.topological_order()


def test_blank_reference_fixture(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "blank_reference.xlsx")
    graph = build_graph(parsed)

    assert set(graph.precedents("Blank Refs!B1")) == {"Blank Refs!A1", "Blank Refs!C1"}
    assert graph.is_empty("Blank Refs!A1") is False
    assert graph.is_empty("Blank Refs!C1") is True
