"""Tests for rules.literal_in_block.LiteralInBlockRule — the first Tier 0 rule.

Covers the README's own C14:N14/H14 scenario end-to-end through the real
pipeline (parser -> tokenizer/AST -> r1c1 -> depgraph -> blocks -> rule),
using the revenue_row_with_hardcode.xlsx fixture built for the block
detector stage. Also covers the precision baseline: a clean block with no
literal must never fire.
"""

from __future__ import annotations

from pathlib import Path

from ssmlint.blocks import detect_blocks
from ssmlint.depgraph import build_graph
from ssmlint.parser import CellRecord, ParsedWorkbook, SheetRecord, parse_workbook
from ssmlint.rules import Issue, LiteralInBlockRule


def _cell(address: str, formula: str | None = None, value=None) -> CellRecord:
    return CellRecord(
        address=address, formula=formula, value=value, number_format="General", merged=False
    )


def _evaluate(sheets: list[SheetRecord]) -> list[Issue]:
    parsed = ParsedWorkbook(workbook="test.xlsx", sheets=sheets, named_ranges=[], skipped=[])
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    return LiteralInBlockRule().evaluate(sheet_blocks)


# ---------------------------------------------------------------------------
# End-to-end: README's own C14:N14/H14 scenario, through the real pipeline
# ---------------------------------------------------------------------------


def test_hardcode_fully_surrounded_by_one_pattern_is_flagged_high(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "revenue_row_with_hardcode.xlsx")
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    issues = LiteralInBlockRule().evaluate(sheet_blocks)

    h14 = next(i for i in issues if i.cell == "Model!H14")
    assert h14.rule_id == "literal-in-formula-block"
    assert h14.severity == "high"

    # The explanation must name the actual pattern and value, not a
    # generic template — assert on the real substance, not just presence.
    assert "4,500,000" in h14.explanation
    assert "(R[0]C[-1]*(1+R[1]C[-1]))" in h14.explanation
    assert "Model!C14:G14" in h14.explanation
    assert "Model!I14:M14" in h14.explanation

    # Suggested fix describes the pattern, never a ready-to-paste formula
    # for this specific cell (no "=" and no concrete H14-relative address).
    assert "(R[0]C[-1]*(1+R[1]C[-1]))" in h14.suggested_fix
    assert not h14.suggested_fix.startswith("=")

    # Exactly one Issue for H14, not two duplicates from the two blocks
    # that both report it as their non-conforming neighbor.
    assert sum(1 for i in issues if i.cell == "Model!H14") == 1


def test_seed_literal_adjacent_to_only_one_block_is_flagged_medium(fixtures_dir: Path) -> None:
    # B14 (the growth chain's seed value) borders only one block
    # (C14:G14) — real, structurally honest medium-severity signal, even
    # though this particular literal is actually intentional (the chain's
    # anchor). Tier 0 flags it anyway by design (see literal_in_block.py's
    # module docstring on NOT building an intentional-override heuristic).
    parsed = parse_workbook(fixtures_dir / "revenue_row_with_hardcode.xlsx")
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    issues = LiteralInBlockRule().evaluate(sheet_blocks)

    b14 = next(i for i in issues if i.cell == "Model!B14")
    assert b14.severity == "medium"
    assert "100,000" in b14.explanation
    assert "Model!C14:G14" in b14.explanation


def test_exactly_two_issues_for_the_revenue_row_fixture(fixtures_dir: Path) -> None:
    # B14 (seed) and H14 (hardcode) — nothing else in this fixture is a
    # literal touching a block.
    parsed = parse_workbook(fixtures_dir / "revenue_row_with_hardcode.xlsx")
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    issues = LiteralInBlockRule().evaluate(sheet_blocks)

    assert {i.cell for i in issues} == {"Model!B14", "Model!H14"}


# ---------------------------------------------------------------------------
# Precision baseline: a clean block with no literal must never fire
# ---------------------------------------------------------------------------


def test_clean_block_with_no_literal_produces_zero_issues() -> None:
    # A block spanning the entire row (A14:C14): no left neighbor exists
    # (A is the first column present), no right neighbor exists (C is the
    # last), so non_conforming is genuinely empty, not just non-literal.
    cells = [
        _cell("S1!A14", formula="=A13+1"),
        _cell("S1!B14", formula="=B13+1"),
        _cell("S1!C14", formula="=C13+1"),
    ]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])
    assert issues == []


def test_block_with_only_blank_or_different_formula_neighbors_produces_zero_issues() -> None:
    # Neighbors exist and are non-conforming, but neither is a literal —
    # this rule must stay silent (that's a different future rule's job).
    cells = [
        _cell("S1!B14"),  # blank neighbor
        _cell("S1!C14", formula="=B13+1"),
        _cell("S1!D14", formula="=C13+1"),
        _cell("S1!E14", formula="=D13+1"),
        _cell("S1!F14", formula="=SUM(A1:A20)"),  # unrelated-formula neighbor
    ]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])
    assert issues == []


# ---------------------------------------------------------------------------
# A literal between two DIFFERENT patterns: ambiguous boundary, not "high"
# ---------------------------------------------------------------------------


def test_literal_between_two_different_patterns_is_medium_not_high() -> None:
    cells = [
        _cell("S1!C14", formula="=B14*(1+B15)"),
        _cell("S1!D14", formula="=C14*(1+C15)"),
        _cell("S1!E14", formula="=D14*(1+D15)"),
        _cell("S1!F14", value=42),
        _cell("S1!G14", formula="=G13+1"),
        _cell("S1!H14", formula="=H13+1"),
        _cell("S1!I14", formula="=I13+1"),
        _cell("S1!B15", value=0.05),
        _cell("S1!C15", value=0.05),
        _cell("S1!D15", value=0.05),
    ]
    issues = _evaluate([SheetRecord(name="S1", cells=cells)])
    f14 = next(i for i in issues if i.cell == "S1!F14")
    assert f14.severity == "medium"
    assert "boundary" in f14.suggested_fix or "review" in f14.suggested_fix
