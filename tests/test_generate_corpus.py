"""Validates the synthetic-corruption corpus generator against the real
Tier 0 rules, end-to-end through the actual pipeline (not the generator's
own bookkeeping). This is the meaningful check: does an injection the
generator claims will trigger a specific rule actually get flagged by
that rule when the workbook it produced is parsed for real, and does the
rule stay silent on everything the generator's own ground truth marked
"clean"?

Builds representative entries directly via the generator's functions
(not the full ~50-model sweep `main()` produces) and saves each to a
temp path, since `parse_workbook` reads from disk.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ssmlint.blocks import detect_blocks
from ssmlint.depgraph import build_graph
from ssmlint.parser import parse_workbook
from ssmlint.rules import InconsistentAnchoringRule, LiteralInBlockRule, RangeBoundaryRule, ReferenceToBlankRule

_GEN_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_corpus.py"
_spec = importlib.util.spec_from_file_location("generate_corpus", _GEN_PATH)
generate_corpus = importlib.util.module_from_spec(_spec)
sys.modules["generate_corpus"] = generate_corpus  # dataclass field-type resolution needs this registered
_spec.loader.exec_module(generate_corpus)


def _cells_by_state(ground_truth, state: str) -> set[str]:
    return {c.cell for c in ground_truth if c.state == state}


def _save(wb, tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.xlsx"
    wb.save(path)
    return path


def test_literal_injection_flags_injected_and_seed_cells(tmp_path: Path) -> None:
    wb, gt = generate_corpus._growth_chain_workbook(8, ("literal", 3))
    parsed = parse_workbook(_save(wb, tmp_path, "literal_case"))
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    issues = LiteralInBlockRule().evaluate(sheet_blocks)

    injected = _cells_by_state(gt, "injected_bug")
    known_intentional = _cells_by_state(gt, "known_intentional")
    clean = _cells_by_state(gt, "clean")
    assert injected == {"Model!F14"}
    assert known_intentional == {"Model!B14"}

    flagged = {i.cell for i in issues}
    assert flagged == injected | known_intentional  # exactly the injected bug + the (expected) seed
    assert flagged.isdisjoint(clean)  # never flags an actually-clean formula cell

    by_cell = {i.cell: i for i in issues}
    assert by_cell["Model!F14"].severity == "high"
    assert by_cell["Model!F14"].rule_id == "literal-in-formula-block"
    assert by_cell["Model!B14"].severity == "medium"  # bordered on one side only


def test_anchoring_injection_flags_exactly_the_injected_cell(tmp_path: Path) -> None:
    wb, gt = generate_corpus._growth_chain_workbook(8, ("anchoring", 3))
    parsed = parse_workbook(_save(wb, tmp_path, "anchoring_case"))
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    issues = InconsistentAnchoringRule().evaluate(sheet_blocks)

    injected = _cells_by_state(gt, "injected_bug")
    clean = _cells_by_state(gt, "clean")
    assert injected == {"Model!F14"}

    flagged = {i.cell for i in issues}
    assert flagged == injected
    assert flagged.isdisjoint(clean)

    issue = issues[0]
    assert issue.severity == "high"  # bordered by same-pattern blocks on both sides
    assert issue.rule_id == "inconsistent-anchoring"
    assert "col axis" in issue.explanation

    # literal-in-formula-block still fires on the seed (B14) -- expected, per the
    # Week 4 design note, and unrelated to this injection -- but never on F14 itself
    # (it's a formula, not a literal), and range-boundary never fires on this shape.
    literal_issues = LiteralInBlockRule().evaluate(sheet_blocks)
    assert {i.cell for i in literal_issues} == {"Model!B14"}
    assert RangeBoundaryRule().evaluate(sheet_blocks) == []


def test_range_boundary_injection_flags_exactly_the_injected_cell(tmp_path: Path) -> None:
    wb, gt = generate_corpus._trailing_sum_workbook(10, ("range_boundary", 3))
    parsed = parse_workbook(_save(wb, tmp_path, "range_boundary_case"))
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    issues = RangeBoundaryRule().evaluate(sheet_blocks)

    injected = _cells_by_state(gt, "injected_bug")
    clean = _cells_by_state(gt, "clean")
    assert injected == {"Rolling!G5"}

    flagged = {i.cell for i in issues}
    assert flagged == injected
    assert flagged.isdisjoint(clean)

    issue = issues[0]
    assert issue.severity == "high"
    assert issue.rule_id == "range-boundary-mismatch"

    assert InconsistentAnchoringRule().evaluate(sheet_blocks) == []


@pytest.mark.parametrize(
    "m, blank_idx, expected_severity",
    [
        (12, 4, "high"),  # clean members > affected members
        (6, 2, "medium"),  # affected members >= clean members
    ],
)
def test_reference_to_blank_injection_flags_exactly_the_affected_cells(
    tmp_path: Path, m: int, blank_idx: int, expected_severity: str
) -> None:
    wb, gt = generate_corpus._trailing_sum_workbook(m, ("reference_to_blank", blank_idx))
    parsed = parse_workbook(_save(wb, tmp_path, f"reference_to_blank_case_{m}_{blank_idx}"))
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)
    issues = ReferenceToBlankRule().evaluate(sheet_blocks, graph=graph)

    injected = _cells_by_state(gt, "injected_bug")
    clean = _cells_by_state(gt, "clean")
    assert injected  # every parametrized case has at least one affected member

    flagged = {i.cell for i in issues}
    assert flagged == injected
    assert flagged.isdisjoint(clean)
    assert all(i.severity == expected_severity for i in issues)
    assert all(i.rule_id == "reference-to-blank" for i in issues)


def test_clean_baselines_produce_zero_issues_from_the_rules_that_need_no_seed_tolerance(tmp_path: Path) -> None:
    """literal-in-formula-block is expected to still fire on the seed literal
    (see CONTRIBUTING.md's Week 4 design note) -- that's checked separately
    above. The other three rules should be completely silent on an
    uninjected clean model.
    """
    wb, _gt = generate_corpus._growth_chain_workbook(8, None)
    parsed = parse_workbook(_save(wb, tmp_path, "clean_growth_chain"))
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)

    assert InconsistentAnchoringRule().evaluate(sheet_blocks) == []
    assert RangeBoundaryRule().evaluate(sheet_blocks) == []
    assert ReferenceToBlankRule().evaluate(sheet_blocks, graph=graph) == []

    wb2, _gt2 = generate_corpus._trailing_sum_workbook(10, None)
    parsed2 = parse_workbook(_save(wb2, tmp_path, "clean_trailing_sum"))
    graph2 = build_graph(parsed2)
    sheet_blocks2 = detect_blocks(parsed2, graph2)

    assert LiteralInBlockRule().evaluate(sheet_blocks2) == []
    assert InconsistentAnchoringRule().evaluate(sheet_blocks2) == []
    assert RangeBoundaryRule().evaluate(sheet_blocks2) == []
    assert ReferenceToBlankRule().evaluate(sheet_blocks2, graph=graph2) == []


def test_full_corpus_sweep_ground_truth_is_internally_consistent() -> None:
    """Every generated entry's ground truth must have at least one cell,
    and every `injected_bug` cell must actually name one of the four
    rule_ids the rule engine knows about -- catches a typo in the
    generator's own rule_id strings before it ever reaches a corpus file.
    """
    known_rule_ids = {
        LiteralInBlockRule.rule_id,
        RangeBoundaryRule.rule_id,
        ReferenceToBlankRule.rule_id,
        InconsistentAnchoringRule.rule_id,
    }
    generate_corpus.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    entries = generate_corpus._growth_chain_entries() + generate_corpus._trailing_sum_entries()
    assert len(entries) == 46

    names = [e.name for e in entries]
    assert len(names) == len(set(names))  # no accidental filename collisions

    for entry in entries:
        assert entry.cells
        for cell_gt in entry.cells:
            if cell_gt.state == "injected_bug":
                assert cell_gt.rule_id in known_rule_ids
            else:
                assert cell_gt.rule_id is None
