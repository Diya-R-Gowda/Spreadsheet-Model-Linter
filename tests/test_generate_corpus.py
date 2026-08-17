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
from ssmlint.evaluation import DEFAULT_RULES, evaluate_rule
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


def test_category_total_shape_raises_zero_issues_from_every_rule(tmp_path: Path) -> None:
    """The subtotal shape's Total-row cells are all ground-truthed
    "subtotal" -- a legitimate, intentionally-different aggregate, never
    expected to be flagged. Checked at both cat_count/col_count extremes
    and a mid-range point, not just one arbitrary size.
    """
    for cat_count, col_count in ((2, 3), (6, 14), (4, 8)):
        wb, gt = generate_corpus._category_total_workbook(cat_count, col_count)
        parsed = parse_workbook(_save(wb, tmp_path, f"category_total_{cat_count}_{col_count}"))
        graph = build_graph(parsed)
        sheet_blocks = detect_blocks(parsed, graph)

        subtotal_cells = _cells_by_state(gt, "subtotal")
        assert len(subtotal_cells) == col_count

        for rule_id, rule in DEFAULT_RULES.items():
            issues = evaluate_rule(rule, sheet_blocks, graph)
            assert issues == [], f"{rule_id} unexpectedly fired on category_total({cat_count}, {col_count}): {issues}"


def test_ambiguous_growth_chain_injection_at_n4_edge_still_gets_flagged(tmp_path: Path) -> None:
    """n=3/edge produces no block at all (verified: only 2 clean cells
    remain, below blocks.py's min_block_size=3) -- n=4/edge is the
    shortest length that actually triggers a flag, so that's what the
    "ambiguous" (unknown-label) entries use. Tier 0 has no concept of
    ambiguity, so it must still flag these deterministically, exactly as
    it would an injected_bug at the same shape -- only the ground-truth
    state differs.
    """
    wb, gt = generate_corpus._growth_chain_workbook(4, ("literal_ambiguous", 0))
    parsed = parse_workbook(_save(wb, tmp_path, "ambiguous_literal_edge_left"))
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)

    ambiguous_cells = _cells_by_state(gt, "ambiguous")
    assert ambiguous_cells == {"Model!C14"}

    issues = LiteralInBlockRule().evaluate(sheet_blocks)
    assert {i.cell for i in issues} == {"Model!C14"}

    wb2, gt2 = generate_corpus._growth_chain_workbook(4, ("anchoring_ambiguous", 0))
    parsed2 = parse_workbook(_save(wb2, tmp_path, "ambiguous_anchoring_edge_left"))
    graph2 = build_graph(parsed2)
    sheet_blocks2 = detect_blocks(parsed2, graph2)

    assert _cells_by_state(gt2, "ambiguous") == {"Model!C14"}
    anchoring_issues = InconsistentAnchoringRule().evaluate(sheet_blocks2)
    assert {i.cell for i in anchoring_issues} == {"Model!C14"}


def test_variance_shape_literal_and_reference_to_blank_injections(tmp_path: Path) -> None:
    """The variance-row shape (`=Actual-Budget`) is a genuinely different
    two-precedent R1C1 pattern from growth-chain and trailing-sum. Its
    literal-in-formula-block and reference-to-blank injections should
    fire exactly the same way those rules already fire on the other two
    shapes -- confirms the injections are correctly wired on new host
    formulas, not just structurally plausible.
    """
    wb, gt = generate_corpus._variance_workbook(8, ("literal", 3))
    parsed = parse_workbook(_save(wb, tmp_path, "variance_literal_interior"))
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)

    injected = _cells_by_state(gt, "injected_bug")
    clean = _cells_by_state(gt, "clean")
    assert injected == {"Variance!E6"}

    issues = LiteralInBlockRule().evaluate(sheet_blocks)
    flagged = {i.cell for i in issues}
    assert flagged == injected
    assert flagged.isdisjoint(clean)
    assert issues[0].severity == "high"  # interior injection, bordered on both sides

    wb2, gt2 = generate_corpus._variance_workbook(8, ("reference_to_blank", 4))
    parsed2 = parse_workbook(_save(wb2, tmp_path, "variance_reference_to_blank"))
    graph2 = build_graph(parsed2)
    sheet_blocks2 = detect_blocks(parsed2, graph2)

    injected2 = _cells_by_state(gt2, "injected_bug")
    assert injected2 == {"Variance!F6"}
    blank_issues = ReferenceToBlankRule().evaluate(sheet_blocks2, graph=graph2)
    assert {i.cell for i in blank_issues} == injected2
    assert blank_issues[0].severity == "high"  # exactly 1 of 8 affected -- clean majority


def test_ambiguous_reference_to_blank_near_tie_still_gets_flagged(tmp_path: Path) -> None:
    """One real entry from the near-tie sweep, spot-checked directly: an
    affected/clean split close to 50/50 is still weaker evidence, not zero
    evidence -- Tier 0 has no ambiguity concept and flags it the same as
    any other reference-to-blank case.
    """
    wb, gt = generate_corpus._trailing_sum_workbook(7, ("reference_to_blank", 1))
    ambiguous_gt = [
        generate_corpus.CellGroundTruth(cell=c.cell, state="ambiguous", rule_id=c.rule_id, note=c.note)
        if c.state == "injected_bug"
        else c
        for c in gt
    ]
    parsed = parse_workbook(_save(wb, tmp_path, "ambiguous_reference_to_blank_near_tie"))
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)

    ambiguous_cells = _cells_by_state(ambiguous_gt, "ambiguous")
    assert ambiguous_cells  # this (m, blank_idx) combo does affect some members

    issues = ReferenceToBlankRule().evaluate(sheet_blocks, graph=graph)
    assert {i.cell for i in issues} == ambiguous_cells


def test_new_shape_entries_have_internally_consistent_ground_truth() -> None:
    """Same consistency check as the original two shapes, extended to the
    three new entry-generating functions: every injected_bug/ambiguous
    cell names a real rule_id, every subtotal/clean cell has none, and no
    entry name collides with an existing one.
    """
    known_rule_ids = {
        LiteralInBlockRule.rule_id,
        RangeBoundaryRule.rule_id,
        ReferenceToBlankRule.rule_id,
        InconsistentAnchoringRule.rule_id,
    }
    generate_corpus.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    new_entries = (
        generate_corpus._ambiguous_growth_chain_entries()
        + generate_corpus._category_total_entries()
        + generate_corpus._variance_entries()
        + generate_corpus._ambiguous_reference_to_blank_entries()
    )
    assert new_entries

    names = [e.name for e in new_entries]
    assert len(names) == len(set(names))

    for entry in new_entries:
        assert entry.cells
        for cell_gt in entry.cells:
            if cell_gt.state in ("injected_bug", "ambiguous"):
                assert cell_gt.rule_id in known_rule_ids
            else:
                assert cell_gt.rule_id is None


def test_cross_block_isolation_corrupting_one_block_never_flags_the_other(tmp_path: Path) -> None:
    """Two independent row blocks on one sheet (row 14: growth-chain with
    a literal injection; row 30: a clean trailing-sum block, far apart).
    Confirms corrupting the first block raises exactly the expected
    issues there, AND raises zero issues anywhere in the second block --
    not just that the injected bug is caught, but that the clean block is
    completely untouched by every rule.
    """
    wb, gt = generate_corpus._cross_block_isolation_workbook()
    parsed = parse_workbook(_save(wb, tmp_path, "cross_block_isolation_case"))
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)

    injected = _cells_by_state(gt, "injected_bug")
    known_intentional = _cells_by_state(gt, "known_intentional")
    clean = _cells_by_state(gt, "clean")
    assert injected == {"Combined!F14"}
    assert known_intentional == {"Combined!B14"}
    row_30_clean_cells = {c for c in clean if c.split("!")[1].lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ") == "30"}
    assert row_30_clean_cells == {
        "Combined!D30", "Combined!E30", "Combined!F30", "Combined!G30",
        "Combined!H30", "Combined!I30", "Combined!J30", "Combined!K30",
    }

    literal_issues = LiteralInBlockRule().evaluate(sheet_blocks)
    anchoring_issues = InconsistentAnchoringRule().evaluate(sheet_blocks)
    range_issues = RangeBoundaryRule().evaluate(sheet_blocks)
    blank_issues = ReferenceToBlankRule().evaluate(sheet_blocks, graph=graph)

    # The corrupted block (row 14): exactly the injected bug plus the
    # expected seed-literal issue -- same shape already proven in
    # test_literal_injection_flags_injected_and_seed_cells.
    assert {i.cell for i in literal_issues} == {"Combined!B14", "Combined!F14"}
    by_cell = {i.cell: i for i in literal_issues}
    assert by_cell["Combined!F14"].severity == "high"
    assert by_cell["Combined!B14"].severity == "medium"

    # The clean block (row 30): zero issues from every rule, not merely
    # zero on its own block members -- the strong claim the prompt asked
    # for, verified directly rather than inferred from the corrupted
    # block's correctness.
    row_30_issues_from_any_rule = [
        i for i in (literal_issues + anchoring_issues + range_issues + blank_issues)
        if i.cell.split("!")[1].lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ") == "30"
    ]
    assert row_30_issues_from_any_rule == []
    assert anchoring_issues == []
    assert range_issues == []
    assert blank_issues == []
