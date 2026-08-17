"""Tests for ssmlint.evaluation -- the Week 4 scoring logic itself.

Most cases use small hand-constructed ground-truth dicts + fake Issue
lists for precise control over each outcome (true_positive, false_
positive, known_intentional, false_negative), per the standing "verify
in isolation before trusting the full-corpus run" discipline. One
integration test runs the real harness against the real corpus and
asserts the exact numbers already verified by hand via
scripts/run_evaluation.py.
"""

from __future__ import annotations

from pathlib import Path

from ssmlint.evaluation import (
    INTERNAL_CONSISTENCY_CAVEAT,
    PRECISION_AT_K,
    count_false_negatives,
    format_report_text,
    is_clean_baseline,
    run_evaluation,
    score_rule_on_entry,
)
from ssmlint.rules import Issue

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def _issue(cell: str, severity: str = "medium", rule_id: str = "literal-in-formula-block") -> Issue:
    return Issue(cell=cell, severity=severity, rule_id=rule_id, explanation="x", suggested_fix="y")


def _gt(*cells: tuple[str, str, str | None]) -> dict:
    """`cells` is (cell, state, rule_id_or_None) tuples."""
    return {"cells": [{"cell": c, "state": s, "rule_id": r} for c, s, r in cells]}


# ---------------------------------------------------------------------------
# score_rule_on_entry: true_positive / false_positive / known_intentional
# ---------------------------------------------------------------------------


def test_true_positive_when_state_and_rule_id_match() -> None:
    gt = _gt(("Model!F14", "injected_bug", "literal-in-formula-block"))
    issues = [_issue("Model!F14", "high", "literal-in-formula-block")]
    scored = score_rule_on_entry("e", issues, gt, "literal-in-formula-block")

    assert len(scored) == 1
    assert scored[0].outcome == "true_positive"
    assert scored[0].severity == "high"


def test_false_positive_when_ground_truth_is_clean() -> None:
    gt = _gt(("Model!C14", "clean", None))
    issues = [_issue("Model!C14")]
    scored = score_rule_on_entry("e", issues, gt, "literal-in-formula-block")

    assert scored[0].outcome == "false_positive"


def test_false_positive_when_wrong_rule_flags_someone_elses_bug() -> None:
    """A cell ground-truthed as an inconsistent-anchoring bug, but flagged
    by literal-in-formula-block instead -- the wrong tool fired, so this
    must count as a false positive for literal-in-formula-block (not a
    true positive just because *some* rule's bug is real there).
    """
    gt = _gt(("Model!F14", "injected_bug", "inconsistent-anchoring"))
    issues = [_issue("Model!F14", "high", "literal-in-formula-block")]
    scored = score_rule_on_entry("e", issues, gt, "literal-in-formula-block")

    assert scored[0].outcome == "false_positive"


def test_known_intentional_is_its_own_outcome_not_tp_or_fp() -> None:
    gt = _gt(("Model!B14", "known_intentional", None))
    issues = [_issue("Model!B14", "medium", "literal-in-formula-block")]
    scored = score_rule_on_entry("e", issues, gt, "literal-in-formula-block")

    assert scored[0].outcome == "known_intentional"
    assert scored[0].outcome not in ("true_positive", "false_positive")


def test_cell_absent_from_ground_truth_scores_as_false_positive() -> None:
    """Structurally never observed against the real corpus (verified
    directly during investigation), but the scoring function must still
    handle it sanely rather than crashing or silently vanishing it.
    """
    gt = _gt(("Model!C14", "clean", None))
    issues = [_issue("Model!Z99")]
    scored = score_rule_on_entry("e", issues, gt, "literal-in-formula-block")

    assert scored[0].outcome == "false_positive"


# ---------------------------------------------------------------------------
# count_false_negatives
# ---------------------------------------------------------------------------


def test_false_negative_counted_when_rules_own_bug_is_missed() -> None:
    gt = _gt(("Model!F14", "injected_bug", "range-boundary-mismatch"))
    issues: list[Issue] = []  # the rule never flagged it
    fn = count_false_negatives(issues, gt, "range-boundary-mismatch")

    assert fn == 1


def test_no_false_negative_when_the_bug_is_flagged() -> None:
    gt = _gt(("Model!F14", "injected_bug", "range-boundary-mismatch"))
    issues = [_issue("Model!F14", "high", "range-boundary-mismatch")]
    fn = count_false_negatives(issues, gt, "range-boundary-mismatch")

    assert fn == 0


def test_false_negative_only_attributed_to_the_owning_rule_id() -> None:
    """A missed bug for rule A must not count as a false negative against
    rule B, even if rule B is also being scored on the same entry.
    """
    gt = _gt(("Model!F14", "injected_bug", "range-boundary-mismatch"))
    issues: list[Issue] = []
    assert count_false_negatives(issues, gt, "range-boundary-mismatch") == 1
    assert count_false_negatives(issues, gt, "inconsistent-anchoring") == 0


# ---------------------------------------------------------------------------
# is_clean_baseline
# ---------------------------------------------------------------------------


def test_is_clean_baseline_true_when_no_injected_bug_cells() -> None:
    gt = _gt(("Model!B14", "known_intentional", None), ("Model!C14", "clean", None))
    assert is_clean_baseline(gt) is True


def test_is_clean_baseline_false_when_any_injected_bug_cell_present() -> None:
    gt = _gt(("Model!B14", "known_intentional", None), ("Model!F14", "injected_bug", "literal-in-formula-block"))
    assert is_clean_baseline(gt) is False


# ---------------------------------------------------------------------------
# precision@10: known_intentional flags must not crowd the ranking pool
# ---------------------------------------------------------------------------


def test_precision_at_10_pool_excludes_known_intentional_entries() -> None:
    """Regression test for a real bug caught while smoke-testing the
    harness: an earlier version pooled ALL outcomes (including
    known_intentional) before ranking, so known_intentional entries that
    happened to sort alphabetically first could crowd real TP/FP entries
    out of the top-10 window, silently shrinking the effective k below
    10 for reasons unrelated to ranking quality. Confirmed directly on
    the original 47-entry corpus: with 20 known_intentional entries and
    11 true positives for literal-in-formula-block, the buggy version
    produced k=5; the fixed version produces k=10. The exact counts below
    reflect the current (post-corpus-expansion) corpus, not those
    original numbers -- ambiguous_excluded, added later, is excluded from
    the ranking pool the same way known_intentional is, for the same
    reason. This test locks the fix in place via the module's own public
    functions -- the real full-corpus numbers are checked separately in
    test_full_corpus_evaluation_matches_verified_numbers.
    """
    report = run_evaluation(CORPUS_DIR)
    literal_rollup = next(r for r in report.rollups if r.rule_id == "literal-in-formula-block")

    assert literal_rollup.known_intentional_excluded == 22  # confirms the crowding scenario is real, not contrived
    assert literal_rollup.precision_at_10_k == PRECISION_AT_K  # not shrunk below 10 by known_intentional entries
    assert literal_rollup.precision_at_10 == 1.0


# ---------------------------------------------------------------------------
# format_report_text: the caveat must actually be printed, not just documented
# ---------------------------------------------------------------------------


def test_formatted_report_includes_the_internal_consistency_caveat_verbatim() -> None:
    report = run_evaluation(CORPUS_DIR)
    text = format_report_text(report)

    assert INTERNAL_CONSISTENCY_CAVEAT in text


# ---------------------------------------------------------------------------
# Integration: the real harness against the real corpus
# ---------------------------------------------------------------------------


def test_full_corpus_evaluation_matches_verified_numbers() -> None:
    """Locks in the exact numbers verified by hand via
    scripts/run_evaluation.py against the real, post-corpus-expansion
    144-entry corpus (originally 47 entries; growing it added subtotal,
    ambiguous, and variance-shape entries -- see
    scripts/generate_corpus.py's module docstring). Every original
    47-entry injection is untouched, so precision/recall stay 1.0/1.0
    exactly as before; only the exclusion-bucket counts and clean-baseline
    entry count grew, since subtotal/ambiguous entries have no
    injected_bug cell and so count as "clean baselines" under
    is_clean_baseline's existing (unchanged) definition.
    """
    report = run_evaluation(CORPUS_DIR)
    rollups_by_id = {r.rule_id: r for r in report.rollups}

    expected = {
        "literal-in-formula-block": (21, 0, 0, 22, 4),
        "range-boundary-mismatch": (10, 0, 0, 0, 0),
        "reference-to-blank": (20, 0, 0, 0, 39),
        "inconsistent-anchoring": (10, 0, 0, 0, 2),
    }
    for rule_id, (tp, fp, fn, known, ambiguous) in expected.items():
        r = rollups_by_id[rule_id]
        assert (r.tp, r.fp, r.fn, r.known_intentional_excluded, r.ambiguous_excluded) == (tp, fp, fn, known, ambiguous)
        assert r.precision == 1.0
        assert r.recall == 1.0

    assert report.clean_baseline.entries_checked == 94
    assert len(report.clean_baseline.raw_flags) == 52
    assert len(report.clean_baseline.unexplained_flags) == 0


def test_entry_names_filter_restricts_scoring_to_the_given_entries() -> None:
    """New in the Tier 1 classifier follow-up: entry_names lets a caller
    (classifier.py, scoring Tier 0 on a held-out test split only) restrict
    which corpus entries get loaded/scored, without reusing the inflated
    full-corpus number. Omitting the parameter must reproduce today's
    existing full-corpus numbers exactly (regression guard); passing it
    must match hand-scoring just those entries.
    """
    full_report = run_evaluation(CORPUS_DIR)
    full_rollups = {r.rule_id: (r.tp, r.fp, r.fn) for r in full_report.rollups}

    two_entries = {"clean__growth_chain__n4", "literal_in_block__edge_left__n4"}
    subset_report = run_evaluation(CORPUS_DIR, entry_names=two_entries)

    # literal_in_block__edge_left__n4 contributes exactly one real TP for
    # literal-in-formula-block (see test_literal_injection_flags_injected_and_seed_cells-
    # style fixtures in test_generate_corpus.py); clean__growth_chain__n4 contributes
    # zero TP/FP on that rule (its own seed literal is known_intentional, excluded).
    literal_rollup = next(r for r in subset_report.rollups if r.rule_id == "literal-in-formula-block")
    assert literal_rollup.tp == 1
    assert literal_rollup.fp == 0

    # Every rule's rollup across the 2-entry subset must be a subset of the full-corpus
    # totals, never exceeding them -- a cheap but real cross-check that filtering actually
    # restricted the entries scored, not silently ignored the parameter.
    for r in subset_report.rollups:
        full_tp, full_fp, full_fn = full_rollups[r.rule_id]
        assert r.tp <= full_tp
        assert r.fp <= full_fp

    # Omitting entry_names entirely must reproduce the existing full-corpus numbers exactly.
    default_report = run_evaluation(CORPUS_DIR)
    assert {r.rule_id: (r.tp, r.fp, r.fn) for r in default_report.rollups} == full_rollups
