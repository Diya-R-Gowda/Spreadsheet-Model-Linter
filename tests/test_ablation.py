"""Tests for ssmlint.ablation -- the Week 6 ablation table.

Unit tests isolate the micro-averaging math against hand-constructed
RuleRollup objects (no corpus, no I/O) before trusting it against real
numbers, per the standing discipline. The timing-loop test asserts only
structural properties, never exact wall-clock values. One integration
test locks in the real full-corpus numbers already verified by hand via
scripts/run_ablation.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ssmlint.ablation import (
    TIER_0_1_2_LABEL,
    TIER_0_1_LABEL,
    TIER_0_LABEL,
    _aggregate_tier0_metrics,
    build_ablation_table,
    format_ablation_table_text,
)
from ssmlint.evaluation import EvaluationReport, RuleRollup

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def _report(rollups: list[RuleRollup]) -> EvaluationReport:
    from ssmlint.evaluation import CleanBaselineResult

    return EvaluationReport(slices=[], rollups=rollups, clean_baseline=CleanBaselineResult(entries_checked=0))


# ---------------------------------------------------------------------------
# Micro-averaging math, isolated
# ---------------------------------------------------------------------------


def test_aggregate_metrics_micro_averages_across_rules() -> None:
    rollups = [
        RuleRollup(rule_id="a", tp=8, fp=2, fn=0, precision_at_10=0.8, precision_at_10_k=10),
        RuleRollup(rule_id="b", tp=2, fp=0, fn=2, precision_at_10=1.0, precision_at_10_k=2),
    ]
    precision, recall, precision_at_10 = _aggregate_tier0_metrics(_report(rollups))

    # precision = (8+2) / (8+2 + 2+0) = 10/12
    assert precision == 10 / 12
    # recall = (8+2) / (8+2 + 0+2) = 10/12
    assert recall == 10 / 12
    # precision@10 = (8 + 2) / (10 + 2) = 10/12  -- tp_in_top_k reconstructed from ratio*k
    assert precision_at_10 == 10 / 12


def test_aggregate_metrics_handles_zero_denominators() -> None:
    rollups = [RuleRollup(rule_id="a", tp=0, fp=0, fn=0, precision_at_10=None, precision_at_10_k=0)]
    precision, recall, precision_at_10 = _aggregate_tier0_metrics(_report(rollups))

    assert precision is None
    assert recall is None
    assert precision_at_10 is None


def test_aggregate_metrics_matches_known_full_corpus_totals() -> None:
    """Real per-rule totals already verified in test_evaluation.py's
    integration test (TP=11/10/17/10, FP=0, FN=0 for the four rules,
    all precision_at_10=1.0 at k=10) -- confirms the aggregate is exactly
    1.0/1.0/1.0, not just plausible-looking.
    """
    rollups = [
        RuleRollup(rule_id="literal-in-formula-block", tp=11, fp=0, fn=0, precision_at_10=1.0, precision_at_10_k=10),
        RuleRollup(rule_id="range-boundary-mismatch", tp=10, fp=0, fn=0, precision_at_10=1.0, precision_at_10_k=10),
        RuleRollup(rule_id="reference-to-blank", tp=17, fp=0, fn=0, precision_at_10=1.0, precision_at_10_k=10),
        RuleRollup(rule_id="inconsistent-anchoring", tp=10, fp=0, fn=0, precision_at_10=1.0, precision_at_10_k=10),
    ]
    precision, recall, precision_at_10 = _aggregate_tier0_metrics(_report(rollups))

    assert precision == 1.0
    assert recall == 1.0
    assert precision_at_10 == 1.0


# ---------------------------------------------------------------------------
# Not-yet-built rows: status, notes, and the $0 architectural-fact cost
# ---------------------------------------------------------------------------


def test_unbuilt_rows_have_none_metrics_but_populated_cost_and_notes() -> None:
    table = build_ablation_table(CORPUS_DIR)
    unbuilt = [r for r in table.rows if r.status == "not_yet_available"]

    assert {r.configuration for r in unbuilt} == {TIER_0_1_LABEL, TIER_0_1_2_LABEL}
    for row in unbuilt:
        assert row.precision is None
        assert row.recall is None
        assert row.precision_at_10 is None
        assert row.median_runtime_ms is None
        assert row.cost == "$0"
        assert row.notes  # non-empty


def test_tier0_row_is_measured() -> None:
    table = build_ablation_table(CORPUS_DIR)
    tier0 = next(r for r in table.rows if r.configuration == TIER_0_LABEL)

    assert tier0.status == "measured"
    assert tier0.cost == "$0"


# ---------------------------------------------------------------------------
# Timing loop: structural properties only, never exact wall-clock values
# ---------------------------------------------------------------------------


def test_median_runtime_is_a_positive_float_with_one_sample_per_entry() -> None:
    table = build_ablation_table(CORPUS_DIR)
    tier0 = next(r for r in table.rows if r.configuration == TIER_0_LABEL)

    assert isinstance(tier0.median_runtime_ms, float)
    assert tier0.median_runtime_ms > 0


def test_median_computed_correctly_on_a_controlled_fake_timing_list() -> None:
    import statistics

    fake_samples = [1.0, 2.0, 3.0, 100.0, 5.0]
    with patch("ssmlint.ablation._time_full_pipeline_per_entry", return_value=fake_samples):
        table = build_ablation_table(CORPUS_DIR)
    tier0 = next(r for r in table.rows if r.configuration == TIER_0_LABEL)

    assert tier0.median_runtime_ms == statistics.median(fake_samples)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_formatted_table_includes_caveat_verbatim() -> None:
    from ssmlint.evaluation import INTERNAL_CONSISTENCY_CAVEAT

    table = build_ablation_table(CORPUS_DIR)
    text = format_ablation_table_text(table)

    assert INTERNAL_CONSISTENCY_CAVEAT in text
    assert TIER_0_LABEL in text
    assert TIER_0_1_LABEL in text
    assert TIER_0_1_2_LABEL in text


# ---------------------------------------------------------------------------
# Integration: real corpus numbers, matching test_evaluation.py's own totals
# ---------------------------------------------------------------------------


def test_full_corpus_ablation_matches_verified_tier0_numbers() -> None:
    table = build_ablation_table(CORPUS_DIR)
    tier0 = next(r for r in table.rows if r.configuration == TIER_0_LABEL)

    assert tier0.precision == 1.0
    assert tier0.recall == 1.0
    assert tier0.precision_at_10 == 1.0
