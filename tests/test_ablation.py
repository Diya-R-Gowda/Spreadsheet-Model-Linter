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
from ssmlint.evaluation import RuleRollup

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


# ---------------------------------------------------------------------------
# Micro-averaging math, isolated
# ---------------------------------------------------------------------------


def test_aggregate_metrics_micro_averages_across_rules() -> None:
    rollups = [
        RuleRollup(rule_id="a", tp=8, fp=2, fn=0, precision_at_10=0.8, precision_at_10_k=10),
        RuleRollup(rule_id="b", tp=2, fp=0, fn=2, precision_at_10=1.0, precision_at_10_k=2),
    ]
    precision, recall, precision_at_10 = _aggregate_tier0_metrics(rollups)

    # precision = (8+2) / (8+2 + 2+0) = 10/12
    assert precision == 10 / 12
    # recall = (8+2) / (8+2 + 0+2) = 10/12
    assert recall == 10 / 12
    # precision@10 = (8 + 2) / (10 + 2) = 10/12  -- tp_in_top_k reconstructed from ratio*k
    assert precision_at_10 == 10 / 12


def test_aggregate_metrics_handles_zero_denominators() -> None:
    rollups = [RuleRollup(rule_id="a", tp=0, fp=0, fn=0, precision_at_10=None, precision_at_10_k=0)]
    precision, recall, precision_at_10 = _aggregate_tier0_metrics(rollups)

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
    precision, recall, precision_at_10 = _aggregate_tier0_metrics(rollups)

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


# ---------------------------------------------------------------------------
# tier1_checkpoint_dir wiring -- mocked (no torch, no real checkpoint needed;
# training itself is deliberately not part of the default pytest run)
# ---------------------------------------------------------------------------


def test_tier1_checkpoint_dir_none_preserves_existing_not_yet_available_behavior() -> None:
    """Omitting the parameter (every existing caller/test) must reproduce
    today's exact behavior -- a real regression guard on the new
    optional parameter's default.
    """
    table = build_ablation_table(CORPUS_DIR)
    tier01 = next(r for r in table.rows if r.configuration == TIER_0_1_LABEL)

    assert tier01.status == "not_yet_available"
    assert tier01.precision is None


def test_tier1_checkpoint_dir_given_produces_a_measured_row_with_fair_comparison_notes() -> None:
    fake_rollups = [
        RuleRollup(rule_id="literal-in-formula-block", tp=1, fp=0, fn=0, precision_at_10=1.0, precision_at_10_k=1),
    ]
    fake_test_entries = ["literal_in_block__edge_left__n4"]

    with patch("ssmlint.classifier.evaluate_tier0_plus_1", return_value=(fake_rollups, fake_test_entries)):
        table = build_ablation_table(CORPUS_DIR, tier1_checkpoint_dir=Path("fake/checkpoint"))

    tier01 = next(r for r in table.rows if r.configuration == TIER_0_1_LABEL)
    assert tier01.status == "measured"
    assert tier01.precision == 1.0
    assert tier01.recall == 1.0
    # the fair-comparison context (Tier 0 alone on the SAME test split) must be visible in the
    # printed notes, not silently computed and discarded
    assert "1-entry held-out test split" in tier01.notes
    assert "Tier 0 ALONE on this same test split" in tier01.notes

    text = format_ablation_table_text(table)
    assert "[Tier 0 + 1 (+ trained classifier)]" in text
    assert "Tier 0 ALONE on this same test split" in text


def test_tier1_checkpoint_dir_given_includes_the_intentional_override_confidence_caveat() -> None:
    """The Tier 0+1 row's headline precision/recall must never stand alone
    without the real, dynamic intentional_override confidence caveat sitting
    next to it -- same standard as Tier 0's own internal-consistency caveat.
    """
    fake_rollups = [
        RuleRollup(rule_id="literal-in-formula-block", tp=1, fp=0, fn=0, precision_at_10=1.0, precision_at_10_k=1),
    ]
    fake_test_entries = ["literal_in_block__edge_left__n4"]
    caveat_text = "CAVEAT: intentional_override had only 1 held-out test example(s) and 0.000 recall..."

    with (
        patch("ssmlint.classifier.evaluate_tier0_plus_1", return_value=(fake_rollups, fake_test_entries)),
        patch("ssmlint.classifier.describe_intentional_override_confidence", return_value=caveat_text),
    ):
        table = build_ablation_table(CORPUS_DIR, tier1_checkpoint_dir=Path("fake/checkpoint"))

    tier01 = next(r for r in table.rows if r.configuration == TIER_0_1_LABEL)
    assert caveat_text in tier01.notes

    text = format_ablation_table_text(table)
    assert caveat_text in text


# ---------------------------------------------------------------------------
# tier2_model wiring -- mocked (no real Ollama server needed; a real live run
# is documented separately in CONTRIBUTING.md with real pasted output)
# ---------------------------------------------------------------------------


def test_tier2_model_none_preserves_existing_not_yet_available_behavior() -> None:
    """Omitting the parameter (every existing caller/test) must reproduce
    today's exact behavior -- a real regression guard on the new
    optional parameter's default."""
    # tier1_checkpoint_dir given WITHOUT tier2_model -- must still be a placeholder Tier 0+1+2 row
    fake_rollups = [
        RuleRollup(rule_id="literal-in-formula-block", tp=1, fp=0, fn=0, precision_at_10=1.0, precision_at_10_k=1),
    ]
    fake_test_entries = ["literal_in_block__edge_left__n4"]
    with patch("ssmlint.classifier.evaluate_tier0_plus_1", return_value=(fake_rollups, fake_test_entries)):
        table = build_ablation_table(CORPUS_DIR, tier1_checkpoint_dir=Path("fake/checkpoint"))

    tier012 = next(r for r in table.rows if r.configuration == TIER_0_1_2_LABEL)
    assert tier012.status == "not_yet_available"
    assert tier012.precision is None


def test_tier2_model_given_without_tier1_checkpoint_dir_raises() -> None:
    """Chained-only design: Tier 2 layers on top of Tier 1's own
    predictions, never a standalone alternative -- matches the README's
    own ablation table, which only ever shows 'Tier 0 + 1 + 2'."""
    try:
        build_ablation_table(CORPUS_DIR, tier2_model="qwen2.5:3b-instruct")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "tier1_checkpoint_dir" in str(exc)


def test_tier2_model_given_produces_a_measured_row_with_fair_comparison_notes() -> None:
    fake_tier1_rollups = [
        RuleRollup(rule_id="literal-in-formula-block", tp=1, fp=0, fn=0, precision_at_10=1.0, precision_at_10_k=1),
    ]
    fake_tier2_rollups = [
        RuleRollup(rule_id="literal-in-formula-block", tp=1, fp=0, fn=0, precision_at_10=1.0, precision_at_10_k=1),
    ]
    fake_test_entries = ["literal_in_block__edge_left__n4"]

    with (
        patch("ssmlint.classifier.evaluate_tier0_plus_1", return_value=(fake_tier1_rollups, fake_test_entries)),
        patch(
            "ssmlint.llm_adjudicator.evaluate_tier0_plus_1_plus_2",
            return_value=(fake_tier2_rollups, fake_test_entries),
        ),
    ):
        table = build_ablation_table(
            CORPUS_DIR, tier1_checkpoint_dir=Path("fake/checkpoint"), tier2_model="qwen2.5:3b-instruct"
        )

    tier012 = next(r for r in table.rows if r.configuration == TIER_0_1_2_LABEL)
    assert tier012.status == "measured"
    assert tier012.precision == 1.0
    assert tier012.recall == 1.0
    assert "1-entry held-out test split" in tier012.notes
    assert "qwen2.5:3b-instruct" in tier012.notes
    assert "Tier 0+1 ALONE on this same test split" in tier012.notes
    assert "zero-shot local LLM" in tier012.notes

    text = format_ablation_table_text(table)
    assert "[Tier 0 + 1 + 2 (+ local LLM)]" in text
    assert "qwen2.5:3b-instruct" in text
