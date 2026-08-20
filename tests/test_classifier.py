"""Tests for ssmlint.classifier -- the Tier 1 classifier's non-training
logic (serialization, Tier-0-suppression). Deliberately fast and free of
any real model/tokenizer: these hand-built-fixture tests are what runs
in the default `pytest` suite. A real fine-tuning run is verified
manually via scripts/train_classifier.py, with real output pasted as
evidence -- see CONTRIBUTING.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from ssmlint.blocks import Block, NearMissCell, NonConformingCell, SheetBlocks
from ssmlint.classifier import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    apply_tier1,
    apply_tier1_to_workbook,
    describe_intentional_override_confidence,
    evaluate_tier0_plus_1,
    serialize_block_example,
)
from ssmlint.labeling import LABELS, Deviation, build_block_example
from ssmlint.rules import Issue

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def _block(cells: list[str], non_conforming: list[NonConformingCell] | None = None) -> Block:
    return Block(
        sheet="Model", row=14, span="C14:E14", pattern="(R[0]C[-1]*(1+R1C2))",
        cells=cells, non_conforming=non_conforming or [], near_misses=[],
    )


def _issue(cell: str, rule_id: str = "literal-in-formula-block") -> Issue:
    return Issue(cell=cell, severity="medium", rule_id=rule_id, explanation="x", suggested_fix="y")


# ---------------------------------------------------------------------------
# LABEL_TO_ID / ID_TO_LABEL
# ---------------------------------------------------------------------------


def test_label_id_mappings_cover_every_label_bijectively() -> None:
    assert set(LABEL_TO_ID.keys()) == set(LABELS)
    assert set(ID_TO_LABEL.values()) == set(LABELS)
    for label, i in LABEL_TO_ID.items():
        assert ID_TO_LABEL[i] == label


# ---------------------------------------------------------------------------
# serialize_block_example
# ---------------------------------------------------------------------------


def test_serialize_includes_shape_pattern_and_conforming_count() -> None:
    example = build_block_example(
        "entry_1",
        _block(["Model!C14", "Model!D14", "Model!E14"]),
        base_shape="growth_chain",
    )
    text = serialize_block_example(example)

    assert "growth_chain" in text
    assert "(R[0]C[-1]*(1+R1C2))" in text
    assert "Conforming cells: 3" in text
    assert "Deviations: none" in text


def test_serialize_includes_each_deviation_cell_kind_and_side() -> None:
    example = build_block_example(
        "entry_1",
        _block(["Model!D14", "Model!E14"], [NonConformingCell(cell="Model!C14", kind="literal")]),
        base_shape="growth_chain",
    )
    text = serialize_block_example(example)

    assert "Model!C14" in text
    assert "non_conforming:literal" in text
    assert "neighbor" in text


def test_serialize_never_includes_cell_values_or_row_col_labels() -> None:
    """Confirms the input the model sees never claims richer context than
    the pipeline actually has -- BlockExample.row_label/col_labels are
    always None (Week 5's confirmed input-shape gap) and no cell value
    ever appears (BlockExample/Deviation carry no value field at all).
    """
    example = build_block_example(
        "entry_1",
        _block(["Model!D14", "Model!E14"], [NonConformingCell(cell="Model!C14", kind="literal", value=4500000)]),
        base_shape="growth_chain",
    )
    text = serialize_block_example(example)

    assert "4500000" not in text
    assert "None" not in text  # row_label/col_labels being None must never leak into the string
    assert example.row_label is None
    assert example.col_labels is None


def test_serialize_is_deterministic() -> None:
    example = build_block_example("entry_1", _block(["Model!C14", "Model!D14", "Model!E14"]), base_shape="growth_chain")
    assert serialize_block_example(example) == serialize_block_example(example)


# ---------------------------------------------------------------------------
# apply_tier1: Tier 0 issue suppression
# ---------------------------------------------------------------------------


def test_subtotal_prediction_suppresses_every_issue_on_that_block() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])
    issues = [_issue("Model!C14"), _issue("Model!D14")]

    result = apply_tier1(issues, [block], ["subtotal"])

    assert result == []


def test_intentional_override_prediction_suppresses_neighbor_deviation_cells_too() -> None:
    """Suppression uses the block's full candidate_cells (members +
    non_conforming + near_misses), not just block.cells -- a
    literal-in-formula-block issue is raised on the NEIGHBOR literal, not
    a block member, so it must still be suppressed when that block is
    predicted intentional_override.
    """
    block = _block(
        ["Model!D14", "Model!E14", "Model!F14"],
        [NonConformingCell(cell="Model!C14", kind="literal")],
    )
    issues = [_issue("Model!C14")]

    result = apply_tier1(issues, [block], ["intentional_override"])

    assert result == []


def test_suspected_error_prediction_leaves_issues_untouched() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])
    issues = [_issue("Model!C14")]

    result = apply_tier1(issues, [block], ["suspected_error"])

    assert result == issues


def test_unknown_prediction_leaves_issues_untouched() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])
    issues = [_issue("Model!C14")]

    result = apply_tier1(issues, [block], ["unknown"])

    assert result == issues


def test_suppression_only_affects_the_predicted_blocks_own_cells() -> None:
    """An issue on a cell that belongs to a DIFFERENT block (predicted
    suspected_error) must survive even when another block in the same
    call is predicted subtotal.
    """
    subtotal_block = _block(["Model!C14", "Model!D14", "Model!E14"])
    other_block = _block(["Model!C20", "Model!D20", "Model!E20"])
    issues = [_issue("Model!C14"), _issue("Model!C20")]

    result = apply_tier1(issues, [subtotal_block, other_block], ["subtotal", "suspected_error"])

    assert {i.cell for i in result} == {"Model!C20"}


def test_issue_on_a_cell_with_no_block_context_is_never_touched() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])
    issues = [_issue("Model!Z99")]  # unrelated to the block entirely

    result = apply_tier1(issues, [block], ["subtotal"])

    assert result == issues


def test_mismatched_blocks_and_predictions_length_raises() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])
    try:
        apply_tier1([], [block], [])
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# evaluate_tier0_plus_1: real corpus entries, mocked classifier (no torch,
# no real checkpoint needed -- training itself is deliberately not part of
# the default pytest run, see CONTRIBUTING.md)
# ---------------------------------------------------------------------------


def _fake_checkpoint(tmp_path: Path, test_entries: list[str]) -> Path:
    checkpoint_dir = tmp_path / "fake_checkpoint"
    checkpoint_dir.mkdir()
    split = {"train": [], "val": [], "test": test_entries}
    (checkpoint_dir / "split.json").write_text(json.dumps(split), encoding="utf-8")
    return checkpoint_dir


def test_evaluate_tier0_plus_1_matches_tier0_when_predictions_never_suppress(tmp_path: Path) -> None:
    """literal_in_block__edge_left__n4 has exactly one real injected_bug
    flag (Model!C14, already locked in by test_generate_corpus.py). If
    every block is predicted suspected_error/unknown (never suppressed),
    Tier 0+1 must score identically to Tier 0 alone on this entry.
    """
    checkpoint_dir = _fake_checkpoint(tmp_path, ["literal_in_block__edge_left__n4"])

    with (
        patch("ssmlint.classifier.load_classifier", return_value=(object(), object())),
        patch("ssmlint.classifier.predict_labels", return_value=["suspected_error"]),
    ):
        rollups, test_entries = evaluate_tier0_plus_1(CORPUS_DIR, checkpoint_dir)

    assert test_entries == ["literal_in_block__edge_left__n4"]
    literal_rollup = next(r for r in rollups if r.rule_id == "literal-in-formula-block")
    assert literal_rollup.tp == 1
    assert literal_rollup.fp == 0
    assert literal_rollup.fn == 0


def test_evaluate_tier0_plus_1_suppresses_a_real_bug_when_predicted_subtotal(tmp_path: Path) -> None:
    """The inverse case: if the classifier (wrongly) predicts subtotal
    for every block, the real injected_bug flag gets suppressed by
    apply_tier1 -- Tier 0+1 must then score it as a missed bug (a false
    negative), not silently drop it from the count entirely. This is the
    real mechanism by which a wrong Tier 1 prediction can make Tier 0+1
    WORSE than Tier 0 alone, not just better.
    """
    checkpoint_dir = _fake_checkpoint(tmp_path, ["literal_in_block__edge_left__n4"])

    with (
        patch("ssmlint.classifier.load_classifier", return_value=(object(), object())),
        patch("ssmlint.classifier.predict_labels", return_value=["subtotal"]),
    ):
        rollups, _test_entries = evaluate_tier0_plus_1(CORPUS_DIR, checkpoint_dir)

    literal_rollup = next(r for r in rollups if r.rule_id == "literal-in-formula-block")
    assert literal_rollup.tp == 0
    assert literal_rollup.fn == 1


# ---------------------------------------------------------------------------
# apply_tier1_to_workbook -- live inference for one real (already-parsed)
# workbook, mocked classifier (no torch, no real checkpoint needed)
# ---------------------------------------------------------------------------


def test_apply_tier1_to_workbook_splits_issues_exhaustively_when_nothing_suppressed() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])
    sheet_blocks = [SheetBlocks(sheet="Model", blocks=[block])]
    issues = [_issue("Model!C14"), _issue("Model!D14")]

    with (
        patch("ssmlint.classifier.load_classifier", return_value=(object(), object())),
        patch("ssmlint.classifier.predict_labels", return_value=["suspected_error"]),
    ):
        surviving, suppressed = apply_tier1_to_workbook(issues, sheet_blocks, Path("fake/checkpoint"))

    assert surviving == issues
    assert suppressed == []


def test_apply_tier1_to_workbook_splits_issues_exhaustively_when_a_block_is_suppressed() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])
    sheet_blocks = [SheetBlocks(sheet="Model", blocks=[block])]
    issues = [_issue("Model!C14"), _issue("Model!D14")]

    with (
        patch("ssmlint.classifier.load_classifier", return_value=(object(), object())),
        patch("ssmlint.classifier.predict_labels", return_value=["subtotal"]),
    ):
        surviving, suppressed = apply_tier1_to_workbook(issues, sheet_blocks, Path("fake/checkpoint"))

    # exhaustive, non-overlapping split -- every real issue lands in exactly one of the two lists
    assert surviving == []
    assert {i.cell for i in suppressed} == {"Model!C14", "Model!D14"}
    assert set(surviving) | set(suppressed) == set(issues)
    assert set(surviving) & set(suppressed) == set()


def test_apply_tier1_to_workbook_flattens_blocks_across_multiple_sheets() -> None:
    block_a = _block(["Model!C14", "Model!D14", "Model!E14"])
    block_b = _block(["Other!C14", "Other!D14", "Other!E14"])
    sheet_blocks = [SheetBlocks(sheet="Model", blocks=[block_a]), SheetBlocks(sheet="Other", blocks=[block_b])]
    issues = [_issue("Model!C14"), _issue("Other!C14")]

    with (
        patch("ssmlint.classifier.load_classifier", return_value=(object(), object())),
        patch("ssmlint.classifier.predict_labels", return_value=["subtotal", "suspected_error"]),
    ):
        surviving, suppressed = apply_tier1_to_workbook(issues, sheet_blocks, Path("fake/checkpoint"))

    assert {i.cell for i in suppressed} == {"Model!C14"}
    assert {i.cell for i in surviving} == {"Other!C14"}


# ---------------------------------------------------------------------------
# describe_intentional_override_confidence -- pure file-parsing, no torch/checkpoint needed
# ---------------------------------------------------------------------------


def _write_classification_report(tmp_path: Path, *, support: int, recall: float | None) -> Path:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    report = {
        "per_label": [
            {"label": "subtotal", "support": 7, "tp": 7, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
            {
                "label": "intentional_override", "support": support, "tp": 0, "fp": 0, "fn": support,
                "precision": None, "recall": recall, "f1": None,
            },
        ],
        "confusion_matrix": {},
        "accuracy": 0.5,
        "test_size": support + 7,
    }
    (checkpoint_dir / "classification_report.json").write_text(json.dumps(report), encoding="utf-8")
    return checkpoint_dir


def test_describe_intentional_override_confidence_thin_support_returns_explicit_caveat(tmp_path: Path) -> None:
    checkpoint_dir = _write_classification_report(tmp_path, support=1, recall=0.0)

    result = describe_intentional_override_confidence(checkpoint_dir)

    assert "CAVEAT" in result
    assert "1 held-out test example" in result
    assert "0.000 recall" in result
    assert "no real evidence" not in result  # exact phrasing lives in report.py's own caveat, not duplicated here
    assert "judgment call Tier 1 exists for" in result


def test_describe_intentional_override_confidence_ample_support_returns_plain_stats(tmp_path: Path) -> None:
    checkpoint_dir = _write_classification_report(tmp_path, support=12, recall=0.833)

    result = describe_intentional_override_confidence(checkpoint_dir)

    assert result == "intentional_override: 12 held-out test examples, 0.833 recall."
    assert "CAVEAT" not in result


def test_describe_intentional_override_confidence_missing_file_returns_unknown(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint_no_report"
    checkpoint_dir.mkdir()

    result = describe_intentional_override_confidence(checkpoint_dir)

    assert "UNKNOWN" in result
    assert "classification_report.json not found" in result
