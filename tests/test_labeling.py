"""Tests for ssmlint.labeling -- the Week 5 plumbing-only label
generation / split / readiness pipeline. This is NOT testing a trained
classifier (none exists); it validates the label-generation logic in
isolation with hand-built scenarios, plus an integration test against
the real corpus locking in the real (honest, mostly-NOT-ready) numbers.
"""

from __future__ import annotations

from pathlib import Path

from ssmlint.blocks import Block, NonConformingCell
from ssmlint.labeling import (
    LABELS,
    check_training_readiness,
    generate_labeled_examples,
    label_for_block,
    split_corpus_entries,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def _block(cells: list[str], non_conforming: list[NonConformingCell] | None = None) -> Block:
    return Block(
        sheet="Model", row=14, span="C14:E14", pattern="(R[0]C[-1]*(1+R1C2))",
        cells=cells, non_conforming=non_conforming or [], near_misses=[],
    )


# ---------------------------------------------------------------------------
# label_for_block: isolated, hand-built scenarios
# ---------------------------------------------------------------------------


def test_block_with_flagged_injected_bug_neighbor_labels_suspected_error() -> None:
    block = _block(
        ["Model!C14", "Model!D14", "Model!E14"],
        [NonConformingCell(cell="Model!F14", kind="literal")],
    )
    gt_by_cell = {"Model!F14": {"cell": "Model!F14", "state": "injected_bug", "rule_id": "literal-in-formula-block"}}
    label, basis = label_for_block(block, gt_by_cell, flagged_cells={"Model!F14"})

    assert label == "suspected_error"
    assert "literal-in-formula-block" in basis


def test_block_with_known_intentional_neighbor_labels_intentional_override() -> None:
    block = _block(
        ["Model!C14", "Model!D14", "Model!E14"],
        [NonConformingCell(cell="Model!B14", kind="literal")],
    )
    gt_by_cell = {"Model!B14": {"cell": "Model!B14", "state": "known_intentional", "rule_id": None}}
    label, basis = label_for_block(block, gt_by_cell, flagged_cells=set())

    assert label == "intentional_override"
    assert "known_intentional" in basis


def test_block_with_no_ground_truth_signal_labels_none() -> None:
    block = _block(
        ["Model!C14", "Model!D14", "Model!E14"],
        [NonConformingCell(cell="Model!B14", kind="blank")],
    )
    gt_by_cell = {"Model!B14": {"cell": "Model!B14", "state": "clean", "rule_id": None}}
    label, basis = label_for_block(block, gt_by_cell, flagged_cells=set())

    assert label is None
    assert basis == "no_ground_truth_signal"


def test_injected_bug_cell_not_flagged_does_not_count_as_suspected_error() -> None:
    """Ground truth says injected_bug, but the rule engine never actually
    flagged it (a real miss) -- the block must not be labeled
    suspected_error off ground truth alone; the flag has to be real.
    """
    block = _block(
        ["Model!C14", "Model!D14", "Model!E14"],
        [NonConformingCell(cell="Model!F14", kind="literal")],
    )
    gt_by_cell = {"Model!F14": {"cell": "Model!F14", "state": "injected_bug", "rule_id": "literal-in-formula-block"}}
    label, basis = label_for_block(block, gt_by_cell, flagged_cells=set())  # F14 NOT in flagged_cells

    assert label is None


def test_suspected_error_takes_precedence_over_intentional_override_on_the_same_block() -> None:
    """A block bordering both a known_intentional seed AND a flagged
    injected_bug cell must resolve to suspected_error -- the documented,
    deliberate precedence (a real bug is the more decision-relevant
    signal for that block), not an arbitrary pick.
    """
    block = _block(
        ["Model!C14", "Model!D14", "Model!E14"],
        [
            NonConformingCell(cell="Model!B14", kind="literal"),
            NonConformingCell(cell="Model!F14", kind="literal"),
        ],
    )
    gt_by_cell = {
        "Model!B14": {"cell": "Model!B14", "state": "known_intentional", "rule_id": None},
        "Model!F14": {"cell": "Model!F14", "state": "injected_bug", "rule_id": "literal-in-formula-block"},
    }
    label, _basis = label_for_block(block, gt_by_cell, flagged_cells={"Model!F14"})

    assert label == "suspected_error"


# ---------------------------------------------------------------------------
# split_corpus_entries
# ---------------------------------------------------------------------------


def test_split_is_deterministic_for_a_fixed_seed() -> None:
    entries = [f"entry_{i}" for i in range(20)]
    split_a = split_corpus_entries(entries, seed=42)
    split_b = split_corpus_entries(entries, seed=42)
    assert split_a == split_b


def test_split_partitions_are_disjoint_and_cover_every_entry() -> None:
    entries = [f"entry_{i}" for i in range(20)]
    split = split_corpus_entries(entries, seed=1)
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])

    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(entries)


def test_split_rejects_ratios_that_do_not_sum_to_one() -> None:
    try:
        split_corpus_entries(["a", "b"], train=0.5, val=0.3, test=0.3)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Integration: the real corpus -- locks in the honest, mostly-not-ready numbers
# ---------------------------------------------------------------------------


def test_full_corpus_labeling_matches_verified_numbers() -> None:
    examples = generate_labeled_examples(CORPUS_DIR)
    assert len(examples) == 61

    counts = {label: sum(1 for e in examples if e.label == label) for label in LABELS}
    assert counts == {"subtotal": 0, "intentional_override": 5, "suspected_error": 50, "unknown": 0}

    unlabeled = sum(1 for e in examples if e.label is None)
    assert unlabeled == 6

    patterns = {e.formula_pattern for e in examples}
    assert len(patterns) == 2

    # row_label/col_labels must be explicitly None everywhere -- the pipeline
    # cannot produce them yet (see module docstring); never fabricated.
    assert all(e.row_label is None and e.col_labels is None for e in examples)


def test_training_readiness_is_honestly_not_ready_on_the_real_corpus() -> None:
    examples = generate_labeled_examples(CORPUS_DIR)
    entry_names = sorted({e.entry for e in examples})
    split = split_corpus_entries(entry_names)
    report = check_training_readiness(examples, split)

    assert report.ready is False  # the honest result -- must not be quietly true
    not_ready_labels = {l.label for l in report.per_label if not l.ready}
    assert not_ready_labels == {"subtotal", "unknown", "intentional_override"}
    ready_labels = {l.label for l in report.per_label if l.ready}
    assert ready_labels == {"suspected_error"}
