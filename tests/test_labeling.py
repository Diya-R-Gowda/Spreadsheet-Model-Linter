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
    UNRECOGNIZED_BASE_SHAPE,
    build_block_example,
    check_training_readiness,
    generate_labeled_examples,
    guess_base_shape,
    label_for_block,
    labeled_entry_names,
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


def test_block_with_subtotal_cell_labels_subtotal() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])
    gt_by_cell = {
        "Model!C14": {"cell": "Model!C14", "state": "subtotal", "rule_id": None},
        "Model!D14": {"cell": "Model!D14", "state": "subtotal", "rule_id": None},
        "Model!E14": {"cell": "Model!E14", "state": "subtotal", "rule_id": None},
    }
    label, basis = label_for_block(block, gt_by_cell, flagged_cells=set())

    assert label == "subtotal"
    assert "subtotal" in basis


def test_block_with_flagged_ambiguous_neighbor_labels_unknown() -> None:
    block = _block(
        ["Model!D14", "Model!E14", "Model!F14"],
        [NonConformingCell(cell="Model!C14", kind="literal")],
    )
    gt_by_cell = {"Model!C14": {"cell": "Model!C14", "state": "ambiguous", "rule_id": "literal-in-formula-block"}}
    label, basis = label_for_block(block, gt_by_cell, flagged_cells={"Model!C14"})

    assert label == "unknown"
    assert "ambiguous" in basis


def test_ambiguous_cell_not_flagged_does_not_count_as_unknown() -> None:
    """Same guard as the injected_bug case: ground truth alone isn't
    enough, the rule engine must have actually flagged it.
    """
    block = _block(
        ["Model!D14", "Model!E14", "Model!F14"],
        [NonConformingCell(cell="Model!C14", kind="literal")],
    )
    gt_by_cell = {"Model!C14": {"cell": "Model!C14", "state": "ambiguous", "rule_id": "literal-in-formula-block"}}
    label, _basis = label_for_block(block, gt_by_cell, flagged_cells=set())  # C14 NOT in flagged_cells

    assert label is None


def test_suspected_error_takes_precedence_over_unknown_on_the_same_block() -> None:
    """Mirrors the suspected_error-over-intentional_override precedence
    test: a block bordering both a flagged ambiguous cell AND a flagged
    real injected_bug cell must resolve to suspected_error, the more
    decision-relevant signal.
    """
    block = _block(
        ["Model!D14", "Model!E14"],
        [
            NonConformingCell(cell="Model!C14", kind="literal"),
            NonConformingCell(cell="Model!F14", kind="literal"),
        ],
    )
    gt_by_cell = {
        "Model!C14": {"cell": "Model!C14", "state": "ambiguous", "rule_id": "literal-in-formula-block"},
        "Model!F14": {"cell": "Model!F14", "state": "injected_bug", "rule_id": "literal-in-formula-block"},
    }
    label, _basis = label_for_block(block, gt_by_cell, flagged_cells={"Model!C14", "Model!F14"})

    assert label == "suspected_error"


def test_subtotal_takes_precedence_over_every_other_signal() -> None:
    """subtotal is checked first -- the most direct, certain signal
    (a block IS the legitimate aggregate) -- even if the same block
    somehow also borders a flagged injected_bug cell.
    """
    block = _block(["Model!C14", "Model!D14", "Model!E14"], [NonConformingCell(cell="Model!F14", kind="literal")])
    gt_by_cell = {
        "Model!C14": {"cell": "Model!C14", "state": "subtotal", "rule_id": None},
        "Model!D14": {"cell": "Model!D14", "state": "subtotal", "rule_id": None},
        "Model!E14": {"cell": "Model!E14", "state": "subtotal", "rule_id": None},
        "Model!F14": {"cell": "Model!F14", "state": "injected_bug", "rule_id": "literal-in-formula-block"},
    }
    label, _basis = label_for_block(block, gt_by_cell, flagged_cells={"Model!F14"})

    assert label == "subtotal"


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
# build_block_example: row_label/col_labels pass through Block's real
# heuristic capture (2026-08-20) instead of always being None
# ---------------------------------------------------------------------------


def test_build_block_example_passes_through_a_blocks_real_row_and_col_labels() -> None:
    block = Block(
        sheet="Model", row=14, span="C14:E14", pattern="(R[0]C[-1]*(1+R1C2))",
        cells=["Model!C14", "Model!D14", "Model!E14"],
        row_label="Revenue", col_labels=["Jan-24", "Feb-24", None],
    )
    example = build_block_example("entry_1", block, "growth_chain")

    assert example.row_label == "Revenue"
    assert example.col_labels == ["Jan-24", "Feb-24", None]


def test_build_block_example_row_and_col_labels_stay_none_when_the_block_has_none() -> None:
    block = _block(["Model!C14", "Model!D14", "Model!E14"])  # default Block: no labels captured
    example = build_block_example("entry_1", block, "growth_chain")

    assert example.row_label is None
    assert example.col_labels == []


# ---------------------------------------------------------------------------
# guess_base_shape -- the live-inference base_shape heuristic (2026-08-20)
# ---------------------------------------------------------------------------


def test_guess_base_shape_growth_chain_corpus_absolute_form() -> None:
    assert guess_base_shape("(R[0]C[-1]*(1+R1C2))") == "growth_chain"


def test_guess_base_shape_growth_chain_real_fixture_relative_form() -> None:
    """revenue_row_with_hardcode.xlsx's real formula (=prev14*(1+prev15))
    normalizes to a RELATIVE growth-rate operand, unlike the corpus's
    absolute $B$1 form -- proves the regex isn't overfit to synthetic data.
    """
    assert guess_base_shape("(R[0]C[-1]*(1+R[1]C[-1]))") == "growth_chain"


def test_guess_base_shape_trailing_sum_horizontal_range() -> None:
    assert guess_base_shape("SUM(R[-1]C[-2]:R[-1]C[0])") == "trailing_sum"


def test_guess_base_shape_category_total_vertical_range() -> None:
    assert guess_base_shape("SUM(R[-13]C[0]:R[-1]C[0])") == "category_total"


def test_guess_base_shape_variance_subtraction() -> None:
    assert guess_base_shape("(R[-2]C[0]-R[-1]C[0])") == "variance"


def test_guess_base_shape_variance_regex_matches_but_identical_offsets_rejected() -> None:
    assert guess_base_shape("(R[-1]C[0]-R[-1]C[0])") == UNRECOGNIZED_BASE_SHAPE


def test_guess_base_shape_unmatched_pattern_returns_unrecognized() -> None:
    assert guess_base_shape("(R[0]C[-1]+R[0]C[-2])") == UNRECOGNIZED_BASE_SHAPE
    assert guess_base_shape("AVERAGE(R[-1]C[0]:R[-3]C[0])") == UNRECOGNIZED_BASE_SHAPE


# ---------------------------------------------------------------------------
# labeled_entry_names -- the split-discrepancy fix (2026-08-19)
# ---------------------------------------------------------------------------


def test_labeled_entry_names_excludes_entries_with_no_labeled_block() -> None:
    labeled = build_block_example("entry_labeled", _block(["Model!C14", "Model!D14", "Model!E14"]), "growth_chain", label="subtotal")
    unlabeled = build_block_example("entry_unlabeled", _block(["Model!C20", "Model!D20", "Model!E20"]), "growth_chain", label=None)

    assert labeled_entry_names([labeled, unlabeled]) == ["entry_labeled"]


def test_labeled_entry_names_includes_an_entry_with_at_least_one_labeled_block() -> None:
    one_labeled = build_block_example("entry_mixed", _block(["Model!C14", "Model!D14", "Model!E14"]), "growth_chain", label="suspected_error")
    one_unlabeled = build_block_example("entry_mixed", _block(["Model!C20", "Model!D20", "Model!E20"]), "growth_chain", label=None)

    assert labeled_entry_names([one_labeled, one_unlabeled]) == ["entry_mixed"]


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
    """Locks in the exact numbers verified by hand via
    scripts/run_labeling.py against the real, post-corpus-expansion
    144-entry corpus (originally 47 entries, 61 blocks -- see
    scripts/generate_corpus.py's module docstring for what the new
    subtotal/ambiguous/variance-shape entries added).
    """
    examples = generate_labeled_examples(CORPUS_DIR)
    assert len(examples) == 162

    counts = {label: sum(1 for e in examples if e.label == label) for label in LABELS}
    assert counts == {"subtotal": 60, "intentional_override": 5, "suspected_error": 67, "unknown": 20}

    unlabeled = sum(1 for e in examples if e.label is None)
    assert unlabeled == 10

    patterns = {e.formula_pattern for e in examples}
    assert len(patterns) == 8  # up from 2 -- the variance + category-total shapes are genuinely new patterns

    # The synthetic corpus's shape builders write no label text anywhere (confirmed by
    # reading scripts/generate_corpus.py directly -- no column-A text, no header row), so
    # even with row_label/col_labels capture now real (2026-08-20), every real corpus block
    # is an honest heuristic MISS: row_label stays None, and col_labels is always a real
    # list (never bare None -- _capture_col_labels always returns one entry per spanned
    # column) whose every entry is None.
    assert all(e.row_label is None for e in examples)
    assert all(label is None for e in examples for label in e.col_labels)


def test_training_readiness_is_honestly_not_ready_on_the_real_corpus() -> None:
    """subtotal now clears the floor (60 >= 50) after corpus expansion;
    intentional_override and unknown remain below it -- both for real,
    documented structural reasons (see scripts/generate_corpus.py and
    ablation.py's _BLOCKED_NOTE), not an oversight.
    """
    examples = generate_labeled_examples(CORPUS_DIR)
    entry_names = sorted({e.entry for e in examples})
    split = split_corpus_entries(entry_names)
    report = check_training_readiness(examples, split)

    assert report.ready is False  # the honest result -- must not be quietly true
    not_ready_labels = {l.label for l in report.per_label if not l.ready}
    assert not_ready_labels == {"unknown", "intentional_override"}
    ready_labels = {l.label for l in report.per_label if l.ready}
    assert ready_labels == {"suspected_error", "subtotal"}
