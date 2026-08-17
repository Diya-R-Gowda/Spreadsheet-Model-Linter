"""Week 5, for real: the Tier 1 trained classifier.

Fine-tunes a small CPU-only encoder (`microsoft/deberta-v3-small`, per
the README's stated default) on the synthetic-corruption labels
`labeling.py` already produces, to predict the 4-way block-level label
(`subtotal | intentional_override | suspected_error | unknown`) the
README specifies. This is the first module in the project that imports
`torch`/`transformers` -- both are an optional dependency
(`pip install -e ".[classifier]"`), never required for the base Tier-0-
only `ssmlint` install, per the README's own "Tier 1/2 are optional
layers" framing.

WHAT "TIER 0 + 1" ACTUALLY MEANS HERE
------------------------------------------------------------------------
The classifier operates on BLOCKS (subtotal / intentional_override /
suspected_error / unknown), while Tier 0's rules flag individual CELLS.
Tier 1 is not an independent detector -- it's a judgment layer over
Tier 0's own candidates, matching the README's framing ("that judgment
call is where a semantic layer... earns its place, on top of a rule
engine that already does most of the work deterministically"):
`apply_tier1` takes Tier 0's real flagged Issues plus a predicted label
per block, and SUPPRESSES issues on any block predicted `subtotal` or
`intentional_override` (Tier 1 overriding a Tier 0 false alarm) while
leaving `suspected_error`/`unknown` predictions untouched. Scoring the
resulting (possibly smaller) issue set the same way `evaluation.py`
already scores Tier 0's is what produces a real, comparable Tier 0+1
precision/recall number for `ablation.py`.

DATA READINESS (confirmed directly with the user before training,
2026-08-17)
------------------------------------------------------------------------
`check_training_readiness()` still reports NOT READY: `subtotal`(60) and
`suspected_error`(67) clear the 50-per-class floor, `intentional_
override`(5) and `unknown`(20) don't. Training proceeds anyway, on all
four labels, with real per-class results reported honestly -- including
if the model ends up unreliable on the two thin classes -- rather than
narrowing scope or deferring further. `train_classifier` prints the
pre-training readiness report for visibility; it is NOT a hard gate.
"""

from __future__ import annotations

from .blocks import Block
from .labeling import LABELS, BlockExample, candidate_cells
from .rules import Issue

LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = {i: label for label, i in LABEL_TO_ID.items()}


def serialize_block_example(example: BlockExample) -> str:
    """Deterministic text template turning a `BlockExample` into the
    string the tokenizer sees. Only fields `BlockExample` actually
    carries -- no cell values (never available, per Week 5's confirmed
    input-shape gap) and no `row_label`/`col_labels` (always `None`
    today, same gap) -- the model sees exactly what the README's compact
    block description promises, nothing fabricated to make the input
    look richer than the pipeline can actually produce.
    """
    lines = [
        f"Block: {example.block_address}",
        f"Base shape: {example.base_shape}",
        f"Formula pattern: {example.formula_pattern}",
        f"Conforming cells: {example.conforming_count}",
    ]
    if example.deviations:
        lines.append("Deviations:")
        for d in example.deviations:
            lines.append(f"  - {d.cell} ({d.kind}, {d.side})")
    else:
        lines.append("Deviations: none")
    return "\n".join(lines)


def apply_tier1(issues: list[Issue], blocks: list[Block], predictions: list[str]) -> list[Issue]:
    """Filters Tier 0's real Issues using Tier 1's predicted label per
    block. `blocks[i]`'s predicted label is `predictions[i]`.

    - `subtotal` / `intentional_override`: every issue whose cell is in
      that block's `candidate_cells` (same cell-scope `label_for_block`
      itself uses -- the cells that would have earned the block that
      label in the first place) is suppressed.
    - `suspected_error` / `unknown`: issues are left untouched. No new
      severity/confidence scheme is invented for `unknown` in this pass
      (a stated scoping choice -- see the classifier-plan roadmap note
      in CONTRIBUTING.md, not an oversight).

    A cell not covered by any block's candidate_cells (e.g. an isolated
    flagged cell with no block context) is never touched -- suppression
    only ever applies to cells Tier 1 actually had an opinion about.
    """
    if len(blocks) != len(predictions):
        raise ValueError(f"blocks and predictions must be the same length, got {len(blocks)} and {len(predictions)}")

    suppressed_cells: set[str] = set()
    for block, predicted_label in zip(blocks, predictions):
        if predicted_label in ("subtotal", "intentional_override"):
            suppressed_cells.update(candidate_cells(block))

    return [issue for issue in issues if issue.cell not in suppressed_cells]
