"""literal-in-formula-block — the first Tier 0 rule.

Flags a hardcoded value sitting inside (or immediately touching) a block
of otherwise-uniform formulas — the README's own worked example: `H14`
hardcoded to `4500000` while `C14:N14` follows `=prev*(1+growth)`.

WHERE THIS READS FROM, AND A CONFIRMED GAP IN IT
-----------------------------------------------------
This rule consumes exactly one thing from blocks.py: each `Block`'s
`non_conforming` list, filtered to `kind == "literal"`. That list only
ever contains the single cell immediately left of a block's start column
and the single cell immediately right of its end column (confirmed by
reading `blocks.py::_build_block` before writing this rule, not assumed).
For the README's actual shape — one hardcoded cell fully surrounded by a
formula pattern — this is complete: the cell shows up as the non-
conforming right-neighbor of the block on its left and the non-conforming
left-neighbor of the block on its right, and this rule merges those two
sightings into one Issue (see `_literal_touchpoints`).

It is NOT complete for a *wider* gap: three or more contiguous literals
sitting between two blocks leave the interior literal(s) invisible to
every block's `non_conforming` list (verified directly — reported to the
user before writing this rule), so this rule cannot flag them either.
This is a known Tier 0 blind spot, not silently patched here — fixing it
means changing blocks.py's neighbor traversal, which is out of scope for
a single-rule stage and was explicitly deferred rather than fixed as a
drive-by.

SEVERITY: EXACTLY TWO LEVELS, FROM STRUCTURE ONLY
------------------------------------------------------
Cells are never recalculated (no `data_only` re-evaluation — out of scope
per the README), so severity cannot be based on computed magnitude; only
structural signals blocks.py actually provides are used:

  - "high": the literal is bordered by a block on BOTH sides, and both
    blocks share the *identical* R1C1 pattern. This is the unambiguous
    case: one continuous pattern with a single interruption — the
    README's own H14 shape, no different interpretation is plausible
    from the structure alone.
  - "medium": everything else this rule still flags — bordered by a
    block on only one side (a pattern that hasn't (yet, or ever)
    resumed on the far side), or bordered by two blocks with
    *different* patterns (a genuine boundary between two distinct
    formula shapes, not obviously a single interrupted pattern).

Two levels, not three or four: adding more would mean inventing a
threshold (e.g. "how many conforming neighbors counts as *more*
convincing") without a specific data-backed reason to place it exactly
there, which is the kind of unjustified heuristic this rule intentionally
avoids. See NOT BUILT below for the same reasoning applied to
intentional-override detection.

NOT BUILT: "IS THIS A TOTAL/SUBTOTAL ROW" HEURISTIC
--------------------------------------------------------
A hardcoded value is not automatically wrong — the README names
`intentional_override` as a first-class Tier 1/2 label specifically
because sometimes a hardcode is deliberate (a plugged total, a manual
override of a driver). This rule does NOT attempt to guess "is this
formula-block plausibly a subtotal/total row" from any proxy signal
(row label text, position, function name, etc.) — there is no reliable
signal for that available at this stage, and inventing one without real
backing is exactly the kind of false-positive risk the README's Failure
Modes table warns against. Every literal-in-block match is flagged as
`suspected_error` by default, conservatively, and it is Tier 1/2's job
to reclassify the intentional ones — this is a deliberate scope decision,
not an oversight.
"""

from __future__ import annotations

from ..blocks import Block, NonConformingCell, SheetBlocks
from .base import Issue, Rule

RULE_ID = "literal-in-formula-block"


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return f"{value:,.0f}" if float(value).is_integer() else f"{value:,}"
    return f'"{value}"' if isinstance(value, str) else str(value)


def _literal_touchpoints(
    sheet_blocks: list[SheetBlocks],
) -> dict[str, list[tuple[Block, NonConformingCell]]]:
    """Every literal-kind non_conforming entry, grouped by the cell it names.

    A given literal cell can appear at most twice (once per side it
    borders), since a row cell only ever has two neighbors — this groups
    those sightings back into one entry per cell so the rule emits one
    Issue, not two duplicates, for the README's own H14 shape.
    """
    touchpoints: dict[str, list[tuple[Block, NonConformingCell]]] = {}
    for sheet in sheet_blocks:
        for block in sheet.blocks:
            for nc in block.non_conforming:
                if nc.kind == "literal":
                    touchpoints.setdefault(nc.cell, []).append((block, nc))
    return touchpoints


def _severity(bordering_blocks: list[Block]) -> str:
    if len(bordering_blocks) == 2 and bordering_blocks[0].pattern == bordering_blocks[1].pattern:
        return "high"
    return "medium"


def _explanation(cell: str, value: object, bordering_blocks: list[Block]) -> str:
    formatted_value = _format_value(value)
    spans = [f"{b.sheet}!{b.span}" for b in bordering_blocks]
    patterns = [b.pattern for b in bordering_blocks]

    if len(bordering_blocks) == 1:
        return (
            f"{cell} is a hardcoded value ({formatted_value}) immediately adjacent to a "
            f"block of formulas ({spans[0]}) following the pattern {patterns[0]}."
        )
    if patterns[0] == patterns[1]:
        return (
            f"{cell} is a hardcoded value ({formatted_value}) inside what would otherwise be "
            f"one continuous block of formulas: {spans[0]} and {spans[1]} both follow the "
            f"pattern {patterns[0]}."
        )
    return (
        f"{cell} is a hardcoded value ({formatted_value}) sitting between two "
        f"differently-patterned formula blocks: {spans[0]} follows {patterns[0]} and "
        f"{spans[1]} follows {patterns[1]}."
    )


def _suggested_fix(bordering_blocks: list[Block]) -> str:
    patterns = sorted({b.pattern for b in bordering_blocks})
    if len(patterns) == 1:
        return (
            f"If this cell were meant to follow the surrounding pattern, it would contain a "
            f"formula matching the R1C1 pattern {patterns[0]} (relative to this cell's own "
            f"position) instead of a hardcoded value."
        )
    return (
        f"This cell sits at a boundary between two different formula patterns "
        f"({patterns[0]} and {patterns[1]}); if it was meant to follow one of them rather than "
        f"being a deliberate override, review which side's pattern actually applies here."
    )


class LiteralInBlockRule(Rule):
    rule_id = RULE_ID

    def evaluate(self, sheet_blocks: list[SheetBlocks]) -> list[Issue]:
        issues: list[Issue] = []
        for cell, pairs in _literal_touchpoints(sheet_blocks).items():
            bordering_blocks = [block for block, _nc in pairs]
            value = pairs[0][1].value
            issues.append(
                Issue(
                    cell=cell,
                    severity=_severity(bordering_blocks),
                    rule_id=RULE_ID,
                    explanation=_explanation(cell, value, bordering_blocks),
                    suggested_fix=_suggested_fix(bordering_blocks),
                )
            )
        return issues
