"""range-boundary-mismatch — the second Tier 0 rule.

Flags a formula whose RANGE ARGUMENT (a `SUM(...)`/`AVERAGE(...)`/bare
range/etc. — anything containing a `RangeRef`) has a different span than
its block-mates', even though the block-mates' own pattern is otherwise
identical — the classic "someone extended the data by a column and forgot
to update one SUM" bug.

WHERE THIS ACTUALLY SURFACES (investigated before writing any rule logic)
--------------------------------------------------------------------------
A block only exists because its members share one exact R1C1 string, so
an off-by-one range can never be an actual block *member* by definition
— it necessarily breaks clustering and shows up as a neighbor. The
question was which neighbor bucket. Constructed 7 deliberate off-by-one
scenarios (end-boundary shift, start-boundary shift, 3-digit rows, bare
ranges with no wrapping function, a shift crossing a single-digit ->
double-digit offset) and printed real `detect_blocks()` output for each:
every one landed in `near_misses` (rapidfuzz scores 93.0-96.4), never in
`non_conforming` — because a single-digit change in an otherwise-
identical string reads as "similar" to `fuzz.ratio`. A genuinely
unrelated formula, by contrast, correctly landed in `non_conforming`
with no near-miss at all.

That said, near-miss placement is a similarity-score artifact, not a
structural guarantee: an adversarial minimal case (`R[-9]` vs `R[-10]`
alone, no surrounding function or second endpoint) scores 72.73 —
*below* the near-miss threshold. No realistic whole-formula pattern in
testing ever produced that, but relying on "off-by-one always lands in
near_misses" would silently blind this rule the day a short-enough
formula doesn't. So this rule inspects BOTH `non_conforming`
(`kind == "different_formula"`) and `near_misses` for every block, and
independently verifies the deviation is *specifically* a single range
endpoint shifted by exactly one — by diffing the R1C1 pattern strings
structurally, not by trusting whichever bucket blocks.py sorted it into.

WHAT COUNTS AS "JUST A RANGE BOUNDARY DEVIATION" (SCOPE)
--------------------------------------------------------------
`_find_single_range_boundary_delta` finds every `RangeRef` span in both
the block's pattern and the candidate's pattern (regex over the R1C1
output format, which is fully regular and produced by exactly one
function in this codebase, `r1c1.py`). It only reports a match when:
  - both patterns have the same number of range spans,
  - everything *outside* those range spans is byte-for-byte identical
    (same function name, same nesting, same other operands — verified by
    substituting a placeholder for every range span and comparing what's
    left),
  - across all range spans, exactly one span differs, and within that
    span exactly one of its four sub-parts (start row, start col, end
    row, end col) differs, by exactly 1, with both sides using the same
    anchoring (both relative-bracketed or both absolute).
Anything else (a different function, a shift of more than one row/col, a
change to more than one boundary at once, a cross-sheet difference) is
left alone — that is a different, unrelated deviation for
`literal-in-formula-block` or a future rule to own, not this one.

SEVERITY: THE SAME TWO-LEVEL SCHEME AS STAGE 6a, REUSED DELIBERATELY
--------------------------------------------------------------------------
"high": the deviant cell borders blocks on both sides and both produce a
valid, *matching* boundary delta against it (a single interrupted range
pattern — unambiguous). "medium": everything else this rule still flags
(bordered on one side only, or the two sides disagree/only one side's
comparison actually resolves to a clean single-boundary delta). This is
the identical structural argument used in `literal_in_block.py` — a
single, doubly-confirmed pattern interruption is the strongest evidence
available regardless of what kind of deviation is being evaluated — so
it's reused rather than inventing a new numeric member-count threshold
that would need its own justification. The number of block-mates sharing
the expected boundary is still real, available signal, and is surfaced
in the explanation text (e.g. "5 other cells... sum a 5-row range") as
supporting evidence, without being turned into an extra, arbitrarily-
placed severity tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..blocks import Block, SheetBlocks
from .base import Issue, Rule

RULE_ID = "range-boundary-mismatch"

_RC_TOKEN = r"R(?:\[-?\d+\]|\d+)C(?:\[-?\d+\]|\d+)"
_RANGE_RE = re.compile(
    r"(?:(?P<sheet>[A-Za-z_][A-Za-z0-9_.]*|'[^']+')!)?"
    rf"(?P<start>{_RC_TOKEN}):(?P<end>{_RC_TOKEN})"
)
_RC_PARTS_RE = re.compile(r"R(\[-?\d+\]|\d+)C(\[-?\d+\]|\d+)")


@dataclass(frozen=True)
class _RangeDelta:
    axis: str  # "row" | "col"
    boundary: str  # "start" | "end"
    expected: int
    actual: int
    expected_count: int
    actual_count: int


def _parse_rc(token: str) -> tuple[str, str]:
    match = _RC_PARTS_RE.match(token)
    assert match is not None  # guaranteed by _RANGE_RE having already matched
    return match.group(1), match.group(2)


def _numeric(part: str) -> int:
    return int(part.strip("[]"))


def _diff_one_range(block_match: re.Match, candidate_match: re.Match) -> _RangeDelta | None:
    if block_match.group("sheet") != candidate_match.group("sheet"):
        return None  # a sheet difference isn't a boundary issue

    b_start_row, b_start_col = _parse_rc(block_match.group("start"))
    b_end_row, b_end_col = _parse_rc(block_match.group("end"))
    c_start_row, c_start_col = _parse_rc(candidate_match.group("start"))
    c_end_row, c_end_col = _parse_rc(candidate_match.group("end"))

    parts = [
        ("start", "row", b_start_row, c_start_row),
        ("start", "col", b_start_col, c_start_col),
        ("end", "row", b_end_row, c_end_row),
        ("end", "col", b_end_col, c_end_col),
    ]
    differing = [(boundary, axis, b, c) for boundary, axis, b, c in parts if b != c]
    if len(differing) != 1:
        return None  # zero, or more than one, sub-part differs

    boundary, axis, b_raw, c_raw = differing[0]
    if b_raw.startswith("[") != c_raw.startswith("["):
        return None  # one side relative, one absolute -> not a clean boundary shift

    b_val, c_val = _numeric(b_raw), _numeric(c_raw)
    if abs(c_val - b_val) != 1:
        return None

    if axis == "row":
        b_other_row, c_other_row = (
            (_numeric(b_end_row), _numeric(c_end_row))
            if boundary == "start"
            else (_numeric(b_start_row), _numeric(c_start_row))
        )
        expected_count = abs(b_val - b_other_row) + 1
        actual_count = abs(c_val - c_other_row) + 1
    else:
        b_other_col, c_other_col = (
            (_numeric(b_end_col), _numeric(c_end_col))
            if boundary == "start"
            else (_numeric(b_start_col), _numeric(c_start_col))
        )
        expected_count = abs(b_val - b_other_col) + 1
        actual_count = abs(c_val - c_other_col) + 1

    return _RangeDelta(
        axis=axis, boundary=boundary, expected=b_val, actual=c_val,
        expected_count=expected_count, actual_count=actual_count,
    )


def _find_single_range_boundary_delta(block_pattern: str, candidate_pattern: str) -> _RangeDelta | None:
    block_ranges = list(_RANGE_RE.finditer(block_pattern))
    candidate_ranges = list(_RANGE_RE.finditer(candidate_pattern))
    if not block_ranges or len(block_ranges) != len(candidate_ranges):
        return None

    block_skeleton = _RANGE_RE.sub("\0", block_pattern)
    candidate_skeleton = _RANGE_RE.sub("\0", candidate_pattern)
    if block_skeleton != candidate_skeleton:
        return None  # something outside the range argument(s) also differs

    deltas: list[_RangeDelta] = []
    for block_match, candidate_match in zip(block_ranges, candidate_ranges):
        if block_match.group(0) == candidate_match.group(0):
            continue
        delta = _diff_one_range(block_match, candidate_match)
        if delta is None:
            return None
        deltas.append(delta)

    return deltas[0] if len(deltas) == 1 else None


def _range_boundary_touchpoints(
    sheet_blocks: list[SheetBlocks],
) -> dict[str, list[tuple[Block, _RangeDelta]]]:
    touchpoints: dict[str, list[tuple[Block, _RangeDelta]]] = {}
    for sheet in sheet_blocks:
        for block in sheet.blocks:
            candidates: list[tuple[str, str | None]] = [
                (nc.cell, nc.formula_r1c1)
                for nc in block.non_conforming
                if nc.kind == "different_formula"
            ]
            candidates += [(nm.cell, nm.formula_r1c1) for nm in block.near_misses]
            for cell, pattern in candidates:
                if pattern is None:
                    continue
                delta = _find_single_range_boundary_delta(block.pattern, pattern)
                if delta is not None:
                    touchpoints.setdefault(cell, []).append((block, delta))
    return touchpoints


def _severity(bordering: list[tuple[Block, _RangeDelta]]) -> str:
    if len(bordering) == 2 and bordering[0][0].pattern == bordering[1][0].pattern:
        return "high"
    return "medium"


def _describe_boundary(delta: _RangeDelta) -> str:
    unit = "row" if delta.axis == "row" else "column"
    direction = "starts" if delta.boundary == "start" else "ends"
    if delta.actual_count < delta.expected_count:
        shift = f"{direction} one {unit} short of"
    else:
        shift = f"{direction} one {unit} past"
    return f"{shift} the expected boundary"


def _explanation(cell: str, bordering: list[tuple[Block, _RangeDelta]]) -> str:
    parts = []
    for block, delta in bordering:
        span = f"{block.sheet}!{block.span}"
        parts.append(
            f"{span} (following {block.pattern}, {delta.expected_count} cells per range argument)"
        )
    joined = " and ".join(parts)
    _block, delta = bordering[0]
    return (
        f"{cell}'s range argument covers {delta.actual_count} cells, but its block-mates in "
        f"{joined} cover {delta.expected_count} cells each — this formula {_describe_boundary(delta)}."
    )


def _suggested_fix(bordering: list[tuple[Block, _RangeDelta]]) -> str:
    patterns = sorted({block.pattern for block, _delta in bordering})
    if len(patterns) == 1:
        return (
            f"If this cell were meant to follow the surrounding pattern, its range argument would "
            f"match the R1C1 pattern {patterns[0]} (relative to this cell's own position), covering "
            f"the same number of cells as its block-mates, instead of the shifted boundary it has now."
        )
    return (
        f"This cell sits between blocks with different patterns ({patterns[0]} and {patterns[1]}); "
        f"review which side's range boundary this cell was actually meant to follow."
    )


class RangeBoundaryRule(Rule):
    rule_id = RULE_ID

    def evaluate(self, sheet_blocks: list[SheetBlocks]) -> list[Issue]:
        issues: list[Issue] = []
        for cell, bordering in _range_boundary_touchpoints(sheet_blocks).items():
            issues.append(
                Issue(
                    cell=cell,
                    severity=_severity(bordering),
                    rule_id=RULE_ID,
                    explanation=_explanation(cell, bordering),
                    suggested_fix=_suggested_fix(bordering),
                )
            )
        return issues
