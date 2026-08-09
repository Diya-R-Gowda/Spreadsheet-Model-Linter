"""reference-to-blank — the third Tier 0 rule.

Flags a block member's formula that reads from a genuinely blank
precedent (`DependencyGraph.is_empty()`), when at least one *sibling*
block member — sharing the identical R1C1 pattern — has a real,
populated precedent at the time this rule runs. A range or cell ref
pointing at nothing is often a copy-paste artifact or an off-by-one that
landed one row/column short of real data.

WHY THIS IS SCOPED TO BLOCK MEMBERS ONLY (investigated before writing
any rule logic, same discipline as stages 6a/6b)
------------------------------------------------------------------------
Real financial models legitimately contain blank cells that are NOT
bugs: spacer rows/columns between sections, label columns, a cell
deliberately left blank as a placeholder for a future period. There is
no reliable "is this a total row" style semantic signal available at
Tier 0 to tell those apart from a genuine data gap — same conclusion as
literal_in_block.py's NOT BUILT section, and the same reason it isn't
invented here either.

`DependencyGraph.is_empty()` was confirmed (not assumed) to conflate two
different things into one `True`: a blank cell that exists within the
used range, and a reference to a cell/sheet that doesn't exist at all.
That conflation turns out not to matter for this rule's design, since
the comparison below never depends on *why* a precedent is empty.

Three scenarios were constructed and their real output inspected before
deciding scope:

  1. A genuine gap in a row of raw data feeding a block of trailing-sum
     formulas: 3 of 7 block members came back with a blank precedent,
     4 came back completely clean — a real minority-vs-majority split.
  2. An isolated one-off formula referencing a blank spacer cell, not
     part of any block: zero blocks formed, so there is nothing to
     compare it against — no signal available at all.
  3. Every member of a block uniformly referencing the same unfilled
     shared driver cell (e.g. `$H$15`, a discount rate nobody filled
     in yet): every member comes back with the identical blank
     precedent — no minority, nothing anomalous.

Only case 1 has real, deterministic supporting evidence: the block's own
membership already proves the position *can* hold data (siblings have
it), so a minority of members lacking it is a genuine anomaly, not a
guess. Cases 2 and 3 have no such evidence and are deliberately never
flagged — this was raised as an explicit scope decision (not decided
silently) and confirmed: ship narrow, block members only.

REQUIRES THE DEPENDENCY GRAPH — WHY base.py's Rule.evaluate() WAS WIDENED
--------------------------------------------------------------------------
This is the first Tier 0 rule that needs `DependencyGraph.precedents()`/
`is_empty()` directly — blocks.py's own output (block membership, plus
non_conforming/near_misses for *neighbors*) has no notion of a member's
own precedents. Rather than re-deriving precedent/empty-cell information
from scratch inside this module (duplicating depgraph.py's formula
re-parsing, range expansion, and named-range resolution), `Rule.evaluate`
in base.py was widened to accept an optional `graph` parameter
(`graph: DependencyGraph | None = None`) — confirmed backward compatible:
`literal_in_block.py`/`range_boundary.py` and their existing tests need
graph, so their call sites are entirely unaffected. This rule requires it
and raises if it isn't supplied.

SEVERITY: TWO LEVELS, FROM THE ACTUAL MEMBER COUNTS
--------------------------------------------------------
"high": clean members (no blank precedent) outnumber affected members
within the block — the populated case is clearly the established norm,
so this is a clear anomaly. "medium": affected members are equal to or
outnumber clean members — a blank precedent is still real, available
evidence (proven possible by the clean members that do exist), but the
argument that this is clearly the *unusual* case is weaker. No new
member-count threshold is invented beyond this direct majority
comparison — the same standard applied to severity in stages 6a/6b.
"""

from __future__ import annotations

from ..blocks import Block, SheetBlocks
from ..depgraph import DependencyGraph
from .base import Issue, Rule

RULE_ID = "reference-to-blank"


def _blank_precedents(cell: str, graph: DependencyGraph) -> list[str]:
    return [p for p in graph.precedents(cell) if graph.is_empty(p)]


def _severity(clean_count: int, affected_count: int) -> str:
    return "high" if clean_count > affected_count else "medium"


def _explanation(cell: str, block: Block, blanks: list[str], clean_count: int, affected_count: int) -> str:
    span = f"{block.sheet}!{block.span}"
    blank_list = ", ".join(blanks)
    return (
        f"{cell} references {'a blank cell' if len(blanks) == 1 else 'blank cells'} "
        f"({blank_list}) with no value or formula, but {clean_count} other cell"
        f"{'s' if clean_count != 1 else ''} in {span} (following {block.pattern}) "
        f"reference real data at the equivalent position; only {affected_count} of "
        f"{len(block.cells)} members in this block lack it."
    )


def _suggested_fix(blanks: list[str]) -> str:
    blank_list = ", ".join(blanks)
    return (
        f"Verify whether {blank_list} should contain a value — most of this block's members "
        f"reference populated data at the equivalent position, so this looks like a gap rather "
        f"than an intentional blank."
    )


class ReferenceToBlankRule(Rule):
    rule_id = RULE_ID

    def evaluate(self, sheet_blocks: list[SheetBlocks], graph: DependencyGraph | None = None) -> list[Issue]:
        if graph is None:
            raise ValueError(f"{RULE_ID} requires the dependency graph; pass graph=...")

        issues: list[Issue] = []
        for sheet in sheet_blocks:
            for block in sheet.blocks:
                member_blanks: dict[str, list[str]] = {}
                for member in block.cells:
                    blanks = _blank_precedents(member, graph)
                    if blanks:
                        member_blanks[member] = blanks

                clean_count = len(block.cells) - len(member_blanks)
                affected_count = len(member_blanks)
                if clean_count == 0 or affected_count == 0:
                    continue  # no cross-sibling contrast -> no evidence, stay silent

                for member, blanks in member_blanks.items():
                    issues.append(
                        Issue(
                            cell=member,
                            severity=_severity(clean_count, affected_count),
                            rule_id=RULE_ID,
                            explanation=_explanation(member, block, blanks, clean_count, affected_count),
                            suggested_fix=_suggested_fix(blanks),
                        )
                    )
        return issues
