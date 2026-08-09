"""Shared types for the Tier 0 rule engine — stage [6] in the architecture.

Every rule turns block-detector (stage [5]) output into `Issue` objects:
the one shared output shape every rule — and the Week 4 evaluation
harness that measures them — produces and consumes. Per the README's
Output spec, an issue is a cell address, a severity, a human-readable
explanation, and a suggested fix; `rule_id` is added so a report (or the
evaluation harness) can group/filter issues by which deterministic check
raised them.

Tier 0 issues are deliberately never labeled `intentional_override`,
`subtotal`, or similar — that 4-way judgment call
(`subtotal | intentional_override | suspected_error | unknown`) belongs
to the Tier 1/2 semantic layer, per the README's own Failure Modes table
("`intentional_override` is a first-class label at Tier 1/2, not lumped
in with errors"). Every Tier 0 issue is implicitly a `suspected_error`
candidate for that later layer to adjudicate, not a final verdict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..blocks import SheetBlocks
from ..depgraph import DependencyGraph


@dataclass(frozen=True)
class Issue:
    cell: str  # sheet-qualified address, e.g. "Model!H14"
    severity: str  # rule-specific levels; see each rule's own docstring for how they're derived
    rule_id: str  # stable identifier, e.g. "literal-in-formula-block"
    explanation: str  # must name the actual pattern/deviation involved, never a generic template
    suggested_fix: str  # describes the conforming pattern in words — never a ready-to-paste formula


class Rule(ABC):
    """Base interface every Tier 0 rule implements.

    Deliberately thin: a stable `rule_id` plus one `evaluate()` method
    turning block-detector output into a list of Issues. Tier 0 rules are
    independent, stateless structural checks — each one is its own
    future prompt/stage, measured in isolation — so there's no shared
    config or lifecycle to justify more than this. The (not-yet-built)
    Week 4 evaluation harness is expected to hold a list of `Rule`
    instances and call `evaluate()` on each uniformly.
    """

    rule_id: str

    @abstractmethod
    def evaluate(
        self, sheet_blocks: list[SheetBlocks], graph: DependencyGraph | None = None
    ) -> list[Issue]:
        """Return zero or more Issues found across all given sheets' blocks.

        `graph` is optional and defaults to None: most Tier 0 rules (e.g.
        literal-in-formula-block, range-boundary-mismatch) need nothing
        beyond blocks.py's own output and never reference it. A rule that
        genuinely needs precedent/empty-cell information (e.g.
        reference-to-blank, which compares block members' precedents
        against each other via DependencyGraph.precedents()/is_empty())
        requires it be passed. Added as an optional parameter rather than
        a required one specifically so existing rules and their call
        sites/tests need no changes.
        """
