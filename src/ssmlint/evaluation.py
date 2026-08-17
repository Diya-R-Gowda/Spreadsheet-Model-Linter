"""Week 4 baseline evaluation harness — stage [after 6] in the build plan.

Scores the Tier 0 rule engine's real output against the synthetic-
corruption corpus's per-cell ground truth (`corpus/*.ground_truth.json`,
built by `scripts/generate_corpus.py`). This module holds the reusable
scoring/metrics logic; `scripts/run_evaluation.py` is the thin CLI that
drives it against `corpus/` and prints the report. Split this way
because Week 5's build plan explicitly re-runs "the Week 4 evaluation
with the classifier layered on top" — the scoring logic needs to be
importable and reusable, not locked inside a one-off script the way
`generate_fixtures.py`/`generate_corpus.py` are (those only ever produce
artifacts; this produces a result other stages will build on).

SCORING DEFINITIONS (investigated and confirmed with the user before
implementation — see the four findings this stage's investigation
turn reported)
------------------------------------------------------------------------
For a given rule's flagged cell, checked against that corpus entry's
ground truth for the SAME cell:
  - true_positive: ground truth state is "injected_bug" AND its
    `rule_id` matches the rule that flagged it.
  - known_intentional (excluded, not scored as an error): ground truth
    state is "known_intentional" — e.g. a growth-chain's seed literal,
    which is structurally indistinguishable from a real hardcode at
    Tier 0. Per the Week 4 design note, folding this into false-positive
    counts would drag precision down for a reason unrelated to real
    detection quality. Tracked and reported separately rather than
    silently dropped, so the count stays visible to anyone reading the
    report, not just someone who read the investigation that decided it.
  - ambiguous_excluded (excluded, not scored as an error; added during
    the 2026-08-17 corpus-expansion follow-up): ground truth state is
    "ambiguous" — a deliberately weak-evidence construction meant to
    carry the `unknown` block-level label (see labeling.py), not a
    Tier-0-detection failure. Tier 0 has no concept of ambiguity and
    flags these deterministically, same as it would an injected_bug at
    the identical shape — same reasoning as known_intentional above,
    handled identically (own bucket, excluded from precision, reported
    not hidden).
  - false_positive: anything else a rule flags — ground truth "clean",
    or an "injected_bug" cell belonging to a DIFFERENT rule_id (the
    wrong rule fired on someone else's bug), or (structurally never
    observed, but handled) a cell absent from ground truth entirely.
  - false_negative: a ground-truth "injected_bug" cell for rule X that
    rule X never flagged. Has no severity (severity is a property of an
    Issue a rule actually raised, not of an abstract missed bug), so
    it's only ever reported at the rolled-up per-rule_id level, never
    sliced by severity.

SLICING: precision is reported both per (rule_id, severity) and rolled
up per rule_id, per the design note's explicit "sliced by (rule_id,
severity)" — matching the README's eventual ablation-table shape
(a granular table plus a headline number). Recall has no severity axis
(see false_negative above) and is reported per rule_id only.

PRECISION@10: no per-issue ranking signal exists anywhere in Stage 6 —
every rule's `evaluate()` returns issues in dict-iteration order, not a
deliberate confidence order. The only real signal available is each
rule's own two-level `severity`. Top-10 is computed per rule_id, pooling
that rule's issues across the whole corpus, sorted by severity
descending then a deterministic (entry, cell) tiebreak (there is nothing
finer to break ties with), taking the first min(10, total) and scoring
precision on just that slice.

THE INTERNAL-CONSISTENCY CAVEAT
------------------------------------------------------------------------
Every injection in this corpus is a surgical, structurally-exact match
to what its target rule's own predicate checks for (verified per-
injection in `tests/test_generate_corpus.py`). A 1.0/1.0 result is
therefore an expected proof that the corpus generator and the rules
agree with each other — not a real-world precision measurement.
`INTERNAL_CONSISTENCY_CAVEAT` below is printed by the report formatter
itself, not left in a docstring or commit message, so anyone running the
harness sees it without having read this investigation.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path

from .blocks import detect_blocks
from .depgraph import DependencyGraph, build_graph
from .parser import parse_workbook
from .rules import (
    InconsistentAnchoringRule,
    Issue,
    LiteralInBlockRule,
    RangeBoundaryRule,
    ReferenceToBlankRule,
    Rule,
)

DEFAULT_RULES: dict[str, Rule] = {
    "literal-in-formula-block": LiteralInBlockRule(),
    "range-boundary-mismatch": RangeBoundaryRule(),
    "reference-to-blank": ReferenceToBlankRule(),
    "inconsistent-anchoring": InconsistentAnchoringRule(),
}
SEVERITIES = ("high", "medium")
PRECISION_AT_K = 10

INTERNAL_CONSISTENCY_CAVEAT = (
    "CAVEAT: every injection in this corpus is a surgical, structurally-exact match to what its "
    "target rule's own predicate checks for (verified per-injection in tests/test_generate_corpus.py). "
    "A 1.0/1.0 precision/recall result reflects internal consistency between the corpus generator and "
    "the rules it was built to validate -- it confirms the rules and the corpus agree with each other, "
    "not that the rules will perform this well against messy, ambiguous, real-world spreadsheets. "
    "Treat this as a correctness check on the pipeline, not a real-world precision claim."
)


@dataclass(frozen=True)
class ScoredIssue:
    entry: str
    cell: str
    severity: str
    outcome: str  # "true_positive" | "false_positive" | "known_intentional" | "ambiguous_excluded"


def is_clean_baseline(ground_truth: dict) -> bool:
    """An entry is a clean baseline iff no cell in it is ground-truthed as
    an injected bug -- generator-naming-agnostic on purpose, so this
    doesn't silently drift out of sync with `generate_corpus.py`'s own
    filename conventions.
    """
    return not any(c["state"] == "injected_bug" for c in ground_truth["cells"])


def score_rule_on_entry(entry_name: str, issues: list[Issue], ground_truth: dict, rule_id: str) -> list[ScoredIssue]:
    """Scores one rule's Issues from one corpus entry against that entry's
    ground truth. False negatives are computed separately (see
    `count_false_negatives`) since a miss has no severity to attach a
    ScoredIssue to.
    """
    gt_by_cell = {c["cell"]: c for c in ground_truth["cells"]}
    scored: list[ScoredIssue] = []
    for issue in issues:
        gt_cell = gt_by_cell.get(issue.cell)
        if gt_cell is not None and gt_cell["state"] == "injected_bug" and gt_cell["rule_id"] == rule_id:
            outcome = "true_positive"
        elif gt_cell is not None and gt_cell["state"] == "known_intentional":
            outcome = "known_intentional"
        elif gt_cell is not None and gt_cell["state"] == "ambiguous":
            outcome = "ambiguous_excluded"
        else:
            outcome = "false_positive"
        scored.append(ScoredIssue(entry=entry_name, cell=issue.cell, severity=issue.severity, outcome=outcome))
    return scored


def count_false_negatives(issues: list[Issue], ground_truth: dict, rule_id: str) -> int:
    flagged_cells = {i.cell for i in issues}
    return sum(
        1
        for c in ground_truth["cells"]
        if c["state"] == "injected_bug" and c["rule_id"] == rule_id and c["cell"] not in flagged_cells
    )


@dataclass
class SeveritySlice:
    rule_id: str
    severity: str
    tp: int = 0
    fp: int = 0
    known_intentional_excluded: int = 0
    ambiguous_excluded: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "tp": self.tp,
            "fp": self.fp,
            "known_intentional_excluded": self.known_intentional_excluded,
            "ambiguous_excluded": self.ambiguous_excluded,
            "precision": self.precision,
        }


@dataclass
class RuleRollup:
    rule_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    known_intentional_excluded: int = 0
    ambiguous_excluded: int = 0
    precision_at_10: float | None = None
    precision_at_10_k: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "known_intentional_excluded": self.known_intentional_excluded,
            "ambiguous_excluded": self.ambiguous_excluded,
            "precision": self.precision,
            "recall": self.recall,
            "precision_at_10": self.precision_at_10,
            "precision_at_10_k": self.precision_at_10_k,
        }


@dataclass
class CleanBaselineResult:
    entries_checked: int
    raw_flags: list[ScoredIssue] = field(default_factory=list)
    unexplained_flags: list[ScoredIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entries_checked": self.entries_checked,
            "raw_flag_count": len(self.raw_flags),
            "unexplained_flag_count": len(self.unexplained_flags),
            "raw_flags": [{"entry": s.entry, "cell": s.cell, "severity": s.severity} for s in self.raw_flags],
            "unexplained_flags": [
                {"entry": s.entry, "cell": s.cell, "severity": s.severity} for s in self.unexplained_flags
            ],
        }


@dataclass
class EvaluationReport:
    slices: list[SeveritySlice]
    rollups: list[RuleRollup]
    clean_baseline: CleanBaselineResult
    caveat: str = INTERNAL_CONSISTENCY_CAVEAT

    def to_dict(self) -> dict:
        return {
            "caveat": self.caveat,
            "slices": [s.to_dict() for s in self.slices],
            "rollups": [r.to_dict() for r in self.rollups],
            "clean_baseline": self.clean_baseline.to_dict(),
        }


def evaluate_rule(rule: Rule, sheet_blocks, graph: DependencyGraph) -> list[Issue]:
    """Calls `rule.evaluate()` with `graph` only if the rule's own
    signature actually accepts it -- `LiteralInBlockRule` and
    `RangeBoundaryRule` don't take a `graph` parameter at all (confirmed
    by reading their current source, not assumed from the shared `Rule`
    ABC's optional-graph signature), so passing it unconditionally would
    raise `TypeError` for those two. Introspecting instead of hardcoding
    a per-rule branch so this keeps working if Week 5 adds more rules.
    """
    params = inspect.signature(rule.evaluate).parameters
    if "graph" in params:
        return rule.evaluate(sheet_blocks, graph=graph)
    return rule.evaluate(sheet_blocks)


def _load_corpus_entries(corpus_dir: Path) -> list[tuple[str, Path, dict]]:
    entries = []
    for gt_path in sorted(corpus_dir.glob("*.ground_truth.json")):
        name = gt_path.stem.removesuffix(".ground_truth")
        xlsx_path = corpus_dir / f"{name}.xlsx"
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        entries.append((name, xlsx_path, ground_truth))
    return entries


def run_evaluation(
    corpus_dir: Path, rules: dict[str, Rule] | None = None, entry_names: set[str] | None = None
) -> EvaluationReport:
    """`entry_names`, when given, restricts scoring to only those corpus
    entries (by their entry name, e.g. "range_boundary__edge_left__n4") --
    used by classifier.py to re-score Tier 0 on only a held-out test
    split, for a fair Tier 0 vs. Tier 0+1 comparison (Week 5's own design
    note: the full-corpus number can't be reused as-is for that
    comparison). `None` (the default) preserves the original whole-corpus
    behavior exactly -- every existing call site is unaffected.
    """
    rules = rules if rules is not None else DEFAULT_RULES
    entries = _load_corpus_entries(corpus_dir)
    if entry_names is not None:
        entries = [e for e in entries if e[0] in entry_names]
    if not entries:
        raise ValueError(f"No corpus entries found in {corpus_dir} -- run scripts/generate_corpus.py first")

    slices: dict[tuple[str, str], SeveritySlice] = {
        (rid, sev): SeveritySlice(rule_id=rid, severity=sev) for rid in rules for sev in SEVERITIES
    }
    fn_by_rule: dict[str, int] = {rid: 0 for rid in rules}
    all_scored_by_rule: dict[str, list[ScoredIssue]] = {rid: [] for rid in rules}
    clean_baseline_scored: list[ScoredIssue] = []
    clean_baseline_count = 0

    for name, xlsx_path, ground_truth in entries:
        parsed = parse_workbook(xlsx_path)
        graph: DependencyGraph = build_graph(parsed)
        sheet_blocks = detect_blocks(parsed, graph)
        entry_is_clean_baseline = is_clean_baseline(ground_truth)
        if entry_is_clean_baseline:
            clean_baseline_count += 1

        for rule_id, rule in rules.items():
            issues = evaluate_rule(rule, sheet_blocks, graph)
            scored = score_rule_on_entry(name, issues, ground_truth, rule_id)
            all_scored_by_rule[rule_id].extend(scored)
            fn_by_rule[rule_id] += count_false_negatives(issues, ground_truth, rule_id)

            for s in scored:
                key = (rule_id, s.severity)
                if s.outcome == "true_positive":
                    slices[key].tp += 1
                elif s.outcome == "known_intentional":
                    slices[key].known_intentional_excluded += 1
                elif s.outcome == "ambiguous_excluded":
                    slices[key].ambiguous_excluded += 1
                else:
                    slices[key].fp += 1

            if entry_is_clean_baseline:
                clean_baseline_scored.extend(scored)

    rollups: list[RuleRollup] = []
    for rule_id in rules:
        tp = sum(slices[(rule_id, sev)].tp for sev in SEVERITIES)
        fp = sum(slices[(rule_id, sev)].fp for sev in SEVERITIES)
        known = sum(slices[(rule_id, sev)].known_intentional_excluded for sev in SEVERITIES)
        ambiguous = sum(slices[(rule_id, sev)].ambiguous_excluded for sev in SEVERITIES)
        fn = fn_by_rule[rule_id]

        # known_intentional flags are excluded from the ranking pool itself, not just from
        # the final precision count -- otherwise they can crowd out real TP/FP entries from
        # the top-10 window purely on alphabetical tiebreak, silently shrinking the effective
        # k below 10 for no reason related to ranking quality. Confirmed this actually happens:
        # an earlier version that pooled all outcomes and filtered afterward produced k=5 for
        # literal-in-formula-block, purely because clean__growth_chain__* (known_intentional)
        # entries sort alphabetically before literal_in_block__* (the real TP/FP entries).
        severity_rank = {"high": 0, "medium": 1}
        scoreable = [s for s in all_scored_by_rule[rule_id] if s.outcome in ("true_positive", "false_positive")]
        pool = sorted(scoreable, key=lambda s: (severity_rank.get(s.severity, 99), s.entry, s.cell))
        top_k = pool[:PRECISION_AT_K]
        k = len(top_k)
        top_k_tp = sum(1 for s in top_k if s.outcome == "true_positive")
        precision_at_10 = top_k_tp / k if k else None

        rollups.append(
            RuleRollup(
                rule_id=rule_id, tp=tp, fp=fp, fn=fn, known_intentional_excluded=known,
                ambiguous_excluded=ambiguous, precision_at_10=precision_at_10, precision_at_10_k=k,
            )
        )

    # ambiguous_excluded counts as an expected, explained flag here too -- same treatment as
    # known_intentional. Since corpus expansion, "clean baseline" (no injected_bug cell) now also
    # covers subtotal/ambiguous-only entries (deliberately different-but-legitimate or
    # deliberately weak-evidence content, not pristine models) -- is_clean_baseline's definition
    # itself is unchanged (still just "no injected_bug cell"), but its real-world population grew.
    raw_flags = [
        s for s in clean_baseline_scored if s.outcome in ("false_positive", "known_intentional", "ambiguous_excluded")
    ]
    unexplained_flags = [s for s in clean_baseline_scored if s.outcome == "false_positive"]

    return EvaluationReport(
        slices=[slices[(rid, sev)] for rid in rules for sev in SEVERITIES],
        rollups=rollups,
        clean_baseline=CleanBaselineResult(
            entries_checked=clean_baseline_count, raw_flags=raw_flags, unexplained_flags=unexplained_flags
        ),
    )


def format_report_text(report: EvaluationReport) -> str:
    lines = []
    lines.append("=== Week 4 Baseline Evaluation: Tier 0 Rule Engine vs. Synthetic-Corruption Corpus ===")
    lines.append("")
    lines.append(report.caveat)
    lines.append("")

    lines.append("--- Per (rule_id, severity) ---")
    header = (
        f"{'rule_id':<28}{'severity':<10}{'TP':>4}{'FP':>4}{'known_intentional_excl':>24}"
        f"{'ambiguous_excl':>16}{'precision':>12}"
    )
    lines.append(header)
    for s in report.slices:
        prec = f"{s.precision:.3f}" if s.precision is not None else "n/a"
        lines.append(
            f"{s.rule_id:<28}{s.severity:<10}{s.tp:>4}{s.fp:>4}{s.known_intentional_excluded:>24}"
            f"{s.ambiguous_excluded:>16}{prec:>12}"
        )
    lines.append("")

    lines.append("--- Rolled up per rule_id ---")
    header2 = (
        f"{'rule_id':<28}{'TP':>4}{'FP':>4}{'FN':>4}{'known_intentional_excl':>24}{'ambiguous_excl':>16}"
        f"{'precision':>12}{'recall':>10}{'precision@10 (k)':>20}"
    )
    lines.append(header2)
    for r in report.rollups:
        prec = f"{r.precision:.3f}" if r.precision is not None else "n/a"
        rec = f"{r.recall:.3f}" if r.recall is not None else "n/a"
        p10 = f"{r.precision_at_10:.3f} (k={r.precision_at_10_k})" if r.precision_at_10 is not None else "n/a"
        lines.append(
            f"{r.rule_id:<28}{r.tp:>4}{r.fp:>4}{r.fn:>4}{r.known_intentional_excluded:>24}{r.ambiguous_excluded:>16}"
            f"{prec:>12}{rec:>10}{p10:>20}"
        )
    lines.append("")

    cb = report.clean_baseline
    lines.append("--- Clean-baseline false positives ---")
    lines.append(f"Entries checked: {cb.entries_checked}")
    lines.append(
        f"Raw flags (README-style, unfiltered -- includes expected known_intentional/ambiguous_excluded "
        f"flags): {len(cb.raw_flags)}"
    )
    for s in cb.raw_flags:
        lines.append(f"  {s.entry}: {s.cell} (severity={s.severity}, outcome={s.outcome})")
    lines.append(
        f"Unexplained flags (excluding known_intentional/ambiguous_excluded -- the real false-positive count): "
        f"{len(cb.unexplained_flags)}"
    )
    for s in cb.unexplained_flags:
        lines.append(f"  {s.entry}: {s.cell} (severity={s.severity})")

    return "\n".join(lines)
