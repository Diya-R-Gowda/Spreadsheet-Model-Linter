"""Week 6 ablation table — the README's headline evaluation deliverable:
"Publish the full ablation table (precision, recall, precision@10,
cost, runtime, per tier)." Scoped Tier 0 only, per the user's explicit
decision to defer corpus expansion and real Tier 1/2 work (see
CONTRIBUTING.md's Week 5 follow-up note) — only the "Tier 0 (rules
only)" row carries real numbers; the other two rows are structurally
reserved, not fabricated.

This module never duplicates Week 4's scoring logic — it reads only
`evaluation.py`'s already-public results (`run_evaluation()`,
`RuleRollup`'s public fields via `EvaluationReport.rollups`) and adds
exactly two things `evaluation.py` doesn't have: (1) a single
micro-averaged Tier-0 number aggregated across all four rules (the
existing data model is per-`rule_id` only, with no "all rules combined"
row), and (2) runtime instrumentation (no wall-clock timing exists
anywhere in `evaluation.py`).

WHAT "MEDIAN RUNTIME/WORKBOOK" ACTUALLY MEASURES (confirmed explicitly,
not left implicit)
------------------------------------------------------------------------
Each sample times the genuinely full per-entry pipeline:
`parser.parse_workbook` (a fresh `ParsedWorkbook`, never reused) ->
`depgraph.build_graph` -> `blocks.detect_blocks` (which internally
performs per-cell formula tokenization/AST building and R1C1
normalization via its own `_row_groups` -- nothing extra needs to be
called separately here) -> `evaluation.evaluate_rule` for all four
default rules. So this covers parse -> tokenize -> R1C1 -> depgraph ->
blocks -> rule evaluation, not just rule evaluation against pre-built
blocks. Every corpus entry gets its own fresh call chain: `Rule`
instances are stateless (per `rules/base.py`'s own ABC docstring), and
no `ParsedWorkbook`/graph/blocks object is reused across entries, so
there is no cross-entry cached state inflating or deflating the
measurement. This does mean the corpus gets walked twice per run (once
inside `run_evaluation()`, once in this module's own timing loop) --
accepted deliberately rather than widening `evaluation.py` to add an
internal timing hook, since that would touch a completed-stage-adjacent
file for a minor optimization (47 corpus entries is trivial either way).

PRECISION@10 AGGREGATION IS AN APPROXIMATION (documented, not silent)
------------------------------------------------------------------------
`evaluation.py`'s precision@10 pools each rule's own flags independently
(there is no single global ranking signal across all four rules beyond
severity). The Tier-0 row's aggregate precision@10 here is a weighted
average of each rule's own top-10 result
(`sum(tp_in_top_k) / sum(precision_at_10_k)`), not a true re-ranked
pooled-across-all-rules top-10. This is a real approximation, stated
here rather than presented as something more rigorous than it is.

COST: "$0" IS AN ARCHITECTURAL FACT, NOT A MEASUREMENT
------------------------------------------------------------------------
Confirmed directly with the user before implementation: the README
states cost is $0 across all tiers "by construction" -- no paid API
anywhere in the architecture. This is true whether or not a tier has
been built yet, so all three rows show `cost="$0"`, while
precision/recall/precision_at_10/median_runtime_ms stay `None` for the
two not-yet-built rows, since those genuinely require running the tier.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import evaluation
from .blocks import detect_blocks
from .depgraph import build_graph
from .parser import parse_workbook

TIER_0_LABEL = "Tier 0 (rules only)"
TIER_0_1_LABEL = "Tier 0 + 1 (+ trained classifier)"
TIER_0_1_2_LABEL = "Tier 0 + 1 + 2 (+ local LLM)"

_ARCHITECTURAL_COST = "$0"
_BLOCKED_NOTE = (
    "Blocked -- see CONTRIBUTING.md's Week 5 design note. Corpus expansion (2026-08-17) closed the "
    "original zero-example gap for 'subtotal' and made real progress on 'unknown', but "
    "check_training_readiness() still reports 'intentional_override' and 'unknown' below the "
    "50-per-class floor -- the former has a real, documented low ceiling in the current corpus design, "
    "the latter a real, verified structural ceiling on its weak-evidence injection mechanisms. "
    "Not started here; see scripts/run_labeling.py's own readiness report for exact current counts."
)
_NO_CHECKPOINT_NOTE = (
    "Not measured this run -- pass tier1_checkpoint_dir (scripts/run_ablation.py's "
    "--tier1-checkpoint flag) to a real checkpoint from scripts/train_classifier.py to fill "
    "this row in. A real trained checkpoint now exists (see CONTRIBUTING.md); this row is "
    "just not wired up unless the flag is actually passed."
)
_NO_TIER2_NOTE = (
    "Not measured this run -- pass tier2_model (scripts/run_ablation.py's --tier2-model flag, "
    "e.g. 'qwen2.5:3b-instruct') together with tier1_checkpoint_dir to fill this row in. A real "
    "local Ollama adjudicator now exists (see CONTRIBUTING.md); this row is just not wired up "
    "unless the flag is actually passed."
)


@dataclass(frozen=True)
class AblationRow:
    configuration: str
    status: str  # "measured" | "not_yet_available"
    precision: float | None
    recall: float | None
    precision_at_10: float | None
    cost: str
    median_runtime_ms: float | None
    notes: str

    def to_dict(self) -> dict:
        return {
            "configuration": self.configuration,
            "status": self.status,
            "precision": self.precision,
            "recall": self.recall,
            "precision_at_10": self.precision_at_10,
            "cost": self.cost,
            "median_runtime_ms": self.median_runtime_ms,
            "notes": self.notes,
        }


@dataclass
class AblationTable:
    rows: list[AblationRow] = field(default_factory=list)
    caveat: str = evaluation.INTERNAL_CONSISTENCY_CAVEAT

    def to_dict(self) -> dict:
        return {"caveat": self.caveat, "rows": [r.to_dict() for r in self.rows]}


def _aggregate_tier0_metrics(
    rollups: list[evaluation.RuleRollup],
) -> tuple[float | None, float | None, float | None]:
    """Micro-averaged precision/recall across all rule rollups, plus the
    approximate weighted-average precision@10 (see module docstring).
    Reads only RuleRollup's public fields -- never reaches into
    evaluation.py's private scoring internals. Takes a plain rollup list
    (not a full EvaluationReport) so it works equally for Week 4's
    full-corpus Tier 0 rollups and classifier.evaluate_tier0_plus_1's
    Tier 0+1 rollups, which have no EvaluationReport wrapper of their own.
    """
    total_tp = sum(r.tp for r in rollups)
    total_fp = sum(r.fp for r in rollups)
    total_fn = sum(r.fn for r in rollups)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else None

    total_k = sum(r.precision_at_10_k for r in rollups)
    total_tp_in_top_k = sum(
        round(r.precision_at_10 * r.precision_at_10_k) for r in rollups if r.precision_at_10 is not None
    )
    precision_at_10 = total_tp_in_top_k / total_k if total_k else None

    return precision, recall, precision_at_10


def _time_full_pipeline_per_entry(corpus_dir: Path, rules: dict) -> list[float]:
    """One elapsed-ms sample per corpus entry, timing the full pipeline
    (parse -> graph -> blocks -> all rule evaluations) fresh each time.
    See module docstring for exactly what this does and doesn't cover.
    """
    samples: list[float] = []
    for gt_path in sorted(corpus_dir.glob("*.ground_truth.json")):
        name = gt_path.stem.removesuffix(".ground_truth")
        xlsx_path = corpus_dir / f"{name}.xlsx"

        start = time.perf_counter()
        parsed = parse_workbook(xlsx_path)
        graph = build_graph(parsed)
        sheet_blocks = detect_blocks(parsed, graph)
        for rule in rules.values():
            evaluation.evaluate_rule(rule, sheet_blocks, graph)
        elapsed_ms = (time.perf_counter() - start) * 1000
        samples.append(elapsed_ms)

    return samples


def build_ablation_table(
    corpus_dir: Path,
    rules: dict | None = None,
    tier1_checkpoint_dir: Path | None = None,
    tier2_model: str | None = None,
    tier2_endpoint: str | None = None,
) -> AblationTable:
    """`tier1_checkpoint_dir`, when given (a real checkpoint saved by
    `scripts/train_classifier.py`), fills in the Tier 0+1 row for real:
    Tier 0+1 is scored via `classifier.evaluate_tier0_plus_1` on the
    checkpoint's own recorded held-out test split, compared fairly
    against Tier 0 alone re-scored on that SAME test split (per Week 5's
    design note -- the full-corpus Tier 0 row above is a different,
    larger, and not directly comparable number). Omitting the parameter
    (the default) preserves the exact prior behavior -- every existing
    caller/test is unaffected.

    `tier2_model`, when ALSO given (e.g. "qwen2.5:3b-instruct", a real
    local Ollama model), fills in the Tier 0+1+2 row for real via
    `llm_adjudicator.evaluate_tier0_plus_1_plus_2`. Chained-only, matching
    the README's own ablation table shape (it only ever shows "Tier 0",
    "Tier 0 + 1", "Tier 0 + 1 + 2" -- never "Tier 0 + 2"): passing
    `tier2_model` without `tier1_checkpoint_dir` raises `ValueError`
    immediately, before any real work runs, rather than doing something
    the README never describes.
    """
    if tier2_model is not None and tier1_checkpoint_dir is None:
        raise ValueError(
            "tier2_model requires tier1_checkpoint_dir to also be given -- Tier 2 is a second "
            "opinion layered on top of Tier 1's own predictions, never a standalone alternative "
            "(matches the README's own ablation table, which only ever shows 'Tier 0 + 1 + 2', "
            "never 'Tier 0 + 2')."
        )

    rules = rules if rules is not None else evaluation.DEFAULT_RULES
    report = evaluation.run_evaluation(corpus_dir, rules=rules)
    precision, recall, precision_at_10 = _aggregate_tier0_metrics(report.rollups)

    runtime_samples = _time_full_pipeline_per_entry(corpus_dir, rules)
    median_runtime_ms = statistics.median(runtime_samples) if runtime_samples else None

    tier0_row = AblationRow(
        configuration=TIER_0_LABEL,
        status="measured",
        precision=precision,
        recall=recall,
        precision_at_10=precision_at_10,
        cost=_ARCHITECTURAL_COST,
        median_runtime_ms=median_runtime_ms,
        notes=f"Measured against the {len(list(corpus_dir.glob('*.ground_truth.json')))}-entry synthetic corpus.",
    )

    if tier1_checkpoint_dir is not None:
        from . import classifier

        tier01_rollups, test_entries = classifier.evaluate_tier0_plus_1(corpus_dir, tier1_checkpoint_dir, rules=rules)
        tier01_precision, tier01_recall, tier01_precision_at_10 = _aggregate_tier0_metrics(tier01_rollups)

        tier0_test_report = evaluation.run_evaluation(corpus_dir, rules=rules, entry_names=set(test_entries))
        tier0_test_precision, tier0_test_recall, tier0_test_precision_at_10 = _aggregate_tier0_metrics(
            tier0_test_report.rollups
        )
        intentional_override_note = classifier.describe_intentional_override_confidence(tier1_checkpoint_dir)

        tier01_row = AblationRow(
            configuration=TIER_0_1_LABEL,
            status="measured",
            precision=tier01_precision,
            recall=tier01_recall,
            precision_at_10=tier01_precision_at_10,
            cost=_ARCHITECTURAL_COST,
            median_runtime_ms=None,  # classifier inference latency not measured this pass -- a real, stated gap
            notes=(
                f"Measured on the {len(test_entries)}-entry held-out test split from {tier1_checkpoint_dir}. "
                f"Tier 0 ALONE on this same test split (the fair comparison point, not the full-corpus row "
                f"above): precision={tier0_test_precision if tier0_test_precision is not None else 'n/a'}, "
                f"recall={tier0_test_recall if tier0_test_recall is not None else 'n/a'}. "
                f"{intentional_override_note}"
            ),
        )

        if tier2_model is not None:
            from . import llm_adjudicator

            endpoint = tier2_endpoint if tier2_endpoint is not None else llm_adjudicator.DEFAULT_OLLAMA_ENDPOINT
            tier012_rollups, tier2_test_entries = llm_adjudicator.evaluate_tier0_plus_1_plus_2(
                corpus_dir, tier1_checkpoint_dir, tier2_model=tier2_model, tier2_endpoint=endpoint, rules=rules
            )
            tier012_precision, tier012_recall, tier012_precision_at_10 = _aggregate_tier0_metrics(tier012_rollups)

            tier012_row = AblationRow(
                configuration=TIER_0_1_2_LABEL,
                status="measured",
                precision=tier012_precision,
                recall=tier012_recall,
                precision_at_10=tier012_precision_at_10,
                cost=_ARCHITECTURAL_COST,
                # LLM inference latency not measured this pass -- a real, stated gap, same
                # honesty standard as the Tier 0+1 row's own median_runtime_ms=None above.
                median_runtime_ms=None,
                notes=(
                    f"Measured on the {len(tier2_test_entries)}-entry held-out test split from "
                    f"{tier1_checkpoint_dir}, using local LLM model '{tier2_model}' via Ollama as a second "
                    f"opinion layered on top of Tier 1's own predictions (chained, not standalone -- see "
                    f"module docstring). Tier 0+1 ALONE on this same test split (the fair comparison point, "
                    f"not the full-corpus row above): precision={tier01_precision if tier01_precision is not None else 'n/a'}, "
                    f"recall={tier01_recall if tier01_recall is not None else 'n/a'}. Tier 2 has NO held-out "
                    "classification report of its own -- it is a zero-shot local LLM, never fine-tuned on "
                    "this corpus, unlike Tier 1's own per-label report -- these ablation numbers are the "
                    "only real evidence of its quality on this corpus."
                ),
            )
        else:
            tier012_row = AblationRow(
                configuration=TIER_0_1_2_LABEL,
                status="not_yet_available",
                precision=None,
                recall=None,
                precision_at_10=None,
                cost=_ARCHITECTURAL_COST,
                median_runtime_ms=None,
                notes=_NO_TIER2_NOTE,
            )
    else:
        tier01_row = AblationRow(
            configuration=TIER_0_1_LABEL,
            status="not_yet_available",
            precision=None,
            recall=None,
            precision_at_10=None,
            cost=_ARCHITECTURAL_COST,
            median_runtime_ms=None,
            notes=_NO_CHECKPOINT_NOTE,
        )
        tier012_row = AblationRow(
            configuration=TIER_0_1_2_LABEL,
            status="not_yet_available",
            precision=None,
            recall=None,
            precision_at_10=None,
            cost=_ARCHITECTURAL_COST,
            median_runtime_ms=None,
            notes="Not measured this run -- requires tier1_checkpoint_dir AND tier2_model both given (Tier 2 "
            "is chained on top of Tier 1, never a standalone alternative).",
        )

    return AblationTable(rows=[tier0_row, tier01_row, tier012_row])


def format_ablation_table_text(table: AblationTable) -> str:
    measured = [r.configuration for r in table.rows if r.status == "measured"]
    pending = [r.configuration for r in table.rows if r.status != "measured"]
    status_summary = "; ".join(
        part for part in [
            f"measured: {', '.join(measured)}" if measured else "",
            f"not yet available: {', '.join(pending)}" if pending else "",
        ] if part
    )

    lines = []
    lines.append(f"=== Week 6 Ablation Table ({status_summary}) ===")
    lines.append("")
    lines.append(table.caveat)
    lines.append("")

    header = (
        f"{'Configuration':<34}{'Precision':>10}{'Recall':>10}{'Precision@10':>14}{'Cost':>8}"
        f"{'Median runtime/workbook':>26}"
    )
    lines.append(header)
    for row in table.rows:
        prec = f"{row.precision:.3f}" if row.precision is not None else "n/a"
        rec = f"{row.recall:.3f}" if row.recall is not None else "n/a"
        p10 = f"{row.precision_at_10:.3f}" if row.precision_at_10 is not None else "n/a"
        runtime = f"{row.median_runtime_ms:.1f} ms" if row.median_runtime_ms is not None else "n/a"
        lines.append(f"{row.configuration:<34}{prec:>10}{rec:>10}{p10:>14}{row.cost:>8}{runtime:>26}")
    lines.append("")

    for row in table.rows:
        if row.notes:
            lines.append(f"[{row.configuration}] {row.notes}")

    return "\n".join(lines)
