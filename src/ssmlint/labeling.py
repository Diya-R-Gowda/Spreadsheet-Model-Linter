"""Week 5 label generation + training-readiness plumbing — NOT a trained
classifier. Per the approved scope for this stage ("Plumbing-only,
corpus-honest"), this module builds the label-generation and dataset-
prep pipeline the README's Tier 1 classifier will eventually consume,
and validates it against the existing corpus as a correctness/plumbing
check — mirroring how Week 4's evaluation harness caveated its 1.0/1.0
result as pipeline-consistency, not real-world performance. No ML
dependency (`transformers`, `torch`, etc.) is imported or required here;
none has been installed, per explicit instruction.

WHY THIS ORIGINALLY COULDN'T HONESTLY PRODUCE A REAL TRAINING SET
(investigated and confirmed with the user before implementation,
2026-08-11)
------------------------------------------------------------------------
The corpus's cell-level ground truth was originally 3-state (clean /
injected_bug / known_intentional); the classifier's target is a 4-state
BLOCK-level label (subtotal | intentional_override | suspected_error |
unknown). Only two of those four labels had any real mapping at the
time: suspected_error (a block touching a rule-engine-flagged
injected_bug cell) and intentional_override (a block bordering a
known_intentional cell). `subtotal` and `unknown` had ZERO examples
anywhere -- nothing in `generate_corpus.py` built a legitimate subtotal
row or a genuinely ambiguous case.

CLOSED (2026-08-17): the corpus-expansion follow-up task widened
`CellGroundTruth.state` to 5 values, adding `subtotal` (a block that IS
a legitimate, intentionally-different aggregate -- a new category-total
base shape) and `ambiguous` (Tier 0 still flags these deterministically,
same predicate as an injected_bug, but they're deliberately weak-evidence
constructions -- short block, edge position, or a near-50/50
affected/clean split on reference-to-blank -- intended to carry the
`unknown` label instead of `suspected_error`). `label_for_block` below
now returns all four labels; see scripts/generate_corpus.py's own
module docstring for exactly how each new state is constructed, and the
real per-label volumes reported below (`unknown` in particular did not
reach the same 50+ floor as the other three -- documented honestly, not
padded to hit a number).

DISCOVERED WHILE BUILDING THIS (not predicted by the investigation):
`intentional_override` is even thinner than the raw ground-truth cell
count (26 `known_intentional` cells) suggested. A single block can
border BOTH a known_intentional seed AND an injected_bug cell at once
(e.g. a growth-chain's C14:E14 block sits between the seed at B14 and
an injected literal at F14 in most non-`edge_left` injection shapes) --
`label_for_block` checks suspected_error first, so whenever both
signals land on the same block, the real bug wins and the seed signal
is dropped for that block (a defensible choice: a block containing an
actual injected bug should be labeled by the bug, not by an incidental
clean neighbor). The practical effect: real block-level
`intentional_override` examples come ONLY from the corpus's 5
uncorrupted growth-chain clean baselines, where the seed has no
competing signal to lose to -- not the ~20-26 a per-cell count would
suggest. Confirmed by running `generate_labeled_examples` for real, not
predicted in advance -- exactly the kind of thing this plumbing check
exists to surface.

INPUT SHAPE: NO row_label/col_labels (confirmed gap, not implemented
here)
------------------------------------------------------------------------
The README's example block description includes human-readable
`row_label`/`col_labels` text. Neither `CellRecord` (parser.py) nor
`Block` (blocks.py) captures any header/label text -- confirmed by
reading both directly, not assumed. `BlockExample` below carries
`row_label`/`col_labels` fields set to `None` on every example (kept
visible rather than omitted, so the gap stays obvious to anyone reading
a real example) rather than fabricating placeholder text. Extracting
real header text would be a new pipeline capability under
`src/ssmlint/parser.py`/`blocks.py` -- a separate, explicitly-scoped
prerequisite task, not built here.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .blocks import Block, SheetBlocks, detect_blocks
from .depgraph import build_graph
from .parser import parse_workbook
from .rules import Rule
from .evaluation import DEFAULT_RULES, evaluate_rule

LABELS = ("subtotal", "intentional_override", "suspected_error", "unknown")


@dataclass(frozen=True)
class Deviation:
    cell: str
    kind: str  # "non_conforming:literal" | "non_conforming:blank" | "non_conforming:different_formula"
    # | "non_conforming:unparseable" | "near_miss"
    side: str  # "left" | "right" | "member" (member = inside the block itself, e.g. reference-to-blank)

    def to_dict(self) -> dict:
        return {"cell": self.cell, "kind": self.kind, "side": self.side}


@dataclass(frozen=True)
class BlockExample:
    """A compact, README-shaped block description -- never a raw cell
    dump. `row_label`/`col_labels` are always None (see module docstring:
    the pipeline cannot produce them yet).
    """

    entry: str
    block_address: str  # e.g. "Model!C14:E14"
    base_shape: str
    formula_pattern: str
    conforming_count: int
    deviations: list[Deviation]
    row_label: str | None
    col_labels: list[str] | None
    label: str | None  # one of LABELS, or None if no ground-truth signal maps this block to any of them
    label_basis: str  # human-readable trace of why this label (or lack of one) was assigned

    def to_dict(self) -> dict:
        return {
            "entry": self.entry,
            "block_address": self.block_address,
            "base_shape": self.base_shape,
            "formula_pattern": self.formula_pattern,
            "conforming_count": self.conforming_count,
            "deviations": [d.to_dict() for d in self.deviations],
            "row_label": self.row_label,
            "col_labels": self.col_labels,
            "label": self.label,
            "label_basis": self.label_basis,
        }


def _block_deviations(block: Block) -> list[Deviation]:
    deviations = []
    start_col = block.cells[0] if block.cells else None
    for nc in block.non_conforming:
        deviations.append(Deviation(cell=nc.cell, kind=f"non_conforming:{nc.kind}", side="neighbor"))
    for nm in block.near_misses:
        deviations.append(Deviation(cell=nm.cell, kind="near_miss", side="neighbor"))
    return deviations


def label_for_block(
    block: Block, gt_by_cell: dict[str, dict], flagged_cells: set[str]
) -> tuple[str | None, str]:
    """Determines the 4-way label for one block from the corpus's 5-state
    cell-level ground truth (widened from 3 states -- see
    scripts/generate_corpus.py's corpus-expansion module docstring for
    "subtotal" and "ambiguous"). Checked in this precedence order:
      1. `subtotal` -- a block IS the legitimate aggregate itself; checked
         first since it's the most direct, certain signal available (no
         dependence on whether Tier 0 happened to flag anything).
      2. `suspected_error` -- an `injected_bug` cell the rule engine
         actually flagged (existing logic, unchanged).
      3. `unknown` -- an `ambiguous` cell the rule engine flagged. Checked
         after suspected_error deliberately: if a block somehow border
         both a real bug and a weak-evidence case, the real bug should
         still win, same precedence rationale as suspected_error already
         winning over intentional_override below.
      4. `intentional_override` -- a `known_intentional` cell (existing
         logic, unchanged).
      5. `None` -- no ground-truth signal maps this block to any label.
    """
    candidate_cells = list(block.cells) + [nc.cell for nc in block.non_conforming] + [nm.cell for nm in block.near_misses]

    for cell in candidate_cells:
        gt = gt_by_cell.get(cell)
        if gt is not None and gt["state"] == "subtotal":
            return "subtotal", f"subtotal@{cell}"

    for cell in candidate_cells:
        gt = gt_by_cell.get(cell)
        if gt is None:
            continue
        if gt["state"] == "injected_bug" and cell in flagged_cells:
            return "suspected_error", f"injected_bug:{gt['rule_id']}@{cell}"

    for cell in candidate_cells:
        gt = gt_by_cell.get(cell)
        if gt is None:
            continue
        if gt["state"] == "ambiguous" and cell in flagged_cells:
            return "unknown", f"ambiguous:{gt['rule_id']}@{cell}"

    for cell in candidate_cells:
        gt = gt_by_cell.get(cell)
        if gt is not None and gt["state"] == "known_intentional":
            return "intentional_override", f"known_intentional@{cell}"

    return None, "no_ground_truth_signal"


def _load_corpus_entries(corpus_dir: Path) -> list[tuple[str, Path, dict]]:
    entries = []
    for gt_path in sorted(corpus_dir.glob("*.ground_truth.json")):
        name = gt_path.stem.removesuffix(".ground_truth")
        xlsx_path = corpus_dir / f"{name}.xlsx"
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        entries.append((name, xlsx_path, ground_truth))
    return entries


def generate_labeled_examples(corpus_dir: Path, rules: dict[str, Rule] | None = None) -> list[BlockExample]:
    """Runs the real Tier 0 rule engine against every corpus entry and
    produces one BlockExample per detected block, labeled from ground
    truth where possible (label=None where the corpus has no signal).
    Reuses the Week 4 evaluation harness's own rule-calling logic
    (`ssmlint.evaluation.evaluate_rule`) so this never re-derives which
    rules need the dependency graph.
    """
    rules = rules if rules is not None else DEFAULT_RULES
    examples: list[BlockExample] = []

    for name, xlsx_path, ground_truth in _load_corpus_entries(corpus_dir):
        gt_by_cell = {c["cell"]: c for c in ground_truth["cells"]}
        parsed = parse_workbook(xlsx_path)
        graph = build_graph(parsed)
        sheet_blocks: list[SheetBlocks] = detect_blocks(parsed, graph)

        flagged_cells: set[str] = set()
        for rule in rules.values():
            for issue in evaluate_rule(rule, sheet_blocks, graph):
                flagged_cells.add(issue.cell)

        for sb in sheet_blocks:
            for block in sb.blocks:
                label, basis = label_for_block(block, gt_by_cell, flagged_cells)
                examples.append(
                    BlockExample(
                        entry=name,
                        block_address=f"{block.sheet}!{block.span}",
                        base_shape=ground_truth["base_shape"],
                        formula_pattern=block.pattern,
                        conforming_count=len(block.cells),
                        deviations=_block_deviations(block),
                        row_label=None,
                        col_labels=None,
                        label=label,
                        label_basis=basis,
                    )
                )

    return examples


def split_corpus_entries(
    entry_names: list[str], train: float = 0.7, val: float = 0.15, test: float = 0.15, seed: int = 42
) -> dict[str, list[str]]:
    """Deterministic entry-level split (never a block-level split -- every
    block in an entry shares that entry's formula pattern, so splitting
    below the entry level would leak near-duplicate examples across
    train/test). Fixed seed for full reproducibility, same rationale as
    generate_corpus.py's no-randomness-without-a-seed discipline.
    """
    if abs((train + val + test) - 1.0) > 1e-9:
        raise ValueError(f"train+val+test must sum to 1.0, got {train + val + test}")

    shuffled = sorted(entry_names)  # sort first so the shuffle is reproducible regardless of glob order
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train)
    n_val = round(n * val)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


@dataclass
class LabelReadiness:
    label: str
    count: int
    min_required: int

    @property
    def ready(self) -> bool:
        return self.count >= self.min_required

    def to_dict(self) -> dict:
        return {"label": self.label, "count": self.count, "min_required": self.min_required, "ready": self.ready}


@dataclass
class TrainingReadinessReport:
    total_examples: int
    unlabeled_examples: int  # label is None -- no ground-truth signal maps this block to any of the 4 labels
    distinct_formula_patterns: int
    per_label: list[LabelReadiness]
    per_split_label_counts: dict[str, dict[str, int]]

    @property
    def ready(self) -> bool:
        return all(l.ready for l in self.per_label)

    def to_dict(self) -> dict:
        return {
            "total_examples": self.total_examples,
            "unlabeled_examples": self.unlabeled_examples,
            "distinct_formula_patterns": self.distinct_formula_patterns,
            "per_label": [l.to_dict() for l in self.per_label],
            "per_split_label_counts": self.per_split_label_counts,
            "ready": self.ready,
        }


def check_training_readiness(
    examples: list[BlockExample], split: dict[str, list[str]], min_required: int = 50
) -> TrainingReadinessReport:
    """Reports real per-label volume against `min_required` (a stated,
    conservative floor for a 4-way small-encoder fine-tune -- not a
    precise published minimum, just a defensible lower bound to make the
    gap concrete) -- never trains anything. This is the plumbing check:
    it proves the label-generation and split pipeline runs correctly
    against the real corpus; `ready == False` is the expected, honest
    result given the current corpus (see module docstring).
    """
    labeled = [e for e in examples if e.label is not None]
    unlabeled = [e for e in examples if e.label is None]
    label_counts = {label: sum(1 for e in labeled if e.label == label) for label in LABELS}

    entry_to_split = {entry: split_name for split_name, entries in split.items() for entry in entries}
    per_split_label_counts: dict[str, dict[str, int]] = {s: dict.fromkeys(LABELS, 0) for s in split}
    for e in labeled:
        split_name = entry_to_split.get(e.entry)
        if split_name is not None:
            per_split_label_counts[split_name][e.label] += 1

    return TrainingReadinessReport(
        total_examples=len(examples),
        unlabeled_examples=len(unlabeled),
        distinct_formula_patterns=len({e.formula_pattern for e in examples}),
        per_label=[LabelReadiness(label=label, count=label_counts[label], min_required=min_required) for label in LABELS],
        per_split_label_counts=per_split_label_counts,
    )


def format_readiness_report(report: TrainingReadinessReport) -> str:
    lines = []
    lines.append("=== Week 5 Training-Readiness Plumbing Check (NOT a trained classifier) ===")
    lines.append("")
    lines.append(
        "This validates the label-generation + train/val/test split pipeline against the real corpus. "
        "It is a correctness check on the plumbing, not a real-accuracy claim -- mirrors Week 4's own "
        "internal-consistency caveat."
    )
    lines.append("")
    lines.append(f"Total blocks examined: {report.total_examples}")
    lines.append(f"Blocks with NO ground-truth label signal (label=None): {report.unlabeled_examples}")
    lines.append(f"Distinct formula patterns across all examples: {report.distinct_formula_patterns}")
    lines.append("")
    lines.append("--- Per-label volume vs. a conservative minimum floor ---")
    for l in report.per_label:
        status = "READY" if l.ready else "NOT READY"
        lines.append(f"  {l.label:<22} count={l.count:<4} min_required={l.min_required:<4} [{status}]")
    lines.append("")
    lines.append("--- Per-split label counts ---")
    for split_name, counts in report.per_split_label_counts.items():
        counts_str = ", ".join(f"{label}={n}" for label, n in counts.items())
        lines.append(f"  {split_name}: {counts_str}")
    lines.append("")
    verdict = "READY for a real fine-tune" if report.ready else "NOT READY for a real fine-tune"
    lines.append(f"OVERALL: {verdict}")
    if not report.ready:
        lines.append(
            "At least one label is below its floor -- see scripts/generate_corpus.py's module docstring "
            "for why (e.g. 'unknown' has a real, verified structural ceiling on how many genuinely distinct "
            "weak-evidence examples the current injection mechanisms can produce). Before training, either "
            "accept the honestly-reported gap or extend generate_corpus.py with a new mechanism for the "
            "specific label(s) still short."
        )
    return "\n".join(lines)
