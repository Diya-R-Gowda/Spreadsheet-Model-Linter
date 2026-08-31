"""Week 6 HTML/JSON report builder — the README's stage [9] deliverable:
"ranked JSON + standalone HTML, tagged with which tier flagged each
issue." Scoped Tier 0 only (see `diya.md`): every issue's `tier` is
currently always `"Tier 0"`, but the field exists so future tiers slot
in without a schema change.

This is a genuinely different consumer than `evaluation.py`/`ablation.py`
(which score the rule engine against the synthetic corpus's ground
truth): `build_report()` runs the real pipeline against ONE real
workbook and renders its actual findings — no ground truth, no corpus.

WHY THIS WRAPS `Issue` INSTEAD OF WIDENING `rules/base.py`
------------------------------------------------------------------------
`Issue` (a completed Week-3 stage file) has no `to_dict()` and no `tier`
field. Confirmed directly with the user before implementation: don't
touch `rules/base.py` — `RankedIssue` below wraps an `Issue` with
`rank`, `tier`, and split `sheet`/`address` fields instead, the same
"wrapper dataclass" pattern `evaluation.py`'s `ScoredIssue` already
established for exactly this situation.

RANKING: ADAPTED FROM evaluation.py, NOT REINVENTED
------------------------------------------------------------------------
`evaluation.py` ranks precision@10 candidates by severity descending
with a deterministic tiebreak, specifically to avoid relying on each
rule's dict-iteration output order. A single-workbook report has no
corpus-entry axis to break ties on (that tiebreak was corpus-specific),
so the tiebreak here is `(rule_id, cell)` instead — both already
deterministic. Sort key: `(severity_rank[severity], rule_id, cell)`.

SHEET HEATMAP: A BOUNDING BOX OVER FLAGGED CELLS, NOT THE FULL SHEET
------------------------------------------------------------------------
`Issue.cell` (e.g. `"Model!H14"`) is the only linkage available between
an issue and its position — nothing threads `Block`/`ParsedWorkbook`
context through to `Issue` today. The heatmap's bounding box is
computed only from flagged-cell addresses, not a sheet's full used
range: this is a deliberate v1 scoping choice ("the region containing
findings," not the whole sheet) to avoid rendering a huge, mostly-empty
grid for a large real workbook. A true full-sheet heatmap would need
`ParsedWorkbook`'s full per-sheet cell list threaded through here — a
bigger change, out of scope for this stage.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from openpyxl.utils import column_index_from_string

from .blocks import detect_blocks
from .depgraph import build_graph
from .evaluation import DEFAULT_RULES, SEVERITIES, evaluate_rule
from .parser import parse_workbook
from .rules.base import Issue
from .tokenizer import parse_cell_addr_parts

_SEVERITY_RANK = {sev: i for i, sev in enumerate(SEVERITIES)}  # "high"=0, "medium"=1

REPORT_CAVEAT = (
    "These findings are Tier 0 structural checks only -- deterministic pattern-matching, not "
    "semantic judgment. Cases like a legitimate subtotal row or a deliberate intentional override "
    "are not yet distinguished from real errors; that judgment is a Tier 1/2 capability not present "
    "in this build. Every flagged cell here deserves a human look, not automatic trust either way."
)

REPORT_CAVEAT_TIER1 = (
    "These findings were reviewed by a trained Tier 1 classifier on top of Tier 0's structural checks. "
    "Issues it judged a legitimate subtotal or intentional override were moved to a separate "
    "'suppressed by Tier 1' section below, never silently dropped. The block 'base shape' Tier 1 saw "
    "for this workbook is a heuristic guess from the block's own formula pattern (guess_base_shape), "
    "not the rigor of the corpus's own ground truth used to train it -- read predictions on any block "
    "guessed 'unrecognized' with extra skepticism."
)

REPORT_CAVEAT_TIER2 = (
    "Additionally, a local LLM adjudicator (a 3B-parameter model run via Ollama) reviewed Tier 1's "
    "own surviving issues as a second opinion -- never a standalone judge on its own, always layered "
    "on top of Tier 1. Its output is grammar-constrained (a JSON schema restricts it to exactly one of "
    "the four labels; it can never invent a cell address), but unlike Tier 1 it is zero-shot -- never "
    "fine-tuned on this project's own corpus, and has no held-out classification report of its own. "
    "Read its suppressions with at least as much skepticism as Tier 1's; the ablation table is the "
    "only real, measured evidence of its quality on this project's synthetic corpus."
)

_TEMPLATE_NAME = "report.html.jinja"


@dataclass(frozen=True)
class RankedIssue:
    rank: int
    sheet: str
    address: str  # bare, e.g. "H14"
    cell: str  # sheet-qualified, e.g. "Model!H14"
    severity: str
    rule_id: str
    tier: str
    explanation: str
    suggested_fix: str

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "sheet": self.sheet,
            "address": self.address,
            "cell": self.cell,
            "severity": self.severity,
            "rule_id": self.rule_id,
            "tier": self.tier,
            "explanation": self.explanation,
            "suggested_fix": self.suggested_fix,
        }


@dataclass(frozen=True)
class HeatmapCell:
    address: str
    severity: str
    issue_count: int
    rule_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "address": self.address,
            "severity": self.severity,
            "issue_count": self.issue_count,
            "rule_ids": self.rule_ids,
        }


@dataclass(frozen=True)
class SheetHeatmap:
    sheet: str
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    cells: dict[str, HeatmapCell]

    def to_dict(self) -> dict:
        return {
            "sheet": self.sheet,
            "min_row": self.min_row,
            "max_row": self.max_row,
            "min_col": self.min_col,
            "max_col": self.max_col,
            "cells": {addr: c.to_dict() for addr, c in self.cells.items()},
        }


@dataclass
class WorkbookReport:
    workbook: str
    generated_at: str
    issues: list[RankedIssue] = field(default_factory=list)
    suppressed_issues: list[RankedIssue] = field(default_factory=list)
    heatmaps: list[SheetHeatmap] = field(default_factory=list)
    severity_counts: dict[str, int] = field(default_factory=dict)
    rule_id_counts: dict[str, int] = field(default_factory=dict)
    caveat: str = REPORT_CAVEAT

    def to_dict(self) -> dict:
        return {
            "workbook": self.workbook,
            "generated_at": self.generated_at,
            "issues": [i.to_dict() for i in self.issues],
            "suppressed_issues": [i.to_dict() for i in self.suppressed_issues],
            "heatmaps": [h.to_dict() for h in self.heatmaps],
            "severity_counts": self.severity_counts,
            "rule_id_counts": self.rule_id_counts,
            "caveat": self.caveat,
        }


def _split_cell(cell: str) -> tuple[str, str]:
    """Sheet-qualified address -> (sheet, bare address). Deliberately
    replicates blocks.py's own one-line `_split_address` logic rather
    than importing a private helper across modules.
    """
    sheet, plain = cell.split("!", 1)
    return sheet, plain


def rank_issues(issues: list[Issue], tier: str = "Tier 0") -> list[RankedIssue]:
    def sort_key(issue: Issue) -> tuple:
        return (_SEVERITY_RANK.get(issue.severity, 99), issue.rule_id, issue.cell)

    ranked: list[RankedIssue] = []
    for rank, issue in enumerate(sorted(issues, key=sort_key), start=1):
        sheet, address = _split_cell(issue.cell)
        ranked.append(
            RankedIssue(
                rank=rank, sheet=sheet, address=address, cell=issue.cell, severity=issue.severity,
                rule_id=issue.rule_id, tier=tier, explanation=issue.explanation, suggested_fix=issue.suggested_fix,
            )
        )
    return ranked


_WORSE_SEVERITY = {"high": 0, "medium": 1}


def build_heatmaps(ranked_issues: list[RankedIssue]) -> list[SheetHeatmap]:
    by_sheet: dict[str, dict[str, HeatmapCell]] = {}
    for ri in ranked_issues:
        sheet_cells = by_sheet.setdefault(ri.sheet, {})
        existing = sheet_cells.get(ri.address)
        if existing is None:
            sheet_cells[ri.address] = HeatmapCell(
                address=ri.address, severity=ri.severity, issue_count=1, rule_ids=[ri.rule_id]
            )
        else:
            worse = ri.severity if _WORSE_SEVERITY.get(ri.severity, 99) < _WORSE_SEVERITY.get(existing.severity, 99) else existing.severity
            sheet_cells[ri.address] = HeatmapCell(
                address=ri.address, severity=worse, issue_count=existing.issue_count + 1,
                rule_ids=[*existing.rule_ids, ri.rule_id],
            )

    heatmaps: list[SheetHeatmap] = []
    for sheet, cells in by_sheet.items():
        rows, cols = [], []
        for address in cells:
            _col_abs, col_letters, _row_abs, row = parse_cell_addr_parts(address)
            rows.append(row)
            cols.append(column_index_from_string(col_letters))
        heatmaps.append(
            SheetHeatmap(
                sheet=sheet, min_row=min(rows), max_row=max(rows), min_col=min(cols), max_col=max(cols), cells=cells
            )
        )
    return heatmaps


def build_report(
    workbook_path: str | Path,
    tier1_checkpoint: str | Path | None = None,
    tier2_model: str | None = None,
    tier2_endpoint: str | None = None,
) -> WorkbookReport:
    """`tier1_checkpoint`, when given (a real checkpoint saved by
    `scripts/train_classifier.py`), runs live Tier 1 inference against
    this workbook's real Tier 0 issues (`classifier.apply_tier1_to_workbook`)
    and splits them into surviving vs. suppressed. Omitting it (the
    default) preserves the exact prior Tier-0-only behavior. A bad or
    missing checkpoint fails loudly (wrapped in a `RuntimeError` naming
    the checkpoint) rather than silently falling back to Tier 0 -- a user
    who explicitly asked for Tier 1 review should never get an
    unmarked-down Tier-0-only report back.

    `tier2_model`, when ALSO given (e.g. "qwen2.5:3b-instruct", a real
    local Ollama model), runs live Tier 2 inference
    (`llm_adjudicator.apply_tier2_to_workbook`) against Tier 1's own
    SURVIVING issues -- a second opinion layered on top, never a
    standalone alternative (chained-only, matching `ablation.py`'s own
    design). `tier2_model` without `tier1_checkpoint` raises `ValueError`
    immediately, before any parsing/rule work runs. A bad checkpoint-free
    server (unreachable Ollama, model not pulled) fails loudly the same
    way a bad Tier 1 checkpoint does -- checked up front via
    `llm_adjudicator.check_ollama_available` for a clearer error than a
    raw connection-refused traceback.
    """
    if tier2_model is not None and tier1_checkpoint is None:
        raise ValueError(
            "tier2_model requires tier1_checkpoint to also be given -- Tier 2 is a second opinion "
            "layered on top of Tier 1's own predictions, never a standalone alternative (matches "
            "ssmlint.ablation's own chained-only design)."
        )

    workbook_path = Path(workbook_path)
    parsed = parse_workbook(workbook_path)
    graph = build_graph(parsed)
    sheet_blocks = detect_blocks(parsed, graph)

    issues: list[Issue] = []
    for rule in DEFAULT_RULES.values():
        issues.extend(evaluate_rule(rule, sheet_blocks, graph))

    tier1_suppressed: list[Issue] = []
    tier2_suppressed: list[Issue] = []

    if tier1_checkpoint is not None:
        from . import classifier

        try:
            surviving_issues, tier1_suppressed = classifier.apply_tier1_to_workbook(
                issues, sheet_blocks, Path(tier1_checkpoint), workbook_name=workbook_path.name
            )
        except Exception as exc:
            raise RuntimeError(f"Tier 1 inference failed using checkpoint '{tier1_checkpoint}': {exc}") from exc
        tier_label = "Tier 0 (Tier 1-reviewed)"
        caveat = f"{REPORT_CAVEAT_TIER1} {classifier.describe_intentional_override_confidence(tier1_checkpoint)}"

        if tier2_model is not None:
            from . import llm_adjudicator

            endpoint = tier2_endpoint if tier2_endpoint is not None else llm_adjudicator.DEFAULT_OLLAMA_ENDPOINT
            try:
                llm_adjudicator.check_ollama_available(endpoint=endpoint, model=tier2_model)
                surviving_issues, tier2_suppressed = llm_adjudicator.apply_tier2_to_workbook(
                    surviving_issues, sheet_blocks, model=tier2_model, endpoint=endpoint,
                    workbook_name=workbook_path.name,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Tier 2 inference failed using model '{tier2_model}' at '{endpoint}': {exc}"
                ) from exc
            tier_label = "Tier 0 (Tier 1+2-reviewed)"
            caveat = f"{caveat} {REPORT_CAVEAT_TIER2}"
    else:
        surviving_issues = issues
        tier_label = "Tier 0"
        caveat = REPORT_CAVEAT

    ranked = rank_issues(surviving_issues, tier=tier_label)
    suppressed_ranked: list[RankedIssue] = []
    if tier1_suppressed:
        suppressed_ranked.extend(rank_issues(tier1_suppressed, tier="Tier 0 (suppressed by Tier 1)"))
    if tier2_suppressed:
        suppressed_ranked.extend(rank_issues(tier2_suppressed, tier="Tier 0 (suppressed by Tier 2)"))
    heatmaps = build_heatmaps(ranked)  # unaffected -- suppressed issues never appear on the heatmap

    severity_counts: dict[str, int] = {}
    rule_id_counts: dict[str, int] = {}
    for ri in ranked:
        severity_counts[ri.severity] = severity_counts.get(ri.severity, 0) + 1
        rule_id_counts[ri.rule_id] = rule_id_counts.get(ri.rule_id, 0) + 1

    return WorkbookReport(
        workbook=workbook_path.name,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        issues=ranked,
        suppressed_issues=suppressed_ranked,
        heatmaps=heatmaps,
        severity_counts=severity_counts,
        rule_id_counts=rule_id_counts,
        caveat=caveat,
    )


def _template_dir() -> Path:
    return Path(str(resources.files("ssmlint") / "templates"))


def render_html(report: WorkbookReport) -> str:
    from openpyxl.utils import get_column_letter

    env = Environment(
        loader=FileSystemLoader(str(_template_dir())),
        autoescape=select_autoescape(["html", "jinja"]),
    )
    env.globals["cell_address"] = lambda row, col: f"{get_column_letter(col)}{row}"
    template = env.get_template(_TEMPLATE_NAME)
    return template.render(report=report)
