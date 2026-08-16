# Week 6: Ablation Table + HTML Report Builder (Tier 0 only)

## Context

Weeks 1-5 are done: the full Tier 0 pipeline (parser → AST → R1C1 → dependency graph → block detector → 4 deterministic rules), a synthetic-corruption corpus + evaluation harness (Week 4, `src/ssmlint/evaluation.py`, 1.0/1.0 precision/recall with an honest internal-consistency caveat), and Week 5's label-generation plumbing (`src/ssmlint/labeling.py`) — with the actual Tier 1 classifier explicitly **blocked** (the corpus can't honestly support a 4-way fine-tune yet) and deliberately deferred, not abandoned, per a direct decision already recorded in `CONTRIBUTING.md`.

The user was asked directly whether to scope corpus expansion now or move to Week 6 with Tier 0 alone, and chose Week 6. Per the README, Tier 1/2 are optional layers on top of a Tier 0 that stands on its own — this stage builds the two remaining Week-6 deliverables that don't require a classifier: **the ablation table** and **the HTML report builder**, both on top of the already-working, already-evaluated Tier 0 engine. The Local LLM (Tier 2) row stays explicitly red/deferred.

Confirmed via direct exploration (Explore + Plan subagents, cross-checked by reading `cli.py`, `rules/base.py`, `pyproject.toml` directly): this is fully greenfield — zero report-building code, no `jinja2`, no HTML/template files anywhere in the repo. `evaluation.py` has no "configuration" axis (Tier 0 vs Tier 0+1 vs Tier 0+1+2) and no runtime/cost instrumentation. `Issue` (in `rules/base.py`, a completed Week-3 stage file) has no `to_dict()` and no `tier` field.

**Four judgment calls were surfaced and confirmed directly with the user before finalizing this plan** (all four recommended defaults accepted):
1. Add `jinja2` as a new dependency — **yes**.
2. `Issue`'s missing `tier`/`to_dict()` — **wrap it** in a new `RankedIssue` in `report.py`; do not touch the completed-stage `rules/base.py`.
3. Ablation table's Cost column for the two not-yet-built rows — **show `"$0"`** (an architectural fact per the README: no paid API anywhere in the design, true regardless of whether the tier is built), while precision/recall/precision@10/runtime stay `None`/"not yet available" for those rows.
4. Report CLI surface — **new `ssmlint report <path>` subcommand** in `cli.py` (matches `dump`'s pattern; this is a single-workbook, end-user-facing operation, unlike the corpus-facing `scripts/run_*.py` dev tools).

---

## Deliverable A: Ablation table

**New files:** `src/ssmlint/ablation.py` (reusable logic), `scripts/run_ablation.py` (thin CLI, mirrors `scripts/run_evaluation.py`), `tests/test_ablation.py`.

**Reuses, never duplicates:** `evaluation.run_evaluation(corpus_dir, rules=None)`, `evaluation.DEFAULT_RULES`, `evaluation.INTERNAL_CONSISTENCY_CAVEAT`, `evaluation.RuleRollup` (via its public `to_dict()`-safe fields on `EvaluationReport.rollups`), `evaluation.SEVERITIES`. For the timing pass: `parser.parse_workbook`, `depgraph.build_graph`, `blocks.detect_blocks`, `evaluation.evaluate_rule` (already public — promoted during Week 5 for exactly this kind of cross-module reuse).

**New dataclasses (`ablation.py`):**
```python
@dataclass(frozen=True)
class AblationRow:
    configuration: str        # "Tier 0 (rules only)" | "Tier 0 + 1 (+ trained classifier)" | "Tier 0 + 1 + 2 (+ local LLM)"
    status: str                # "measured" | "not_yet_available"
    precision: float | None
    recall: float | None
    precision_at_10: float | None
    cost: str                  # "$0" for all three rows
    median_runtime_ms: float | None
    notes: str                 # e.g. "Blocked — see CONTRIBUTING.md Week 5 design note" for rows 2/3
    def to_dict(self) -> dict: ...

@dataclass
class AblationTable:
    rows: list[AblationRow]
    caveat: str = evaluation.INTERNAL_CONSISTENCY_CAVEAT
    def to_dict(self) -> dict: ...
```

**Aggregating the 4 rules into one Tier-0 row** (micro-averaged, computed only from `RuleRollup`'s already-public fields — never reaches into `evaluation.py`'s private scoring internals):
- Precision = `sum(tp) / sum(tp + fp)` across all rollups.
- Recall = `sum(tp) / sum(tp + fn)` across all rollups.
- Precision@10 = `sum(tp_in_top_k) / sum(precision_at_10_k)` across rollups — document explicitly in the docstring that this is a weighted-average **approximation** of one global pooled top-10 ranking, not the same thing as a true cross-rule pooled rank (each rule currently pools its own top-10 independently; there's no cross-rule ranking signal to pool on beyond severity, same limitation `evaluation.py` already documented for precision@10 in general).

**Runtime instrumentation (new — nothing like this exists today):** `ablation.py` runs its own separate timing loop over the corpus (re-parsing/re-evaluating, since `run_evaluation()` has no internal timing hook and touching it would mean widening a completed-stage-adjacent file) — one `time.perf_counter()` sample per corpus entry, `statistics.median(...)` across entries for the "median runtime/workbook" column. Accept the minor cost of walking the corpus twice (47 entries — trivial) rather than modifying `evaluation.py`.

**Confirmed scope of what's timed (stated explicitly in `ablation.py`'s module docstring, not left implicit):** each sample times the genuinely full per-entry pipeline — `parser.parse_workbook` (fresh `ParsedWorkbook`, no reuse) → `depgraph.build_graph` → `blocks.detect_blocks` (which internally invokes per-cell formula tokenization/AST building and R1C1 normalization via its own `_row_groups`, not something the timing loop needs to call separately) → `evaluation.evaluate_rule` for all four rules. So "median runtime/workbook" genuinely covers parse → tokenize → R1C1 → depgraph → blocks → rule evaluation, not just rule evaluation against pre-built blocks. Each corpus entry gets its own fresh call chain — `Rule` instances are stateless per `rules/base.py`'s own ABC docstring, and no `ParsedWorkbook`/graph/blocks object is reused across entries — so there's no cross-entry cached state inflating or deflating the measurement.

**Output:** `build_ablation_table(corpus_dir) -> AblationTable`, `format_ablation_table_text(table) -> str` (fixed-width columns matching `evaluation.py`'s `format_report_text` style, README's exact column order: Configuration, Precision, Recall, Precision@10, Cost, Median runtime/workbook). `scripts/run_ablation.py`: `--json` flag for machine-readable output, human-readable table by default — same shape as `run_evaluation.py`/`run_labeling.py`.

**Tests (`test_ablation.py`, mirrors `test_evaluation.py`'s pattern):**
- Unit tests on hand-constructed `RuleRollup` objects (no corpus, no I/O) for the micro-average math in isolation.
- Unit test: unbuilt rows have `status="not_yet_available"`, `None` metrics, `cost="$0"`, non-empty `notes`.
- Timing-loop test: assert structural properties only (positive float, one sample per entry, correct median on a controlled fake list) — never assert exact wall-clock values.
- Integration test against real `corpus/`: Tier 0 row's aggregated precision/recall must exactly match the already-verified 1.0/1.0 full-corpus numbers from `test_evaluation.py` — a real cross-check that the aggregation math is correct, not just plausible. Assert the caveat text is present verbatim in formatted output.
- Manual step before calling this done: actually run `scripts/run_ablation.py` and paste the real printed table (same verification standard as every prior stage).

---

## Deliverable B: HTML report builder

**New files:** `src/ssmlint/report.py`, `src/ssmlint/templates/report.html.jinja` (packaged inside `src/ssmlint/` so the installed console script finds it via `Path(__file__).resolve().parent / "templates"` regardless of cwd — needs a `[tool.setuptools.package-data]` entry in `pyproject.toml`, verified at build/install time, not just assumed), `tests/test_report.py`.

**Reuses:** `parser.parse_workbook`, `depgraph.build_graph`, `blocks.detect_blocks`, `evaluation.DEFAULT_RULES`, `evaluation.evaluate_rule`, `evaluation.SEVERITIES`.

**New dataclasses (`report.py`):**
```python
@dataclass(frozen=True)
class RankedIssue:              # wrapper around Issue — see Decision 2, confirmed
    rank: int
    sheet: str
    address: str                 # bare, e.g. "H14"
    cell: str                    # sheet-qualified, e.g. "Model!H14"
    severity: str
    rule_id: str
    tier: str                    # "Tier 0" for now; field exists for future tiers
    explanation: str
    suggested_fix: str
    def to_dict(self) -> dict: ...

@dataclass(frozen=True)
class HeatmapCell:
    address: str
    severity: str                 # worst severity if multiple issues land on one cell
    issue_count: int
    rule_ids: list[str]
    def to_dict(self) -> dict: ...

@dataclass(frozen=True)
class SheetHeatmap:
    sheet: str
    min_row: int; max_row: int; min_col: int; max_col: int   # bounding box over flagged cells only
    cells: dict[str, HeatmapCell]   # keyed by bare address, only flagged cells present
    def to_dict(self) -> dict: ...

@dataclass
class WorkbookReport:
    workbook: str
    generated_at: str              # ISO 8601
    issues: list[RankedIssue]
    heatmaps: list[SheetHeatmap]   # one per sheet with >=1 issue
    severity_counts: dict[str, int]
    rule_id_counts: dict[str, int]
    caveat: str                     # new, report-specific — see below
    def to_dict(self) -> dict: ...
```

**Ranking:** adapted from `evaluation.py`'s severity-desc + deterministic-tiebreak convention (state this explicitly in the docstring — it's a deliberate adaptation, not a reinvention). Sort key: `(severity_rank[severity], rule_id, cell)` — tiebreak changes from `evaluation.py`'s `(entry, cell)` to `(rule_id, cell)` since a single-workbook report has no corpus-entry axis. `rank` = position after sort, 1-indexed.

**Sheet heatmap:** group `RankedIssue`s by `sheet` (split on `!`, same one-liner logic `blocks.py` already uses internally — replicate it, don't import a private helper), then by bare address within each sheet. A cell hit by more than one rule gets `issue_count > 1` and the worse of the two severities. **Bounding box, not full sheet extent** — computed only from flagged-cell addresses, not the sheet's full used range (avoids a huge mostly-empty grid on a large real workbook; this is a deliberate v1 scoping choice — "region containing findings," not the whole sheet — worth noting in the module docstring as a known limitation, same as prior stages' documented scope boundaries).

**JSON output:** `WorkbookReport.to_dict()` → `json.dumps(..., indent=2)`. Satisfies the README's "ranked JSON" half of the output spec independently of the HTML render.

**HTML template (`report.html.jinja`):** `autoescape=True` (required — explanation/suggested-fix text derives from formula strings and must not allow injection into the page), all CSS/JS inlined (no CDN references, no external files — "must open without a server"). Structure: header (workbook name, timestamp, summary counts, caveat banner) → severity-filter checkboxes (client-side vanilla JS toggling visibility via `data-severity` attributes, no server round trip) → per-sheet sections (heatmap grid with click-through anchors `href="#issue-<sheet>-<address>"` → that sheet's issue table with matching `id`/`data-severity` attributes) → footer repeating the caveat.

**Report-specific caveat** (distinct from `evaluation.py`'s corpus-specific one, since that one doesn't apply to an arbitrary real workbook): states plainly that these are Tier 0 structural checks only — deterministic pattern-matching, not semantic judgment — and that `intentional_override` cases (legitimate subtotals, deliberate overrides) aren't yet distinguished from real errors, since that's a Tier 1/2 capability not present in this build. Printed directly in the HTML/JSON output itself, not left in a docstring — same "must actually be visible to whoever runs it" standard `evaluation.py`'s caveat was held to (and tested for).

**CLI:** new `ssmlint report <path> [--html OUT.html] [--json OUT.json]` subcommand in `cli.py`, following `_cmd_dump`'s exact pattern (`_cmd_report(args) -> int`, `subparsers.add_parser(...).set_defaults(func=...)`). If neither flag given, default to writing HTML to `<stem>.report.html` in the cwd and printing a short confirmation line — a deliberate UX difference from `dump`'s stdout-first default, since dumping a full HTML document to stdout isn't useful; state this explicitly rather than silently deviating from the established pattern.

**Tests (`test_report.py`, mirrors `test_evaluation.py`/`test_labeling.py`'s pattern):**
- Unit tests on hand-constructed `Issue` lists (no workbook, no I/O): ranking order, heatmap building (including the "two rules flag the same cell" case), bounding-box math on a small known address set.
- `WorkbookReport.to_dict()` round-trip test, **including an explicit assertion that the `caveat` field is present and non-empty in the JSON output** — the HTML test checks the caveat renders visibly on the page; this is the independent check that the JSON consumer gets it too, so both output formats are verified to carry it, not just the human-facing one.
- HTML rendering test on a small hand-built `WorkbookReport`: structural string-containment checks (workbook name, each cell address, `data-severity` attributes present), plus a **deliberate XSS/escaping test** — construct a `RankedIssue` with `<script>alert(1)</script>` in `explanation`, render, assert the raw tag never appears unescaped.
- Integration test against the real `tests/fixtures/revenue_row_with_hardcode.xlsx` fixture (already known-flagged per `CONTRIBUTING.md`: `H14: high`, `B14: medium` from `LiteralInBlockRule`) — run `build_report()` through the full real pipeline, assert the exact ranked order and heatmap contents, not a hand-built stub.
- CLI integration test added to `tests/test_cli.py` (mirrors its existing `dump`-to-file test): run `main(["report", str(fixture), "--html", ..., "--json", ...])`, assert both files exist, JSON parses, HTML contains expected markers.
- Manual verification: actually open the generated HTML in a real browser before calling this done — the one deliverable in this whole project whose entire point is a human-facing artifact, so string-containment tests alone aren't sufficient sign-off.

---

## Sequencing

1. Add `jinja2` to `pyproject.toml`'s main `dependencies` (confirmed — not a dev-only tool, it's core pipeline stage [9]).
2. Build the ablation table first (`ablation.py` + `scripts/run_ablation.py` + tests) — no new dependency needed for this half, pure reuse of Week 4 code, delivers the README's "publish the ablation table" ask standalone.
3. Build the report builder (`report.py` + template + `cli.py` subcommand + tests).
4. Update `CONTRIBUTING.md`'s Week 6 table (flip "Full ablation table" and "HTML report builder" to 🟢 with the established how-it-was-solved/bugs-fixed narrative pattern; "Local LLM adjudicator" stays 🔴, explicitly deferred) once both are verified end-to-end with real output.
5. Commit + push at fine granularity per the standing git workflow (checkpoint after each logical piece, not one giant commit at the end).

## Verification

Run the full test suite after each new file lands (`pytest`, currently 173 passing — expect it to grow with `test_ablation.py`/`test_report.py`/`test_cli.py` additions). For each deliverable, paste real printed output before considering it done: the actual `scripts/run_ablation.py` table, and the actual generated HTML file's content (plus confirmation it was opened and visually checked, not just string-matched) — matching this project's standing discipline of showing real computed output rather than pass/fail counts alone.
