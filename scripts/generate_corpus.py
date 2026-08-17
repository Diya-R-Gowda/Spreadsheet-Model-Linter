"""Synthetic-corruption corpus generator — the last Week 3 item.

Produces the corpus that makes precision/recall measurable at all (Week
4's evaluation harness) and later feeds Week 5's classifier training.
Two structurally distinct base shapes, each parametrized by formula-run
length and (optionally) an injected corruption, cover all four Tier 0
rules:

  - growth-chain row (`_growth_chain_workbook`): a seed literal followed
    by a chain of `=prev*(1+$B$1)` formulas. Feeds `literal-in-formula-
    block` (replace one formula with its computed value) and
    `inconsistent-anchoring` (drop the column anchor on one formula's
    reference to $B$1).
  - trailing-window-sum row (`_trailing_sum_workbook`): a row of raw
    monthly data plus a row of trailing-3-month `SUM` formulas. Feeds
    `range-boundary-mismatch` (shift one SUM's range start by one) and
    `reference-to-blank` (leave one raw data cell blank instead of
    editing any formula — see below for why).

WHY reference-to-blank'S INJECTION NEVER TOUCHES A FORMULA
------------------------------------------------------------------------
`ReferenceToBlankRule.evaluate()` only ever inspects `block.cells` (real
block members) — never `non_conforming`/`near_misses` (confirmed by
reading `rules/reference_to_blank.py`). Editing a formula's own reference
text would change its R1C1 pattern and evict it from the block entirely,
so the rule would never see it. The correct injection is the inverse:
leave every formula in the row untouched and blank one *raw data* cell
that only some (not all, not zero) of the row's trailing-window formulas
read from — exactly the shape `reference_to_blank_gap.xlsx` already
proved out by hand in stage 6c.

SEVERITY CONTROL, VIA BLOCK-ADJACENCY POSITION (three of the four rules)
------------------------------------------------------------------------
literal-in-formula-block, range-boundary-mismatch, and inconsistent-
anchoring all derive severity the same way: `high` when the injected
cell borders a same-pattern block on BOTH sides, `medium` otherwise (see
each rule's own `_severity`). This is controlled purely by where the
injection sits in the formula run:
  - "interior": injected at index 3 of a run of >= 7, leaving >= 3 clean
    cells on both sides -> both sides re-cluster into blocks -> `high`.
  - "edge_left" / "edge_right": injected at the first/last index of a
    shorter run, leaving a clean block on only one side -> `medium`.
reference-to-blank has no adjacency concept (it compares block members'
own precedents, not neighbors) — its severity instead comes directly
from clean-member-count vs. affected-member-count, controlled by where
the blanked raw-data cell sits relative to the trailing window.

DETERMINISTIC, NO RANDOMNESS
------------------------------------------------------------------------
Every corpus entry is produced from an explicit (shape, injection,
length) parameter tuple enumerated in `main()` — no `random` module use
anywhere — so regeneration is always byte-identical, the same guarantee
`generate_fixtures.py` already relies on. Corpus output is regenerated
on demand (`python scripts/generate_corpus.py`), never committed as
binaries, and gitignored exactly like `tests/fixtures/*.xlsx`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

GROWTH_RATE = 0.05
SEED_VALUE = 100000
RAW_MONTHLY_BASE = 1000


@dataclass(frozen=True)
class CellGroundTruth:
    cell: str  # sheet-qualified address, e.g. "Model!H14"
    state: str  # "clean" | "injected_bug" | "known_intentional" | "subtotal" | "ambiguous"
    rule_id: str | None = None  # set iff state in ("injected_bug", "ambiguous") -- which rule's predicate this cell matches
    note: str | None = None  # set iff state == "known_intentional"


@dataclass(frozen=True)
class CorpusEntry:
    name: str
    base_shape: str  # "growth_chain" | "trailing_sum"
    cells: list[CellGroundTruth]


# ---------------------------------------------------------------------------
# Shape: growth-chain row
# ---------------------------------------------------------------------------


def _write_growth_chain_row(
    ws, sheet: str, row: int, n: int, injection: tuple[str, int] | None = None
) -> list[CellGroundTruth]:
    """Writes a seed literal at column B of `row`, followed by `n` formula
    cells (C onward), each `=prev*(1+$B$1)`. Assumes `ws["B1"]` (the
    shared growth-rate driver) is already written by the caller. `injection`
    is `(kind, idx)` where `idx` is a 0-based index into the n formula cells
    (0 = the first formula cell) and `kind` is one of:
      - "literal" / "anchoring": the standard injected_bug shapes.
      - "literal_ambiguous" / "anchoring_ambiguous": structurally identical
        writes to the two above, but ground-truthed `state="ambiguous"`
        instead of `"injected_bug"` -- Tier 0 still flags these
        deterministically (it has no concept of ambiguity), but they're
        intended to carry the `unknown` block-level label downstream, not
        `suspected_error`. Only meaningful at small `n` / edge `idx` --
        weak block evidence is what makes the ambiguity label defensible
        (see scripts/generate_corpus.py module docstring).
    """
    ws.cell(row=row, column=2, value=SEED_VALUE)
    ground_truth = [
        CellGroundTruth(cell=f"{sheet}!B{row}", state="known_intentional", note="growth-chain seed literal")
    ]

    inject_kind, inject_idx = injection if injection else (None, None)
    value = SEED_VALUE
    for i in range(n):
        col = 3 + i
        prev_letter = get_column_letter(col - 1)
        this_letter = get_column_letter(col)
        addr = f"{sheet}!{this_letter}{row}"
        value = value * (1 + GROWTH_RATE)

        if i == inject_idx and inject_kind in ("literal", "literal_ambiguous"):
            state = "ambiguous" if inject_kind == "literal_ambiguous" else "injected_bug"
            ws.cell(row=row, column=col, value=round(value, 2))
            ground_truth.append(CellGroundTruth(cell=addr, state=state, rule_id="literal-in-formula-block"))
        elif i == inject_idx and inject_kind in ("anchoring", "anchoring_ambiguous"):
            state = "ambiguous" if inject_kind == "anchoring_ambiguous" else "injected_bug"
            ws.cell(row=row, column=col, value=f"={prev_letter}{row}*(1+B$1)")
            ground_truth.append(CellGroundTruth(cell=addr, state=state, rule_id="inconsistent-anchoring"))
        else:
            ws.cell(row=row, column=col, value=f"={prev_letter}{row}*(1+$B$1)")
            ground_truth.append(CellGroundTruth(cell=addr, state="clean"))

    return ground_truth


def _growth_chain_workbook(
    n: int, injection: tuple[str, int] | None = None
) -> tuple[openpyxl.Workbook, list[CellGroundTruth]]:
    """A single growth-chain row (row 14) on its own workbook. See
    `_write_growth_chain_row` for the row-construction details.
    """
    sheet = "Model"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ws["B1"] = GROWTH_RATE
    ground_truth = _write_growth_chain_row(ws, sheet, 14, n, injection)
    return wb, ground_truth


# ---------------------------------------------------------------------------
# Shape: trailing-3-month-sum row
# ---------------------------------------------------------------------------


def _write_trailing_sum_row(
    ws, sheet: str, data_row: int, sum_row: int, m: int, injection: tuple[str, int] | None = None
) -> list[CellGroundTruth]:
    """Writes `m` months of raw data at `data_row` (column B onward), and a
    trailing-3-month `SUM` row at `sum_row` starting at the 3rd data
    column (`n = m - 2` formula cells). `injection` is `(kind, idx)`:
      - ("range_boundary", idx): idx is a 0-based index into the n SUM
        cells; that one cell's range start shifts by one (2-month window
        instead of 3).
      - ("reference_to_blank", idx): idx is a 0-based index into the m
        raw data columns; that one raw cell is left genuinely blank.
    """
    n = m - 2
    inject_kind, inject_idx = injection if injection else (None, None)
    blank_idx = inject_idx if inject_kind == "reference_to_blank" else None
    range_boundary_idx = inject_idx if inject_kind == "range_boundary" else None

    for j in range(m):
        col = 2 + j
        if j == blank_idx:
            continue  # intentionally left blank -- the injected data gap
        ws.cell(row=data_row, column=col, value=RAW_MONTHLY_BASE * (j + 1))

    affected_formula_indices: set[int] = set()
    if blank_idx is not None:
        affected_formula_indices = {i for i in range(n) if blank_idx - 2 <= i <= blank_idx}

    ground_truth: list[CellGroundTruth] = []
    for i in range(n):
        col = 4 + i
        this_letter = get_column_letter(col)
        addr = f"{sheet}!{this_letter}{sum_row}"

        if i == range_boundary_idx:
            start_letter = get_column_letter(col - 1)  # off-by-one: 2-month window
            ws.cell(row=sum_row, column=col, value=f"=SUM({start_letter}{data_row}:{this_letter}{data_row})")
            ground_truth.append(CellGroundTruth(cell=addr, state="injected_bug", rule_id="range-boundary-mismatch"))
            continue

        start_letter = get_column_letter(col - 2)
        ws.cell(row=sum_row, column=col, value=f"=SUM({start_letter}{data_row}:{this_letter}{data_row})")
        if i in affected_formula_indices:
            ground_truth.append(CellGroundTruth(cell=addr, state="injected_bug", rule_id="reference-to-blank"))
        else:
            ground_truth.append(CellGroundTruth(cell=addr, state="clean"))

    return ground_truth


def _trailing_sum_workbook(
    m: int, injection: tuple[str, int] | None = None
) -> tuple[openpyxl.Workbook, list[CellGroundTruth]]:
    """A single trailing-3-month-sum row (data row 4, sum row 5) on its
    own workbook. See `_write_trailing_sum_row` for the row-construction
    details.
    """
    sheet = "Rolling"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ground_truth = _write_trailing_sum_row(ws, sheet, 4, 5, m, injection)
    return wb, ground_truth


# ---------------------------------------------------------------------------
# Shape: category totals (feeds the "subtotal" label)
# ---------------------------------------------------------------------------


def _write_category_total_row(
    ws, sheet: str, first_cat_row: int, cat_count: int, total_row: int, col_count: int
) -> list[CellGroundTruth]:
    """Writes `cat_count` rows of raw literal data (`first_cat_row` onward,
    column B onward) and one "Total" row (`total_row`) whose formula per
    column is `=SUM(<top of category rows>:<bottom of category rows>)` --
    a vertical aggregate, genuinely different in R1C1 shape from both the
    growth-chain (single relative precedent) and trailing-window-sum
    (fixed-width horizontal SUM) patterns. Category rows are raw data, not
    formulas, so they never form a blocks.py block themselves -- only the
    Total row does, and every one of its cells is ground-truthed
    `state="subtotal"`: a legitimate, intentionally-different aggregate
    that no Tier 0 rule is expected to flag (confirmed empirically, not
    assumed -- see tests/test_generate_corpus.py).
    """
    last_cat_row = first_cat_row + cat_count - 1
    ground_truth: list[CellGroundTruth] = []
    for j in range(col_count):
        col = 2 + j
        col_letter = get_column_letter(col)
        for r in range(cat_count):
            ws.cell(row=first_cat_row + r, column=col, value=RAW_MONTHLY_BASE * (r + 1) * (j + 1))
        addr = f"{sheet}!{col_letter}{total_row}"
        ws.cell(row=total_row, column=col, value=f"=SUM({col_letter}{first_cat_row}:{col_letter}{last_cat_row})")
        ground_truth.append(CellGroundTruth(cell=addr, state="subtotal"))
    return ground_truth


def _category_total_workbook(cat_count: int, col_count: int) -> tuple[openpyxl.Workbook, list[CellGroundTruth]]:
    """A single category-total block (category rows 14.., total row right
    below them) on its own workbook. See `_write_category_total_row`.
    """
    sheet = "Totals"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ground_truth = _write_category_total_row(ws, sheet, 14, cat_count, 14 + cat_count, col_count)
    return wb, ground_truth


# ---------------------------------------------------------------------------
# Shape: variance row (feeds pattern diversity -- a two-precedent formula,
# unlike growth-chain's single relative precedent or trailing-sum's SUM)
# ---------------------------------------------------------------------------


def _write_variance_row(
    ws, sheet: str, actual_row: int, budget_row: int, variance_row: int, n: int,
    injection: tuple[str, int] | None = None,
) -> list[CellGroundTruth]:
    """Writes two raw-data rows (Actual, Budget) and one variance row
    (`=Actual-Budget` per column) -- a genuinely different two-precedent
    R1C1 pattern from both the growth-chain (single relative precedent)
    and trailing-sum (SUM over a range) shapes. `injection` is `(kind, idx)`:
      - ("literal" / "literal_ambiguous", idx): idx is a 0-based column
        index into the n variance cells; that cell becomes a hardcoded
        literal instead of a formula (injected_bug / ambiguous -- same
        literal-in-formula-block predicate as the growth-chain shape).
      - ("reference_to_blank", idx): the Actual cell at column idx is left
        genuinely blank instead of written, leaving exactly one variance
        cell (the one at that column) with a blank precedent. Unlike
        trailing-sum's 3-month window, each variance cell here has exactly
        one Actual + one Budget precedent, so a single blank always
        affects exactly 1 of n members -- this shape structurally cannot
        produce a near-50/50 "ambiguous" split the way trailing-sum's
        windowed version can (see _ambiguous_reference_to_blank_entries).
    """
    inject_kind, inject_idx = injection if injection else (None, None)
    blank_idx = inject_idx if inject_kind == "reference_to_blank" else None

    ground_truth: list[CellGroundTruth] = []
    for j in range(n):
        col = 2 + j
        letter = get_column_letter(col)
        addr = f"{sheet}!{letter}{variance_row}"

        if j != blank_idx:
            ws.cell(row=actual_row, column=col, value=RAW_MONTHLY_BASE * 2 * (j + 1))
        ws.cell(row=budget_row, column=col, value=RAW_MONTHLY_BASE * (j + 1))

        if j == inject_idx and inject_kind in ("literal", "literal_ambiguous"):
            state = "ambiguous" if inject_kind == "literal_ambiguous" else "injected_bug"
            ws.cell(row=variance_row, column=col, value=RAW_MONTHLY_BASE * (j + 1))
            ground_truth.append(CellGroundTruth(cell=addr, state=state, rule_id="literal-in-formula-block"))
        else:
            ws.cell(row=variance_row, column=col, value=f"={letter}{actual_row}-{letter}{budget_row}")
            if j == blank_idx:
                ground_truth.append(CellGroundTruth(cell=addr, state="injected_bug", rule_id="reference-to-blank"))
            else:
                ground_truth.append(CellGroundTruth(cell=addr, state="clean"))

    return ground_truth


def _variance_workbook(
    n: int, injection: tuple[str, int] | None = None
) -> tuple[openpyxl.Workbook, list[CellGroundTruth]]:
    """A single variance row (Actual row 4, Budget row 5, Variance row 6)
    on its own workbook. See `_write_variance_row`.
    """
    sheet = "Variance"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ground_truth = _write_variance_row(ws, sheet, 4, 5, 6, n, injection)
    return wb, ground_truth


# ---------------------------------------------------------------------------
# Shape: cross-block isolation (one sheet, two independent blocks)
# ---------------------------------------------------------------------------


def _cross_block_isolation_workbook() -> tuple[openpyxl.Workbook, list[CellGroundTruth]]:
    """One sheet, two independent row blocks far apart (row 14, row 30):
    a corrupted growth-chain block (literal injection) and a clean
    trailing-sum block. Proves corrupting one block raises nothing in
    the other. blocks.py clusters strictly row-wise, so two different
    rows are already fully independent by construction -- the row gap
    is for sheet realism, not a detection requirement. The clean block
    deliberately uses the trailing-sum shape, not a second growth-chain,
    because growth-chain always has a seed-literal neighbor that
    legitimately fires literal-in-formula-block on its own (see the
    Week 4 design note) -- trailing-sum has no such neighbor, so it's
    the shape that can genuinely produce zero issues from every rule,
    proving isolation with no caveat.
    """
    sheet = "Combined"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ws["B1"] = GROWTH_RATE

    corrupted_gt = _write_growth_chain_row(ws, sheet, 14, 8, ("literal", 3))
    clean_gt = _write_trailing_sum_row(ws, sheet, 29, 30, 10, None)
    return wb, corrupted_gt + clean_gt


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


def _growth_chain_entries() -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []

    for rule_kind, rule_tag in (("literal", "literal_in_block"), ("anchoring", "inconsistent_anchoring")):
        for n in (4, 5, 6):
            for position, idx in (("edge_left", 0), ("edge_right", n - 1)):
                wb, gt = _growth_chain_workbook(n, (rule_kind, idx))
                name = f"{rule_tag}__{position}__n{n}"
                wb.save(CORPUS_DIR / f"{name}.xlsx")
                entries.append(CorpusEntry(name=name, base_shape="growth_chain", cells=gt))

        for n in (7, 8, 10, 12):
            wb, gt = _growth_chain_workbook(n, (rule_kind, 3))
            name = f"{rule_tag}__interior__n{n}"
            wb.save(CORPUS_DIR / f"{name}.xlsx")
            entries.append(CorpusEntry(name=name, base_shape="growth_chain", cells=gt))

    for n in (4, 6, 8, 10, 12):
        wb, gt = _growth_chain_workbook(n, None)
        name = f"clean__growth_chain__n{n}"
        wb.save(CORPUS_DIR / f"{name}.xlsx")
        entries.append(CorpusEntry(name=name, base_shape="growth_chain", cells=gt))

    return entries


def _trailing_sum_entries() -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []

    for n in (4, 5, 6):
        m = n + 2
        for position, idx in (("edge_left", 0), ("edge_right", n - 1)):
            wb, gt = _trailing_sum_workbook(m, ("range_boundary", idx))
            name = f"range_boundary__{position}__n{n}"
            wb.save(CORPUS_DIR / f"{name}.xlsx")
            entries.append(CorpusEntry(name=name, base_shape="trailing_sum", cells=gt))

    for n in (7, 8, 10, 12):
        m = n + 2
        wb, gt = _trailing_sum_workbook(m, ("range_boundary", 3))
        name = f"range_boundary__interior__n{n}"
        wb.save(CORPUS_DIR / f"{name}.xlsx")
        entries.append(CorpusEntry(name=name, base_shape="trailing_sum", cells=gt))

    # reference-to-blank, "medium" severity: affected members >= clean members
    for m, blank_idx in ((6, 1), (6, 2), (7, 2)):
        wb, gt = _trailing_sum_workbook(m, ("reference_to_blank", blank_idx))
        name = f"reference_to_blank__medium__m{m}_blank{blank_idx}"
        wb.save(CORPUS_DIR / f"{name}.xlsx")
        entries.append(CorpusEntry(name=name, base_shape="trailing_sum", cells=gt))

    # reference-to-blank, "high" severity: clean members > affected members
    for m, blank_idx in ((10, 4), (12, 4), (14, 5)):
        wb, gt = _trailing_sum_workbook(m, ("reference_to_blank", blank_idx))
        name = f"reference_to_blank__high__m{m}_blank{blank_idx}"
        wb.save(CORPUS_DIR / f"{name}.xlsx")
        entries.append(CorpusEntry(name=name, base_shape="trailing_sum", cells=gt))

    for m in (6, 8, 10, 12, 14):
        wb, gt = _trailing_sum_workbook(m, None)
        name = f"clean__trailing_sum__m{m}"
        wb.save(CORPUS_DIR / f"{name}.xlsx")
        entries.append(CorpusEntry(name=name, base_shape="trailing_sum", cells=gt))

    return entries


def _ambiguous_growth_chain_entries() -> list[CorpusEntry]:
    """`unknown`-label entries: weak-evidence injections at n=4/edge -- the
    shortest length that produces any flag at all (verified: n=3/edge
    leaves only 2 clean cells, below blocks.py's min_block_size=3, so no
    block forms and nothing gets flagged). n=4/edge is also the exact
    shape the existing `suspected_error` corpus entries above already use.
    Confirmed directly with the user rather than silently avoided: new
    entries here deliberately share that same (length=3-cell-block,
    edge-position) feature signature with some existing suspected_error
    entries -- real-world ambiguity plausibly does cluster at a decision
    boundary rather than being cleanly separable by block length alone,
    as long as the bulk of each label's distribution stays separable
    (suspected_error's n=5/6/interior entries have no ambiguous
    counterpart at those lengths).
    """
    entries: list[CorpusEntry] = []
    for rule_kind, rule_tag in (("literal_ambiguous", "literal_in_block"), ("anchoring_ambiguous", "inconsistent_anchoring")):
        for position, idx in (("edge_left", 0), ("edge_right", 3)):
            wb, gt = _growth_chain_workbook(4, (rule_kind, idx))
            name = f"ambiguous__{rule_tag}__weak_{position}__n4"
            wb.save(CORPUS_DIR / f"{name}.xlsx")
            entries.append(CorpusEntry(name=name, base_shape="growth_chain", cells=gt))
    return entries


def _category_total_entries() -> list[CorpusEntry]:
    """`subtotal`-label entries. Parametrized by category-row count (2-6)
    and column count (3-14, the minimum needed for the Total row itself to
    form a >=3-cell block) -- 5 x 12 = 60 entries, each contributing one
    genuinely subtotal-labeled block. No injected corruption anywhere in
    this shape; every Total-row cell is ground-truthed "subtotal" by
    construction, verified empirically to raise zero Tier 0 issues (see
    tests/test_generate_corpus.py).
    """
    entries: list[CorpusEntry] = []
    for cat_count in range(2, 7):
        for col_count in range(3, 15):
            wb, gt = _category_total_workbook(cat_count, col_count)
            name = f"subtotal__category_total__cat{cat_count}_col{col_count}"
            wb.save(CORPUS_DIR / f"{name}.xlsx")
            entries.append(CorpusEntry(name=name, base_shape="category_total", cells=gt))
    return entries


def _variance_entries() -> list[CorpusEntry]:
    """Third base shape, for R1C1 pattern diversity: `suspected_error`
    (literal-in-formula-block, reference-to-blank) and `unknown`
    (literal_ambiguous at n=4/edge, same rationale as
    _ambiguous_growth_chain_entries) examples hosted on a genuinely
    different two-precedent formula pattern (`=Actual-Budget`), not on
    growth-chain or trailing-sum.
    """
    entries: list[CorpusEntry] = []

    for n in (4, 5, 6):
        for position, idx in (("edge_left", 0), ("edge_right", n - 1)):
            wb, gt = _variance_workbook(n, ("literal", idx))
            name = f"variance__literal_in_block__{position}__n{n}"
            wb.save(CORPUS_DIR / f"{name}.xlsx")
            entries.append(CorpusEntry(name=name, base_shape="variance", cells=gt))

    for n in (7, 8, 10, 12):
        wb, gt = _variance_workbook(n, ("literal", 3))
        name = f"variance__literal_in_block__interior__n{n}"
        wb.save(CORPUS_DIR / f"{name}.xlsx")
        entries.append(CorpusEntry(name=name, base_shape="variance", cells=gt))

    for position, idx in (("edge_left", 0), ("edge_right", 3)):
        wb, gt = _variance_workbook(4, ("literal_ambiguous", idx))
        name = f"ambiguous__variance__literal_in_block__weak_{position}__n4"
        wb.save(CORPUS_DIR / f"{name}.xlsx")
        entries.append(CorpusEntry(name=name, base_shape="variance", cells=gt))

    for n, blank_idx in ((5, 2), (8, 4), (10, 5)):
        wb, gt = _variance_workbook(n, ("reference_to_blank", blank_idx))
        name = f"variance__reference_to_blank__n{n}_blank{blank_idx}"
        wb.save(CORPUS_DIR / f"{name}.xlsx")
        entries.append(CorpusEntry(name=name, base_shape="variance", cells=gt))

    for n in (4, 6, 8, 10):
        wb, gt = _variance_workbook(n, None)
        name = f"clean__variance__n{n}"
        wb.save(CORPUS_DIR / f"{name}.xlsx")
        entries.append(CorpusEntry(name=name, base_shape="variance", cells=gt))

    return entries


def _ambiguous_reference_to_blank_entries() -> list[CorpusEntry]:
    """More `unknown`-label volume via reference-to-blank's own natural
    ambiguity axis: how close the affected-vs-clean member split is to
    50/50 (a near-tie is weaker evidence that the "clean" pattern is
    really the established norm). Sweeps trailing-sum's (m, blank_idx)
    parameter space programmatically -- rather than hand-picking values --
    filters to entries whose affected/(affected+clean) ratio lands in
    [0.4, 0.6], and skips any (m, blank_idx) pair the existing
    `suspected_error` entries in _trailing_sum_entries already use, so
    nothing here ever duplicates or silently reclassifies a shipped entry.
    Real achieved count is reported by scripts/generate_corpus.py's own
    printed total, not assumed in advance.
    """
    already_shipped_medium = {(6, 1), (6, 2), (7, 2)}
    already_shipped_high = {(10, 4), (12, 4), (14, 5)}
    already_shipped = already_shipped_medium | already_shipped_high

    entries: list[CorpusEntry] = []
    for m in range(6, 21):
        n = m - 2
        for blank_idx in range(m):
            if (m, blank_idx) in already_shipped:
                continue
            affected = len({i for i in range(n) if blank_idx - 2 <= i <= blank_idx})
            clean = n - affected
            if affected == 0 or clean == 0:
                continue
            ratio = affected / n
            if not (0.4 <= ratio <= 0.6):
                continue

            wb, gt = _trailing_sum_workbook(m, ("reference_to_blank", blank_idx))
            gt = [
                CellGroundTruth(cell=c.cell, state="ambiguous", rule_id=c.rule_id, note=c.note)
                if c.state == "injected_bug"
                else c
                for c in gt
            ]
            name = f"ambiguous__reference_to_blank__near_tie__m{m}_blank{blank_idx}"
            wb.save(CORPUS_DIR / f"{name}.xlsx")
            entries.append(CorpusEntry(name=name, base_shape="trailing_sum", cells=gt))
    return entries


def _cross_block_isolation_entries() -> list[CorpusEntry]:
    wb, gt = _cross_block_isolation_workbook()
    name = "cross_block_isolation__literal_in_block"
    wb.save(CORPUS_DIR / f"{name}.xlsx")
    return [CorpusEntry(name=name, base_shape="cross_block_isolation", cells=gt)]


def _write_ground_truth(entry: CorpusEntry) -> None:
    payload = {
        "workbook": f"{entry.name}.xlsx",
        "base_shape": entry.base_shape,
        "cells": [asdict(c) for c in entry.cells],
    }
    (CORPUS_DIR / f"{entry.name}.ground_truth.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    entries = (
        _growth_chain_entries()
        + _trailing_sum_entries()
        + _cross_block_isolation_entries()
        + _ambiguous_growth_chain_entries()
        + _category_total_entries()
        + _variance_entries()
        + _ambiguous_reference_to_blank_entries()
    )
    for entry in entries:
        _write_ground_truth(entry)
    print(f"Wrote {len(entries)} corpus entries ({len(entries) * 2} files) to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
