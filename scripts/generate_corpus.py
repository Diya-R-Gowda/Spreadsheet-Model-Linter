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
    state: str  # "clean" | "injected_bug" | "known_intentional"
    rule_id: str | None = None  # set iff state == "injected_bug"
    note: str | None = None  # set iff state == "known_intentional"


@dataclass(frozen=True)
class CorpusEntry:
    name: str
    base_shape: str  # "growth_chain" | "trailing_sum"
    cells: list[CellGroundTruth]


# ---------------------------------------------------------------------------
# Shape: growth-chain row
# ---------------------------------------------------------------------------


def _growth_chain_workbook(
    n: int, injection: tuple[str, int] | None = None
) -> tuple[openpyxl.Workbook, list[CellGroundTruth]]:
    """A seed literal at B14 followed by `n` formula cells (C14 onward),
    each `=prev*(1+$B$1)`. `injection` is `(kind, idx)` where `kind` is
    "literal" or "anchoring" and `idx` is a 0-based index into the n
    formula cells (0 = the first formula cell, C14).
    """
    sheet = "Model"
    row = 14
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet

    ws["B1"] = GROWTH_RATE
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

        if i == inject_idx and inject_kind == "literal":
            ws.cell(row=row, column=col, value=round(value, 2))
            ground_truth.append(CellGroundTruth(cell=addr, state="injected_bug", rule_id="literal-in-formula-block"))
        elif i == inject_idx and inject_kind == "anchoring":
            ws.cell(row=row, column=col, value=f"={prev_letter}{row}*(1+B$1)")
            ground_truth.append(CellGroundTruth(cell=addr, state="injected_bug", rule_id="inconsistent-anchoring"))
        else:
            ws.cell(row=row, column=col, value=f"={prev_letter}{row}*(1+$B$1)")
            ground_truth.append(CellGroundTruth(cell=addr, state="clean"))

    return wb, ground_truth


# ---------------------------------------------------------------------------
# Shape: trailing-3-month-sum row
# ---------------------------------------------------------------------------


def _trailing_sum_workbook(
    m: int, injection: tuple[str, int] | None = None
) -> tuple[openpyxl.Workbook, list[CellGroundTruth]]:
    """`m` months of raw data at row 4 (B4 onward), and a trailing-3-month
    `SUM` row at row 5 starting at the 3rd data column (`n = m - 2`
    formula cells). `injection` is `(kind, idx)`:
      - ("range_boundary", idx): idx is a 0-based index into the n SUM
        cells; that one cell's range start shifts by one (2-month window
        instead of 3).
      - ("reference_to_blank", idx): idx is a 0-based index into the m
        raw data columns; that one raw cell is left genuinely blank.
    """
    sheet = "Rolling"
    data_row, sum_row = 4, 5
    n = m - 2
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet

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

    return wb, ground_truth


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


def _write_ground_truth(entry: CorpusEntry) -> None:
    payload = {
        "workbook": f"{entry.name}.xlsx",
        "base_shape": entry.base_shape,
        "cells": [asdict(c) for c in entry.cells],
    }
    (CORPUS_DIR / f"{entry.name}.ground_truth.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    entries = _growth_chain_entries() + _trailing_sum_entries()
    for entry in entries:
        _write_ground_truth(entry)
    print(f"Wrote {len(entries)} corpus entries ({len(entries) * 2} files) to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
