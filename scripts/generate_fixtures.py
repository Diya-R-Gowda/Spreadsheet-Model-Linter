"""Generate the synthetic .xlsx fixtures used by the test suite.

Fixtures are generated rather than hand-crafted so they stay reproducible
and in sync with whatever this script says they contain. Run directly:

    python scripts/generate_fixtures.py

The test suite also runs this automatically via tests/conftest.py.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.formula import ArrayFormula

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

MONTHS = ["Jan-24", "Feb-24", "Mar-24", "Apr-24"]


def build_clean_model() -> openpyxl.Workbook:
    """A simple model: a header row, a revenue row of consistent formulas, a driver row."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Model"

    ws["A1"] = "Line Item"
    for i, month in enumerate(MONTHS):
        ws.cell(row=1, column=2 + i, value=month)

    ws["A2"] = "Revenue"
    ws["B2"] = 100000
    ws["B2"].number_format = "#,##0"
    for col_idx in range(3, 2 + len(MONTHS)):
        prev_letter = get_column_letter(col_idx - 1)
        this_letter = get_column_letter(col_idx)
        cell = ws[f"{this_letter}2"]
        cell.value = f"={prev_letter}2*1.05"
        cell.number_format = "#,##0"

    ws["A3"] = "Growth Rate"
    for col_idx in range(2, 2 + len(MONTHS)):
        letter = get_column_letter(col_idx)
        ws[f"{letter}3"] = 0.05

    return wb


def build_merged_cells() -> openpyxl.Workbook:
    """A budget sheet with a merged title spanning a header row."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Budget"

    ws.merge_cells("A1:D1")
    ws["A1"] = "FY2024 Budget"

    ws["A2"] = "Category"
    ws["B2"] = "Q1"
    ws["C2"] = "Q2"
    ws["D2"] = "Total"

    ws["A3"] = "Opex"
    ws["B3"] = 1000
    ws["C3"] = 1200
    ws["D3"] = "=B3+C3"

    return wb


def build_named_range_cross_sheet() -> openpyxl.Workbook:
    """A two-sheet model with a workbook-level name, a sheet-level name, and a cross-sheet formula."""
    wb = openpyxl.Workbook()
    assumptions = wb.active
    assumptions.title = "Assumptions"
    assumptions["A1"] = "Tax Rate"
    assumptions["B1"] = 0.21

    model = wb.create_sheet("Model")
    model["A1"] = "Pre-tax Income"
    model["B1"] = 500000
    model["A2"] = "Tax"
    model["B2"] = "=B1*Assumptions!B1"
    model["A3"] = "Net Income"
    model["B3"] = "=B1-B2"

    wb.defined_names["TaxRate"] = DefinedName("TaxRate", attr_text="Assumptions!$B$1")
    model.defined_names["LocalNet"] = DefinedName("LocalNet", attr_text="Model!$B$3")

    return wb


def build_skip_cases() -> openpyxl.Workbook:
    """Formula shapes the parsing spine should log to `skipped` rather than crash on."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Edge Cases"

    ws["A1"] = "Array Formula"
    ws["B1"] = ArrayFormula("B1", "=SUM(C1:C3*D1:D3)")

    ws["A2"] = "LET Formula"
    ws["B2"] = "=LET(x, 5, x*2)"

    ws["A3"] = "Structured Table Ref"
    ws["B3"] = "=SUM(Table1[Column1])"

    ws["A4"] = "Normal Formula"
    ws["B4"] = '=A4&"!"'

    return wb


REVENUE_MONTHS = [
    "Jan-24", "Feb-24", "Mar-24", "Apr-24", "May-24", "Jun-24",
    "Jul-24", "Aug-24", "Sep-24", "Oct-24", "Nov-24", "Dec-24",
]


def build_revenue_row_with_hardcode() -> openpyxl.Workbook:
    """The README's own C14:N14 example: a 12-month revenue row (row 14)
    with an identical growth formula in every column except H14, which is
    a hardcoded literal breaking the pattern in the middle of the row.

    Row 13 holds month headers, row 14 the revenue formula chain (seeded
    from B14), row 15 a flat growth-rate driver each formula reads from.
    Column range is B (seed) through M (12 formula columns after the
    seed), i.e. the formula run lives in C14:M14 with H14 (the 6th
    formula column) hardcoded — deliberately mirroring the README's
    "H14 hardcoded while neighbors are formulas" framing without needing
    exactly 14 columns to land on letter N.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Model"

    ws["A13"] = "Line Item"
    for i, month in enumerate(REVENUE_MONTHS):
        ws.cell(row=13, column=2 + i, value=month)

    ws["A14"] = "Revenue"
    ws["B14"] = 100000
    ws["B14"].number_format = "#,##0"

    ws["A15"] = "Growth Rate"
    for col_idx in range(2, 2 + len(REVENUE_MONTHS)):
        letter = get_column_letter(col_idx)
        ws[f"{letter}15"] = 0.05

    for col_idx in range(3, 2 + len(REVENUE_MONTHS)):
        prev_letter = get_column_letter(col_idx - 1)
        this_letter = get_column_letter(col_idx)
        cell = ws[f"{this_letter}14"]
        if this_letter == "H":
            cell.value = 4500000  # hardcoded — breaks the pattern, on purpose
        else:
            cell.value = f"={prev_letter}14*(1+{prev_letter}15)"
        cell.number_format = "#,##0"

    return wb


def build_range_boundary_mismatch() -> openpyxl.Workbook:
    """A rolling 3-month trailing-sum row where one cell's range is off by one.

    Row 4 holds 12 months of raw data (B4:M4). Row 5 sums the trailing
    3 months for each column from D5 onward — except H5, which sums only
    the trailing 2 months (a hardcoded off-by-one instead of extending
    the range to match its neighbors). D5:G5 and I5:M5 should cluster
    into two blocks sharing the identical `SUM(3-trailing)` pattern, with
    H5 sitting between them as the deliberate off-by-one deviation.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Rolling"

    ws["A4"] = "Monthly Data"
    for i, month in enumerate(REVENUE_MONTHS):
        col = get_column_letter(2 + i)
        ws[f"{col}3"] = month
        ws[f"{col}4"] = 1000 * (i + 1)

    ws["A5"] = "Trailing 3-Month Sum"
    for col_idx in range(4, 2 + len(REVENUE_MONTHS)):
        this_letter = get_column_letter(col_idx)
        cell = ws[f"{this_letter}5"]
        if this_letter == "H":
            start_letter = get_column_letter(col_idx - 1)  # off-by-one: only 2 months
        else:
            start_letter = get_column_letter(col_idx - 2)
        cell.value = f"=SUM({start_letter}4:{this_letter}4)"

    return wb


def build_reference_to_blank_gap() -> openpyxl.Workbook:
    """A rolling 3-month trailing-sum row where one month's raw data is blank.

    Row 4 holds 12 months of raw data (B4:M4) — except F4, which is left
    genuinely blank (no value, no formula). Row 5 sums the trailing 3
    months for each column from D5 onward, all sharing the identical
    `SUM(3-trailing)` pattern (unlike range_boundary_mismatch.xlsx, no
    boundary is off — every formula's range shape is correct). Because
    F4 is blank, the three formulas whose trailing window includes it
    (F5, G5, H5) each read a blank precedent, while the rest of the
    block's members (D5, E5, I5..M5) read real data throughout — a
    genuine majority-clean/minority-affected split for
    reference_to_blank.py to catch.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Rolling"

    ws["A4"] = "Monthly Data"
    for i, month in enumerate(REVENUE_MONTHS):
        col = get_column_letter(2 + i)
        ws[f"{col}3"] = month
        if col != "F":
            ws[f"{col}4"] = 1000 * (i + 1)
        # F4 intentionally left blank -- a genuine gap in the monthly data

    ws["A5"] = "Trailing 3-Month Sum"
    for col_idx in range(4, 2 + len(REVENUE_MONTHS)):
        this_letter = get_column_letter(col_idx)
        start_letter = get_column_letter(col_idx - 2)
        ws[f"{this_letter}5"] = f"=SUM({start_letter}4:{this_letter}4)"

    return wb


def build_inconsistent_anchoring() -> openpyxl.Workbook:
    """A growth-rate-driven revenue row where one cell's anchoring drifts.

    B1 holds a fixed growth-rate assumption. Row 14 is a chain of
    "=prev*(1+$B$1)" formulas, each dragged across from the previous
    column, correctly holding B1 fixed with both axes anchored — except
    H14, which lost its column anchor (`B$1` instead of `$B$1`). Right
    now H14 still computes correctly (it's still literally reading B1),
    but this is exactly the mistake horizontal fill-drag causes for real:
    without the column anchor, filling further right would silently walk
    the reference to C1, D1, and so on instead of staying pinned to B1.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Model"

    ws["A1"] = "Growth Rate Assumption"
    ws["B1"] = 0.05

    ws["A13"] = "Line Item"
    for i, month in enumerate(REVENUE_MONTHS):
        ws.cell(row=13, column=2 + i, value=month)

    ws["A14"] = "Revenue"
    ws["B14"] = 100000
    ws["B14"].number_format = "#,##0"

    for col_idx in range(3, 2 + len(REVENUE_MONTHS)):
        prev_letter = get_column_letter(col_idx - 1)
        this_letter = get_column_letter(col_idx)
        cell = ws[f"{this_letter}14"]
        if this_letter == "H":
            cell.value = f"={prev_letter}14*(1+B$1)"  # anchoring drift: lost the column anchor
        else:
            cell.value = f"={prev_letter}14*(1+$B$1)"
        cell.number_format = "#,##0"

    return wb


def build_circular_reference() -> openpyxl.Workbook:
    """A direct two-cell circular reference: A1 depends on B1, B1 depends on A1."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Circular"

    ws["A1"] = "=B1+1"
    ws["B1"] = "=A1+1"

    return wb


def build_blank_reference() -> openpyxl.Workbook:
    """A formula referencing a genuinely blank cell within the sheet's used range.

    D1 is set purely to widen the sheet's used range so C1 — never
    written to — still shows up as a real (blank) cell when iterated,
    rather than simply falling outside the workbook's dimensions.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Blank Refs"

    ws["A1"] = 10
    ws["B1"] = "=A1+C1"  # C1 is intentionally never written to
    ws["D1"] = "marker"

    return wb


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    build_clean_model().save(FIXTURES_DIR / "clean_model.xlsx")
    build_merged_cells().save(FIXTURES_DIR / "merged_cells.xlsx")
    build_named_range_cross_sheet().save(FIXTURES_DIR / "named_range_cross_sheet.xlsx")
    build_skip_cases().save(FIXTURES_DIR / "skip_cases.xlsx")
    build_circular_reference().save(FIXTURES_DIR / "circular_reference.xlsx")
    build_blank_reference().save(FIXTURES_DIR / "blank_reference.xlsx")
    build_revenue_row_with_hardcode().save(FIXTURES_DIR / "revenue_row_with_hardcode.xlsx")
    build_range_boundary_mismatch().save(FIXTURES_DIR / "range_boundary_mismatch.xlsx")
    build_reference_to_blank_gap().save(FIXTURES_DIR / "reference_to_blank_gap.xlsx")
    build_inconsistent_anchoring().save(FIXTURES_DIR / "inconsistent_anchoring.xlsx")
    print(f"Wrote fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
