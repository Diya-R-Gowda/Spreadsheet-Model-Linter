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
    print(f"Wrote fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
