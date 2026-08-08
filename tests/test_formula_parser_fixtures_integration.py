"""Sanity-check the tokenizer/parser against real extracted formulas.

Read-only integration check: reuses the Prompt-1 parsing spine
(ssmlint.parser.parse_workbook) purely to pull formula strings out of the
synthetic fixtures, and feeds each one through parse_formula. Does not
modify parser.py, cli.py, or the fixtures, and does not wire the AST back
into the parsing spine — that wiring is a later step.
"""

from __future__ import annotations

from pathlib import Path

from ssmlint.errors import FormulaError, UnsupportedFormulaError
from ssmlint.formula_parser import parse_formula
from ssmlint.parser import parse_workbook

ALL_FIXTURES = [
    "clean_model.xlsx",
    "merged_cells.xlsx",
    "named_range_cross_sheet.xlsx",
    "skip_cases.xlsx",
]

# skip_cases.xlsx deliberately contains formula shapes our v1 whitelist
# rejects (LET, a structured table reference). Everything else in it, and
# every formula in the other three fixtures, is expected to parse cleanly.
EXPECTED_REJECTIONS = {
    "Edge Cases!B2": "LET",
    "Edge Cases!B3": "structured table reference",
}


def _all_formula_cells(fixtures_dir: Path):
    for filename in ALL_FIXTURES:
        parsed = parse_workbook(fixtures_dir / filename)
        for sheet in parsed.sheets:
            for cell in sheet.cells:
                if cell.formula is not None:
                    yield filename, cell


def test_no_formula_in_the_fixtures_crashes_the_parser(fixtures_dir: Path) -> None:
    """The real sanity check: every formula must either parse or raise a
    known FormulaError — never an unrelated exception."""
    for filename, cell in _all_formula_cells(fixtures_dir):
        try:
            parse_formula(cell.formula)
        except FormulaError:
            pass  # a clean, recognized rejection is fine
        except Exception as exc:  # pragma: no cover - this is what we're guarding against
            raise AssertionError(
                f"{filename} {cell.address} ({cell.formula!r}) raised an unexpected "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def test_expected_rejections_raise_unsupported_formula_error(fixtures_dir: Path) -> None:
    for filename, cell in _all_formula_cells(fixtures_dir):
        if cell.address not in EXPECTED_REJECTIONS:
            continue
        try:
            parse_formula(cell.formula)
        except UnsupportedFormulaError:
            continue
        else:
            raise AssertionError(
                f"{cell.address} ({cell.formula!r}) was expected to raise "
                "UnsupportedFormulaError but parsed successfully"
            )


def test_non_edge_case_fixtures_parse_with_no_rejections(fixtures_dir: Path) -> None:
    for filename, cell in _all_formula_cells(fixtures_dir):
        if filename == "skip_cases.xlsx":
            continue
        # must not raise at all
        node = parse_formula(cell.formula)
        assert node is not None
