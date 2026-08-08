from __future__ import annotations

from pathlib import Path

from ssmlint.parser import CellRecord, SheetRecord, parse_workbook


def _sheet(sheets: list[SheetRecord], name: str) -> SheetRecord:
    for sheet in sheets:
        if sheet.name == name:
            return sheet
    raise AssertionError(f"sheet {name!r} not found among {[s.name for s in sheets]}")


def _cell(sheet: SheetRecord, coordinate: str, sheet_name: str) -> CellRecord:
    address = f"{sheet_name}!{coordinate}"
    for cell in sheet.cells:
        if cell.address == address:
            return cell
    raise AssertionError(f"cell {address!r} not found")


def test_basic_formula_extraction(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "clean_model.xlsx")
    model = _sheet(parsed.sheets, "Model")

    c2 = _cell(model, "C2", "Model")
    assert c2.formula == "=B2*1.05"
    assert c2.value is None
    assert c2.number_format == "#,##0"

    d2 = _cell(model, "D2", "Model")
    assert d2.formula == "=C2*1.05"

    e2 = _cell(model, "E2", "Model")
    assert e2.formula == "=D2*1.05"


def test_literal_values(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "clean_model.xlsx")
    model = _sheet(parsed.sheets, "Model")

    b2 = _cell(model, "B2", "Model")
    assert b2.formula is None
    assert b2.value == 100000
    assert b2.number_format == "#,##0"

    a2 = _cell(model, "A2", "Model")
    assert a2.formula is None
    assert a2.value == "Revenue"


def test_merged_cells(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "merged_cells.xlsx")
    budget = _sheet(parsed.sheets, "Budget")

    for coordinate in ("A1", "B1", "C1", "D1"):
        cell = _cell(budget, coordinate, "Budget")
        assert cell.merged is True, f"{coordinate} should be part of the merged range"

    anchor = _cell(budget, "A1", "Budget")
    assert anchor.value == "FY2024 Budget"

    non_merged = _cell(budget, "A2", "Budget")
    assert non_merged.merged is False


def test_named_ranges(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "named_range_cross_sheet.xlsx")
    by_name = {n.name: n for n in parsed.named_ranges}

    assert "TaxRate" in by_name
    assert by_name["TaxRate"].scope == "workbook"
    assert by_name["TaxRate"].refers_to == "Assumptions!$B$1"

    assert "LocalNet" in by_name
    assert by_name["LocalNet"].scope == "Model"
    assert by_name["LocalNet"].refers_to == "Model!$B$3"


def test_cross_sheet_reference_detection(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "named_range_cross_sheet.xlsx")
    model = _sheet(parsed.sheets, "Model")

    tax_cell = _cell(model, "B2", "Model")
    assert tax_cell.formula == "=B1*Assumptions!B1"
    assert tax_cell.cross_sheet_refs == ["Assumptions"]

    net_cell = _cell(model, "B3", "Model")
    assert net_cell.formula == "=B1-B2"
    assert net_cell.cross_sheet_refs == []


def test_skip_and_log_behavior(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "skip_cases.xlsx")
    edge = _sheet(parsed.sheets, "Edge Cases")

    reasons_by_address = {}
    for item in parsed.skipped:
        reasons_by_address.setdefault(item.address, []).append(item.reason)

    assert "Edge Cases!B1" in reasons_by_address
    assert any("array formula" in r for r in reasons_by_address["Edge Cases!B1"])

    assert "Edge Cases!B2" in reasons_by_address
    assert any("LET/LAMBDA" in r for r in reasons_by_address["Edge Cases!B2"])

    assert "Edge Cases!B3" in reasons_by_address
    assert any("structured table reference" in r for r in reasons_by_address["Edge Cases!B3"])

    # Parsing must not crash or stop: every cell, including the skipped ones,
    # is still captured with best-effort data.
    b1 = _cell(edge, "B1", "Edge Cases")
    assert b1.formula == "=SUM(C1:C3*D1:D3)"

    b4 = _cell(edge, "B4", "Edge Cases")
    assert b4.formula == '=A4&"!"'
    assert "Edge Cases!B4" not in reasons_by_address


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "not_a_workbook.csv"
    bogus.write_text("a,b,c\n1,2,3\n")

    try:
        parse_workbook(bogus)
    except ValueError as exc:
        assert ".csv" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported extension")


def test_to_dict_is_json_shaped(fixtures_dir: Path) -> None:
    parsed = parse_workbook(fixtures_dir / "clean_model.xlsx")
    data = parsed.to_dict()

    assert data["workbook"] == "clean_model.xlsx"
    assert isinstance(data["sheets"], list)
    assert data["sheets"][0]["name"] == "Model"
    assert isinstance(data["sheets"][0]["cells"], list)
    assert isinstance(data["named_ranges"], list)
    assert isinstance(data["skipped"], list)
