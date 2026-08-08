from __future__ import annotations

import json
from pathlib import Path

from ssmlint.cli import main


def test_dump_to_stdout(fixtures_dir: Path, capsys) -> None:
    exit_code = main(["dump", str(fixtures_dir / "clean_model.xlsx")])
    assert exit_code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert data["workbook"] == "clean_model.xlsx"
    sheet_names = [s["name"] for s in data["sheets"]]
    assert "Model" in sheet_names


def test_dump_to_file(fixtures_dir: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "out.json"
    exit_code = main(["dump", str(fixtures_dir / "merged_cells.xlsx"), "-o", str(out_path)])
    assert exit_code == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["workbook"] == "merged_cells.xlsx"

    budget = next(s for s in data["sheets"] if s["name"] == "Budget")
    a1 = next(c for c in budget["cells"] if c["address"] == "Budget!A1")
    assert a1["merged"] is True
    assert a1["value"] == "FY2024 Budget"


def test_dump_includes_skipped_and_named_ranges(fixtures_dir: Path, capsys) -> None:
    exit_code = main(["dump", str(fixtures_dir / "named_range_cross_sheet.xlsx")])
    assert exit_code == 0

    data = json.loads(capsys.readouterr().out)
    named_range_names = {n["name"] for n in data["named_ranges"]}
    assert {"TaxRate", "LocalNet"} <= named_range_names
    assert data["skipped"] == []
