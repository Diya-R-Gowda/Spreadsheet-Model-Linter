from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from ssmlint.cli import main
from ssmlint.rules import Issue


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


def test_report_writes_html_and_json(fixtures_dir: Path, tmp_path: Path) -> None:
    html_path = tmp_path / "out.report.html"
    json_path = tmp_path / "out.report.json"
    exit_code = main(
        [
            "report",
            str(fixtures_dir / "revenue_row_with_hardcode.xlsx"),
            "--html",
            str(html_path),
            "--json",
            str(json_path),
        ]
    )
    assert exit_code == 0

    assert html_path.exists()
    assert "Model!H14" in html_path.read_text(encoding="utf-8")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["workbook"] == "revenue_row_with_hardcode.xlsx"
    assert {i["cell"] for i in data["issues"]} == {"Model!H14", "Model!B14"}


def test_report_defaults_html_path_to_workbook_stem(fixtures_dir: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = main(["report", str(fixtures_dir / "revenue_row_with_hardcode.xlsx")])
    assert exit_code == 0

    default_path = tmp_path / "revenue_row_with_hardcode.report.html"
    assert default_path.exists()


def test_report_tier1_checkpoint_flag_reports_suppressed_count(fixtures_dir: Path, tmp_path: Path, capsys) -> None:
    html_path = tmp_path / "out.report.html"
    json_path = tmp_path / "out.report.json"
    fake_surviving = [Issue(cell="Model!H14", severity="high", rule_id="literal-in-formula-block",
                             explanation="e", suggested_fix="f")]
    fake_suppressed = [Issue(cell="Model!B14", severity="medium", rule_id="literal-in-formula-block",
                              explanation="e", suggested_fix="f")]

    with (
        patch("ssmlint.classifier.apply_tier1_to_workbook", return_value=(fake_surviving, fake_suppressed)),
        patch("ssmlint.classifier.describe_intentional_override_confidence", return_value="note"),
    ):
        exit_code = main(
            [
                "report",
                str(fixtures_dir / "revenue_row_with_hardcode.xlsx"),
                "--html", str(html_path),
                "--json", str(json_path),
                "--tier1-checkpoint", "fake/checkpoint",
            ]
        )
    assert exit_code == 0

    stdout = capsys.readouterr().out
    assert "1 suppressed by Tier 1" in stdout

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["suppressed_issues"]
    assert data["suppressed_issues"][0]["cell"] == "Model!B14"
