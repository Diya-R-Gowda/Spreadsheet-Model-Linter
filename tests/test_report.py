"""Tests for ssmlint.report -- the Week 6 HTML/JSON report builder.

Unit tests isolate ranking, heatmap building, and rendering against
hand-constructed Issue lists (no workbook, no I/O) before trusting the
full pipeline. Includes a deliberate XSS/escaping test since
`autoescape` is doing real work here, not just a formality, and an
integration test against the real `revenue_row_with_hardcode.xlsx`
fixture (already known-flagged per CONTRIBUTING.md: H14 high, B14
medium from LiteralInBlockRule).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ssmlint.report import (
    REPORT_CAVEAT,
    REPORT_CAVEAT_TIER1,
    build_heatmaps,
    build_report,
    rank_issues,
    render_html,
)
from ssmlint.rules import Issue


def _issue(cell: str, severity: str = "medium", rule_id: str = "literal-in-formula-block") -> Issue:
    return Issue(cell=cell, severity=severity, rule_id=rule_id, explanation=f"explains {cell}", suggested_fix="fix it")


# ---------------------------------------------------------------------------
# rank_issues: severity-desc + (rule_id, cell) deterministic tiebreak
# ---------------------------------------------------------------------------


def test_rank_issues_orders_high_before_medium() -> None:
    issues = [_issue("Model!B14", "medium"), _issue("Model!H14", "high")]
    ranked = rank_issues(issues)

    assert [r.cell for r in ranked] == ["Model!H14", "Model!B14"]
    assert [r.rank for r in ranked] == [1, 2]


def test_rank_issues_tiebreak_is_rule_id_then_cell() -> None:
    issues = [
        _issue("Model!D14", "high", rule_id="range-boundary-mismatch"),
        _issue("Model!C14", "high", rule_id="inconsistent-anchoring"),
        _issue("Model!B14", "high", rule_id="inconsistent-anchoring"),
    ]
    ranked = rank_issues(issues)

    # same severity -> sorted by (rule_id, cell): inconsistent-anchoring before range-boundary-mismatch,
    # and within inconsistent-anchoring, B14 before C14
    assert [(r.rule_id, r.cell) for r in ranked] == [
        ("inconsistent-anchoring", "Model!B14"),
        ("inconsistent-anchoring", "Model!C14"),
        ("range-boundary-mismatch", "Model!D14"),
    ]


def test_rank_issues_splits_sheet_and_bare_address() -> None:
    ranked = rank_issues([_issue("Model!H14", "high")])
    assert ranked[0].sheet == "Model"
    assert ranked[0].address == "H14"
    assert ranked[0].cell == "Model!H14"
    assert ranked[0].tier == "Tier 0"


# ---------------------------------------------------------------------------
# build_heatmaps
# ---------------------------------------------------------------------------


def test_heatmap_bounding_box_covers_only_flagged_cells() -> None:
    ranked = rank_issues([_issue("Model!C5", "medium"), _issue("Model!F9", "high")])
    heatmaps = build_heatmaps(ranked)

    assert len(heatmaps) == 1
    hm = heatmaps[0]
    assert hm.sheet == "Model"
    assert (hm.min_row, hm.max_row) == (5, 9)
    assert (hm.min_col, hm.max_col) == (3, 6)  # C=3, F=6


def test_heatmap_cell_hit_by_two_rules_takes_worse_severity_and_counts_both() -> None:
    ranked = rank_issues(
        [
            _issue("Model!H14", "medium", rule_id="inconsistent-anchoring"),
            _issue("Model!H14", "high", rule_id="literal-in-formula-block"),
        ]
    )
    heatmaps = build_heatmaps(ranked)
    cell = heatmaps[0].cells["H14"]

    assert cell.severity == "high"  # worse of medium/high
    assert cell.issue_count == 2
    assert set(cell.rule_ids) == {"inconsistent-anchoring", "literal-in-formula-block"}


def test_heatmaps_grouped_separately_per_sheet() -> None:
    ranked = rank_issues([_issue("Model!B14", "high"), _issue("Rolling!D5", "medium")])
    heatmaps = build_heatmaps(ranked)

    assert {h.sheet for h in heatmaps} == {"Model", "Rolling"}


# ---------------------------------------------------------------------------
# HTML rendering: structure + a deliberate XSS/escaping test
# ---------------------------------------------------------------------------


def _small_report(issues):
    from ssmlint.report import WorkbookReport

    ranked = rank_issues(issues)
    heatmaps = build_heatmaps(ranked)
    severity_counts: dict[str, int] = {}
    for r in ranked:
        severity_counts[r.severity] = severity_counts.get(r.severity, 0) + 1
    return WorkbookReport(
        workbook="fake.xlsx", generated_at="2026-01-01T00:00:00+00:00",
        issues=ranked, heatmaps=heatmaps, severity_counts=severity_counts,
    )


def test_rendered_html_contains_workbook_name_and_cell_addresses() -> None:
    report = _small_report([_issue("Model!H14", "high")])
    html = render_html(report)

    assert "fake.xlsx" in html
    assert "Model!H14" in html
    assert 'data-severity="high"' in html


def test_rendered_html_escapes_malicious_explanation_text() -> None:
    malicious = Issue(
        cell="Model!H14", severity="high", rule_id="literal-in-formula-block",
        explanation="<script>alert(1)</script>", suggested_fix="also <b>bold</b> nonsense",
    )
    report = _small_report([malicious])
    html = render_html(report)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<b>bold</b>" not in html


def test_rendered_html_shows_no_issues_message_when_empty() -> None:
    report = _small_report([])
    html = render_html(report)
    assert "No issues found." in html


def test_rendered_html_shows_suppressed_issues_section_when_present() -> None:
    from ssmlint.report import WorkbookReport

    ranked = rank_issues([_issue("Model!H14", "high")])
    suppressed = rank_issues([_issue("Model!B14", "medium")], tier="Tier 0 (suppressed by Tier 1)")
    report = WorkbookReport(
        workbook="fake.xlsx", generated_at="2026-01-01T00:00:00+00:00",
        issues=ranked, suppressed_issues=suppressed, heatmaps=build_heatmaps(ranked),
        severity_counts={"high": 1},
    )
    html = render_html(report)

    assert "Suppressed by Tier 1" in html
    assert "1 suppressed by Tier 1" in html
    assert "Model!B14" in html
    assert "Tier 0 (suppressed by Tier 1)" in html


def test_rendered_html_omits_suppressed_section_when_empty() -> None:
    report = _small_report([_issue("Model!H14", "high")])
    html = render_html(report)
    assert "Suppressed by Tier 1" not in html


# ---------------------------------------------------------------------------
# JSON output caveat -- independently asserted, not just the HTML's
# ---------------------------------------------------------------------------


def test_workbook_report_to_dict_includes_nonempty_caveat() -> None:
    report = _small_report([_issue("Model!H14", "high")])
    data = report.to_dict()

    assert "caveat" in data
    assert data["caveat"]  # non-empty
    assert data["caveat"] == REPORT_CAVEAT


# ---------------------------------------------------------------------------
# Integration: the real, already-known-flagged fixture through the full pipeline
# ---------------------------------------------------------------------------


def test_build_report_against_real_fixture(fixtures_dir: Path) -> None:
    report = build_report(fixtures_dir / "revenue_row_with_hardcode.xlsx")

    assert report.workbook == "revenue_row_with_hardcode.xlsx"
    assert [i.cell for i in report.issues] == ["Model!H14", "Model!B14"]  # high before medium
    assert report.issues[0].severity == "high"
    assert report.issues[1].severity == "medium"
    assert report.severity_counts == {"high": 1, "medium": 1}

    assert len(report.heatmaps) == 1
    hm = report.heatmaps[0]
    assert hm.sheet == "Model"
    assert "H14" in hm.cells and hm.cells["H14"].severity == "high"
    assert "B14" in hm.cells and hm.cells["B14"].severity == "medium"

    assert report.caveat == REPORT_CAVEAT


# ---------------------------------------------------------------------------
# tier1_checkpoint wiring -- mocked (no torch, no real checkpoint needed;
# apply_tier1_to_workbook itself is unit-tested against a real model in
# tests/test_classifier.py)
# ---------------------------------------------------------------------------


def test_tier1_checkpoint_none_preserves_existing_behavior_exactly(fixtures_dir: Path) -> None:
    """Omitting the parameter (every existing caller/test) must reproduce
    today's exact behavior -- a real regression guard on the new
    optional parameter's default.
    """
    report = build_report(fixtures_dir / "revenue_row_with_hardcode.xlsx")

    assert report.suppressed_issues == []
    assert report.issues[0].tier == "Tier 0"
    assert report.caveat == REPORT_CAVEAT


def test_tier1_checkpoint_given_moves_suppressed_issues_to_their_own_section(fixtures_dir: Path) -> None:
    fake_confidence_note = "intentional_override: 12 held-out test examples, 0.833 recall."

    with (
        patch(
            "ssmlint.classifier.apply_tier1_to_workbook",
            return_value=([Issue(cell="Model!H14", severity="high", rule_id="literal-in-formula-block",
                                  explanation="e", suggested_fix="f")],
                          [Issue(cell="Model!B14", severity="medium", rule_id="literal-in-formula-block",
                                 explanation="e", suggested_fix="f")]),
        ),
        patch("ssmlint.classifier.describe_intentional_override_confidence", return_value=fake_confidence_note),
    ):
        report = build_report(fixtures_dir / "revenue_row_with_hardcode.xlsx", tier1_checkpoint="fake/checkpoint")

    assert [i.cell for i in report.issues] == ["Model!H14"]
    assert report.issues[0].tier == "Tier 0 (Tier 1-reviewed)"
    assert [i.cell for i in report.suppressed_issues] == ["Model!B14"]
    assert report.suppressed_issues[0].tier == "Tier 0 (suppressed by Tier 1)"
    assert report.caveat == f"{REPORT_CAVEAT_TIER1} {fake_confidence_note}"
    assert "Model!B14" not in [h for hm in report.heatmaps for h in hm.cells]  # suppressed cell never on the heatmap


def test_tier1_checkpoint_bad_checkpoint_fails_loudly_not_silently(fixtures_dir: Path) -> None:
    """A user who explicitly asked for Tier 1 review must never silently
    get back an unmarked Tier-0-only report on a bad/missing checkpoint.
    """
    with patch("ssmlint.classifier.apply_tier1_to_workbook", side_effect=OSError("bad checkpoint path")):
        try:
            build_report(fixtures_dir / "revenue_row_with_hardcode.xlsx", tier1_checkpoint="bogus/checkpoint")
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "Tier 1 inference failed" in str(exc)
            assert "bogus/checkpoint" in str(exc)
