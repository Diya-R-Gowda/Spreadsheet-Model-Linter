"""CLI entry point for ssmlint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .parser import parse_workbook
from .report import build_report, render_html


def _cmd_dump(args: argparse.Namespace) -> int:
    parsed = parse_workbook(args.path)
    output = json.dumps(parsed.to_dict(), indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    report = build_report(args.path)

    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {args.json}")

    # Deliberate UX difference from `dump`: a full HTML document dumped to stdout isn't
    # useful, so `report` always writes an HTML file -- explicit path if given, otherwise
    # `<stem>.report.html` in the cwd -- rather than defaulting to stdout like `dump` does.
    html_path = args.html or f"{Path(args.path).stem}.report.html"
    Path(html_path).write_text(render_html(report), encoding="utf-8")
    print(f"Wrote HTML report to {html_path} ({len(report.issues)} issue(s) found)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssmlint",
        description="Static analysis for financial models in .xlsx files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_parser = subparsers.add_parser("dump", help="Parse a workbook and dump its structure as JSON")
    dump_parser.add_argument("path", help="Path to the .xlsx/.xlsm workbook")
    dump_parser.add_argument("-o", "--output", help="Write JSON to this file instead of stdout")
    dump_parser.set_defaults(func=_cmd_dump)

    report_parser = subparsers.add_parser(
        "report", help="Run Tier 0 checks against a workbook and write a ranked JSON + HTML report"
    )
    report_parser.add_argument("path", help="Path to the .xlsx/.xlsm workbook")
    report_parser.add_argument("--html", help="Write the HTML report to this path (default: <stem>.report.html)")
    report_parser.add_argument("--json", help="Also write the JSON report to this path")
    report_parser.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
