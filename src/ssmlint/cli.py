"""CLI entry point for ssmlint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .parser import parse_workbook


def _cmd_dump(args: argparse.Namespace) -> int:
    parsed = parse_workbook(args.path)
    output = json.dumps(parsed.to_dict(), indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
