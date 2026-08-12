"""Week 4 baseline evaluation CLI — thin driver over `ssmlint.evaluation`.

Runs the Tier 0 rule engine against `corpus/` (built by
`scripts/generate_corpus.py`; run that first if `corpus/` is empty) and
prints the scored precision/recall/precision@10 report. Pass `--json` to
print the same report as JSON instead of the human-readable table.

    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ssmlint.evaluation import format_report_text, run_evaluation  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the report as JSON instead of a text table")
    args = parser.parse_args()

    report = run_evaluation(CORPUS_DIR)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report_text(report))


if __name__ == "__main__":
    main()
