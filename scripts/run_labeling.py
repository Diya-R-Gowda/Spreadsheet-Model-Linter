"""Week 5 training-readiness CLI — thin driver over `ssmlint.labeling`.

Runs label generation + a deterministic train/val/test split against
`corpus/` and prints the readiness report. This is NOT a training run --
no ML dependency is imported anywhere in this path. Pass `--json` to
print the same report as JSON instead of the human-readable table.

    python scripts/run_labeling.py
    python scripts/run_labeling.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ssmlint.labeling import (  # noqa: E402
    check_training_readiness,
    format_readiness_report,
    generate_labeled_examples,
    split_corpus_entries,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the report as JSON instead of a text table")
    args = parser.parse_args()

    examples = generate_labeled_examples(CORPUS_DIR)
    entry_names = sorted({e.entry for e in examples})
    split = split_corpus_entries(entry_names)
    report = check_training_readiness(examples, split)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_readiness_report(report))


if __name__ == "__main__":
    main()
