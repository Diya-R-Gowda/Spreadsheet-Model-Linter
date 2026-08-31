"""Week 6 ablation table CLI — thin driver over `ssmlint.ablation`.

Runs the Tier 0 rule engine against `corpus/` (built by
`scripts/generate_corpus.py`; run that first if `corpus/` is empty),
aggregates precision/recall/precision@10 across all four rules into one
Tier-0 row, times the full pipeline per corpus entry, and prints the
README's ablation table shape. Pass --tier1-checkpoint pointing at a
real checkpoint from scripts/train_classifier.py to also fill in a real,
measured Tier 0+1 row (scored on that checkpoint's own held-out test
split, compared against Tier 0 re-scored on the same entries). Also pass
--tier2-model (a real local Ollama model, e.g. "qwen2.5:3b-instruct") to
additionally fill in a real, measured Tier 0+1+2 row -- requires
--tier1-checkpoint too, since Tier 2 is a second opinion layered on top
of Tier 1, never a standalone alternative (see `ablation.py`'s module
docstring). Without --tier1-checkpoint, both rows stay structurally
present but not yet available.

    python scripts/run_ablation.py
    python scripts/run_ablation.py --json
    python scripts/run_ablation.py --tier1-checkpoint models/tier1
    python scripts/run_ablation.py --tier1-checkpoint models/tier1 --tier2-model qwen2.5:3b-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ssmlint.ablation import build_ablation_table, format_ablation_table_text  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the report as JSON instead of a text table")
    parser.add_argument(
        "--tier1-checkpoint", default=None, help="path to a real checkpoint from scripts/train_classifier.py"
    )
    parser.add_argument(
        "--tier2-model",
        default=None,
        help="a real local Ollama model (e.g. 'qwen2.5:3b-instruct') -- requires --tier1-checkpoint too",
    )
    parser.add_argument("--tier2-endpoint", default=None, help="Ollama server URL (default: http://localhost:11434)")
    args = parser.parse_args()

    tier1_checkpoint_dir = Path(args.tier1_checkpoint) if args.tier1_checkpoint else None
    table = build_ablation_table(
        CORPUS_DIR,
        tier1_checkpoint_dir=tier1_checkpoint_dir,
        tier2_model=args.tier2_model,
        tier2_endpoint=args.tier2_endpoint,
    )
    if args.json:
        print(json.dumps(table.to_dict(), indent=2))
    else:
        print(format_ablation_table_text(table))


if __name__ == "__main__":
    main()
