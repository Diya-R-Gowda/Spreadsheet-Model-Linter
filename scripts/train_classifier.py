"""Tier 1 classifier training CLI -- thin driver over `ssmlint.classifier`.

Trains microsoft/deberta-v3-small on the corpus's real labeled blocks and
saves the checkpoint to `models/tier1/` (gitignored, not committed).
Requires the "classifier" optional dependency (`pip install -e
".[classifier]"`) -- imports torch/transformers, unlike every other
script in this project.

    python scripts/train_classifier.py
    python scripts/train_classifier.py --epochs 3 --output-dir models/tier1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ssmlint.classifier import TrainingConfig, format_classification_report, train_classifier  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "tier1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="where to save the trained checkpoint")
    parser.add_argument("--epochs", type=float, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--json", action="store_true", help="also write the classification report as JSON")
    args = parser.parse_args()

    config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)

    print(f"Training {config.model_name} on {CORPUS_DIR} -> {args.output_dir}\n")
    result = train_classifier(CORPUS_DIR, Path(args.output_dir), config)

    print(result.readiness_report_text)
    print()
    print(f"Split sizes -- train: {len(result.train_entries)} entries, val: {len(result.val_entries)} entries, "
          f"test: {len(result.test_entries)} entries")
    print()
    print(format_classification_report(result.classification_report))
    print()
    print(f"Checkpoint saved to {result.output_dir}")

    if args.json:
        json_path = Path(args.output_dir) / "classification_report.json"
        json_path.write_text(json.dumps(result.classification_report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote classification report JSON to {json_path}")


if __name__ == "__main__":
    main()
