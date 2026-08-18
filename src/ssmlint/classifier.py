"""Week 5, for real: the Tier 1 trained classifier.

Fine-tunes a small CPU-only encoder on the synthetic-corruption labels
`labeling.py` already produces, to predict the 4-way block-level label
(`subtotal | intentional_override | suspected_error | unknown`) the
README specifies. This is the first module in the project that imports
`torch`/`transformers` -- both are an optional dependency
(`pip install -e ".[classifier]"`), never required for the base Tier-0-
only `ssmlint` install, per the README's own "Tier 1/2 are optional
layers" framing.

MODEL CHOICE: distilbert-base-uncased, not DeBERTa-v3-small (changed
2026-08-18, mid-implementation, from what the approved plan originally
named)
------------------------------------------------------------------------
The plan's original reasoning -- "DeBERTa-v3-small is safer/more
compatible for CPU-only fine-tuning" -- was wrong in practice, caught by
actually running it, not assumed correct from the plan. Real measurement:
DeBERTa-v3-small never completed a single forward+backward step in over
15 minutes of wall-clock time on this machine (confirmed still actively
consuming real CPU the whole time, not hung -- process CPU-time was
tracked directly). DeBERTa-v2/v3's disentangled-attention architecture is
known to perform very poorly in plain CPU eager-mode PyTorch (the
relative-position-bucket gather operations don't vectorize well without
specialized kernels). Isolated the cause by running the identical loop
with `distilbert-base-uncased` (a comparable 66M-param size class):
~1.1s/step, real and measured. Confirmed directly with the user before
switching -- this is now the real, working default.

WHAT "TIER 0 + 1" ACTUALLY MEANS HERE
------------------------------------------------------------------------
The classifier operates on BLOCKS (subtotal / intentional_override /
suspected_error / unknown), while Tier 0's rules flag individual CELLS.
Tier 1 is not an independent detector -- it's a judgment layer over
Tier 0's own candidates, matching the README's framing ("that judgment
call is where a semantic layer... earns its place, on top of a rule
engine that already does most of the work deterministically"):
`apply_tier1` takes Tier 0's real flagged Issues plus a predicted label
per block, and SUPPRESSES issues on any block predicted `subtotal` or
`intentional_override` (Tier 1 overriding a Tier 0 false alarm) while
leaving `suspected_error`/`unknown` predictions untouched. Scoring the
resulting (possibly smaller) issue set the same way `evaluation.py`
already scores Tier 0's is what produces a real, comparable Tier 0+1
precision/recall number for `ablation.py`.

DATA READINESS (confirmed directly with the user before training,
2026-08-17)
------------------------------------------------------------------------
`check_training_readiness()` still reports NOT READY: `subtotal`(60) and
`suspected_error`(67) clear the 50-per-class floor, `intentional_
override`(5) and `unknown`(20) don't. Training proceeds anyway, on all
four labels, with real per-class results reported honestly -- including
if the model ends up unreliable on the two thin classes -- rather than
narrowing scope or deferring further. `train_classifier` prints the
pre-training readiness report for visibility; it is NOT a hard gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .blocks import Block
from .labeling import (
    LABELS,
    BlockExample,
    candidate_cells,
    check_training_readiness,
    format_readiness_report,
    generate_labeled_examples,
    split_corpus_entries,
)
from .rules import Issue

LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = {i: label for label, i in LABEL_TO_ID.items()}

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 256

# torch/transformers are imported LAZILY, inside the functions below that actually need
# them -- never at module level. serialize_block_example/apply_tier1/LABEL_TO_ID above
# must stay usable without the "classifier" optional dependency installed at all, since
# Tier 1 is explicitly optional (README) and the base Tier-0-only `ssmlint` install
# shouldn't require a ~1GB ML dependency it doesn't need.


def serialize_block_example(example: BlockExample) -> str:
    """Deterministic text template turning a `BlockExample` into the
    string the tokenizer sees. Only fields `BlockExample` actually
    carries -- no cell values (never available, per Week 5's confirmed
    input-shape gap) and no `row_label`/`col_labels` (always `None`
    today, same gap) -- the model sees exactly what the README's compact
    block description promises, nothing fabricated to make the input
    look richer than the pipeline can actually produce.
    """
    lines = [
        f"Block: {example.block_address}",
        f"Base shape: {example.base_shape}",
        f"Formula pattern: {example.formula_pattern}",
        f"Conforming cells: {example.conforming_count}",
    ]
    if example.deviations:
        lines.append("Deviations:")
        for d in example.deviations:
            lines.append(f"  - {d.cell} ({d.kind}, {d.side})")
    else:
        lines.append("Deviations: none")
    return "\n".join(lines)


def apply_tier1(issues: list[Issue], blocks: list[Block], predictions: list[str]) -> list[Issue]:
    """Filters Tier 0's real Issues using Tier 1's predicted label per
    block. `blocks[i]`'s predicted label is `predictions[i]`.

    - `subtotal` / `intentional_override`: every issue whose cell is in
      that block's `candidate_cells` (same cell-scope `label_for_block`
      itself uses -- the cells that would have earned the block that
      label in the first place) is suppressed.
    - `suspected_error` / `unknown`: issues are left untouched. No new
      severity/confidence scheme is invented for `unknown` in this pass
      (a stated scoping choice -- see the classifier-plan roadmap note
      in CONTRIBUTING.md, not an oversight).

    A cell not covered by any block's candidate_cells (e.g. an isolated
    flagged cell with no block context) is never touched -- suppression
    only ever applies to cells Tier 1 actually had an opinion about.
    """
    if len(blocks) != len(predictions):
        raise ValueError(f"blocks and predictions must be the same length, got {len(blocks)} and {len(predictions)}")

    suppressed_cells: set[str] = set()
    for block, predicted_label in zip(blocks, predictions):
        if predicted_label in ("subtotal", "intentional_override"):
            suppressed_cells.update(candidate_cells(block))

    return [issue for issue in issues if issue.cell not in suppressed_cells]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    epochs: float = 5.0
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = DEFAULT_MAX_LENGTH
    seed: int = 42


@dataclass(frozen=True)
class LabelMetrics:
    label: str
    support: int  # number of real test examples with this true label
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def to_dict(self) -> dict:
        return {
            "label": self.label, "support": self.support, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
        }


@dataclass(frozen=True)
class ClassificationReport:
    per_label: list[LabelMetrics]
    confusion_matrix: dict[str, dict[str, int]]  # confusion_matrix[true_label][predicted_label] = count
    accuracy: float
    test_size: int

    def to_dict(self) -> dict:
        return {
            "per_label": [m.to_dict() for m in self.per_label],
            "confusion_matrix": self.confusion_matrix,
            "accuracy": self.accuracy,
            "test_size": self.test_size,
        }


def _compute_classification_report(y_true: list[str], y_pred: list[str]) -> ClassificationReport:
    """Manual precision/recall/F1-per-label + confusion matrix -- no
    scikit-learn dependency (not part of the approved dependency list for
    this pass; the math is simple enough to not need it).
    """
    confusion: dict[str, dict[str, int]] = {t: dict.fromkeys(LABELS, 0) for t in LABELS}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    per_label: list[LabelMetrics] = []
    for label in LABELS:
        support = sum(confusion[label].values())
        tp = confusion[label][label]
        fp = sum(confusion[t][label] for t in LABELS if t != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        per_label.append(LabelMetrics(label=label, support=support, tp=tp, fp=fp, fn=fn))

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    return ClassificationReport(per_label=per_label, confusion_matrix=confusion, accuracy=accuracy, test_size=len(y_true))


def format_classification_report(report: ClassificationReport) -> str:
    lines = [f"=== Tier 1 Classifier: held-out test-split classification report ({report.test_size} examples) ==="]
    lines.append(f"Overall accuracy: {report.accuracy:.3f}")
    lines.append("")
    header = f"{'label':<22}{'support':>8}{'precision':>12}{'recall':>10}{'f1':>10}"
    lines.append(header)
    for m in report.per_label:
        prec = f"{m.precision:.3f}" if m.precision is not None else "n/a"
        rec = f"{m.recall:.3f}" if m.recall is not None else "n/a"
        f1 = f"{m.f1:.3f}" if m.f1 is not None else "n/a"
        lines.append(f"{m.label:<22}{m.support:>8}{prec:>12}{rec:>10}{f1:>10}")
    lines.append("")
    lines.append("Confusion matrix (rows = true label, columns = predicted label):")
    col_header = "".join(f"{label[:10]:>12}" for label in LABELS)
    lines.append(f"{'':<22}{col_header}")
    for true_label in LABELS:
        row = "".join(f"{report.confusion_matrix[true_label][pred_label]:>12}" for pred_label in LABELS)
        lines.append(f"{true_label:<22}{row}")
    return "\n".join(lines)


@dataclass(frozen=True)
class TrainingResult:
    config: TrainingConfig
    readiness_report_text: str
    classification_report: ClassificationReport
    train_entries: list[str]
    val_entries: list[str]
    test_entries: list[str]
    output_dir: Path


def _serialize_examples(examples: list[BlockExample]) -> tuple[list[str], list[str]]:
    texts = [serialize_block_example(e) for e in examples]
    labels = [e.label for e in examples]
    return texts, labels


def train_classifier(corpus_dir: Path, output_dir: Path, config: TrainingConfig | None = None) -> TrainingResult:
    """Fine-tunes `config.model_name` on the corpus's real labeled
    blocks. Prints nothing itself (see scripts/train_classifier.py for
    the CLI that does) -- returns everything a caller needs to print or
    inspect. `check_training_readiness()`'s report is computed and
    carried on the result for visibility; it is NOT a hard gate (per the
    confirmed data-readiness decision -- training proceeds on all four
    labels regardless of which ones clear the 50-per-class floor).
    """
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    config = config or TrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_examples = generate_labeled_examples(corpus_dir)
    labeled_examples = [e for e in all_examples if e.label is not None]
    entry_names = sorted({e.entry for e in labeled_examples})
    split = split_corpus_entries(entry_names, seed=config.seed)
    readiness_report_text = format_readiness_report(check_training_readiness(labeled_examples, split))

    train_examples = [e for e in labeled_examples if e.entry in set(split["train"])]
    val_examples = [e for e in labeled_examples if e.entry in set(split["val"])]
    test_examples = [e for e in labeled_examples if e.entry in set(split["test"])]
    if not train_examples or not test_examples:
        raise ValueError(
            f"train/test split produced an empty split (train={len(train_examples)}, "
            f"test={len(test_examples)}) -- corpus_dir may be missing entries; regenerate the corpus first"
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    class _TextDataset(Dataset):
        def __init__(self, examples: list[BlockExample]):
            texts, labels = _serialize_examples(examples)
            self._encodings = tokenizer(texts, truncation=True, max_length=config.max_length, padding=True)
            self._label_ids = [LABEL_TO_ID[label] for label in labels]

        def __len__(self) -> int:
            return len(self._label_ids)

        def __getitem__(self, idx: int) -> dict:
            item = {k: torch.tensor(v[idx]) for k, v in self._encodings.items()}
            item["labels"] = torch.tensor(self._label_ids[idx])
            return item

    train_dataset = _TextDataset(train_examples)
    val_dataset = _TextDataset(val_examples) if val_examples else _TextDataset(train_examples)

    # Class-weighted cross-entropy: a deliberate mitigation for the real label imbalance
    # (subtotal/suspected_error in the 60s, intentional_override/unknown at 5/20) -- inverse
    # label frequency on the TRAIN split only, standard "balanced" formula. Stated explicitly:
    # this makes the loss not ignore the thin classes during training, it does not manufacture
    # more real examples of them -- the held-out test report below still reports the real,
    # possibly weak, resulting quality on those classes honestly.
    train_label_counts = Counter(e.label for e in train_examples)
    n_train = len(train_examples)
    n_labels = len(LABELS)
    class_weights = torch.tensor(
        [n_train / (n_labels * train_label_counts.get(label, 1)) for label in LABELS], dtype=torch.float
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=n_labels, id2label=ID_TO_LABEL, label2id=LABEL_TO_ID
    )

    class _WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # dtype cast is deliberate, not incidental: some model/precision configs load
            # logits in a lower precision (e.g. fp16) than this float32 weights tensor, and
            # CrossEntropyLoss requires both operands to match exactly.
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(device=logits.device, dtype=logits.dtype))
            loss = loss_fct(logits.view(-1, n_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(output_dir / "_trainer_state"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=config.seed,
        use_cpu=True,
    )

    trainer = _WeightedTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    test_texts, test_true_labels = _serialize_examples(test_examples)
    test_pred_labels = predict_labels(model, tokenizer, test_examples, max_length=config.max_length)
    classification_report = _compute_classification_report(test_true_labels, test_pred_labels)

    return TrainingResult(
        config=config,
        readiness_report_text=readiness_report_text,
        classification_report=classification_report,
        train_entries=split["train"],
        val_entries=split["val"],
        test_entries=split["test"],
        output_dir=output_dir,
    )


def load_classifier(checkpoint_dir: Path):
    """Loads a previously-trained checkpoint. Returns (model, tokenizer)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()
    return model, tokenizer


def predict_labels(model, tokenizer, examples: list[BlockExample], max_length: int = DEFAULT_MAX_LENGTH) -> list[str]:
    """Batched, deterministic inference (`model.eval()`, no sampling) --
    caller's responsibility to have called `model.eval()` already if
    reusing a model across many calls (train_classifier's own internal
    call does this via the freshly-trained `model` object; a model loaded
    via `load_classifier` is already in eval mode).
    """
    import torch

    if not examples:
        return []

    texts = [serialize_block_example(e) for e in examples]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            logits = model(**encodings).logits
            predicted_ids = torch.argmax(logits, dim=-1).tolist()
    finally:
        if was_training:
            model.train()

    return [ID_TO_LABEL[i] for i in predicted_ids]
