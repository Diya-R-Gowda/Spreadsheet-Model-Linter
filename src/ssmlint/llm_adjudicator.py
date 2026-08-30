"""Project wrap-up: the Tier 2 Local LLM Adjudicator (README stage [8]) --
the last unbuilt stage of the original 9-stage architecture. A local 3B
model (Qwen2.5-3B-Instruct via Ollama) reviews Tier 1's surviving issues
one more time, using the SAME 4-way closed label set and the SAME compact
block-description text Tier 1 uses -- per the README's own design
principle ("the same block representation feeds Tier 1 and Tier 2
interchangeably"), never a pasted sheet, never an open-ended judgment.

NO NEW PIP DEPENDENCY (confirmed directly with the user before writing
this module): talks to Ollama's real HTTP API using only the Python
stdlib (`urllib.request`) -- matches this project's established
minimal-dependency ethos. `torch`/`transformers` from the `classifier`
extra are never imported here; this module works with the base,
Tier-0-only `ssmlint` install, as long as Ollama itself (a separate,
non-pip local server) is installed and running.

GRAMMAR-CONSTRAINED OUTPUT, VERIFIED LIVE (not assumed) BEFORE THIS WAS
WRITTEN
------------------------------------------------------------------------
Ollama's `format` request field accepts a JSON Schema, not just the
string `"json"`. Passing a schema whose `label` property is constrained
to `enum: [...]` was tested directly against the real, already-running
local server (Ollama 0.33.1, `qwen2.5:3b-instruct` already pulled) with
two real block descriptions before any of this module was written:
every response came back as a syntactically valid JSON object with
`label` inside the enum, never free text and never an invented cell
address -- satisfying the README's own Tier 2 requirements ("grammar-
constrained output", "never let a model emit addresses -- it selects
from a provided list") for real, not by construction alone.

REAL MEASURED LATENCY (README: "seconds/block -- use sparingly")
------------------------------------------------------------------------
Cold call (model not yet loaded into RAM): ~37.7s total (~24.2s one-time
load + ~9s generation + ~4s prompt eval). Warm call (model already
loaded, Ollama keeps it resident for a few idle minutes by default):
~10.3s total (~0.05s load + ~8.2s generation + ~1.9s prompt eval).
Confirms the README's own cost framing is accurate on this hardware
class, not aspirational.

WHY apply_tier1() IS REUSED HERE, NOT REIMPLEMENTED
------------------------------------------------------------------------
`classifier.apply_tier1(issues, blocks, predictions)` has zero ML-
specific logic -- it only ever checks whether each block's predicted
label string is `"subtotal"` or `"intentional_override"`. It is
completely agnostic to which judge produced that label string, so it is
reused verbatim below for Tier 2's own suppression step. `classifier.py`
needed zero code changes for this module to exist.

CHAINED-ONLY, NOT A STANDALONE "TIER 0 + 2" CONFIGURATION
------------------------------------------------------------------------
Confirmed directly with the user: the README's own ablation table only
ever shows "Tier 0", "Tier 0 + 1", "Tier 0 + 1 + 2" -- never "Tier 0 +
2". Callers here (`ablation.py`, `report.py`) enforce that Tier 2 always
runs on top of Tier 1's already-surviving issues, never as an
independent alternative to it.

CACHE IS IN-MEMORY ONLY, NOT PERSISTED TO DISK (a stated, deliberate
scope decision, not an oversight)
------------------------------------------------------------------------
`evaluate_tier0_plus_1_plus_2` mirrors Tier 1's own precedent of scoring
only a checkpoint's held-out test split (~21-24 blocks, not the full
144-entry corpus), so one full ablation run costs roughly 21 blocks x 2
tiers' worth of real LLM calls x ~10s warm each -- a few minutes, not
hours -- acceptable without persistence. A real single-workbook
`ssmlint report` run has even fewer blocks. Cross-run persistent caching
is a real, deferred future scope (same honesty standard as Tier 1's own
ablation row stating "runtime not measured this pass" as a named gap,
not a silent omission).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .blocks import SheetBlocks
    from .labeling import BlockExample
    from .rules import Issue

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_TIER2_MODEL = "qwen2.5:3b-instruct"

_TIER2_SYSTEM_PROMPT = (
    "You are a spreadsheet model auditor. You will be given a compact, structural description "
    "of one formula block from a real financial model -- never raw cell values, never a pasted "
    "sheet. Classify it into EXACTLY ONE of these four labels:\n"
    "- subtotal: the block IS a legitimate aggregate (e.g. a total/subtotal row), not a bug.\n"
    "- intentional_override: a deliberate seed value or manual override, not a bug.\n"
    "- suspected_error: a real hardcode or formula bug that breaks the block's own pattern.\n"
    "- unknown: genuinely ambiguous -- not enough evidence in the description either way.\n"
    "Never invent a cell address that is not already mentioned in the block description. "
    "Respond only via the provided JSON schema."
)


def _json_schema_for_labels() -> dict:
    """Built from `labeling.LABELS` directly, never a second hardcoded
    list -- the exact class of divergence bug the 2026-08-19 split-
    discrepancy fix addressed (two places deriving "the same" set of
    strings independently, silently drifting apart).
    """
    from .labeling import LABELS

    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(LABELS)},
            "reasoning": {"type": "string"},
        },
        "required": ["label", "reasoning"],
    }


def _post_chat(payload: dict, endpoint: str, timeout: float) -> dict:
    """The one real HTTP boundary in this module -- the sole mock point
    for every test in tests/test_llm_adjudicator.py. Deliberately stdlib-
    only (`urllib.request`), per the confirmed no-new-dependency decision.
    """
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Could not reach Ollama at {endpoint}: {exc}") from exc


def check_ollama_available(endpoint: str = DEFAULT_OLLAMA_ENDPOINT, model: str = DEFAULT_TIER2_MODEL) -> None:
    """Fails loudly with a real, actionable explanation -- server
    unreachable vs. model not pulled are genuinely different problems
    with different fixes, so they get different messages, matching this
    project's established "explain root cause, don't paper over it"
    convention (e.g. the CI corpus-reproducibility bug, the DeBERTa-vs-
    DistilBERT switch).
    """
    import urllib.error
    import urllib.request

    url = f"{endpoint.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10.0) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Tier 2 (local LLM adjudicator) is configured but Ollama is not reachable at "
            f"{endpoint}. Is `ollama serve` running? Original error: {exc}"
        ) from exc

    pulled_models = {m.get("name", "") for m in data.get("models", [])}
    # Ollama's /api/tags returns names with an explicit ":latest" tag suffix for models pulled
    # without one -- compare both the exact name and its bare (no-tag) form so a model pulled as
    # "qwen2.5:3b-instruct" matches whether or not the API happens to echo a trailing tag.
    bare_pulled = {name.split(":")[0] for name in pulled_models}
    if model not in pulled_models and model.split(":")[0] not in bare_pulled:
        raise RuntimeError(
            f"Tier 2 (local LLM adjudicator) is configured to use model '{model}', but it is not "
            f"in Ollama's pulled model list at {endpoint} ({sorted(pulled_models)}). "
            f"Run `ollama pull {model}` first."
        )


def query_ollama_label(
    prompt_text: str, model: str = DEFAULT_TIER2_MODEL, endpoint: str = DEFAULT_OLLAMA_ENDPOINT, timeout: float = 90.0
) -> str:
    """One real, grammar-constrained classification call for one block's
    already-serialized text (see `classifier.serialize_block_example` --
    reused verbatim as this function's input, never a second prompt-
    building template). Returns the predicted label string.
    """
    from .labeling import LABELS

    payload = {
        "model": model,
        "stream": False,
        "format": _json_schema_for_labels(),
        "messages": [
            {"role": "system", "content": _TIER2_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
    }
    response = _post_chat(payload, endpoint, timeout)
    content = response["message"]["content"]
    parsed = json.loads(content)
    label = parsed["label"]
    if label not in LABELS:  # defensive: belt-and-suspenders even though the schema already
        # constrains this -- a model could technically still emit something outside the enum if
        # a future Ollama version relaxes enforcement, and this should fail loudly, not silently
        # accept it.
        raise ValueError(f"Ollama returned an out-of-enum label {label!r} for model {model!r}")
    return label


def predict_labels_llm(
    examples: list[BlockExample],
    model: str = DEFAULT_TIER2_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    cache: dict[str, str] | None = None,
) -> list[str]:
    """Mirrors `classifier.predict_labels`'s exact contract
    (`list[BlockExample] -> list[str]`) so `classifier.apply_tier1` can
    be reused verbatim regardless of which tier produced the predictions.
    Cache key is a sha256 of `serialize_block_example(example)` -- the
    README's own "block signature", built from the one shared text
    representation both tiers already use, never a second ad-hoc scheme.
    """
    from .classifier import serialize_block_example

    if not examples:
        return []

    labels: list[str] = []
    for example in examples:
        text = serialize_block_example(example)
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache is not None and key in cache:
            labels.append(cache[key])
            continue
        label = query_ollama_label(text, model=model, endpoint=endpoint)
        if cache is not None:
            cache[key] = label
        labels.append(label)
    return labels


def apply_tier2_to_workbook(
    issues: list[Issue],
    sheet_blocks: list[SheetBlocks],
    model: str = DEFAULT_TIER2_MODEL,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    workbook_name: str = "workbook",
    cache: dict[str, str] | None = None,
) -> tuple[list[Issue], list[Issue]]:
    """Live Tier 2 inference for one real, already-parsed workbook --
    structurally mirrors `classifier.apply_tier1_to_workbook` exactly
    (same `guess_base_shape` heuristic, same `build_block_example` call),
    but predicts via the local LLM instead of the trained encoder, and
    reuses `classifier.apply_tier1` verbatim for suppression (confirmed
    label-agnostic -- see module docstring). Callers are expected to pass
    Tier 1's SURVIVING issues here, never Tier 0's raw issues directly --
    Tier 2 is a second opinion layered on top of Tier 1, not a
    replacement for it (the confirmed chained-only design).
    """
    from .classifier import apply_tier1
    from .labeling import build_block_example, guess_base_shape

    blocks = [block for sb in sheet_blocks for block in sb.blocks]
    block_examples = [
        build_block_example(workbook_name, block, guess_base_shape(block.pattern)) for block in blocks
    ]
    predictions = predict_labels_llm(block_examples, model=model, endpoint=endpoint, cache=cache)
    surviving = apply_tier1(issues, blocks, predictions)
    surviving_set = set(surviving)
    suppressed = [issue for issue in issues if issue not in surviving_set]
    return surviving, suppressed


def evaluate_tier0_plus_1_plus_2(
    corpus_dir: Path,
    checkpoint_dir: Path,
    tier2_model: str = DEFAULT_TIER2_MODEL,
    tier2_endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    rules: dict | None = None,
    cache: dict[str, str] | None = None,
):
    """Lives HERE, not `classifier.py` -- this function composes
    `classifier.py`'s Tier 1 internals AND this module's Tier 2 internals
    (this module depends on `classifier`, never the reverse -- no
    circular import). Mirrors `classifier.evaluate_tier0_plus_1`'s exact
    structure (same held-out `split.json` test entries, same
    `evaluation.py` scoring reuse), but per entry: Tier 0's real issues ->
    `apply_tier1` with Tier 1's own trained-classifier predictions ->
    `apply_tier1` AGAIN with Tier 2's local-LLM predictions over the same
    blocks -- two sequential suppression passes, since either judge could
    independently flag a block as `subtotal`/`intentional_override`.
    Returns `(rollups, test_entries)`, the identical shape
    `evaluate_tier0_plus_1` returns, so `ablation.py`'s existing
    `_aggregate_tier0_metrics` works completely unchanged.
    """
    from . import classifier
    from .blocks import detect_blocks
    from .depgraph import build_graph
    from .evaluation import (
        DEFAULT_RULES,
        PRECISION_AT_K,
        SEVERITIES,
        RuleRollup,
        ScoredIssue,
        SeveritySlice,
        count_false_negatives,
        evaluate_rule,
        score_rule_on_entry,
    )
    from .labeling import build_block_example, guess_base_shape
    from .parser import parse_workbook

    rules = rules if rules is not None else DEFAULT_RULES
    checkpoint_dir = Path(checkpoint_dir)
    test_entries = json.loads((checkpoint_dir / "split.json").read_text(encoding="utf-8"))["test"]
    model, tokenizer = classifier.load_classifier(checkpoint_dir)

    slices: dict[tuple[str, str], SeveritySlice] = {
        (rid, sev): SeveritySlice(rule_id=rid, severity=sev) for rid in rules for sev in SEVERITIES
    }
    fn_by_rule: dict[str, int] = {rid: 0 for rid in rules}
    all_scored_by_rule: dict[str, list[ScoredIssue]] = {rid: [] for rid in rules}

    for entry in sorted(test_entries):
        xlsx_path = corpus_dir / f"{entry}.xlsx"
        ground_truth = json.loads((corpus_dir / f"{entry}.ground_truth.json").read_text(encoding="utf-8"))
        parsed = parse_workbook(xlsx_path)
        graph = build_graph(parsed)
        sheet_blocks = detect_blocks(parsed, graph)

        all_issues = []
        for rule in rules.values():
            all_issues.extend(evaluate_rule(rule, sheet_blocks, graph))

        blocks = [block for sb in sheet_blocks for block in sb.blocks]

        # Tier 1: the real trained classifier, ground-truth base_shape (mirrors evaluate_tier0_plus_1
        # exactly -- corpus entries have real ground truth, unlike a live workbook's guessed shape).
        tier1_examples = [build_block_example(entry, block, ground_truth["base_shape"]) for block in blocks]
        tier1_predictions = classifier.predict_labels(model, tokenizer, tier1_examples)
        tier01_issues = classifier.apply_tier1(all_issues, blocks, tier1_predictions)

        # Tier 2: the local LLM, layered on top of Tier 1's survivors -- same blocks, same
        # base_shape (still ground truth here, for the same corpus-fairness reason as Tier 1
        # above; guess_base_shape is only needed for a real, non-synthetic workbook).
        tier2_examples = [build_block_example(entry, block, ground_truth["base_shape"]) for block in blocks]
        tier2_predictions = predict_labels_llm(tier2_examples, model=tier2_model, endpoint=tier2_endpoint, cache=cache)
        tier012_issues = classifier.apply_tier1(tier01_issues, blocks, tier2_predictions)

        for rule_id in rules:
            filtered_for_rule = [i for i in tier012_issues if i.rule_id == rule_id]
            scored = score_rule_on_entry(entry, filtered_for_rule, ground_truth, rule_id)
            all_scored_by_rule[rule_id].extend(scored)
            fn_by_rule[rule_id] += count_false_negatives(filtered_for_rule, ground_truth, rule_id)

            for s in scored:
                key = (rule_id, s.severity)
                if s.outcome == "true_positive":
                    slices[key].tp += 1
                elif s.outcome == "known_intentional":
                    slices[key].known_intentional_excluded += 1
                elif s.outcome == "ambiguous_excluded":
                    slices[key].ambiguous_excluded += 1
                else:
                    slices[key].fp += 1

    rollups: list[RuleRollup] = []
    for rule_id in rules:
        tp = sum(slices[(rule_id, sev)].tp for sev in SEVERITIES)
        fp = sum(slices[(rule_id, sev)].fp for sev in SEVERITIES)
        known = sum(slices[(rule_id, sev)].known_intentional_excluded for sev in SEVERITIES)
        ambiguous = sum(slices[(rule_id, sev)].ambiguous_excluded for sev in SEVERITIES)
        fn = fn_by_rule[rule_id]

        severity_rank = {"high": 0, "medium": 1}
        scoreable = [s for s in all_scored_by_rule[rule_id] if s.outcome in ("true_positive", "false_positive")]
        pool = sorted(scoreable, key=lambda s: (severity_rank.get(s.severity, 99), s.entry, s.cell))
        top_k = pool[:PRECISION_AT_K]
        k = len(top_k)
        top_k_tp = sum(1 for s in top_k if s.outcome == "true_positive")
        precision_at_10 = top_k_tp / k if k else None

        rollups.append(
            RuleRollup(
                rule_id=rule_id, tp=tp, fp=fp, fn=fn, known_intentional_excluded=known,
                ambiguous_excluded=ambiguous, precision_at_10=precision_at_10, precision_at_10_k=k,
            )
        )

    return rollups, test_entries
