# Spreadsheet Model Linter

**Static analysis for financial models. Finds errors in .xlsx files the way a linter finds bugs in code.**

---

## The Problem

Financial models run companies, and they are catastrophically error-prone. Studies of production spreadsheets consistently find errors in 80–90% of them. Real-world consequences have included multi-billion-dollar trading losses and retracted academic economics papers.

Nobody lints spreadsheets. The tooling that exists is either enterprise-priced audit software or trivial "find hardcoded cells" macros. Meanwhile every LLM spreadsheet project built in the last two years is "chat with your CSV" — a natural language query layer, which is a solved and commoditized problem.

This project inverts it: the LLM never answers questions about the data. It reconstructs *intent* from structure, and flags cells that violate that intent.

## The Core Insight

A spreadsheet has an implicit contract that is never written down. If row 14 is "Revenue" and columns C through N are months, then `C14:N14` should all contain structurally identical formulas. If `H14` is a hardcoded `4500000` while its neighbors are `=G14*(1+G15)`, that is a bug — someone plugged a number to make a total tie out and never removed it.

You cannot detect this with rules alone, because the definition of "structurally identical" depends on what the block *means*. `=SUM(...)` breaking a pattern is fine in a subtotal row and wrong in a driver row. That semantic judgment is where the LLM earns its place — and only there.

---

## Scope

**In scope**
- `.xlsx` / `.xlsm` input, single workbook, multiple sheets
- Detection of: hardcoded values inside formula ranges, inconsistent formulas across a row/column block, off-by-one range errors, references to empty cells, circular-ish logic, sign convention breaks, unit/scale mismatches (thousands vs millions)
- Output: a ranked issue report (JSON + HTML) with cell address, severity, explanation, and suggested fix

**Explicitly out of scope**
- Editing or repairing the workbook (report only — do not touch someone's model)
- Google Sheets, .xls (legacy binary), CSV
- Charts, pivot tables, macros, external workbook links
- Anything requiring the workbook to be recalculated

Keeping repair out of scope is not laziness. It halves the trust burden and removes an entire class of destructive failure.

---

## Architecture

```
.xlsx file
   │
   ▼
[1] Workbook Parser ─────────► cells, formulas, formats, named ranges
   │                            (openpyxl, data_only=False)
   ▼
[2] Formula Tokenizer ───────► AST per formula
   │                            (functions, refs, literals, operators)
   ▼
[3] Dependency Graph ────────► DAG of cell → precedents/dependents
   │                            (networkx)
   ▼
[4] Block Detector ──────────► contiguous regions with shared structure
   │                            (formula R1C1 normalization + clustering)
   ▼
[5] Rule Engine ─────────────► deterministic checks, high precision
   │                            (no LLM — catches ~60% of issues)
   ▼
[6] LLM Semantic Layer ──────► block intent + anomaly judgment
   │                            (only on blocks the rule engine flags
   │                             as ambiguous, plus a sampled sweep)
   ▼
[7] Report Builder ──────────► ranked JSON + standalone HTML
```

### The critical design decision

**The LLM sees normalized structure, never raw cell dumps.** Do not paste a sheet into a prompt. Instead, send a compact block description:

```json
{
  "block": "Sheet1!C14:N14",
  "row_label": "Revenue - Enterprise",
  "col_labels": ["Jan-24", "Feb-24", ...],
  "formula_pattern": "=R[0]C[-1]*(1+R[1]C[0])",
  "conforming": 11,
  "deviations": [
    {"cell": "H14", "kind": "literal", "value": 4500000},
    {"cell": "N14", "formula_r1c1": "=SUM(R[0]C[-12]:R[0]C[-1])"}
  ]
}
```

This makes the token cost independent of workbook size, keeps the model focused on judgment rather than extraction, and makes results reproducible.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Parsing | `openpyxl` | Only mature lib exposing raw formulas |
| Formula AST | `formulas` or hand-rolled tokenizer | `formulas` is fuller; hand-rolling is more predictable |
| Graph | `networkx` | Cycle detection, topological ordering |
| Clustering | R1C1 string equality + `rapidfuzz` | Exact match covers most; fuzzy catches near-misses |
| LLM | Any frontier model, temperature 0 | Judgment layer only |
| Report | Jinja2 → single-file HTML | Must open without a server |
| Tests | `pytest` + fixture workbooks | Build a corrupted-workbook corpus |

---

## Data

You need real models, not toy files.

1. **SEC EDGAR full-text search** — filter for `.xlsx` exhibits. Public company filings sometimes include supporting models.
2. **University finance course materials** — LBO/DCF templates are widely posted and realistically messy.
3. **Kaggle / GitHub** — search `filetype:xlsx` for budget, forecast, and model templates.
4. **Synthetic corruption (your evaluation backbone)** — take 50 clean models, programmatically inject known bugs (replace a formula with its computed value, shift a range by one, flip a sign). You now have perfect ground truth.

The synthetic corpus is what makes this project rigorous. Everything else is anecdote.

---

## Build Plan

**Week 1 — Parsing spine.** Load workbooks, extract every cell's formula, value, and format. Handle merged cells, named ranges, and cross-sheet references. Ship a CLI that dumps a workbook to JSON. *Expect this to be harder than it sounds.*

**Week 2 — Dependency graph + R1C1 normalization.** Convert A1 formulas to relative R1C1 so `=B4*C4` in row 4 and `=B5*C5` in row 5 become the same string. This single transformation is the heart of the project.

**Week 3 — Block detection + rule engine.** Cluster contiguous cells by normalized formula. Implement deterministic rules: literal-in-formula-block, range boundary mismatch, reference to blank, inconsistent anchoring (`$`).

**Week 4 — LLM semantic layer.** Prompt design, block summarization, intent inference. Add severity scoring. Cache aggressively — same block, same verdict.

**Week 5 — Evaluation harness.** Run against the synthetic corpus. Compute precision/recall per bug class. This week will tell you your rule engine is over-firing; fix it.

**Week 6 — Report UI + polish.** HTML report with a sheet heatmap, click-through to cells, and severity filtering. Write up findings.

---

## Evaluation

Track these, per bug class, on the synthetic corpus:

- **Precision** — of flagged cells, what fraction are injected bugs? *Target > 0.85.* This matters far more than recall. An auditor who gets 40 false alarms stops using the tool permanently.
- **Recall** — of injected bugs, what fraction were caught? *Target > 0.70.*
- **Ranking quality** — is the true bug in the top 10 flags? Report precision@10.
- **Cost and latency per workbook** — should be under a few cents and under 30 seconds.

Also run against clean, uncorrupted models and count flags. Every flag there is a false positive on a real file. This number is your credibility.

---

## Failure Modes

| Risk | Reality | Mitigation |
|---|---|---|
| Formula parsing eats the schedule | Excel's grammar has array formulas, structured table refs, `LET`/`LAMBDA`, locale-dependent separators | Whitelist a formula subset in v1; skip and log what you can't parse |
| False positive flood | Real models legitimately break patterns constantly | Rule engine must be conservative; require LLM confirmation before surfacing |
| LLM hallucinating cell addresses | It will invent `H15` when it means `H14` | Never let the model emit addresses — it selects from a provided list |
| Huge workbooks | 50-sheet, 500k-cell models exist | Stream sheet by sheet; cap LLM calls per workbook |
| "Is this even a bug?" | Sometimes a hardcode is intentional | Add a severity tier for "intentional-looking overrides" rather than calling everything an error |

---

## Stretch

- VS Code / Excel add-in surface
- Diff mode: what changed between two versions of a model, semantically
- Learn a firm's house conventions from a corpus of their clean models
- Extend to Google Sheets via the Sheets API

## Reading

- Panko, "What We Know About Spreadsheet Errors" — the foundational error-rate research
- EuSpRIG (European Spreadsheet Risks Interest Group) conference archive — the only community that takes this seriously
- `openpyxl` formula parsing docs, and the `formulas` library source
