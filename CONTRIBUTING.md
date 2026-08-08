# Contributing to Spreadsheet Model Linter

This file tracks what's actually built versus what's still on the plan, week by week, plus the architecture and tech-stack decisions a contributor needs to know before touching the code. If you're picking this project back up after a break, start here, not the README — the README sells the idea, this file tells you where the code actually stands.

**Legend:** 🟢 done and tested · 🔴 not started

---

## Dev setup

```bash
# from the repo root
python -m venv .venv
./.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -e ".[dev]"

pytest                          # regenerates synthetic fixture workbooks automatically
ssmlint dump path/to/workbook.xlsx
```

Project layout:
```
src/ssmlint/
  parser.py          # [1] workbook parsing spine
  cli.py              # `ssmlint dump` entry point
  tokenizer.py         # [2] formula lexer
  formula_parser.py    # [2] formula AST builder
  ast_nodes.py          # [2] AST node dataclasses
  errors.py              # shared FormulaError hierarchy
scripts/generate_fixtures.py   # builds synthetic .xlsx test fixtures
tests/                          # pytest suite; fixtures regenerated per session, not committed
```

---

## Architecture

Nine-stage pipeline, each stage consuming the previous stage's output:

```
.xlsx file
   │
   ▼
[1] Workbook Parser ─────────► cells, formulas, formats, named ranges       🟢
   │                            (openpyxl, data_only=False)
   ▼
[2] Formula Tokenizer/AST ───► hand-rolled lexer + recursive-descent parser  🟢
   │                            (functions, refs, literals, operators)
   ▼
[3] R1C1 Normalizer ─────────► relative formulas → position-independent    🔴
   │                            strings (the heart of the project)
   ▼
[4] Dependency Graph ────────► DAG of cell → precedents/dependents          🔴
   │                            (networkx)
   ▼
[5] Block Detector ──────────► contiguous regions with shared structure    🔴
   │                            (R1C1 equality + rapidfuzz for near-misses)
   ▼
[6] Rule Engine (Tier 0) ────► deterministic checks, always on, $0         🔴
   │
   ▼
[7] Trained Classifier (Tier 1, optional) ─► subtotal | intentional_override | 🔴
   │                            suspected_error | unknown
   │                            small encoder, CPU-only, trained on synthetic
   │                            corruption labels
   ▼
[8] Local LLM Adjudicator (Tier 2, optional, flag-gated) ─► same 4-way label 🔴
   │                            3B model via Ollama, grammar-constrained
   ▼
[9] Report Builder ──────────► ranked JSON + standalone HTML                🔴
```

**Design principle:** everything downstream of stage [2] consumes the AST, never raw formula strings or raw cell dumps. Any semantic layer (Tier 1 or Tier 2) sees a compact normalized block description, never a pasted sheet.

**Zero-cost, offline-first:** no hosted API is a required dependency anywhere in this pipeline. Tier 0 (rules) and Tier 1 (trained classifier) are both $0 and CPU-only by construction; Tier 2 (local LLM) is optional and flag-gated, capped at a 3B model — sized for the target dev machine (13th-gen Intel i5-1335U, 16GB RAM, Iris Xe iGPU, no dedicated VRAM), which is memory-bandwidth-bound and rules out 7B+ models for interactive use.

### Tech stack

| Layer | Choice | Why |
|---|---|---|
| Parsing | `openpyxl` | Only mature lib exposing raw formulas |
| Formula AST | hand-rolled tokenizer/parser | Full control, no third-party formula-parsing dependency |
| R1C1 normalization | hand-rolled (planned) | Core transformation of the project |
| Graph | `networkx` (planned) | Cycle detection, topological ordering |
| Clustering | R1C1 equality + `rapidfuzz` (planned) | Exact match covers most; fuzzy catches near-misses |
| Semantic layer — Tier 1 (planned) | DeBERTa-v3-small / ModernBERT-base, fine-tuned on synthetic-corruption labels | CPU-only, ms/block, $0 |
| Semantic layer — Tier 2 (planned, optional) | Local 3B model via Ollama (Qwen2.5-3B-Instruct or Phi-3-mini) | Fits in 16GB RAM; 7B+ excluded for latency reasons |
| Report | Jinja2 → single-file HTML (planned) | Must open without a server |
| Tests | `pytest` + generated fixture workbooks | Fixtures regenerated per session, not committed as binaries |

---

## Progress Tracker

### Week 1 — Parsing spine

| Task | Status | Detail |
|---|:---:|---|
| Workbook parser (`parser.py`) | 🟢 | Loads `.xlsx`/`.xlsm` via `openpyxl` with `data_only=False` so raw formula strings are captured, never Excel's cached computed values. Per cell: sheet-qualified address (`Sheet1!C14`), formula or literal value, number format string, merged-range membership, and cross-sheet reference detection. |
| Named range resolution | 🟢 | Resolves both workbook-scoped (`wb.defined_names`) and sheet-scoped (`ws.defined_names`, distinguished by `localSheetId`) named ranges. |
| Skip-and-log for unparseable cells | 🟢 | Array formulas (`ArrayFormula` objects), `LET`/`LAMBDA`, and structured table references are detected and logged to a `skipped` list with cell address + reason, rather than crashing the parse. A catch-all wraps every cell so one malformed cell can't abort the whole workbook. |
| CLI (`ssmlint dump`) | 🟢 | `argparse`-based, `dump` subcommand, JSON to stdout or to a file via `-o`. Installed as a console script (`ssmlint`). |
| Synthetic test fixtures | 🟢 | `scripts/generate_fixtures.py` builds 4 workbooks (clean formula row, merged cells, named ranges + cross-sheet formula, unparseable-formula edge cases) programmatically — regenerated every `pytest` session via `conftest.py`, not committed as binaries, so they can never drift out of sync with the generator. |
| Pytest suite | 🟢 | 11 tests: formula extraction, literals, merged cells, named ranges, cross-sheet detection, skip-and-log behavior, unsupported file extensions, JSON shape, CLI stdout/file output. |

**How it was solved:** Built directly on `openpyxl`'s cell/worksheet model. The trickiest part was merged cells — only the top-left cell of a merged range holds a real value in openpyxl; the rest are `MergedCell` instances with `None` values but still readable `number_format` — handled by checking `coordinate in merged_range` for every cell rather than relying on cell identity. Named ranges needed two different lookup paths (workbook dict vs. per-worksheet dict) since openpyxl doesn't unify them.

**Bugs/defects encountered:** None. The parser and CLI passed all 11 tests on first full run and were verified end-to-end against the installed console script before being marked done.

---

### Week 2 — R1C1 normalization + dependency graph

The README's original Build Plan bundled "R1C1 normalization + dependency graph" into one week, but building the AST they both depend on turned out to be substantial enough to track as its own row.

| Task | Status | Detail |
|---|:---:|---|
| Formula tokenizer (`tokenizer.py`) | 🟢 | Hand-rolled lexer, no third-party formula-parsing library. Single ordered master regex splits a formula string into cell references (with independent `$` anchoring per axis), sheet-qualified refs (bare and quoted, e.g. `'My Sheet'!A1`), named ranges, function names, all Excel operators, parens, commas, and string/number/boolean literals. |
| Formula AST parser (`formula_parser.py`) | 🟢 | Recursive-descent parser with correct Excel operator precedence and associativity: `^` before `* /` before `+ -` before comparisons, string concat with `&`, and Excel's specific unary-minus-before-`^` rule. Builds `CellRef` / `RangeRef` / `NamedRangeRef` / `Literal` / `FunctionCall` / `BinaryOp` / `UnaryOp` nodes (`ast_nodes.py`). |
| Whitelist enforcement | 🟢 | Array formulas, `LET`/`LAMBDA`, structured table references, full-column/full-row ranges (`A:A`, `1:1`, including `$`-anchored forms like `$A:$B`), 3-D sheet ranges, and the percent operator all raise `UnsupportedFormulaError` cleanly. Genuinely malformed input raises a separate `FormulaSyntaxError`. Both share a common `FormulaError` base so callers can catch broadly or narrowly. |
| Standalone/independently testable | 🟢 | Not wired into `parser.py` yet, by design — this stage is testable in isolation before the pipeline commits to using it. |
| R1C1 Normalizer | 🔴 | Not started. Will convert A1-style formulas in the AST into relative R1C1 strings so that `=B4*C4` in row 4 and `=B5*C5` in row 5 normalize to the same string — the core transformation the block detector depends on. |
| Dependency Graph | 🔴 | Not started. Will build a `networkx` DAG of cell → precedents/dependents from the AST's `CellRef`/`RangeRef` nodes, enabling cycle detection and topological ordering. |

**How it was solved:** The lexer uses one big alternation-based regex tried in a fixed order (multi-char operators like `<=` before their single-char prefixes; sheet-prefix and cell-address patterns before the generic identifier pattern, since identifiers would otherwise swallow them whole). The parser is a standard precedence-climbing recursive descent, with one deliberate deviation from "normal" language grammars: Excel evaluates `-2^2` as `4`, not `-4` — unary minus binds *tighter* than `^`, the opposite of Python/most languages — so `unary` sits *below* `power` in the grammar instead of wrapping it. `^` itself is left-associative in Excel (`2^3^2 = 64`, not `512`), confirmed with dedicated tests. A small literal-only AST evaluator was written in the test suite specifically to pin down these precedence/associativity cases numerically rather than just asserting tree shape.

**Bugs/defects encountered and fixed:**
1. **`$`-anchored full-column/row ranges crashed instead of being rejected cleanly.** Formulas like `VLOOKUP(A1,Sheet2!$A:$B,2,FALSE)` — common in real `VLOOKUP`/`SUMIFS` usage — raised a raw `"unexpected character '$'"` `FormulaSyntaxError` instead of a proper `UnsupportedFormulaError`, because no token pattern consumed a `$` that wasn't immediately followed by row digits. Found during manual spot-checking with realistic formulas beyond the adversarial test suite (the test suite itself hadn't covered this shape). Fixed by adding dedicated `COL_ONLY`/`ROW_ONLY` token kinds and routing them through the same rejection path as plain `A:A`/`1:1` ranges. Added regression tests (`test_rejects_anchored_full_column_range`, `test_rejects_anchored_full_row_range`).
2. **Inconsistent error type for a stray `:` depending on where it appeared.** A colon inside function-call arguments (e.g. `SUM(1:1)`) raised a generic `FormulaSyntaxError` via the parser's `_expect(RPAREN)` check, while the same construct at the top level correctly raised `UnsupportedFormulaError`. Caught while writing the rejection tests, before it shipped. Fixed by centralizing all "unexpected token" handling into one `_raise_unexpected` helper that every mismatch path now calls, so a stray `:` is always classified as a whitelist rejection regardless of where the parser encounters it.

---

### Week 3 — Block detection + rule engine

| Task | Status | Detail |
|---|:---:|---|
| Block detector | 🔴 | Not started. Will cluster contiguous cells by normalized (R1C1) formula using exact string equality plus `rapidfuzz` for near-misses. |
| Rule engine — literal-in-formula-block | 🔴 | Not started. Flags a hardcoded value sitting inside an otherwise-formula-driven row/column block. |
| Rule engine — range boundary mismatch | 🔴 | Not started. Flags off-by-one errors in range references (e.g. a `SUM` that excludes the last row of a block). |
| Rule engine — reference to blank | 🔴 | Not started. Flags formulas referencing genuinely empty cells. |
| Rule engine — inconsistent anchoring | 🔴 | Not started. Flags `$`-anchoring that breaks the pattern established by the rest of a block. |
| Synthetic-corruption corpus (build now, not later) | 🔴 | Not started. 50 clean models with programmatically injected known bugs — the evaluation *and* future classifier-training backbone. |

---

### Week 4 — Baseline evaluation (Tier 0 only)

| Task | Status | Detail |
|---|:---:|---|
| Evaluation harness | 🔴 | Not started. Runs the rule engine alone against the synthetic corpus. |
| Precision/recall/precision@10 per bug class | 🔴 | Not started. Target: precision > 0.85 (matters more than recall — false alarms kill trust in the tool). This is meant to be a standalone, publishable result, not skipped past. |

---

### Week 5 — Trained classifier (Tier 1)

| Task | Status | Detail |
|---|:---:|---|
| Label generation from synthetic corruption | 🔴 | Not started. Reuses the Week 3/4 corruption pipeline to produce labeled training examples, not just eval ground truth. |
| Small encoder classifier | 🔴 | Not started. DeBERTa-v3-small or ModernBERT-base, fine-tuned on ambiguous blocks, classifying into `subtotal \| intentional_override \| suspected_error \| unknown`. CPU-only, ms/block, $0. |
| Tier 0 → Tier 0+1 comparison | 🔴 | Not started. Re-run the Week 4 evaluation with the classifier layered on top; expect recall to rise and precision to dip slightly. |

---

### Week 6 — Local LLM (Tier 2) + ablation table + report UI

| Task | Status | Detail |
|---|:---:|---|
| Local LLM adjudicator | 🔴 | Not started, optional/flag-gated. 3B model via Ollama (Qwen2.5-3B-Instruct or Phi-3-mini), grammar-constrained to the same 4-way label, cached by normalized block signature, calls capped per workbook. |
| Full ablation table | 🔴 | Not started. Precision, recall, precision@10, cost, and median runtime for Tier 0 / Tier 0+1 / Tier 0+1+2 — this is the project's headline deliverable, not a single precision/recall number. |
| HTML report builder | 🔴 | Not started. Jinja2 → single-file HTML, sheet heatmap, click-through to cells, severity filtering, tagged with which tier flagged each issue. |
