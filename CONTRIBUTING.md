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
  r1c1.py                # [3] position-independent formula normalization
  errors.py                # shared FormulaError hierarchy
  depgraph.py               # [4] workbook-wide precedent/dependent DAG
  blocks.py                  # [5] row-wise block clustering + near-miss detection
  rules/                      # [6] Tier 0 rule engine
    base.py                    # Issue dataclass, Rule base interface
    literal_in_block.py         # literal-in-formula-block rule
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
[3] R1C1 Normalizer ─────────► relative formulas → position-independent    🟢
   │                            strings (the heart of the project)
   ▼
[4] Dependency Graph ────────► DAG of cell → precedents/dependents          🟢
   │                            (networkx)
   ▼
[5] Block Detector ──────────► contiguous regions with shared structure    🟢
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
| R1C1 normalization | hand-rolled | Core transformation of the project |
| Graph | `networkx` | Cycle detection, topological ordering |
| Clustering | R1C1 equality + `rapidfuzz` | Exact match covers most; fuzzy catches near-misses |
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
| R1C1 Normalizer (`r1c1.py`) | 🟢 | Converts a formula's AST plus its origin cell address into a position-independent R1C1-style string. Each `CellRef` axis becomes a bracketed relative offset (`R[1]C[-1]`) if unanchored, or an unbracketed absolute position (`R14C2`) if `$`-anchored — independently per axis, so `$B14` and `B$14` normalize differently from each other and from `B14`. `RangeRef` normalizes both endpoints; cross-sheet refs get a `Sheet!` prefix kept structurally separate from the offset part; `NamedRangeRef` passes through unresolved (case preserved — unlike function names, named ranges are case-sensitive in Excel); function names are upper-cased; every `BinaryOp`/`UnaryOp` is explicitly parenthesized in the output so structurally different ASTs can never coincidentally flatten to the same string. |
| Dependency Graph (`depgraph.py`) | 🟢 | Workbook-wide `networkx.DiGraph` where nodes are sheet-qualified addresses and an edge `precedent -> dependent` means "dependent's formula reads from precedent" — chosen specifically so `topological_sort()` doubles as a valid calculation order with no reversal. Walks every formula's AST, expands `RangeRef`s to individual cell edges (a `SUM(B4:B10)` is 7 edges, never one range-shaped edge), resolves `NamedRangeRef`s against `parser.py`'s captured named ranges (sheet-scoped names shadow workbook-scoped ones, matching Excel), and cross-sheet refs become ordinary edges into the target sheet. `precedents()`/`dependents()`/`find_cycles()`/`topological_order()` query methods; the last raises a dedicated `CyclicDependencyError` (carrying the offending cycles) instead of leaking networkx's own exception. Every node also carries an `empty` attribute (computed once, not re-derived per query) and a `parse_error` attribute so cells with unparseable or catastrophically-failed formulas still exist as nodes with no outgoing edges rather than vanishing. |

**Formula tokenizer/parser — how it was solved:** The lexer uses one big alternation-based regex tried in a fixed order (multi-char operators like `<=` before their single-char prefixes; sheet-prefix and cell-address patterns before the generic identifier pattern, since identifiers would otherwise swallow them whole). The parser is a standard precedence-climbing recursive descent, with one deliberate deviation from "normal" language grammars: Excel evaluates `-2^2` as `4`, not `-4` — unary minus binds *tighter* than `^`, the opposite of Python/most languages — so `unary` sits *below* `power` in the grammar instead of wrapping it. `^` itself is left-associative in Excel (`2^3^2 = 64`, not `512`), confirmed with dedicated tests. A small literal-only AST evaluator was written in the test suite specifically to pin down these precedence/associativity cases numerically rather than just asserting tree shape.

**Formula tokenizer/parser — bugs/defects encountered and fixed:**
1. **`$`-anchored full-column/row ranges crashed instead of being rejected cleanly.** Formulas like `VLOOKUP(A1,Sheet2!$A:$B,2,FALSE)` — common in real `VLOOKUP`/`SUMIFS` usage — raised a raw `"unexpected character '$'"` `FormulaSyntaxError` instead of a proper `UnsupportedFormulaError`, because no token pattern consumed a `$` that wasn't immediately followed by row digits. Found during manual spot-checking with realistic formulas beyond the adversarial test suite (the test suite itself hadn't covered this shape). Fixed by adding dedicated `COL_ONLY`/`ROW_ONLY` token kinds and routing them through the same rejection path as plain `A:A`/`1:1` ranges. Added regression tests (`test_rejects_anchored_full_column_range`, `test_rejects_anchored_full_row_range`).
2. **Inconsistent error type for a stray `:` depending on where it appeared.** A colon inside function-call arguments (e.g. `SUM(1:1)`) raised a generic `FormulaSyntaxError` via the parser's `_expect(RPAREN)` check, while the same construct at the top level correctly raised `UnsupportedFormulaError`. Caught while writing the rejection tests, before it shipped. Fixed by centralizing all "unexpected token" handling into one `_raise_unexpected` helper that every mismatch path now calls, so a stray `:` is always classified as a whitelist rejection regardless of where the parser encounters it.

**R1C1 normalizer — how it was solved:** Every `CellRef` axis is formatted independently by its own `$`-anchoring flag: relative → `[delta]` where `delta = target - origin`; absolute → the bare literal position, ignoring the origin entirely. Cross-sheet handling falls out of the API design for free — `normalize()` only ever receives the *origin cell's address* (e.g. `"C14"`), never its sheet — so a formula's own sheet can't influence the output, which is exactly what makes "same relative shape, different host sheet, same explicit cross-sheet target" normalize identically without any special-case code. Verified with a manual audit printing actual before/after strings for 4 required pairs (all matched expectations) before being marked done, not just a green pytest run.

**R1C1 normalizer — bugs/defects encountered and fixed:**
1. **Test suite asserted the wrong expected value for named ranges.** Two tests initially expected `TaxRate` to normalize to `TAXRATE` (upper-cased, mirroring the function-name rule), and failed. Root cause was the test's assumption, not the implementation: Excel function names are case-insensitive but named-range names are case-sensitive, and `r1c1.py` already preserved the original case correctly per spec ("passes through as the name itself"). Fixed by correcting the two test expectations (`test_named_range_passes_through_unresolved`, `test_sheet_scoped_named_range_is_prefixed`) to preserve case rather than changing the implementation.

**Known, documented limitation (not a bug):** a formula that explicitly writes its own sheet name (e.g. `=Sheet1!B14*C14` while physically living on `Sheet1`) is treated as cross-sheet and will *not* normalize identically to the equivalent implicit form `=B14*C14`, because `normalize()`'s API deliberately doesn't take the origin's sheet name as an argument (only its row/column). Fixing this would require widening the public API; documented in `r1c1.py`'s module docstring rather than silently mishandled.

**Dependency graph — how it was solved:** Two design decisions did most of the work. First, the edge direction (`precedent -> dependent`) was picked specifically so `networkx.topological_sort()` needs no reversal to produce a valid Excel calculation order, and so `precedents(cell)`/`dependents(cell)` map directly onto `predecessors()`/`successors()` with no adapter logic. Second, rather than trusting `parser.py`'s own `skipped` log to decide which cells get no outgoing edges, this module re-parses every formula itself with `formula_parser.parse_formula()` — the two layers don't always agree (see the array-formula case below), so re-parsing is the more accurate source of truth. Named-range resolution reuses `formula_parser.parse_formula()` a second time, on the named range's own `refers_to` text, instead of writing a second reference-parsing grammar. The "is this cell empty" question was deliberately made a node attribute computed once at build time rather than a live lookup against `ParsedWorkbook`, since the rule engine (next stage) will ask it repeatedly per edge and the answer can never change after the graph is built. Verified with hand-built `ParsedWorkbook` objects for the cases a real `.xlsx` can't easily produce (circular refs, dangling named ranges, a cell that never got a `CellRecord` at all), plus real-fixture integration tests, plus manual spot-checks beyond the test suite (self-loops, 3-node cycles, duplicate refs in one formula, reversed ranges like `B10:B4`, and a formula combining a range + a named range + a cross-sheet ref in one expression) — all matched expectations with no further bugs found.

**Dependency graph — bugs/defects encountered and fixed:** None. Every hand-built test case, every fixture integration test, and every manual spot-check (including several scenarios beyond the required edge cases — self-loops, 3-node cycles, duplicate same-cell references, reversed range bounds) passed on first implementation.

---

### Week 3 — Block detection + rule engine

| Task | Status | Detail |
|---|:---:|---|
| Block detector (`blocks.py`) | 🟢 | Clusters each sheet's rows into blocks of contiguous cells sharing one exact R1C1 pattern (row-wise only — see "how it was solved"). A gap of even one non-conforming cell ends a block; nothing is ever merged across it. For each block, the immediate left/right neighbor is classified as either `non_conforming` (blank, literal, unparseable, or a formula scoring below the near-miss threshold) or `near_miss` (a different-but-similar formula, via `rapidfuzz.fuzz.ratio`, with its score attached). Minimum block size is 3 cells. |
| Rule engine — literal-in-formula-block (`rules/literal_in_block.py`) | 🟢 | Flags a literal touching a block's `non_conforming` list (blocks.py's own output — no new traversal). Severity is exactly two levels from structure alone: `high` when the literal is bordered by blocks on *both* sides sharing the identical pattern (the README's own H14 shape — one continuous pattern with a single interruption), `medium` for everything else this rule still flags (bordered on one side only, or two different patterns on each side). Always labeled `suspected_error`, never `intentional_override` — that judgment is explicitly deferred to Tier 1/2. `Issue` dataclass (`cell`, `severity`, `rule_id`, `explanation`, `suggested_fix`) and a `Rule` ABC live in the new `rules/base.py`, shared by every future Tier 0 rule. |
| Rule engine — range boundary mismatch | 🔴 | Not started. Flags off-by-one errors in range references (e.g. a `SUM` that excludes the last row of a block). |
| Rule engine — reference to blank | 🔴 | Not started. Flags formulas referencing genuinely empty cells. |
| Rule engine — inconsistent anchoring | 🔴 | Not started. Flags `$`-anchoring that breaks the pattern established by the rest of a block. |
| Synthetic-corruption corpus (build now, not later) | 🔴 | Not started. 50 clean models with programmatically injected known bugs — the evaluation *and* future classifier-training backbone. |

**Block detector — how it was solved:** Clusters along rows only, not columns — the README's own worked example (`C14:N14`, a revenue row with a month per column) is row-wise, and it's the documented primary shape for these models; column-wise clustering was deliberately left out of scope for this stage rather than half-built. Two judgment calls without a derivable "correct" answer were made explicit rather than picked silently: **minimum block size = 3** (2 adjacent same-pattern cells is weak evidence of an intentional pattern — could easily be coincidence — while 3+ is a much stronger signal), flagged in the delivery summary; and the **rapidfuzz near-miss threshold = 85** (`fuzz.ratio`, character-level and order-sensitive — deliberately not `token_sort_ratio`, which would wrongly call `R[0]C[-1]-R[0]C[-2]` and `R[0]C[-2]-R[0]C[-1]` identical when they mean different things), which was raised as a direct question rather than folded into the summary, per instruction that a second judgment call this significant should be asked, not assumed. Blank-cell detection defers to `DependencyGraph.is_empty()` when the neighbor is a graph node, reusing `depgraph.py`'s single already-documented definition of "empty" instead of re-deriving it a third time; cells that never became graph nodes (unreferenced literals) fall back to deriving it straight from the `CellRecord`. Verified by printing actual block spans/patterns/deviations (not just pass/fail) for every required scenario — a clean row block, a gap producing two separate blocks, a literal in the middle of what would've been one run, a near-miss with its real computed score (92.59 for a `+1` tacked onto an otherwise-identical formula, 46.34 for a genuinely unrelated one), and the N-1-vs-N minimum-size boundary — plus the full 12-column README-style revenue row fixture (`revenue_row_with_hardcode.xlsx`), which split cleanly into `C14:G14` and `I14:M14` around the hardcoded `H14`, with `H14` correctly reported as a non-conforming neighbor of *both* blocks.

**Block detector — bugs/defects encountered and fixed:** None in the implementation itself. While writing the test suite, three of my own test assertions had hand-computed R1C1 pattern strings with row/column offsets transposed (e.g. asserting `R[-1]C[0]` where the module actually and correctly produces `R[0]C[-1]`) — caught immediately by cross-checking against `r1c1.normalize()`'s real output before trusting the assertion, per the standard set during the R1C1 stage of verifying actual values rather than assuming derivations. Two other early test assertions asserted a block's `non_conforming` list was empty without accounting for a seed literal cell (`B14`) that was always present as that block's left neighbor in the test data — a test-data oversight, not a code bug; fixed by removing the irrelevant seed cell from those two tests so they isolate the one behavior each was meant to check.

**FIXED (post-stage bugfix): the r1c1.py self-sheet-reference limitation.** The limitation described in the original version of this write-up — a formula explicitly spelling out its own sheet name (`=Sheet1!B14*C14` while living on `Sheet1`) normalizing differently from the equivalent implicit form (`=B14*C14`) — was confirmed as a real, reproducible bug during an audit (constructed a `C14:I14` row where `F14` used the explicit self-sheet form; it split into two 3-cell blocks with `F14` excluded from both). Fixed in `r1c1.py`: `normalize()` now takes a required `origin_sheet` parameter and collapses a reference to the origin's own sheet to the same output as the implicit form, leaving genuine cross-sheet references untouched. `blocks.py` was updated to pass each cell's own sheet through (its one call site was the only caller besides `r1c1.py`'s own tests). Regression test added (`test_explicit_self_sheet_reference_does_not_cause_a_false_split` in `test_blocks.py`) confirming the same `C14:I14` construction now produces one 7-cell block with empty `non_conforming`/`near_misses`. Full suite: 110 → 114 passing, zero regressions. `NamedRangeRef` sheet-prefixing was deliberately left untouched — never reported as a bug, out of scope for this fix.

**Known limitation, new to this stage:** a cell that fails so catastrophically in `parser.py`'s per-cell catch-all that it never gets a `CellRecord` at all (only an entry in `ParsedWorkbook.skipped`, plus a bare node in the dependency graph) is invisible to this module's row grouping, since `blocks.py` walks `ParsedWorkbook.sheets[*].cells` directly. Such a cell could never be reported as a block's non-conforming neighbor even though the graph knows about it. Judged acceptable — this requires an `openpyxl` per-cell read to fail outright, which essentially never happens in practice — and documented in `blocks.py`'s module docstring rather than silently patched around with graph-splicing logic no fixture or test currently needs.

**Rule engine (literal-in-formula-block) — how it was solved:** Before writing any rule logic, confirmed by reading `blocks.py::_build_block` (not assuming) exactly what `non_conforming` captures: precisely the one cell immediately left of a block's start column and the one immediately right of its end column, nothing further. Verified live with three constructed scenarios: a single literal fully between two runs (the README's H14 shape) is completely captured by both bordering blocks; two adjacent literals in a gap are each captured by exactly one side; three adjacent literals in a gap leave the *middle* one invisible to every block. Reported this finding and asked before proceeding rather than fixing it as a drive-by inside a completed stage — user chose to ship v1 on `non_conforming` as-is and document the gap, rather than widening `blocks.py`'s traversal now. The rule itself groups a literal's sightings across up to two bordering blocks (a cell has at most two sides) into one `Issue`, never a duplicate. Severity uses only structural data blocks.py actually provides (no computed-magnitude signal exists, since cells are never recalculated) — deliberately kept to two levels rather than adding a third/fourth tier that would need its own unjustified numeric threshold, consistent with how `min_block_size` and the rapidfuzz threshold were handled in the previous stage. The "is this a total/subtotal row" intentional-override heuristic was deliberately **not** built — there's no reliable signal for it at this stage, and the README itself names `intentional_override` as a Tier 1/2 label, not Tier 0's to guess at. Verified against the real `revenue_row_with_hardcode.xlsx` fixture through the full pipeline (not a hand-built stub): `H14` correctly comes back `high` with both bordering blocks' spans and the shared pattern named in the explanation; `B14` (the growth chain's seed literal) also fires, at `medium` — a real, expected precision consequence of flagging conservatively by default, not a bug, since the seed of a series is structurally indistinguishable from an accidental hardcode at Tier 0.

**Rule engine (literal-in-formula-block) — bugs/defects encountered and fixed:** None in the implementation. One test-construction mistake caught before it shipped: an early version of the "clean block, zero issues" test used formulas like `=Z13+1` intending "the cell diagonally up-left," which doesn't exist for a block starting at column A — caught by printing the actual block/pattern output first (per this project's standing verification habit) rather than trusting the hand-written formula, and fixed by using a same-column-previous-row pattern (`=A13+1`, `=B13+1`, `=C13+1`) instead.

**Known limitation, confirmed and deferred (not fixed) in this stage:** a literal sitting in the interior of a run of 3+ contiguous non-conforming cells between two blocks is invisible to `non_conforming` entirely, so `literal-in-formula-block` cannot flag it either — reported to the user before writing the rule; ship-as-is was the explicit choice, to be revisited only if a future stage needs it, not patched into `blocks.py` as a side effect of this one rule.

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
