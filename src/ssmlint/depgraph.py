"""Dependency graph — stage [4] in the architecture.

Builds a workbook-wide directed graph where nodes are sheet-qualified cell
addresses ("Sheet1!C14") and an edge precedent -> dependent means
"dependent's formula reads from precedent" (equivalently: precedent must
be computed before dependent). The direction is deliberate: it makes
networkx.topological_sort() usable directly as a valid Excel calculation
order, no reversal needed, and it makes the query names read naturally:

    precedents(cell) == predecessors(cell)   # who `cell` reads from
    dependents(cell) == successors(cell)     # who reads from `cell`

NODE SCOPE
-----------
Not every cell in the workbook becomes a node — a literal-valued cell
nothing ever references would add bulk with no value. A cell becomes a
node when at least one of:
  - it has a formula (parseable or not) in the parsing spine's output,
  - some other cell's formula references it (as a plain CellRef, part of
    an expanded RangeRef, or resolved from a NamedRangeRef) — this covers
    referenced literal cells and referenced-but-blank cells alike, since
    add_edge() auto-creates the target node,
  - the parsing spine's per-cell catch-all failed on it so badly the cell
    never made it into ParsedWorkbook.sheets at all (see UNPARSEABLE
    CELLS below) — it still must exist as a node, or anything that
    references it would silently point at nothing.

EMPTY-CELL DETECTION: NODE ATTRIBUTE, NOT A LIVE LOOKUP
----------------------------------------------------------
Every node gets an `empty` boolean attribute, computed once at build time
rather than re-derived per query. The rule engine (a later stage) will
ask "is this precedent empty?" for potentially every edge in the graph,
repeatedly, after the graph is built and never mutated again — so a node
attribute is an O(1) dict read on every later call, while re-checking
against ParsedWorkbook each time would mean threading that reference
through every query call for no benefit, since the answer can't change.
A cell counts as empty if it has neither a formula nor a literal value,
OR if it was referenced but never appears in the parsed workbook at all
(a reference past the used range, or to a sheet/cell that doesn't exist,
is exactly as "empty" as a blank cell within range).

UNPARSEABLE CELLS
--------------------
A cell can fail to contribute outgoing edges for two different reasons,
both surfaced the same way: a node with a non-None `parse_error`
attribute and zero outgoing edges.
  1. It has a formula, but formula_parser.parse_formula() rejects it
     (UnsupportedFormulaError or FormulaSyntaxError). This module
     re-parses every formula itself rather than trusting parser.py's own
     `skipped` log, because the two layers don't always agree — an
     openpyxl ArrayFormula-wrapped formula that parser.py flags as
     "array formula" often parses just fine here, since it was only
     flagged upstream because of *how* Excel stored it, not its syntax.
  2. parser.py's per-cell catch-all failed on it so completely that no
     CellRecord was ever produced at all (an address appears in
     ParsedWorkbook.skipped with no matching cell in
     ParsedWorkbook.sheets). There is no formula text to even attempt
     here, so the node is added bare, with the skip reason as its error.

NAMED RANGE RESOLUTION
-------------------------
A NamedRangeRef is resolved against ParsedWorkbook.named_ranges using
Excel's own scoping rule: a sheet-scoped name (defined on the referencing
cell's own host sheet) shadows a workbook-scoped name of the same
spelling when the reference is unqualified; an explicitly sheet-qualified
reference (`Sheet!Name`) only ever looks at that sheet's local names, no
workbook-scope fallback (matching Excel's own behavior). The resolved
`refers_to` text is itself just reference syntax, so it's re-parsed with
formula_parser.parse_formula() — reusing the exact same grammar rather
than writing a second one — and expected to come back as a CellRef or
RangeRef. Anything else (a dynamic name built from a function call, a
literal "named constant", a name that isn't defined anywhere) is logged
to `unresolved_named_ranges` rather than raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
from openpyxl.utils import get_column_letter

from .ast_nodes import ASTNode, BinaryOp, CellRef, FunctionCall, NamedRangeRef, RangeRef, UnaryOp
from .errors import FormulaError
from .formula_parser import parse_formula
from .parser import CellRecord, NamedRangeRecord, ParsedWorkbook


class CyclicDependencyError(Exception):
    """Raised by DependencyGraph.topological_order() when the graph has cycles.

    Carries the offending cycles so callers don't have to re-derive them.
    """

    def __init__(self, message: str, cycles: list[list[str]]) -> None:
        super().__init__(message)
        self.cycles = cycles


def _resolve_sheet(ref_sheet: str | None, host_sheet: str) -> str:
    return ref_sheet if ref_sheet is not None else host_sheet


def _cell_address(ref: CellRef, host_sheet: str) -> str:
    sheet = _resolve_sheet(ref.sheet, host_sheet)
    return f"{sheet}!{ref.col}{ref.row}"


def _expand_range(ref: RangeRef, host_sheet: str) -> list[str]:
    sheet = _resolve_sheet(ref.sheet, host_sheet)
    col_lo, col_hi = sorted((ref.start.col_index, ref.end.col_index))
    row_lo, row_hi = sorted((ref.start.row, ref.end.row))
    return [
        f"{sheet}!{get_column_letter(col)}{row}"
        for row in range(row_lo, row_hi + 1)
        for col in range(col_lo, col_hi + 1)
    ]


def _collect_refs(node: ASTNode) -> list[CellRef | RangeRef | NamedRangeRef]:
    """Every CellRef/RangeRef/NamedRangeRef reachable anywhere in the AST."""
    if isinstance(node, (CellRef, RangeRef, NamedRangeRef)):
        return [node]
    if isinstance(node, FunctionCall):
        refs: list[CellRef | RangeRef | NamedRangeRef] = []
        for arg in node.args:
            refs.extend(_collect_refs(arg))
        return refs
    if isinstance(node, BinaryOp):
        return _collect_refs(node.left) + _collect_refs(node.right)
    if isinstance(node, UnaryOp):
        return _collect_refs(node.operand)
    return []  # Literal: no refs


def _resolve_named_range(
    ref: NamedRangeRef,
    host_sheet: str,
    named_ranges_by_scope: dict[str, dict[str, NamedRangeRecord]],
    unresolved: list[str],
) -> CellRef | RangeRef | None:
    if ref.sheet is not None:
        record = named_ranges_by_scope.get(ref.sheet, {}).get(ref.name)
    else:
        record = named_ranges_by_scope.get(host_sheet, {}).get(
            ref.name
        ) or named_ranges_by_scope.get("workbook", {}).get(ref.name)

    if record is None:
        unresolved.append(f"{host_sheet}: named range {ref.name!r} could not be resolved")
        return None

    try:
        resolved_node = parse_formula(record.refers_to)
    except FormulaError as exc:
        unresolved.append(
            f"{host_sheet}: named range {ref.name!r} refers to {record.refers_to!r}, "
            f"which could not be parsed as a reference ({exc})"
        )
        return None

    if isinstance(resolved_node, (CellRef, RangeRef)):
        return resolved_node

    unresolved.append(
        f"{host_sheet}: named range {ref.name!r} refers to {record.refers_to!r}, "
        "which is not a plain cell/range reference"
    )
    return None


def _add_edges_for_formula(
    graph: nx.DiGraph,
    dependent_address: str,
    host_sheet: str,
    ast_node: ASTNode,
    named_ranges_by_scope: dict[str, dict[str, NamedRangeRecord]],
    unresolved_named_ranges: list[str],
) -> None:
    for ref in _collect_refs(ast_node):
        if isinstance(ref, CellRef):
            graph.add_edge(_cell_address(ref, host_sheet), dependent_address)
        elif isinstance(ref, RangeRef):
            for precedent in _expand_range(ref, host_sheet):
                graph.add_edge(precedent, dependent_address)
        else:  # NamedRangeRef
            resolved = _resolve_named_range(
                ref, host_sheet, named_ranges_by_scope, unresolved_named_ranges
            )
            if resolved is None:
                continue
            if isinstance(resolved, CellRef):
                graph.add_edge(_cell_address(resolved, host_sheet), dependent_address)
            else:
                for precedent in _expand_range(resolved, host_sheet):
                    graph.add_edge(precedent, dependent_address)


def build_graph(parsed: ParsedWorkbook) -> "DependencyGraph":
    graph: nx.DiGraph = nx.DiGraph()
    cell_index: dict[str, CellRecord] = {
        cell.address: cell for sheet in parsed.sheets for cell in sheet.cells
    }
    named_ranges_by_scope: dict[str, dict[str, NamedRangeRecord]] = {}
    for named_range in parsed.named_ranges:
        named_ranges_by_scope.setdefault(named_range.scope, {})[named_range.name] = named_range

    unresolved_named_ranges: list[str] = []
    parse_errors: dict[str, str] = {}

    for sheet in parsed.sheets:
        for cell in sheet.cells:
            if cell.formula is None:
                continue
            graph.add_node(cell.address)
            try:
                ast_node = parse_formula(cell.formula)
            except FormulaError as exc:
                parse_errors[cell.address] = str(exc)
                continue
            _add_edges_for_formula(
                graph, cell.address, sheet.name, ast_node,
                named_ranges_by_scope, unresolved_named_ranges,
            )

    # Cells parser.py's catch-all failed on so hard they never got a
    # CellRecord at all (see module docstring, UNPARSEABLE CELLS #2).
    for item in parsed.skipped:
        if item.address not in cell_index:
            graph.add_node(item.address)
            parse_errors.setdefault(item.address, item.reason)

    # Empty-cell detection, computed once as a node attribute (see module
    # docstring, EMPTY-CELL DETECTION).
    for address in graph.nodes:
        record = cell_index.get(address)
        graph.nodes[address]["empty"] = record is None or (
            record.formula is None and record.value is None
        )
        graph.nodes[address]["parse_error"] = parse_errors.get(address)

    return DependencyGraph(graph, tuple(unresolved_named_ranges))


@dataclass
class DependencyGraph:
    graph: nx.DiGraph
    unresolved_named_ranges: tuple[str, ...] = field(default_factory=tuple)

    def precedents(self, cell: str) -> list[str]:
        """Cells `cell`'s formula reads from."""
        return list(self.graph.predecessors(cell))

    def dependents(self, cell: str) -> list[str]:
        """Cells whose formulas read from `cell`."""
        return list(self.graph.successors(cell))

    def is_empty(self, cell: str) -> bool:
        return self.graph.nodes[cell]["empty"]

    def parse_error(self, cell: str) -> str | None:
        return self.graph.nodes[cell].get("parse_error")

    def find_cycles(self) -> list[list[str]]:
        return list(nx.simple_cycles(self.graph))

    def topological_order(self) -> list[str]:
        """A valid Excel calculation order (precedents before dependents).

        Raises CyclicDependencyError, carrying the offending cycles,
        instead of letting networkx's own NetworkXUnfeasible escape.
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible as exc:
            cycles = self.find_cycles()
            raise CyclicDependencyError(
                f"workbook has {len(cycles)} circular reference chain(s); cannot "
                "compute a calculation order. Use find_cycles() to inspect them.",
                cycles=cycles,
            ) from exc

    def nodes(self) -> list[str]:
        return list(self.graph.nodes)

    def __contains__(self, cell: str) -> bool:
        return cell in self.graph

    def __len__(self) -> int:
        return self.graph.number_of_nodes()
