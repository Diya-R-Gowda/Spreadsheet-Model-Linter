"""Hand-rolled tokenizer for the Excel formula subset this project supports.

Splits a raw formula string (as extracted by parser.py, with or without
its leading "=") into a flat token stream. Decoding happens here, not in
the parser: STRING/NUMBER/BOOL literal values and sheet-prefix names are
already unescaped/converted by the time a Token reaches formula_parser.py.

Formula constructs that are out of scope for v1 — array formula braces,
structured table references, and the percent operator — are rejected here
with UnsupportedFormulaError as soon as their marker character is seen,
so the parser never has to know about them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .errors import FormulaSyntaxError, UnsupportedFormulaError


class TokenKind(str, Enum):
    NUMBER = "NUMBER"
    STRING = "STRING"
    BOOL = "BOOL"
    CELL_ADDR = "CELL_ADDR"
    COL_ONLY = "COL_ONLY"  # e.g. "$A" — half of an unsupported full-column range
    ROW_ONLY = "ROW_ONLY"  # e.g. "$1" — half of an unsupported full-row range
    SHEET_PREFIX = "SHEET_PREFIX"
    IDENT = "IDENT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    COLON = "COLON"
    PLUS = "PLUS"
    MINUS = "MINUS"
    MUL = "MUL"
    DIV = "DIV"
    POW = "POW"
    CONCAT = "CONCAT"
    EQ = "EQ"
    NE = "NE"
    LE = "LE"
    GE = "GE"
    LT = "LT"
    GT = "GT"
    EOF = "EOF"


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str  # raw matched substring, for error messages
    pos: int
    value: object = None  # decoded payload for NUMBER/STRING/BOOL/SHEET_PREFIX


# Order matters: alternation picks the first pattern that matches at a
# given position, not the longest one. Multi-char operators (<=, >=, <>)
# must precede their single-char prefixes (<, >, =); SHEET_*/CELL_ADDR
# must precede IDENT, which would otherwise swallow them whole.
_MASTER_RE = re.compile(
    r"""
      (?P<WS>\s+)
    | (?P<NUMBER>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
    | (?P<STRING>"(?:[^"]|"")*")
    | (?P<SHEET_QUOTED>'(?:[^']|'')+'!)
    | (?P<SHEET_BARE>[A-Za-z_][A-Za-z0-9_.]*!)
    | (?P<CELL_ADDR>\$?[A-Za-z]{1,3}\$?\d+(?![A-Za-z0-9_.]))
    | (?P<COL_ONLY>\$[A-Za-z]{1,3}(?![A-Za-z0-9_.]))
    | (?P<ROW_ONLY>\$\d+(?![A-Za-z0-9_.]))
    | (?P<BOOL>(?:TRUE|FALSE)(?![A-Za-z0-9_.]))
    | (?P<IDENT>[A-Za-z_][A-Za-z0-9_.]*)
    | (?P<LE><=)
    | (?P<GE>>=)
    | (?P<NE><>)
    | (?P<LPAREN>\()
    | (?P<RPAREN>\))
    | (?P<COMMA>,)
    | (?P<COLON>:)
    | (?P<PLUS>\+)
    | (?P<MINUS>-)
    | (?P<MUL>\*)
    | (?P<DIV>/)
    | (?P<POW>\^)
    | (?P<CONCAT>&)
    | (?P<EQ>=)
    | (?P<LT><)
    | (?P<GT>>)
    | (?P<PERCENT>%)
    | (?P<LBRACKET>\[)
    | (?P<LBRACE>\{)
    | (?P<RBRACE>\})
    """,
    re.VERBOSE | re.IGNORECASE,
)

_CELL_ADDR_PARTS_RE = re.compile(r"^(\$?)([A-Za-z]{1,3})(\$?)(\d+)$")

_UNSUPPORTED_MARKERS = {
    "PERCENT": "the percent operator ('%') is not supported in v1",
    "LBRACKET": "structured table references (e.g. Table1[Column]) are not supported in v1",
    "LBRACE": "array formula literals ('{...}') are not supported in v1",
    "RBRACE": "array formula literals ('{...}') are not supported in v1",
}


def parse_cell_addr_parts(text: str) -> tuple[bool, str, bool, int]:
    """Split a CELL_ADDR token's raw text into (col_abs, col_letters, row_abs, row)."""
    match = _CELL_ADDR_PARTS_RE.match(text)
    if not match:  # pragma: no cover - guarded by the tokenizer's own regex
        raise FormulaSyntaxError(f"malformed cell reference {text!r}")
    col_dollar, col_letters, row_dollar, row_digits = match.groups()
    return bool(col_dollar), col_letters.upper(), bool(row_dollar), int(row_digits)


def _strip_leading_equals(formula: str) -> str:
    return formula[1:] if formula.startswith("=") else formula


def _reject_array_braces(formula: str) -> None:
    stripped = formula.strip()
    if stripped.startswith("{") or stripped.endswith("}"):
        raise UnsupportedFormulaError(
            f"array formula literals ('{{...}}') are not supported in v1 (in {formula!r})"
        )


def tokenize(formula: str) -> list[Token]:
    """Tokenize a formula string, raising FormulaSyntaxError/UnsupportedFormulaError on failure.

    Accepts the formula with or without its leading "=" so bare
    expressions (e.g. in tests) work the same as raw cell formulas.
    """
    _reject_array_braces(formula)
    body = _strip_leading_equals(formula)

    tokens: list[Token] = []
    pos = 0
    length = len(body)
    while pos < length:
        match = _MASTER_RE.match(body, pos)
        if not match:
            raise FormulaSyntaxError(
                f"unexpected character {body[pos]!r} at position {pos} in {formula!r}"
            )
        kind_name = match.lastgroup
        text = match.group()
        start = pos
        pos = match.end()

        if kind_name == "WS":
            continue
        if kind_name in _UNSUPPORTED_MARKERS:
            raise UnsupportedFormulaError(
                f"{_UNSUPPORTED_MARKERS[kind_name]} (in {formula!r} at position {start})"
            )

        if kind_name == "NUMBER":
            tokens.append(Token(TokenKind.NUMBER, text, start, value=float(text)))
        elif kind_name == "STRING":
            tokens.append(Token(TokenKind.STRING, text, start, value=text[1:-1].replace('""', '"')))
        elif kind_name == "SHEET_QUOTED":
            tokens.append(
                Token(TokenKind.SHEET_PREFIX, text, start, value=text[1:-2].replace("''", "'"))
            )
        elif kind_name == "SHEET_BARE":
            tokens.append(Token(TokenKind.SHEET_PREFIX, text, start, value=text[:-1]))
        elif kind_name == "BOOL":
            tokens.append(Token(TokenKind.BOOL, text, start, value=text.upper() == "TRUE"))
        else:
            # CELL_ADDR, COL_ONLY, ROW_ONLY, IDENT, and all punctuation/operator
            # groups carry no decoded payload beyond their raw text.
            tokens.append(Token(TokenKind[kind_name], text, start))

    tokens.append(Token(TokenKind.EOF, "", length))
    return tokens
