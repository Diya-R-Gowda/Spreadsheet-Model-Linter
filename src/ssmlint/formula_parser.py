"""Hand-rolled recursive-descent parser: token stream -> formula AST.

Grammar, loosest to tightest binding (top-level entry is `comparison`):

    comparison := concat ( (= | <> | <= | >= | < | >) concat )*
    concat     := additive ( & additive )*
    additive   := term ( (+ | -) term )*
    term       := power ( (* | /) power )*
    power      := unary ( ^ unary )*
    unary      := (+ | -) unary | primary
    primary    := NUMBER | STRING | BOOL | '(' comparison ')'
                | function_call | cell_ref | range_ref | named_range_ref

Note the unary/power ordering: in Excel, unary minus binds *tighter* than
exponentiation, so `-2^2` evaluates to `4`, not `-4` as in most languages
(and Python). `power` therefore calls `unary` for both of its operands
rather than the more usual unary-wraps-power arrangement. `^` is
left-associative (`2^3^2` is `(2^3)^2 = 64` in Excel), so `power` loops
rather than recursing on the right.
"""

from __future__ import annotations

from .ast_nodes import (
    ASTNode,
    BinaryOp,
    CellRef,
    FunctionCall,
    Literal,
    NamedRangeRef,
    RangeRef,
    UnaryOp,
)
from .errors import FormulaSyntaxError, UnsupportedFormulaError
from .tokenizer import Token, TokenKind, parse_cell_addr_parts, tokenize

_COMPARISON_KINDS = (
    TokenKind.EQ,
    TokenKind.NE,
    TokenKind.LE,
    TokenKind.GE,
    TokenKind.LT,
    TokenKind.GT,
)
_ADDITIVE_KINDS = (TokenKind.PLUS, TokenKind.MINUS)
_TERM_KINDS = (TokenKind.MUL, TokenKind.DIV)
_UNSUPPORTED_FUNCTION_NAMES = {"LET", "LAMBDA"}


def _cell_ref_from_token(token: Token, sheet: str | None) -> CellRef:
    col_abs, col, row_abs, row = parse_cell_addr_parts(token.text)
    return CellRef(col=col, row=row, col_abs=col_abs, row_abs=row_abs, sheet=sheet)


class _Parser:
    def __init__(self, tokens: list[Token], original: str) -> None:
        self._tokens = tokens
        self._pos = 0
        self._original = original

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        token = self._tokens[self._pos]
        if token.kind != TokenKind.EOF:
            self._pos += 1
        return token

    def _raise_unexpected(self, token: Token, *, expected: TokenKind | None = None) -> None:
        # A stray ':' is virtually always a full-row/full-column or 3-D
        # range attempt, not a random syntax typo, and it can surface here
        # from anywhere (top-level trailing token, mid function-call args,
        # ...) — always classify it as a whitelist rejection so callers
        # can distinguish "known unsupported" from "actually malformed".
        if token.kind == TokenKind.COLON:
            raise UnsupportedFormulaError(
                "unexpected ':' — full-row/full-column ranges (e.g. 'A:A', '1:1'), "
                "3-D sheet ranges, and ranges between named ranges are not supported "
                f"in v1 (in {self._original!r})"
            )
        if expected is not None:
            raise FormulaSyntaxError(
                f"expected {expected.value} but found {token.kind.value} ({token.text!r}) "
                f"at position {token.pos} in {self._original!r}"
            )
        raise FormulaSyntaxError(
            f"unexpected token {token.kind.value} ({token.text!r}) at position {token.pos} "
            f"in {self._original!r}"
        )

    def _expect(self, kind: TokenKind) -> Token:
        token = self._peek()
        if token.kind != kind:
            self._raise_unexpected(token, expected=kind)
        return self._advance()

    def parse(self) -> ASTNode:
        node = self._comparison()
        trailing = self._peek()
        if trailing.kind != TokenKind.EOF:
            self._raise_unexpected(trailing)
        return node

    def _comparison(self) -> ASTNode:
        node = self._concat()
        while self._peek().kind in _COMPARISON_KINDS:
            op_token = self._advance()
            node = BinaryOp(op=op_token.text, left=node, right=self._concat())
        return node

    def _concat(self) -> ASTNode:
        node = self._additive()
        while self._peek().kind == TokenKind.CONCAT:
            self._advance()
            node = BinaryOp(op="&", left=node, right=self._additive())
        return node

    def _additive(self) -> ASTNode:
        node = self._term()
        while self._peek().kind in _ADDITIVE_KINDS:
            op_token = self._advance()
            node = BinaryOp(op=op_token.text, left=node, right=self._term())
        return node

    def _term(self) -> ASTNode:
        node = self._power()
        while self._peek().kind in _TERM_KINDS:
            op_token = self._advance()
            node = BinaryOp(op=op_token.text, left=node, right=self._power())
        return node

    def _power(self) -> ASTNode:
        # See module docstring: unary binds tighter than ^ in Excel, and ^
        # is left-associative, so this loops on `unary` rather than
        # recursing on the right the way most languages' `**` would.
        node = self._unary()
        while self._peek().kind == TokenKind.POW:
            self._advance()
            node = BinaryOp(op="^", left=node, right=self._unary())
        return node

    def _unary(self) -> ASTNode:
        token = self._peek()
        if token.kind in (TokenKind.PLUS, TokenKind.MINUS):
            self._advance()
            return UnaryOp(op=token.text, operand=self._unary())
        return self._primary()

    def _primary(self) -> ASTNode:
        token = self._peek()

        if token.kind == TokenKind.NUMBER:
            self._advance()
            return Literal(value=token.value)
        if token.kind == TokenKind.STRING:
            self._advance()
            return Literal(value=token.value)
        if token.kind == TokenKind.BOOL:
            self._advance()
            return Literal(value=token.value)
        if token.kind == TokenKind.LPAREN:
            self._advance()
            node = self._comparison()
            self._expect(TokenKind.RPAREN)
            return node
        if token.kind in (
            TokenKind.SHEET_PREFIX,
            TokenKind.CELL_ADDR,
            TokenKind.IDENT,
            TokenKind.COL_ONLY,
            TokenKind.ROW_ONLY,
        ):
            return self._ref_or_call()

        self._raise_unexpected(token)
        raise AssertionError("unreachable")  # _raise_unexpected always raises

    def _ref_or_call(self) -> ASTNode:
        sheet: str | None = None
        token = self._peek()
        if token.kind == TokenKind.SHEET_PREFIX:
            sheet = token.value
            self._advance()
            token = self._peek()

        if token.kind in (TokenKind.COL_ONLY, TokenKind.ROW_ONLY):
            raise UnsupportedFormulaError(
                f"full-column/full-row references with absolute anchoring (e.g. {token.text!r} "
                f"as in '$A:$B' or '$1:$1') are not supported in v1 (in {self._original!r})"
            )

        if token.kind == TokenKind.CELL_ADDR:
            self._advance()
            start = _cell_ref_from_token(token, sheet)
            if self._peek().kind == TokenKind.COLON:
                self._advance()
                end = self._range_end(sheet)
                return RangeRef(start=start, end=end)
            return start

        if token.kind == TokenKind.IDENT:
            self._advance()
            name = token.text
            if self._peek().kind == TokenKind.LPAREN:
                if sheet is not None:
                    raise UnsupportedFormulaError(
                        f"sheet-qualified function calls ({sheet!r}!{name}) look like a "
                        "custom/macro function reference, which is not supported in v1"
                    )
                return self._function_call(name)
            if self._peek().kind == TokenKind.COLON:
                raise UnsupportedFormulaError(
                    f"colon-joined reference starting from identifier {name!r} — "
                    "full-column/full-row ranges (e.g. 'A:A'), 3-D sheet ranges, and "
                    f"ranges between named ranges are not supported in v1 (in {self._original!r})"
                )
            return NamedRangeRef(name=name, sheet=sheet)

        raise FormulaSyntaxError(
            f"expected a cell reference, range, or name after sheet prefix {sheet!r}, "
            f"found {token.kind.value} ({token.text!r}) in {self._original!r}"
        )

    def _range_end(self, left_sheet: str | None) -> CellRef:
        token = self._peek()
        right_sheet: str | None = None
        if token.kind == TokenKind.SHEET_PREFIX:
            right_sheet = token.value
            self._advance()
            token = self._peek()

        if token.kind != TokenKind.CELL_ADDR:
            raise FormulaSyntaxError(
                f"expected a cell reference after ':', found {token.kind.value} "
                f"({token.text!r}) in {self._original!r}"
            )
        self._advance()

        if right_sheet is not None and right_sheet != left_sheet:
            raise UnsupportedFormulaError(
                f"3-D range references spanning sheets ({left_sheet!r}:{right_sheet!r}) "
                f"are not supported in v1 (in {self._original!r})"
            )
        return _cell_ref_from_token(token, left_sheet)

    def _function_call(self, name: str) -> FunctionCall:
        if name.upper() in _UNSUPPORTED_FUNCTION_NAMES:
            raise UnsupportedFormulaError(
                f"{name.upper()} is not supported in v1 (in {self._original!r})"
            )
        self._expect(TokenKind.LPAREN)
        args: list[ASTNode] = []
        if self._peek().kind != TokenKind.RPAREN:
            args.append(self._comparison())
            while self._peek().kind == TokenKind.COMMA:
                self._advance()
                args.append(self._comparison())
        self._expect(TokenKind.RPAREN)
        return FunctionCall(name=name, args=tuple(args))


def parse_formula(formula: str) -> ASTNode:
    """Tokenize and parse a formula string into an AST.

    Accepts the formula with or without its leading "=". Raises
    FormulaSyntaxError for malformed input and UnsupportedFormulaError for
    constructs deliberately out of scope for v1 (array formulas,
    LET/LAMBDA, structured table references, full-column/full-row and 3-D
    ranges, the percent operator).
    """
    tokens = tokenize(formula)
    return _Parser(tokens, formula).parse()
