from __future__ import annotations

import pytest

from ssmlint.errors import FormulaSyntaxError, UnsupportedFormulaError
from ssmlint.tokenizer import TokenKind, tokenize


def _kinds(formula: str) -> list[TokenKind]:
    return [t.kind for t in tokenize(formula)]


def test_strips_leading_equals() -> None:
    with_eq = tokenize("=A1+1")
    without_eq = tokenize("A1+1")
    assert [t.kind for t in with_eq] == [t.kind for t in without_eq]


def test_basic_operators_and_punctuation() -> None:
    assert _kinds("(1,2)") == [
        TokenKind.LPAREN,
        TokenKind.NUMBER,
        TokenKind.COMMA,
        TokenKind.NUMBER,
        TokenKind.RPAREN,
        TokenKind.EOF,
    ]


def test_multichar_comparison_operators_are_not_split() -> None:
    tokens = tokenize("A1<=B1")
    kinds = [t.kind for t in tokens]
    assert kinds == [TokenKind.CELL_ADDR, TokenKind.LE, TokenKind.CELL_ADDR, TokenKind.EOF]

    tokens = tokenize("A1<>B1")
    assert tokens[1].kind == TokenKind.NE

    tokens = tokenize("A1>=B1")
    assert tokens[1].kind == TokenKind.GE


def test_cell_addr_vs_ident_disambiguation() -> None:
    # "A1" is a cell address; "ABCD123" has 4 leading letters so it can't
    # be a valid column and must fall back to a plain identifier.
    assert tokenize("A1")[0].kind == TokenKind.CELL_ADDR
    assert tokenize("ABCD123")[0].kind == TokenKind.IDENT
    assert tokenize("SUM")[0].kind == TokenKind.IDENT


def test_dollar_anchoring_captured_in_raw_text() -> None:
    for text in ("A1", "$A1", "A$1", "$A$1"):
        tok = tokenize(text)[0]
        assert tok.kind == TokenKind.CELL_ADDR
        assert tok.text == text


def test_string_literal_unescaping() -> None:
    tok = tokenize('"hello ""world"""')[0]
    assert tok.kind == TokenKind.STRING
    assert tok.value == 'hello "world"'


def test_number_literal_decoding() -> None:
    assert tokenize("42")[0].value == 42.0
    assert tokenize("3.14")[0].value == 3.14
    assert tokenize("1.5E+3")[0].value == 1500.0


def test_bool_literal_decoding() -> None:
    assert tokenize("TRUE")[0].value is True
    assert tokenize("FALSE")[0].value is False
    # case-insensitive, but must not swallow a longer identifier
    assert tokenize("TRUELY")[0].kind == TokenKind.IDENT


def test_quoted_sheet_prefix_with_spaces_and_escaped_quote() -> None:
    tok = tokenize("'My Sheet'!A1")[0]
    assert tok.kind == TokenKind.SHEET_PREFIX
    assert tok.value == "My Sheet"

    tok = tokenize("'It''s Mine'!A1")[0]
    assert tok.value == "It's Mine"


def test_bare_sheet_prefix() -> None:
    tok = tokenize("Sheet2!A1")[0]
    assert tok.kind == TokenKind.SHEET_PREFIX
    assert tok.value == "Sheet2"


@pytest.mark.parametrize(
    "formula",
    [
        "=SUM(Table1[Column1])",
        "=A1%",
        "={SUM(A1:A10)}",
        "{=SUM(A1:A10)}",
    ],
)
def test_whitelist_rejections_raise_unsupported(formula: str) -> None:
    with pytest.raises(UnsupportedFormulaError):
        tokenize(formula)


def test_unrecognized_character_raises_syntax_error() -> None:
    with pytest.raises(FormulaSyntaxError):
        tokenize("A1 @ B1")
