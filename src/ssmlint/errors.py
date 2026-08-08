"""Exceptions raised by the formula tokenizer/parser."""

from __future__ import annotations


class FormulaError(Exception):
    """Base class for all formula tokenizing/parsing failures."""


class UnsupportedFormulaError(FormulaError):
    """A formula construct that is deliberately out of scope for v1.

    Covers array formulas, LET/LAMBDA, structured table references,
    full-column/full-row and 3-D ranges, and the percent operator. Callers
    (e.g. parser.py's future skip-and-log wiring) should catch this
    specifically to distinguish "known, not supported yet" from a genuine
    parsing bug.
    """


class FormulaSyntaxError(FormulaError):
    """A formula that does not parse as valid Excel formula syntax at all."""
