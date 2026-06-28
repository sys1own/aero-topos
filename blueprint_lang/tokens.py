# -*- coding: utf-8 -*-
"""Token definitions for the ``blueprint.aero`` lexer."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from .positions import Span


class TokenKind(enum.Enum):
    """The kinds of tokens the lexer produces.

    The string value is a human-readable description used directly in error
    messages (e.g. ``expected '{'``).
    """

    IDENT = "an identifier"
    STRING = "a string"
    NUMBER = "a number"
    BOOL = "a boolean"
    LBRACE = "'{'"
    RBRACE = "'}'"
    LBRACKET = "'['"
    RBRACKET = "']'"
    EQUALS = "'='"
    COMMA = "','"
    EOF = "end of file"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Token:
    """A lexical token.

    ``text`` is the exact slice of source the token came from; ``value`` is the
    decoded payload (e.g. an unescaped string, or an ``int``/``float`` for
    numbers).  For punctuation and identifiers ``value`` mirrors ``text``.
    """

    kind: TokenKind
    text: str
    value: Any
    span: Span

    def describe(self) -> str:
        """A friendly noun phrase for this token, for use in error messages."""
        if self.kind is TokenKind.EOF:
            return "end of file"
        if self.kind is TokenKind.STRING:
            return f"the string {self.text}"
        if self.kind is TokenKind.NUMBER:
            return f"the number {self.text}"
        if self.kind is TokenKind.BOOL:
            return f"the keyword '{self.text}'"
        if self.kind is TokenKind.IDENT:
            return f"the identifier '{self.text}'"
        return f"'{self.text}'"
