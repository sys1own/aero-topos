# -*- coding: utf-8 -*-
"""The lexer (tokenizer) for ``blueprint.aero``.

Turns raw source text into a flat list of :class:`~blueprint_lang.tokens.Token`
objects, tracking precise line/column positions as it goes.  Whitespace and
comments (``#`` or ``//`` to end of line) are skipped.  The lexer is where the
"unterminated string" and "invalid escape" diagnostics originate.
"""

from __future__ import annotations

from typing import List

from .errors import BlueprintSyntaxError
from .positions import Position, Span
from .tokens import Token, TokenKind

_STRING_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "/": "/",
}

_PUNCTUATION = {
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    "=": TokenKind.EQUALS,
    ",": TokenKind.COMMA,
}


class Lexer:
    """Convert ``source`` into tokens.  One-shot: call :meth:`tokenize` once."""

    def __init__(self, source: str, filename: str = "blueprint.aero") -> None:
        self.source = source
        self.filename = filename
        self._pos = 0
        self._line = 1
        self._col = 1

    # -- low-level cursor helpers -----------------------------------------

    def _here(self) -> Position:
        return Position(self._line, self._col, self._pos)

    def _peek(self, ahead: int = 0) -> str:
        index = self._pos + ahead
        if 0 <= index < len(self.source):
            return self.source[index]
        return ""

    def _at_end(self) -> bool:
        return self._pos >= len(self.source)

    def _advance(self) -> str:
        ch = self.source[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
            self._col = 1
        else:
            self._col += 1
        return ch

    # -- trivia ------------------------------------------------------------

    def _skip_trivia(self) -> None:
        while not self._at_end():
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance()
            elif ch == "#" or (ch == "/" and self._peek(1) == "/"):
                while not self._at_end() and self._peek() != "\n":
                    self._advance()
            else:
                break

    # -- public API --------------------------------------------------------

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while True:
            self._skip_trivia()
            if self._at_end():
                eof = self._here()
                tokens.append(Token(TokenKind.EOF, "", None, Span.point(eof)))
                return tokens
            tokens.append(self._next_token())

    # -- token producers ---------------------------------------------------

    def _next_token(self) -> Token:
        ch = self._peek()
        if ch == '"':
            return self._lex_string()
        if ch.isdigit() or (ch == "-" and self._peek(1).isdigit()):
            return self._lex_number()
        if ch.isalpha() or ch == "_":
            return self._lex_ident()
        if ch in _PUNCTUATION:
            start = self._here()
            self._advance()
            return Token(_PUNCTUATION[ch], ch, ch, Span(start, self._here()))

        # Anything else is not part of the language.
        start = self._here()
        self._advance()
        raise BlueprintSyntaxError(
            f"unexpected character {ch!r}",
            Span(start, self._here()),
            label="not valid here",
            hint='blueprints are made of blocks like:  target "name" { key = value }',
        )

    def _lex_string(self) -> Token:
        start = self._here()
        self._advance()  # opening quote
        decoded: List[str] = []
        while True:
            if self._at_end() or self._peek() == "\n":
                raise BlueprintSyntaxError(
                    "unterminated string literal",
                    Span.point(start),
                    label="this string is never closed",
                    hint='strings cannot span lines; add a closing double quote (")',
                )
            ch = self._peek()
            if ch == '"':
                self._advance()  # closing quote
                end = self._here()
                text = self.source[start.offset : self._pos]
                return Token(TokenKind.STRING, text, "".join(decoded), Span(start, end))
            if ch == "\\":
                esc_start = self._here()
                self._advance()  # backslash
                if self._at_end() or self._peek() == "\n":
                    raise BlueprintSyntaxError(
                        "unterminated string literal",
                        Span.point(start),
                        label="this string is never closed",
                        hint='strings cannot span lines; add a closing double quote (")',
                    )
                esc = self._peek()
                if esc in _STRING_ESCAPES:
                    decoded.append(_STRING_ESCAPES[esc])
                    self._advance()
                else:
                    self._advance()
                    raise BlueprintSyntaxError(
                        f"invalid escape sequence '\\{esc}'",
                        Span(esc_start, self._here()),
                        label="unknown escape",
                        hint=r'valid escapes are  \"  \\  \n  \t  \r  \/',
                    )
            else:
                decoded.append(ch)
                self._advance()

    def _lex_number(self) -> Token:
        start = self._here()
        if self._peek() == "-":
            self._advance()
        while self._peek().isdigit():
            self._advance()
        is_float = False
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self._advance()  # dot
            while self._peek().isdigit():
                self._advance()
        if self._peek() in ("e", "E"):
            nxt, nxt2 = self._peek(1), self._peek(2)
            if nxt.isdigit() or (nxt in "+-" and nxt2.isdigit()):
                is_float = True
                self._advance()  # e/E
                if self._peek() in "+-":
                    self._advance()
                while self._peek().isdigit():
                    self._advance()
        end = self._here()
        text = self.source[start.offset : self._pos]
        value: object = float(text) if is_float else int(text)
        return Token(TokenKind.NUMBER, text, value, Span(start, end))

    def _lex_ident(self) -> Token:
        start = self._here()
        while True:
            ch = self._peek()
            if ch.isalnum() or ch == "_":
                self._advance()
            else:
                break
        end = self._here()
        text = self.source[start.offset : self._pos]
        if text == "true":
            return Token(TokenKind.BOOL, text, True, Span(start, end))
        if text == "false":
            return Token(TokenKind.BOOL, text, False, Span(start, end))
        return Token(TokenKind.IDENT, text, text, Span(start, end))


def tokenize(source: str, filename: str = "blueprint.aero") -> List[Token]:
    """Convenience wrapper: tokenize ``source`` in one call."""
    return Lexer(source, filename).tokenize()
