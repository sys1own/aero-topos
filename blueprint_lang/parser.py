# -*- coding: utf-8 -*-
"""Recursive-descent parser for ``blueprint.aero``.

Consumes the token stream from :mod:`blueprint_lang.lexer` and builds the AST
in :mod:`blueprint_lang.nodes`.  Every failure path raises a
:class:`~blueprint_lang.errors.BlueprintSyntaxError` carrying a span, so callers
get a precise ``line:column`` and ``^`` pointer.  The parser handles purely
*structural* problems (missing braces/brackets, missing ``=``, unquoted values,
duplicate keys); semantic checks live in :mod:`blueprint_lang.validator`.
"""

from __future__ import annotations

from typing import Dict, List, NoReturn

from .errors import BlueprintSyntaxError
from .lexer import Lexer
from .nodes import (
    AnyValue,
    Block,
    Blueprint,
    BoolValue,
    Field,
    ListValue,
    NumberValue,
    StringValue,
)
from .positions import Span
from .tokens import Token, TokenKind


class Parser:
    """Parse a token list into a :class:`~blueprint_lang.nodes.Blueprint`."""

    def __init__(
        self,
        tokens: List[Token],
        source: str,
        filename: str = "blueprint.aero",
    ) -> None:
        self.tokens = tokens
        self.source = source
        self.filename = filename
        self._index = 0

    # -- token cursor ------------------------------------------------------

    def _peek(self) -> Token:
        return self.tokens[self._index]

    def _advance(self) -> Token:
        tok = self.tokens[self._index]
        if tok.kind is not TokenKind.EOF:
            self._index += 1
        return tok

    def _at(self, kind: TokenKind) -> bool:
        return self._peek().kind is kind

    def _unexpected(self, tok: Token, expected: str) -> NoReturn:
        if tok.kind is TokenKind.EOF:
            raise BlueprintSyntaxError(
                f"unexpected end of file, expected {expected}",
                tok.span,
                label="the file ends here",
            )
        raise BlueprintSyntaxError(
            f"expected {expected}, but found {tok.describe()}",
            tok.span,
            label="unexpected here",
        )

    def _expect(self, kind: TokenKind, expected: str) -> Token:
        if self._peek().kind is not kind:
            self._unexpected(self._peek(), expected)
        return self._advance()

    # -- grammar -----------------------------------------------------------

    def parse(self) -> Blueprint:
        blocks: List[Block] = []
        while not self._at(TokenKind.EOF):
            blocks.append(self._parse_block())
        return Blueprint(blocks, self.source, self.filename)

    def _parse_block(self) -> Block:
        type_tok = self._peek()
        if type_tok.kind is not TokenKind.IDENT:
            self._unexpected(
                type_tok, "a block type such as 'project' or 'target'"
            )
        self._advance()

        name_tok = self._peek()
        if name_tok.kind is not TokenKind.STRING:
            self._unexpected(
                name_tok, f"a quoted name after '{type_tok.text}'"
            )
        self._advance()

        lbrace = self._expect(
            TokenKind.LBRACE, f"'{{' to open the '{type_tok.text}' block"
        )

        fields: List[Field] = []
        seen: Dict[str, Field] = {}
        while not self._at(TokenKind.RBRACE):
            if self._at(TokenKind.EOF):
                raise BlueprintSyntaxError(
                    f"missing closing '}}': the '{type_tok.text}' block is never closed",
                    lbrace.span,
                    label="this block is opened but never closed",
                    hint="add a '}' to close the block",
                )
            field = self._parse_field()
            previous = seen.get(field.key)
            if previous is not None:
                raise BlueprintSyntaxError(
                    f"duplicate key '{field.key}' in '{type_tok.text}' block",
                    field.key_span,
                    label="defined a second time here",
                    hint=f"'{field.key}' was already set on line "
                    f"{previous.key_span.start.line}",
                )
            seen[field.key] = field
            fields.append(field)

        rbrace = self._advance()  # the '}'
        block_span = Span(type_tok.span.start, rbrace.span.end)
        return Block(
            type=type_tok.text,
            type_span=type_tok.span,
            name=name_tok.value,
            name_span=name_tok.span,
            fields=fields,
            span=block_span,
        )

    def _parse_field(self) -> Field:
        key_tok = self._peek()
        if key_tok.kind is not TokenKind.IDENT:
            if key_tok.kind is TokenKind.STRING:
                self._unexpected(key_tok, "a key name (keys are written without quotes)")
            self._unexpected(key_tok, "a key name")
        self._advance()

        self._expect(TokenKind.EQUALS, f"'=' after the key '{key_tok.text}'")
        value = self._parse_value()
        return Field(
            key=key_tok.text,
            key_span=key_tok.span,
            value=value,
            span=Span(key_tok.span.start, value.span.end),
        )

    def _parse_value(self) -> AnyValue:
        tok = self._peek()
        if tok.kind is TokenKind.STRING:
            self._advance()
            return StringValue(tok.value, tok.span)
        if tok.kind is TokenKind.NUMBER:
            self._advance()
            return NumberValue(tok.value, tok.span)
        if tok.kind is TokenKind.BOOL:
            self._advance()
            return BoolValue(tok.value, tok.span)
        if tok.kind is TokenKind.LBRACKET:
            return self._parse_list()
        if tok.kind is TokenKind.IDENT:
            # A bareword where a value is expected -- almost always a missing
            # pair of quotes (e.g. `language = cpp`).
            raise BlueprintSyntaxError(
                f"expected a value but found the bareword '{tok.text}'",
                tok.span,
                label="values must be quoted",
                hint=f'did you mean "{tok.text}"?  text values need double quotes',
            )
        self._unexpected(tok, "a value (string, number, boolean, or list)")

    def _parse_list(self) -> ListValue:
        lbracket = self._expect(TokenKind.LBRACKET, "'['")
        items: List[AnyValue] = []
        while not self._at(TokenKind.RBRACKET):
            # EOF or a block-closing '}' here means the ']' was forgotten
            # (commonly after a trailing comma).
            if self._at(TokenKind.EOF) or self._at(TokenKind.RBRACE):
                raise BlueprintSyntaxError(
                    "missing closing ']': this list is never closed",
                    lbracket.span,
                    label="this list is opened but never closed",
                    hint="add a ']' to close the list",
                )
            items.append(self._parse_value())
            if self._at(TokenKind.COMMA):
                self._advance()
                continue
            if self._at(TokenKind.RBRACKET):
                break
            self._unexpected(self._peek(), "',' to separate items, or ']' to close the list")
        rbracket = self._advance()  # the ']'
        return ListValue(items, Span(lbracket.span.start, rbracket.span.end))


def parse(source: str, filename: str = "blueprint.aero") -> Blueprint:
    """Tokenize and parse ``source`` into a :class:`Blueprint` (no validation)."""
    tokens = Lexer(source, filename).tokenize()
    return Parser(tokens, source, filename).parse()
