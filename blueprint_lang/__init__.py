# -*- coding: utf-8 -*-
"""``blueprint_lang`` -- the parser/validator for the ``blueprint.aero`` DSL.

A small, declarative, block-structured configuration language for describing a
multi-language build::

    project "my_universal_app" {
        version = "1.0.0"
    }

    target "core_engine" {
        language = "cpp"
        sources  = ["src/core/**/*.cpp", "src/core/**/*.hpp"]
        flags    = ["-O3", "-std=c++20"]
    }

    target "bindings" {
        language = "python"
        requires = ["core_engine"]
        sources  = ["src/bindings/*.py"]
    }

Pipeline: :class:`~blueprint_lang.lexer.Lexer` -> :class:`~blueprint_lang.parser.Parser`
-> :class:`~blueprint_lang.validator.Validator`.  Any problem raises a
:class:`~blueprint_lang.errors.BlueprintError` that
:func:`~blueprint_lang.errors.format_error` renders with a ``line:column`` and a
``^`` pointer.

Typical use as a strict pre-build gate::

    err = blueprint_lang.check_file("blueprint.aero")
    if err is not None:
        print(err, file=sys.stderr)
        raise SystemExit(2)          # abort BEFORE any build step
    blueprint = blueprint_lang.load_file("blueprint.aero")
"""

from __future__ import annotations

import os
import re
from typing import Optional

from .errors import (
    BlueprintError,
    BlueprintSyntaxError,
    BlueprintValidationError,
    format_error,
    format_errors,
)
from .lexer import Lexer, tokenize
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
from .parser import Parser, parse
from .positions import Position, Span
from .tokens import Token, TokenKind
from .validator import SUPPORTED_LANGUAGES, Validator, validate

__all__ = [
    # positions / tokens
    "Position",
    "Span",
    "Token",
    "TokenKind",
    # errors
    "BlueprintError",
    "BlueprintSyntaxError",
    "BlueprintValidationError",
    "format_error",
    "format_errors",
    # AST
    "Blueprint",
    "Block",
    "Field",
    "AnyValue",
    "StringValue",
    "NumberValue",
    "BoolValue",
    "ListValue",
    # stages
    "Lexer",
    "Parser",
    "Validator",
    "tokenize",
    "parse",
    "validate",
    "SUPPORTED_LANGUAGES",
    # high-level API
    "parse_source",
    "load_source",
    "load_file",
    "check_source",
    "check_file",
    "looks_like_blueprint_dsl",
]

__version__ = "1.0.0"

_DEFAULT_FILENAME = "blueprint.aero"


def parse_source(source: str, filename: str = _DEFAULT_FILENAME) -> Blueprint:
    """Lex + parse ``source`` into a :class:`Blueprint` (no semantic validation)."""
    return parse(source, filename)


def load_source(source: str, filename: str = _DEFAULT_FILENAME) -> Blueprint:
    """Parse *and* validate ``source``; raise on the first problem."""
    blueprint = parse(source, filename)
    Validator(blueprint).validate()
    return blueprint


def load_file(path: str) -> Blueprint:
    """Read, parse, and validate the blueprint at ``path``."""
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    return load_source(source, filename=path)


def check_source(source: str, filename: str = _DEFAULT_FILENAME) -> Optional[str]:
    """Validate ``source``.  Return ``None`` if valid, else a rendered error."""
    try:
        load_source(source, filename)
        return None
    except BlueprintError as error:
        return format_error(error, source, filename)


def check_file(path: str) -> Optional[str]:
    """Validate the blueprint at ``path``.  Return ``None`` if valid, else a message."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
    except OSError as exc:
        return f"error: cannot read blueprint file\n  --> {path}\n  = help: {exc.strerror or exc}"
    return check_source(source, filename=path)


# A top-level `ident "name" {` is the signature of the block DSL.  This lets the
# rest of the tool tell a block-format blueprint apart from the legacy INI
# (`[section]`) and JSON (`{`) formats without committing to a full parse.
_DSL_SIGNATURE = re.compile(r"^[ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]+\"[^\"]*\"[ \t]*\{", re.MULTILINE)


def looks_like_blueprint_dsl(source: str) -> bool:
    """Heuristically detect the block DSL (vs. the legacy INI/JSON formats)."""
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//") or line.startswith(";"):
            continue
        # First meaningful character decides INI ('[') and JSON ('{') quickly.
        if line[0] in "[{":
            return False
        break
    return _DSL_SIGNATURE.search(source) is not None
