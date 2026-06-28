# -*- coding: utf-8 -*-
"""Polyglot target emitters for UAST-to-source code generation.

This package provides a generic code-generation interface that accepts a
linearized UAST node list and a scope graph, then emits syntactically valid
source code in any supported target language.
"""

from .base import BaseEmitter, EmitterError, EmitterRegistry, get_emitter
from .python_emitter import PythonEmitter
from .rust_emitter import RustEmitter
from .cpp_emitter import CppEmitter

__all__ = [
    "BaseEmitter",
    "EmitterError",
    "EmitterRegistry",
    "get_emitter",
    "PythonEmitter",
    "RustEmitter",
    "CppEmitter",
]
