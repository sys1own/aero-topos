"""
Domain-Agnostic Semantic Fluidity engine.

Ingests entirely unstructured textual context (medical papers, math write-ups,
economics prose) alongside raw source code, extracts state variables,
algorithmic boundaries and mathematical equations into a JSON-based
"Invariant Schema", and maps everything into a single :class:`SystemGraph` so
that wildly different domains (genomics text, a game-engine library, ...)
never have their definitions conflict -- each domain is namespaced -- while
still being connected as interdependent nodes the synthesis layer can reason
about together.

Typical use::

    from src.semantic_fluidity import ContextIngestionEngine

    engine = ContextIngestionEngine()
    schema = engine.ingest_directory("context_sources/")
    compilation_inputs = engine.to_compilation_inputs(schema)
"""

from src.semantic_fluidity.documents import DocumentLoader, IngestedDocument
from src.semantic_fluidity.domain import DOMAIN_KEYWORDS, infer_domain
from src.semantic_fluidity.engine import ContextIngestionEngine
from src.semantic_fluidity.extractors import (
    CodeRuleExtractor,
    JsonRuleExtractor,
    LLMAssistedExtractor,
    LLMClient,
    NullLLMClient,
    RuleExtractor,
    TextRuleExtractor,
)
from src.semantic_fluidity.graph import SystemGraph
from src.semantic_fluidity.schema import (
    JSON_SCHEMA,
    AlgorithmicBoundary,
    Equation,
    InvariantSchema,
    SourceRef,
    StateVariable,
    make_id,
    validate_invariant_document,
)

__all__ = [
    "ContextIngestionEngine",
    "DocumentLoader",
    "IngestedDocument",
    "infer_domain",
    "DOMAIN_KEYWORDS",
    "RuleExtractor",
    "LLMClient",
    "NullLLMClient",
    "LLMAssistedExtractor",
    "TextRuleExtractor",
    "CodeRuleExtractor",
    "JsonRuleExtractor",
    "SystemGraph",
    "InvariantSchema",
    "StateVariable",
    "AlgorithmicBoundary",
    "Equation",
    "SourceRef",
    "make_id",
    "JSON_SCHEMA",
    "validate_invariant_document",
]

__version__ = "1.0.0"
