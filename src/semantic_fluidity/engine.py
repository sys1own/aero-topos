"""
``ContextIngestionEngine`` -- the Domain-Agnostic Semantic Fluidity engine.

Ingests a directory of mixed, unstructured files (papers, notes, structured
JSON, and raw source code) and turns them into a single, domain-namespaced
:class:`~src.semantic_fluidity.schema.InvariantSchema` plus a
:class:`~src.semantic_fluidity.graph.SystemGraph`, ready to be exposed to code
generator nodes as high-level compilation inputs (see
:meth:`ContextIngestionEngine.to_compilation_inputs`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.semantic_fluidity.documents import DocumentLoader, IngestedDocument
from src.semantic_fluidity.domain import infer_domain
from src.semantic_fluidity.extractors.base import LLMClient
from src.semantic_fluidity.extractors.code_rules import CodeRuleExtractor
from src.semantic_fluidity.extractors.json_rules import JsonRuleExtractor
from src.semantic_fluidity.extractors.text_rules import TextRuleExtractor
from src.semantic_fluidity.graph import SystemGraph
from src.semantic_fluidity.schema import InvariantSchema, validate_invariant_document


class ContextIngestionEngine:
    """Reads a directory of mixed files and extracts the Invariant Schema."""

    REPORT_NAME = "invariant_schema_report.json"

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.loader = DocumentLoader()
        self.text_extractor = TextRuleExtractor()
        self.code_extractor = CodeRuleExtractor()
        self.json_extractor = JsonRuleExtractor()
        # Extensibility point (requirement #2): a real LLMClient can be wired
        # in by a caller; every extractor above remains fully offline.
        self.llm_client = llm_client
        self.last_documents: list = []
        self.last_graph: Optional[SystemGraph] = None
        self.last_errors: list = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_directory(self, directory: Path) -> InvariantSchema:
        documents = self.loader.load_directory(Path(directory))
        self.last_documents = documents
        self.last_errors = [f"{doc.path}: {doc.error}" for doc in documents if doc.error]

        schema = InvariantSchema()
        for document in documents:
            if document.error:
                continue
            schema.merge(self._extract_document(document))
        schema.finalize()

        graph = SystemGraph()
        graph.add_schema(schema)
        graph.auto_link_shared_symbols()
        self.last_graph = graph
        return schema

    def _extract_document(self, document: IngestedDocument) -> InvariantSchema:
        domain = infer_domain(document)
        extractor = self._extractor_for(document)
        return extractor.extract(document, domain)

    def _extractor_for(self, document: IngestedDocument):
        if document.format.startswith("code:"):
            return self.code_extractor
        if document.format == "json":
            return self.json_extractor
        return self.text_extractor

    # ------------------------------------------------------------------
    # Code-generator-facing output
    # ------------------------------------------------------------------

    def to_compilation_inputs(self, schema: InvariantSchema, graph: Optional[SystemGraph] = None) -> Dict[str, Any]:
        """Shape the engine's output as high-level compilation inputs.

        This is the dict a code generator node consumes: the invariant schema
        itself, the system graph it lives in, and any per-file ingestion
        errors that were skipped rather than aborting the whole run.
        """
        active_graph = graph or self.last_graph or SystemGraph()
        document = schema.to_dict()
        errors = validate_invariant_document(document)
        if errors:  # pragma: no cover - defensive; schema.to_dict() is internally consistent
            raise ValueError(f"generated invariant document failed validation: {errors}")
        return {
            "invariant_schema": document,
            "system_graph": active_graph.to_dict(),
            "graph_statistics": active_graph.statistics(),
            "ingestion_errors": list(self.last_errors),
        }

    def ingest_and_export(self, directory: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Ingest ``directory`` and write the compilation inputs to disk as JSON."""
        schema = self.ingest_directory(directory)
        payload = self.to_compilation_inputs(schema)
        target = Path(output_path) if output_path else Path(directory) / self.REPORT_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
