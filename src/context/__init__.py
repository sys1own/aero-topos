"""Context ingestion: import, analyse, and repair external source trees.

This package implements the ``[context]`` blueprint feature: it copies (or
symlinks) external source directories into the workspace, runs a static analysis
pass, applies a configurable set of repair rules (add missing imports, drop
unused imports, infer simple type hints), and writes a
``context_analysis_report.json`` describing everything it did.
"""

from src.context.analyser import CodeAnalyser, FileFindings
from src.context.ingest import ContextIngestor
from src.context.repair import CodeRepairer

__all__ = ["ContextIngestor", "CodeAnalyser", "FileFindings", "CodeRepairer"]
