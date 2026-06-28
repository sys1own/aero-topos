"""
Static analysis of ingested code.

Provides a lightweight, dependency-free analysis of Python sources (via the
``ast`` module) and a best-effort regex scan of Rust sources.  The findings feed
the repair pass: undefined names suggest missing imports, unused imports can be
removed, and un-annotated functions are candidates for type inference.

The analysis is intentionally conservative and heuristic -- it is meant to catch
common, obvious issues in imported code, not to be a full type checker.
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.utils.serialization import dataclass_to_dict

_BUILTINS: Set[str] = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "self", "cls"}


@dataclass
class FileFindings:
    """Static-analysis findings for a single source file."""

    path: str
    language: str
    undefined_names: List[str] = field(default_factory=list)
    unused_imports: List[str] = field(default_factory=list)
    functions_missing_return_type: List[str] = field(default_factory=list)
    syntax_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


class CodeAnalyser:
    """Analyses ingested source files and reports repairable issues."""

    def analyse(self, source: str, path: str, language: str) -> FileFindings:
        if language == "python":
            return self._analyse_python(source, path)
        if language == "rust":
            return self._analyse_rust(source, path)
        return FileFindings(path=path, language=language)

    # ------------------------------------------------------------------
    # Python
    # ------------------------------------------------------------------

    def _analyse_python(self, source: str, path: str) -> FileFindings:
        findings = FileFindings(path=path, language="python")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            findings.syntax_error = f"line {exc.lineno}: {exc.msg}"
            return findings

        imported: Dict[str, ast.AST] = {}
        defined: Set[str] = set()
        used: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    imported[bound] = node
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    imported[bound] = node
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined.add(node.id)
                else:
                    used.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                used.add(node.value.id)

        findings.undefined_names = sorted(
            name
            for name in used
            if name not in defined and name not in imported and name not in _BUILTINS
        )
        findings.unused_imports = sorted(
            name for name in imported if name not in used
        )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is None:
                findings.functions_missing_return_type.append(node.name)
        return findings

    # ------------------------------------------------------------------
    # Rust (best-effort regex scan)
    # ------------------------------------------------------------------

    def _analyse_rust(self, source: str, path: str) -> FileFindings:
        findings = FileFindings(path=path, language="rust")
        # Surface unused `use` statements heuristically: a `use a::b::Name;`
        # whose final identifier never appears again in the file.
        for match in re.finditer(r"^\s*use\s+[\w:]+::(\w+);", source, re.MULTILINE):
            name = match.group(1)
            # Count occurrences beyond the import line itself.
            if len(re.findall(rf"\b{re.escape(name)}\b", source)) <= 1:
                findings.unused_imports.append(name)
        return findings
