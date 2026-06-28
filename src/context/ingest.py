"""
Context Ingestion.

Reads the ``[context]`` blueprint section -- a list of external source trees,
each with a ``path``, ``language``, ``purpose``, ``repair_rules`` and
``target_mapping`` -- copies (or symlinks) the matching files into the
workspace, runs the analyser/repair passes on them, and writes a
``context_analysis_report.json`` describing everything that happened.

Missing sources and per-file errors are recorded in the report rather than
raising, so one bad source never aborts the whole ingestion.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.context.analyser import CodeAnalyser
from src.context.repair import CodeRepairer
from src.utils.serialization import dataclass_to_dict

# Default include globs per language.
_LANG_GLOBS = {
    "python": ["**/*.py"],
    "rust": ["**/*.rs"],
    "c": ["**/*.c", "**/*.h"],
    "cpp": ["**/*.cpp", "**/*.hpp", "**/*.cc"],
}
_IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", "target", "node_modules", ".aero"}


@dataclass
class IngestedFile:
    source: str
    destination: str
    language: str
    mode: str
    repairs: List[str] = field(default_factory=list)
    findings: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


class ContextIngestor:
    """Imports, analyses and repairs external source trees into the workspace."""

    REPORT_NAME = "context_analysis_report.json"

    def __init__(self, config: Dict[str, Any], workspace: Path) -> None:
        self.config = config or {}
        self.workspace = Path(workspace)
        self.analyser = CodeAnalyser()
        self.repairer = CodeRepairer()

    # ------------------------------------------------------------------

    def _sources(self) -> List[Dict[str, Any]]:
        context = self.config.get("context", {}) or {}
        if isinstance(context, list):
            return [s for s in context if isinstance(s, dict)]
        return [s for s in context.get("sources", []) if isinstance(s, dict)]

    def ingest_all(self, write_report: bool = True) -> Dict[str, Any]:
        """Ingest every configured source; return (and optionally write) a report."""
        sources = self._sources()
        report: Dict[str, Any] = {
            "timestamp": time.time(),
            "workspace": str(self.workspace),
            "source_count": len(sources),
            "sources": [],
            "files_ingested": 0,
            "files_repaired": 0,
            "errors": [],
        }

        for spec in sources:
            source_report = self._ingest_source(spec)
            report["sources"].append(source_report)
            report["files_ingested"] += source_report["files_ingested"]
            report["files_repaired"] += source_report["files_repaired"]
            if source_report.get("error"):
                report["errors"].append(source_report["error"])

        if write_report:
            self.workspace.mkdir(parents=True, exist_ok=True)
            (self.workspace / self.REPORT_NAME).write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
        return report

    # ------------------------------------------------------------------

    def _ingest_source(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        raw_path = spec.get("path", "")
        language = str(spec.get("language", "python")).lower()
        purpose = spec.get("purpose", "")
        repair_rules = list(spec.get("repair_rules", []))
        target_mapping = spec.get("target_mapping", f"imported/{Path(raw_path).name}")
        mode = str(spec.get("mode", "copy")).lower()

        source_root = Path(raw_path)
        if not source_root.is_absolute():
            source_root = (self.workspace / raw_path).resolve()

        result: Dict[str, Any] = {
            "path": str(source_root),
            "language": language,
            "purpose": purpose,
            "target_mapping": target_mapping,
            "repair_rules": repair_rules,
            "files_ingested": 0,
            "files_repaired": 0,
            "files": [],
            "error": None,
        }

        if not source_root.exists():
            result["error"] = f"context source not found: {source_root}"
            return result

        # A source entry may be a single file living anywhere on the filesystem
        # (e.g. /content/lib.rs, ../data/core.rs) -- read it from that exact
        # location and copy it into the workspace without requiring a directory
        # layout or any file relocation.
        if source_root.is_file():
            dest = self._single_file_destination(source_root, target_mapping)
            ingested = self._ingest_file(
                source_root, source_root.parent, dest.parent, language, repair_rules, mode,
                dest_override=dest,
            )
            result["files"].append(ingested.to_dict())
            result["files_ingested"] += 1
            if ingested.repairs:
                result["files_repaired"] += 1
            return result

        target_root = self.workspace / target_mapping
        include = spec.get("include") or _LANG_GLOBS.get(language, ["**/*"])
        exclude = spec.get("exclude") or []

        for file_path in self._collect_files(source_root, include, exclude):
            ingested = self._ingest_file(file_path, source_root, target_root, language, repair_rules, mode)
            result["files"].append(ingested.to_dict())
            result["files_ingested"] += 1
            if ingested.repairs:
                result["files_repaired"] += 1
        return result

    def _single_file_destination(self, source_file: Path, target_mapping: str) -> Path:
        """Resolve where a single-file source should land in the workspace.

        ``target_mapping`` may name a file (has a suffix) or a directory; either
        way the result is the concrete destination *file* path.
        """
        mapping = Path(target_mapping)
        if mapping.suffix:  # e.g. "src/lib.rs" -> a file destination
            return self.workspace / mapping
        return self.workspace / mapping / source_file.name

    def _ingest_file(
        self,
        file_path: Path,
        source_root: Path,
        target_root: Path,
        language: str,
        repair_rules: List[str],
        mode: str,
        dest_override: Optional[Path] = None,
    ) -> IngestedFile:
        dest = dest_override if dest_override is not None else target_root / file_path.relative_to(source_root)
        record = IngestedFile(
            source=str(file_path), destination=str(dest), language=language, mode=mode
        )
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Python files with repair rules are always copied (repaired content
            # cannot be a symlink); everything else honours the requested mode.
            if language == "python" and repair_rules:
                source_text = file_path.read_text(encoding="utf-8")
                findings = self.analyser.analyse(source_text, str(file_path), language)
                record.findings = findings.to_dict()
                repaired = self.repairer.repair(source_text, repair_rules, str(file_path))
                dest.write_text(repaired.source, encoding="utf-8")
                record.repairs = repaired.changes
            elif mode == "symlink":
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                os.symlink(file_path.resolve(), dest)
            else:
                shutil.copy2(file_path, dest)
                if language in ("python", "rust"):
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    record.findings = self.analyser.analyse(text, str(file_path), language).to_dict()
        except Exception as exc:  # noqa: BLE001 - per-file isolation
            record.error = f"{type(exc).__name__}: {exc}"
        return record

    @staticmethod
    def _collect_files(root: Path, include: List[str], exclude: List[str]) -> List[Path]:
        seen: List[Path] = []
        for pattern in include:
            for path in sorted(root.glob(pattern)):
                if not path.is_file():
                    continue
                if any(part in _IGNORED_DIRS for part in path.parts):
                    continue
                if any(path.match(ex) for ex in exclude):
                    continue
                if path not in seen:
                    seen.append(path)
        return seen
