"""SQLite-backed storage for ingested AST metadata.

The registry is the immutable record of every source file AeroNova has ingested:
its language, on-disk path, a structural *semantic hash* (stable across
whitespace/comment edits), the serialised AST, and the functions/types it
declares.  It is the foundation the higher-level federation / dedup features
build on.

The table schema is intentionally close to the spec:

    id             INTEGER PRIMARY KEY
    context_name   TEXT       -- the [context_registry] entry this file belongs to
    semantic_hash  TEXT
    language       TEXT
    path           TEXT
    ast            TEXT       -- JSON
    functions      TEXT       -- JSON list
    types          TEXT       -- JSON list
    timestamp      DATETIME

``context_name`` is an addition to the bare spec so that ``ingest --list`` can
group files by the context they were registered under.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Union

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ast_entries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    context_name  TEXT NOT NULL,
    semantic_hash TEXT NOT NULL,
    language      TEXT NOT NULL,
    path          TEXT NOT NULL,
    ast           TEXT,
    functions     TEXT,
    types         TEXT,
    timestamp     DATETIME NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ast_context ON ast_entries (context_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ast_context_path ON ast_entries (context_name, path);
"""


@dataclass
class ASTEntry:
    """One ingested source file's metadata."""

    context_name: str
    semantic_hash: str
    language: str
    path: str
    ast: Any = None
    functions: Optional[List[str]] = None
    types: Optional[List[str]] = None
    timestamp: Optional[str] = None
    id: Optional[int] = None

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> "ASTEntry":
        return cls(
            id=row["id"],
            context_name=row["context_name"],
            semantic_hash=row["semantic_hash"],
            language=row["language"],
            path=row["path"],
            ast=json.loads(row["ast"]) if row["ast"] else None,
            functions=json.loads(row["functions"]) if row["functions"] else [],
            types=json.loads(row["types"]) if row["types"] else [],
            timestamp=row["timestamp"],
        )


class ASTDatabase:
    """A thin SQLite wrapper for storing and querying :class:`ASTEntry` records.

    Usable as a context manager::

        with ASTDatabase(path) as db:
            db.upsert(entry)
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        if self.db_path.parent and str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ASTDatabase":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- writes ---------------------------------------------------------------
    def upsert(self, entry: ASTEntry) -> int:
        """Insert (or replace) the entry keyed by (context_name, path).

        Re-ingesting the same file updates its row in place rather than
        accumulating duplicates.  Returns the row id.
        """
        timestamp = entry.timestamp or datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO ast_entries
                (context_name, semantic_hash, language, path, ast, functions, types, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(context_name, path) DO UPDATE SET
                semantic_hash = excluded.semantic_hash,
                language      = excluded.language,
                ast           = excluded.ast,
                functions     = excluded.functions,
                types         = excluded.types,
                timestamp     = excluded.timestamp
            """,
            (
                entry.context_name,
                entry.semantic_hash,
                entry.language,
                entry.path,
                json.dumps(entry.ast) if entry.ast is not None else None,
                json.dumps(entry.functions or []),
                json.dumps(entry.types or []),
                timestamp,
            ),
        )
        self._conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = self._conn.execute(
            "SELECT id FROM ast_entries WHERE context_name = ? AND path = ?",
            (entry.context_name, entry.path),
        ).fetchone()
        return int(row["id"])

    # -- reads ----------------------------------------------------------------
    def all_entries(self) -> List[ASTEntry]:
        rows = self._conn.execute(
            "SELECT * FROM ast_entries ORDER BY context_name, path"
        ).fetchall()
        return [ASTEntry._from_row(r) for r in rows]

    def entries_for_context(self, context_name: str) -> List[ASTEntry]:
        rows = self._conn.execute(
            "SELECT * FROM ast_entries WHERE context_name = ? ORDER BY path",
            (context_name,),
        ).fetchall()
        return [ASTEntry._from_row(r) for r in rows]

    def context_names(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT context_name FROM ast_entries ORDER BY context_name"
        ).fetchall()
        return [r["context_name"] for r in rows]
