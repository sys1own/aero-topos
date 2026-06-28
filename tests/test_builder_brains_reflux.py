"""Tests for AeroDependencyRefluxEngine's __future__ import precedence."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from builder_brains.reflux import AeroDependencyRefluxEngine


class TestFutureImportPrecedence(unittest.TestCase):
    """Injected imports must never displace an existing __future__ statement."""

    def _apply(self, source: str, action: dict) -> str:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as handle:
            handle.write(source)
            path = handle.name
        engine = AeroDependencyRefluxEngine()
        out = engine.apply_reflux_patches(path, [action])
        Path(path).unlink(missing_ok=True)
        return out.decode("utf-8")

    def test_docstring_then_future_then_inject_stays_valid(self):
        source = (
            '"""Mod."""\n'
            "from __future__ import annotations\n\n"
            "import sys\n\n"
            "def f():\n"
            "    return zzz()\n"
        )
        result = self._apply(
            source, {"action": "RESOLVE_UNDEFINED_SYMBOL", "symbol": "zzz"}
        )
        ast.parse(result)  # raises SyntaxError on failure
        self.assertTrue(result.lstrip().startswith('"""Mod."""'))

    def test_future_immediately_followed_by_docstring_layout(self):
        # Mirrors the layout produced by src.decomposition.splitter for
        # auto-decomposed leaf modules: future import directly on line 1,
        # generated docstring on line 2, no blank line between them.
        source = (
            "from __future__ import annotations\n"
            '"""Auto-decomposed."""\n\n'
            "import os\n\n"
            "class Bar:\n"
            "    pass\n"
        )
        result = self._apply(
            source, {"action": "AUTO_REFLUX_IMPORT", "target": "main_foo"}
        )
        tree = ast.parse(result)
        first_stmt = tree.body[0]
        self.assertIsInstance(first_stmt, ast.ImportFrom)
        self.assertEqual(first_stmt.module, "__future__")

    def test_no_docstring_future_only(self):
        source = (
            "from __future__ import annotations\n\n"
            "def f():\n"
            "    return zzz()\n"
        )
        result = self._apply(
            source, {"action": "RESOLVE_UNDEFINED_SYMBOL", "symbol": "zzz"}
        )
        tree = ast.parse(result)
        first_stmt = tree.body[0]
        self.assertIsInstance(first_stmt, ast.ImportFrom)
        self.assertEqual(first_stmt.module, "__future__")

    def test_multiple_future_statements_preserved_and_pinned(self):
        source = (
            "from __future__ import annotations\n"
            "from __future__ import division\n\n"
            "import os\n\n"
            "def f():\n"
            "    return zzz()\n"
        )
        result = self._apply(
            source, {"action": "RESOLVE_UNDEFINED_SYMBOL", "symbol": "zzz"}
        )
        tree = ast.parse(result)
        future_stmts = [
            n
            for n in tree.body
            if isinstance(n, ast.ImportFrom) and n.module == "__future__"
        ]
        self.assertEqual(len(future_stmts), 2)
        self.assertIsInstance(tree.body[0], ast.ImportFrom)
        self.assertEqual(tree.body[0].module, "__future__")
        self.assertEqual(tree.body[1].module, "__future__")


if __name__ == "__main__":
    unittest.main()
