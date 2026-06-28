"""Tests for the dependency reflux engine (src.decomposition.reflux)."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from src.decomposition import reflux


class TestAnnotationAwareDependencyScan(unittest.TestCase):
    """Type-signature names are treated as mandatory module parameters."""

    def _names(self, source: str) -> set:
        tree = ast.parse(source)
        return reflux._used_names(tree)

    def test_argument_annotations_are_collected(self):
        source = "def f(x: Optional[int]) -> None: pass\n"
        self.assertIn("Optional", self._names(source))

    def test_return_annotation_is_collected(self):
        source = "def f() -> Path: pass\n"
        self.assertIn("Path", self._names(source))

    def test_kwonly_and_vararg_annotations_are_collected(self):
        source = "def f(*args: List[int], **kwargs: Dict[str, Any]) -> None: pass\n"
        names = self._names(source)
        self.assertIn("List", names)
        self.assertIn("Dict", names)
        self.assertIn("Any", names)

    def test_async_function_annotations_are_collected(self):
        source = "async def f(x: Iterator[int]) -> Optional[int]: pass\n"
        names = self._names(source)
        self.assertIn("Iterator", names)
        self.assertIn("Optional", names)

    def test_pep563_string_annotations_are_collected(self):
        source = "def f(x: 'Optional[int]') -> 'Path': pass\n"
        self.assertIn("Optional", self._names(source))
        self.assertIn("Path", self._names(source))


class TestTypeHintFallbackInjection(unittest.TestCase):
    """Blanket typing/pathlib fallback is injected when type hints lack coverage."""

    def test_fallback_injects_when_type_hints_are_uncovered(self):
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "pkg"
            pkg.mkdir()
            (pkg / "leaf.py").write_text(
                "from __future__ import annotations\n\n"
                "def helper(p: Path) -> Optional[int]:\n    return None\n",
                encoding="utf-8",
            )
            result = reflux.run_reflux(pkg)
            text = (pkg / "leaf.py").read_text()
            self.assertIn("from typing import Optional, Union, Any, List, Dict, Callable, Iterator", text)
            self.assertIn("from pathlib import Path", text)
            self.assertLess(text.index("from __future__"), text.index("from typing"))
            self.assertLess(text.index("from typing"), text.index("def helper"))
            self.assertIn(str(pkg / "leaf.py"), result.files_patched)

    def test_fallback_skipped_when_typing_already_imported(self):
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "pkg"
            pkg.mkdir()
            (pkg / "leaf.py").write_text(
                "from typing import Optional\n\n"
                "def helper(p: Optional[int]) -> int:\n    return 1\n",
                encoding="utf-8",
            )
            result = reflux.run_reflux(pkg)
            text = (pkg / "leaf.py").read_text()
            self.assertNotIn("from pathlib import Path", text)
            self.assertNotIn(
                "from typing import Optional, Union, Any, List, Dict, Callable, Iterator", text
            )

    def test_fallback_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "pkg"
            pkg.mkdir()
            (pkg / "leaf.py").write_text(
                "from __future__ import annotations\n\n"
                "from typing import Optional, Union, Any, List, Dict, Callable, Iterator\n"
                "from pathlib import Path\n\n"
                "def helper(p: Path) -> Optional[int]:\n    return None\n",
                encoding="utf-8",
            )
            reflux.run_reflux(pkg)
            text = (pkg / "leaf.py").read_text()
            self.assertEqual(
                text.count("from typing import Optional, Union, Any, List, Dict, Callable, Iterator"), 1
            )
            self.assertEqual(text.count("from pathlib import Path"), 1)


class TestFutureImportHoistingInUtils(unittest.TestCase):
    """Compiler-flag __future__ imports are never duplicated into utils.py."""

    def test_future_imports_stay_in_leaf_modules(self):
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "pkg"
            pkg.mkdir()
            duplicate_body = "from __future__ import annotations\n\ndef shared():\n    return 1\n"
            (pkg / "a.py").write_text(duplicate_body, encoding="utf-8")
            (pkg / "b.py").write_text(duplicate_body, encoding="utf-8")
            result, _ = reflux.consolidate_shared_utils(pkg)
            self.assertTrue(result.utils_created)
            utils_text = (pkg / "utils.py").read_text()
            self.assertNotIn("from __future__", utils_text)
            for leaf in (pkg / "a.py", pkg / "b.py"):
                text = leaf.read_text()
                self.assertIn("from __future__ import annotations", text)


if __name__ == "__main__":
    unittest.main()
