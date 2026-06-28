# -*- coding: utf-8 -*-
"""Tests for AST-driven modular decomposition (``decomposition_mode``).

Layers:
* ``TestModularDecomposer``      -- direct AST extraction, imports, orchestrator.
* ``TestDecomposerErrors``       -- missing nodes / collisions / bad input.
* ``TestEngineModular``          -- ScaffoldEngine 'modular_package' routing.
* ``TestPipelineModular``        -- blueprint-driven decomposition pipeline.
* ``TestParserModuleMapping``    -- [scaffold] module_mapping / decomposition_mode.
"""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from blueprint_parser import (
    BlueprintParseError,
    normalize_analysis_block,
    normalize_optional_sections,
)
from src.scaffold import ScaffoldEngine
from src.scaffold.decomposition import (
    DecompositionError,
    ImportCollisionError,
    MissingASTNodeError,
    ModularDecomposer,
)
from src.scaffold.decomposition import merge_source_asts, resolve_cross_imports
from src.scaffold.import_pruner import prune_dead_imports, render_imports
from src.scaffold.pipeline import ScaffoldBuildPipeline
from src.scaffold.test_matrix import generate_test_matrix

MONOLITH = '''\
# -*- coding: utf-8 -*-
"""A monolithic script to be decomposed."""
from __future__ import annotations
import os
import json

GLOBAL_CONST = 42


class SchemaValidator:
    """Validate things."""

    def check(self, blob):
        return json.dumps(blob)


@staticmethod
def shield(value):
    return os.fspath(value)


def parse_blueprint(text):
    return SchemaValidator().check({"text": text})


def main():
    return parse_blueprint("hello")
'''


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        self.addCleanup(self._t.cleanup)


# ---------------------------------------------------------------------------
# Direct decomposer
# ---------------------------------------------------------------------------


class TestModularDecomposer(_Tmp):
    def _decompose(self, mapping, **kwargs):
        dest = self.tmp / "dist"
        decomposer = ModularDecomposer(verbose=False)
        result = decomposer.decompose(
            MONOLITH, mapping, source_filename="main.py", dest_dir=dest, **kwargs
        )
        return dest, result

    def test_extracts_classes_and_functions_into_targets(self):
        dest, result = self._decompose(
            {"parser": ["SchemaValidator", "parse_blueprint"], "shielder": ["shield"]}
        )
        self.assertTrue((dest / "parser.py").exists())
        self.assertTrue((dest / "shielder.py").exists())
        parser_src = (dest / "parser.py").read_text()
        self.assertIn("class SchemaValidator", parser_src)
        self.assertIn("def parse_blueprint", parser_src)
        self.assertIn("def shield", (dest / "shielder.py").read_text())
        # parse_blueprint did not leak into the shielder module.
        self.assertNotIn("parse_blueprint", (dest / "shielder.py").read_text())

    def test_decorators_are_preserved_during_extraction(self):
        dest, _ = self._decompose({"shielder": ["shield"]})
        self.assertIn("@staticmethod", (dest / "shielder.py").read_text())

    def test_global_imports_duplicated_into_every_module(self):
        dest, _ = self._decompose(
            {"parser": ["SchemaValidator"], "shielder": ["shield"]}
        )
        for name in ("parser.py", "shielder.py"):
            text = (dest / name).read_text()
            self.assertIn("import os", text)
            self.assertIn("import json", text)
            self.assertIn("from __future__ import annotations", text)
            # __future__ must precede any other import.
            self.assertLess(text.index("from __future__"), text.index("import os"))

    def test_generates_empty_package_init(self):
        dest, result = self._decompose({"parser": ["SchemaValidator"]})
        init = dest / "__init__.py"
        self.assertTrue(init.exists())
        self.assertEqual(init.read_text(), "")
        self.assertEqual(result.package_init, "__init__.py")

    def test_orchestrator_rewrites_internal_imports(self):
        dest, result = self._decompose(
            {"parser": ["SchemaValidator", "parse_blueprint"], "shielder": ["shield"]}
        )
        main_src = (dest / "main.py").read_text()
        self.assertIn("from .parser import SchemaValidator, parse_blueprint", main_src)
        self.assertIn("from .shielder import shield", main_src)
        # Moved defs are gone from the orchestrator, but the non-mapped code stays.
        self.assertNotIn("class SchemaValidator", main_src)
        self.assertNotIn("def parse_blueprint", main_src)
        self.assertIn("GLOBAL_CONST = 42", main_src)
        self.assertIn("def main():", main_src)

    def test_orchestrator_preserves_module_docstring_first(self):
        dest, _ = self._decompose({"parser": ["SchemaValidator"]})
        main_src = (dest / "main.py").read_text()
        # __future__ stays ahead of the generated import block.
        self.assertLess(
            main_src.index("from __future__"),
            main_src.index("from .parser import"),
        )

    def test_orchestrator_hoists_future_above_docstring(self):
        """Future imports are hoisted above the preserved module docstring."""
        dest, _ = self._decompose({"parser": ["SchemaValidator"]})
        main_src = (dest / "main.py").read_text()
        self.assertLess(
            main_src.index("from __future__"),
            main_src.index("A monolithic script to be decomposed."),
        )

    def test_all_generated_files_are_syntactically_valid(self):
        import ast as _ast

        dest, result = self._decompose(
            {"parser": ["SchemaValidator", "parse_blueprint"], "shielder": ["shield"]}
        )
        for name in result.files:
            _ast.parse((dest / name).read_text())  # raises on bad syntax

    def test_filename_normalization_accepts_dot_py(self):
        dest, _ = self._decompose({"parser.py": ["SchemaValidator"]})
        self.assertTrue((dest / "parser.py").exists())

    def test_result_to_dict_reports_structure(self):
        _, result = self._decompose(
            {"parser": ["SchemaValidator", "parse_blueprint"]}
        )
        data = result.to_dict()
        self.assertEqual(data["mode"], "modular_package")
        self.assertIn("__init__.py", data["files"])
        module = next(m for m in data["modules"] if m["filename"] == "parser.py")
        self.assertEqual(module["classes"], ["SchemaValidator"])
        self.assertEqual(module["functions"], ["parse_blueprint"])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestDecomposerErrors(_Tmp):
    def _run(self, source, mapping):
        ModularDecomposer(verbose=False).decompose(
            source, mapping, dest_dir=self.tmp / "out"
        )

    def test_missing_ast_node_raises(self):
        with self.assertRaises(MissingASTNodeError):
            self._run(MONOLITH, {"parser": ["DoesNotExist"]})

    def test_cross_module_collision_raises(self):
        with self.assertRaises(ImportCollisionError):
            self._run(MONOLITH, {"a": ["SchemaValidator"], "b": ["SchemaValidator"]})

    def test_empty_mapping_raises(self):
        with self.assertRaises(DecompositionError):
            self._run(MONOLITH, {})

    def test_unparseable_source_raises(self):
        with self.assertRaises(DecompositionError):
            self._run("def broken(:\n", {"x": ["broken"]})

    def test_init_target_is_rejected(self):
        with self.assertRaises(DecompositionError):
            self._run(MONOLITH, {"__init__": ["SchemaValidator"]})


# ---------------------------------------------------------------------------
# Engine routing
# ---------------------------------------------------------------------------


class TestEngineModular(_Tmp):
    def test_engine_routes_to_modular_decomposition(self):
        src = self.tmp / "main.py"
        src.write_text(MONOLITH)
        dist = self.tmp / "pkg"
        msgs = []
        engine = ScaffoldEngine(logger=msgs.append, verbose=True)
        result = engine.scaffold(
            source_entry=str(src),
            distribution_directory=dist,
            language="python",
            module_mapping={"parser": ["SchemaValidator", "parse_blueprint"]},
            decomposition_mode="modular_package",
            build=True,
        )
        self.assertEqual(result.language, "python")
        self.assertTrue((dist / "parser.py").exists())
        self.assertTrue((dist / "__init__.py").exists())
        self.assertIn("decomposition", result.repo)
        self.assertTrue(result.build["succeeded"])
        # Structured decomposition logs surfaced.
        joined = "\n".join(msgs)
        self.assertIn("[Decomposing] Extracted class 'SchemaValidator'", joined)
        self.assertIn("[Scaffold   ] Initialized package boundary __init__.py", joined)

    def test_engine_without_mode_uses_plain_python_layout(self):
        src = self.tmp / "main.py"
        src.write_text(MONOLITH)
        dist = self.tmp / "plain"
        result = ScaffoldEngine().scaffold(
            source_entry=str(src),
            distribution_directory=dist,
            language="python",
        )
        # Single-file copy, not a decomposed package.
        self.assertTrue((dist / "pyproject.toml").exists())
        self.assertFalse((dist / "__init__.py").exists())


# ---------------------------------------------------------------------------
# Blueprint pipeline
# ---------------------------------------------------------------------------


class TestPipelineModular(_Tmp):
    def test_pipeline_decomposes_from_blueprint_context(self):
        src = self.tmp / "main.py"
        src.write_text(MONOLITH)
        dist = self.tmp / "dist"
        context = {
            "frameworks": {"language": "python"},
            "scaffold": {
                "source_entry": str(src),
                "distribution_directory": str(dist),
                "decomposition_mode": "modular_package",
                "module_mapping": {
                    "parser": ["SchemaValidator", "parse_blueprint"],
                    "shielder": ["shield"],
                },
            },
        }
        pipeline = ScaffoldBuildPipeline(logger=lambda _m: None, verbose=False)
        result = pipeline.run(context, build=True)
        self.assertTrue(result.succeeded)
        self.assertTrue((dist / "parser.py").exists())
        self.assertTrue((dist / "shielder.py").exists())
        self.assertTrue((dist / "__init__.py").exists())
        self.assertIn(
            "from .parser import SchemaValidator", (dist / "main.py").read_text()
        )


# ---------------------------------------------------------------------------
# Blueprint parser
# ---------------------------------------------------------------------------


class TestParserModuleMapping(unittest.TestCase):
    def test_module_mapping_and_mode_parsed(self):
        normalized = normalize_optional_sections(
            {
                "scaffold": {
                    "source_entry": "main.py",
                    "decomposition_mode": "modular_package",
                    "module_mapping": {
                        "parser": ["SchemaValidator", "parse_blueprint"],
                        "cli": "main, create_parser",
                    },
                }
            }
        )
        scaffold = normalized["scaffold"]
        self.assertEqual(scaffold["decomposition_mode"], "modular_package")
        self.assertEqual(
            scaffold["module_mapping"]["parser"],
            ["SchemaValidator", "parse_blueprint"],
        )
        # Comma-separated string values are normalised to lists.
        self.assertEqual(scaffold["module_mapping"]["cli"], ["main", "create_parser"])

    def test_unknown_mode_rejected(self):
        with self.assertRaises(BlueprintParseError):
            normalize_optional_sections(
                {"scaffold": {"decomposition_mode": "teleport"}}
            )

    def test_defaults_when_absent(self):
        normalized = normalize_optional_sections({})
        self.assertEqual(normalized["scaffold"]["decomposition_mode"], "")
        self.assertEqual(normalized["scaffold"]["module_mapping"], {})


# ---------------------------------------------------------------------------
# Static import pruning (unit)
# ---------------------------------------------------------------------------


class TestImportPruner(unittest.TestCase):
    def _prune(self, src):
        return prune_dead_imports(ast.parse(src))

    def test_prunes_fully_unused_import(self):
        out = self._prune("import os\nimport json\nx = json.dumps({})\n")
        self.assertEqual(out.pruned, ["os"])
        self.assertEqual(render_imports(out.kept_imports), ["import json"])

    def test_prunes_unused_alias(self):
        out = self._prune("from sys import argv, exit\nprint(argv)\n")
        self.assertEqual(out.pruned, ["exit"])
        self.assertEqual(render_imports(out.kept_imports), ["from sys import argv"])

    def test_keeps_aliased_import_when_used(self):
        out = self._prune("import numpy as np\ny = np.zeros(3)\n")
        self.assertEqual(out.pruned, [])

    def test_future_import_never_pruned(self):
        out = self._prune("from __future__ import annotations\nimport os\nx = 1\n")
        self.assertIn("from __future__ import annotations", render_imports(out.kept_imports))
        self.assertEqual(out.pruned, ["os"])

    def test_star_import_never_pruned(self):
        out = self._prune("from os import *\nimport json\nz = getcwd()\n")
        self.assertEqual(out.pruned, ["json"])
        self.assertIn("from os import *", render_imports(out.kept_imports))

    def test_string_literal_safeguard_keeps_import(self):
        out = self._prune("import os\nimport json\nname = 'json'\nos.getcwd()\n")
        self.assertEqual(out.pruned, [])

    def test_sys_modules_suppresses_pruning(self):
        out = self._prune("import os\nimport sys\nsys.modules.get('os')\n")
        self.assertTrue(out.skipped_dynamic)
        self.assertEqual(out.pruned, [])

    def test_dunder_import_suppresses_pruning(self):
        out = self._prune("import os\nimport json\n__import__('os')\n")
        self.assertTrue(out.skipped_dynamic)

    def test_importlib_import_module_suppresses_pruning(self):
        out = self._prune(
            "import importlib\nimport os\nimportlib.import_module('os')\n"
        )
        self.assertTrue(out.skipped_dynamic)

    def test_attribute_root_counts_as_use(self):
        out = self._prune("import os\nos.path.join('a', 'b')\n")
        self.assertEqual(out.pruned, [])

    def test_no_imports_is_noop(self):
        out = self._prune("x = 1\n")
        self.assertEqual(out.kept_imports, [])
        self.assertEqual(out.pruned, [])


# ---------------------------------------------------------------------------
# Static import pruning (decomposer / engine integration)
# ---------------------------------------------------------------------------


class TestDecomposerPruning(_Tmp):
    def test_pruning_strips_per_file_imports(self):
        dest = self.tmp / "dist"
        decomposer = ModularDecomposer(verbose=False, prune_imports=True)
        decomposer.decompose(
            MONOLITH,
            {"parser": ["SchemaValidator", "parse_blueprint"], "shielder": ["shield"]},
            dest_dir=dest,
        )
        # shielder uses os (via shield) but not json.
        shielder = (dest / "shielder.py").read_text()
        self.assertIn("import os", shielder)
        self.assertNotIn("import json", shielder)
        # parser uses json (via check) but not os.
        parser = (dest / "parser.py").read_text()
        self.assertIn("import json", parser)
        self.assertNotIn("import os", parser)
        # All generated files stay syntactically valid.
        for name in ("parser.py", "shielder.py", "main.py"):
            ast.parse((dest / name).read_text())

    def test_pruning_disabled_keeps_all_imports(self):
        dest = self.tmp / "dist"
        ModularDecomposer(verbose=False, prune_imports=False).decompose(
            MONOLITH, {"shielder": ["shield"]}, dest_dir=dest
        )
        shielder = (dest / "shielder.py").read_text()
        self.assertIn("import os", shielder)
        self.assertIn("import json", shielder)

    def test_engine_emits_optimize_logs(self):
        src = self.tmp / "main.py"
        src.write_text(MONOLITH)
        dist = self.tmp / "pkg"
        msgs = []
        ScaffoldEngine(logger=msgs.append, verbose=True).scaffold(
            source_entry=str(src),
            distribution_directory=dist,
            language="python",
            module_mapping={"parser": ["SchemaValidator", "parse_blueprint"]},
            decomposition_mode="modular_package",
            prune_imports=True,
        )
        joined = "\n".join(msgs)
        self.assertIn("[Optimize   ] Pruned unused 'os' import from parser.py", joined)

    def test_pipeline_honors_analysis_flag(self):
        src = self.tmp / "main.py"
        src.write_text(MONOLITH)
        dist = self.tmp / "dist"
        context = {
            "frameworks": {"language": "python"},
            "analysis": {"static_import_pruning": True},
            "scaffold": {
                "source_entry": str(src),
                "distribution_directory": str(dist),
                "decomposition_mode": "modular_package",
                "module_mapping": {"shielder": ["shield"]},
            },
        }
        ScaffoldBuildPipeline(logger=lambda _m: None, verbose=False).run(context, build=False)
        shielder = (dist / "shielder.py").read_text()
        self.assertNotIn("import json", shielder)


class TestAnalysisBlockParsing(unittest.TestCase):
    def test_flags_parsed_and_defaulted(self):
        normalized = normalize_optional_sections(
            {
                "analysis": {
                    "ast_scanning": "aggressive",
                    "dead_code_elimination": True,
                    "static_import_pruning": True,
                    "macro_expansion": "pass_through",
                }
            }
        )
        analysis = normalized["analysis"]
        self.assertTrue(analysis["static_import_pruning"])
        self.assertTrue(analysis["dead_code_elimination"])
        self.assertEqual(analysis["ast_scanning"], "aggressive")

    def test_default_flag_is_false(self):
        self.assertFalse(
            normalize_optional_sections({})["analysis"]["static_import_pruning"]
        )

    def test_unknown_keys_preserved(self):
        merged = normalize_analysis_block({"static_import_pruning": True, "knob": 7})
        self.assertEqual(merged["knob"], 7)

    def test_non_bool_flag_rejected(self):
        with self.assertRaises(BlueprintParseError):
            normalize_analysis_block({"static_import_pruning": "yes"})


# ---------------------------------------------------------------------------
# Feature A — intra-module interlinking (cross-module calls)
# ---------------------------------------------------------------------------


class TestCrossImports(unittest.TestCase):
    def test_resolve_cross_imports_basic(self):
        body = "def main():\n    return BlueprintParser()\n"
        sym = {"BlueprintParser": "parser.py", "main": "cli.py"}
        lines, pairs = resolve_cross_imports(body, "cli.py", sym)
        self.assertEqual(lines, ["from .parser import BlueprintParser"])
        self.assertEqual(pairs, [("BlueprintParser", "parser")])

    def test_no_self_import(self):
        body = "class A:\n    def m(self):\n        return A()\n"
        sym = {"A": "parser.py"}
        lines, pairs = resolve_cross_imports(body, "parser.py", sym)
        self.assertEqual(lines, [])
        self.assertEqual(pairs, [])

    def test_groups_multiple_symbols_per_module(self):
        body = "def f():\n    return X() + Y()\n"
        sym = {"X": "core.py", "Y": "core.py", "f": "cli.py"}
        lines, _ = resolve_cross_imports(body, "cli.py", sym)
        self.assertEqual(lines, ["from .core import X, Y"])

    def test_decomposer_injects_cross_import(self):
        src = (
            "class BlueprintParser:\n    pass\n"
            "def main():\n    return BlueprintParser()\n"
        )
        dest = Path(tempfile.mkdtemp())
        ModularDecomposer(verbose=False).decompose(
            src,
            {"parser": ["BlueprintParser"], "cli": ["main"]},
            dest_dir=dest,
        )
        cli = (dest / "cli.py").read_text()
        self.assertIn("from .parser import BlueprintParser", cli)
        # parser.py does not import itself.
        self.assertNotIn("from .parser import", (dest / "parser.py").read_text())


# ---------------------------------------------------------------------------
# Feature B — multi-file source ingestion matrix
# ---------------------------------------------------------------------------


class TestMergeSourceAsts(unittest.TestCase):
    def test_merges_and_dedups_imports(self):
        a = "import os\nimport json\ndef f():\n    return os.getcwd()\n"
        b = "import os\nimport sys\ndef g():\n    return sys.argv\n"
        merged = merge_source_asts([("a.py", a), ("b.py", b)])
        # os appears once despite being in both files.
        self.assertEqual(merged.source.count("import os"), 1)
        self.assertIn("import json", merged.source)
        self.assertIn("import sys", merged.source)
        self.assertIn("f", merged.definitions)
        self.assertIn("g", merged.definitions)
        ast.parse(merged.source)  # merged result is valid

    def test_future_import_hoisted_first(self):
        a = "from __future__ import annotations\nimport os\nx = 1\n"
        b = "import sys\ny = 2\n"
        merged = merge_source_asts([("a.py", a), ("b.py", b)])
        self.assertTrue(merged.source.lstrip().startswith("from __future__"))

    def test_bad_source_raises(self):
        with self.assertRaises(DecompositionError):
            merge_source_asts([("bad.py", "def (:\n")])

    def test_engine_multifile_decomposition(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "core.py").write_text("class A:\n    pass\n")
        (tmp / "extra.py").write_text("def helper():\n    return A()\n")
        dist = tmp / "dist"
        ScaffoldEngine(verbose=False).scaffold(
            source_entry=[str(tmp / "core.py"), str(tmp / "extra.py")],
            distribution_directory=dist,
            language="python",
            module_mapping={"parser": ["A"], "utils": ["helper"]},
            decomposition_mode="modular_package",
        )
        self.assertTrue((dist / "parser.py").exists())
        self.assertTrue((dist / "utils.py").exists())
        # cross-module reference resolved across merged files.
        self.assertIn("from .parser import A", (dist / "utils.py").read_text())
        # multi-file orchestrator is named main.py
        self.assertTrue((dist / "main.py").exists())


# ---------------------------------------------------------------------------
# Feature C — autonomous self-testing suite scaffolding
# ---------------------------------------------------------------------------


class TestTestMatrix(_Tmp):
    def test_python_matrix_imports_submodules(self):
        result = generate_test_matrix(
            "python", self.tmp, package="", modules=["parser.py", "cli.py"]
        )
        self.assertEqual(result.files, ["tests/test_package_loading.py"])
        content = (self.tmp / "tests" / "test_package_loading.py").read_text()
        self.assertIn("'parser'", content)
        self.assertIn("'cli'", content)
        self.assertIn("importlib.import_module", content)
        ast.parse(content)

    def test_rust_matrix_writes_binding_test(self):
        result = generate_test_matrix("rust", self.tmp, crate="anyon-sim")
        self.assertEqual(result.files, ["tests/binding_validation.rs"])
        content = (self.tmp / "tests" / "binding_validation.rs").read_text()
        self.assertIn("#[test]", content)
        self.assertIn("binding_validation", content)

    def test_engine_generates_tests_for_modular(self):
        src = self.tmp / "main.py"
        src.write_text(MONOLITH)
        dist = self.tmp / "pkg"
        ScaffoldEngine(verbose=False).scaffold(
            source_entry=str(src),
            distribution_directory=dist,
            language="python",
            module_mapping={"parser": ["SchemaValidator"]},
            decomposition_mode="modular_package",
            generate_tests=True,
        )
        loader = dist / "tests" / "test_package_loading.py"
        self.assertTrue(loader.exists())
        self.assertIn("'parser'", loader.read_text())

    def test_pipeline_honors_generate_test_shims(self):
        src = self.tmp / "main.py"
        src.write_text(MONOLITH)
        dist = self.tmp / "dist"
        context = {
            "frameworks": {"language": "python"},
            "validation": {"generate_test_shims": True},
            "scaffold": {
                "source_entry": str(src),
                "distribution_directory": str(dist),
                "decomposition_mode": "modular_package",
                "module_mapping": {"parser": ["SchemaValidator"]},
            },
        }
        ScaffoldBuildPipeline(logger=lambda _m: None, verbose=False).run(context, build=False)
        self.assertTrue((dist / "tests" / "test_package_loading.py").exists())


class TestMultiFileAndShimParsing(unittest.TestCase):
    def test_source_entry_list_parsed(self):
        normalized = normalize_optional_sections(
            {"scaffold": {"source_entry": ["/a/core.py", "/b/utils.py"]}}
        )
        self.assertEqual(
            normalized["scaffold"]["source_entry"], ["/a/core.py", "/b/utils.py"]
        )

    def test_source_entry_string_still_supported(self):
        normalized = normalize_optional_sections(
            {"scaffold": {"source_entry": "/a/main.py"}}
        )
        self.assertEqual(normalized["scaffold"]["source_entry"], "/a/main.py")

    def test_generate_test_shims_flag_parsed(self):
        normalized = normalize_optional_sections(
            {"validation": {"generate_test_shims": True}}
        )
        self.assertTrue(normalized["validation"]["generate_test_shims"])

    def test_generate_test_shims_defaults_false(self):
        normalized = normalize_optional_sections({})
        self.assertFalse(normalized["validation"]["generate_test_shims"])


# ---------------------------------------------------------------------------
# Strict __future__ hoisting (both decomposition engines)
# ---------------------------------------------------------------------------

# A monolith whose docstring precedes the future statement: when a generator
# adds its own docstring/imports above the future import, the compiler rejects
# the result.  ``compile`` (unlike ``ast.parse``) enforces the future position.
_FUTURE_MONOLITH = '''\
"""Original monolith docstring."""
from __future__ import annotations

import os
import json

GLOBAL = 42


class Engine:
    """An engine."""

    def run(self):
        return os.getcwd()


def helper(payload):
    return json.dumps(payload)
'''


class TestFutureHoistingScaffold(_Tmp):
    """ModularDecomposer (src/scaffold/decomposition.py) hoists __future__."""

    def _decompose(self):
        dest = self.tmp / "scaf"
        ModularDecomposer(verbose=False).decompose(
            _FUTURE_MONOLITH,
            {"engine": ["Engine"], "helpers": ["helper"]},
            source_filename="main.py",
            dest_dir=dest,
        )
        return dest

    def test_future_is_absolute_first_line(self):
        dest = self._decompose()
        for name in ("engine.py", "helpers.py"):
            text = (dest / name).read_text()
            self.assertTrue(
                text.startswith("from __future__ import annotations"),
                f"{name} must begin with the future import, got:\n{text[:120]}",
            )
            # ...and precede the generated banner/docstring + other imports.
            self.assertLess(text.index("from __future__"), text.index('"""'))
            self.assertLess(text.index("from __future__"), text.index("import os"))

    def test_generated_modules_compile(self):
        dest = self._decompose()
        for name in ("engine.py", "helpers.py"):
            text = (dest / name).read_text()
            compile(text, name, "exec")  # raises if the future import is trapped

    def test_future_not_pruned_when_unused(self):
        dest = self.tmp / "scaf_prune"
        # `annotations` has no runtime use, so a dead-import pruner would drop it
        # unless future imports are protected.
        ModularDecomposer(verbose=False, prune_imports=True).decompose(
            _FUTURE_MONOLITH,
            {"engine": ["Engine"]},
            source_filename="main.py",
            dest_dir=dest,
        )
        text = (dest / "engine.py").read_text()
        self.assertIn("from __future__ import annotations", text)
        compile(text, "engine.py", "exec")


class TestFutureHoistingSplitter(_Tmp):
    """decompose_source (src/decomposition/splitter.py) hoists __future__."""

    def _split(self):
        from src.decomposition.splitter import decompose_source

        src = self.tmp / "mono.py"
        src.write_text(_FUTURE_MONOLITH)
        out = self.tmp / "split"
        result = decompose_source(src, out, min_lines=1)
        self.assertEqual(result.errors, [])
        return [Path(p) for p in result.files_written]

    def test_future_is_absolute_first_line(self):
        for path in self._split():
            text = path.read_text()
            self.assertTrue(
                text.startswith("from __future__ import annotations"),
                f"{path.name} must begin with the future import, got:\n{text[:120]}",
            )

    def test_init_orchestrator_starts_with_future(self):
        out = self.tmp / "split"
        self._split()
        init_text = (out / "__init__.py").read_text()
        self.assertTrue(
            init_text.startswith("from __future__ import annotations"),
            f"__init__.py must begin with the future import, got:\n{init_text[:120]}",
        )

    def test_split_modules_compile(self):
        for path in self._split():
            compile(path.read_text(), path.name, "exec")

    def test_future_not_duplicated_into_body(self):
        for path in self._split():
            text = path.read_text()
            # Exactly one future statement — hoisted, never also trapped below.
            self.assertEqual(text.count("from __future__ import annotations"), 1)

    def test_future_hoisting_idempotent_under_recursive_sweep(self):
        from src.decomposition.splitter import decompose_source

        for path in self._split():
            if Path(path).name == "__init__.py":
                continue
            sub_out = self.tmp / f"resplit_{Path(path).stem}"
            sub_result = decompose_source(path, sub_out, min_lines=1)
            self.assertEqual(sub_result.errors, [])
            for sub_path in (Path(p) for p in sub_result.files_written):
                if sub_path.name == "__init__.py":
                    continue
                text = sub_path.read_text()
                self.assertEqual(text.count("from __future__ import annotations"), 1)
                self.assertTrue(
                    text.startswith("from __future__ import annotations"),
                    f"{sub_path.name} must begin with the future import, got:\n{text[:120]}",
                )


# ---------------------------------------------------------------------------
# Feature D — type-hint blanket fallback layer
# ---------------------------------------------------------------------------


class TestTypeHintFallbackSplitter(_Tmp):
    """decompose_source injects typing/pathlib fallbacks when type hints are uncovered."""

    def _split(self, source):
        from src.decomposition.splitter import decompose_source

        src = self.tmp / "mono.py"
        src.write_text(source, encoding="utf-8")
        out = self.tmp / "split"
        result = decompose_source(src, out, min_lines=1)
        self.assertEqual(result.errors, [])
        return [Path(p) for p in result.files_written]

    def test_fallback_injected_for_uncovered_annotations(self):
        source = (
            "from __future__ import annotations\n\n"
            "def helper(p: Path) -> Optional[int]:\n"
            "    return None\n"
        )
        for path in self._split(source):
            if path.name == "__init__.py":
                continue
            text = path.read_text()
            self.assertIn("from typing import Optional, Union, Any, List, Dict, Callable, Iterator", text)
            self.assertIn("from pathlib import Path", text)
            self.assertLess(text.index("from __future__"), text.index("from typing"))
            self.assertLess(text.index("from typing"), text.index("""Auto-decomposed"""))


class TestTypeHintFallbackScaffold(_Tmp):
    """ModularDecomposer injects typing/pathlib fallbacks when type hints are uncovered."""

    def _decompose(self, source):
        dest = self.tmp / "dist"
        ModularDecomposer(verbose=False).decompose(
            source,
            {"helpers": ["helper"]},
            source_filename="main.py",
            dest_dir=dest,
        )
        return dest

    def test_fallback_injected_for_uncovered_annotations(self):
        source = (
            "from __future__ import annotations\n\n"
            "def helper(p: Path) -> Optional[int]:\n"
            "    return None\n"
        )
        dest = self._decompose(source)
        text = (dest / "helpers.py").read_text()
        self.assertIn("from typing import Optional, Union, Any, List, Dict, Callable, Iterator", text)
        self.assertIn("from pathlib import Path", text)
        self.assertLess(text.index("from __future__"), text.index("from typing"))


if __name__ == "__main__":
    unittest.main()
