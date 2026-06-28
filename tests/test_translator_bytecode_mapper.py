# -*- coding: utf-8 -*-
"""Unit tests for translator.bytecode_mapper."""

import unittest

from translator.bytecode_mapper import (
    MeshRecipe,
    _task_block,
    hotpath_to_recipe,
    _MAX_FAMILY_INSTANCES,
    _MAX_NODES_PER_MESH,
)
from translator.hotpath_scanner import HotPath


class TestTaskBlock(unittest.TestCase):
    def test_basic_block(self):
        block = _task_block("init", "print", 'text = "hello"')
        self.assertIn("[task:init]", block)
        self.assertIn("op = print", block)
        self.assertIn('text = "hello"', block)
        self.assertNotIn("needs", block)

    def test_block_with_needs(self):
        block = _task_block("node1", "call", "fn = foo", needs="init")
        self.assertIn("[task:node1]", block)
        self.assertIn("needs = init", block)


class TestHotpathToRecipe(unittest.TestCase):
    def test_single_source(self):
        hp = HotPath(
            source_files=["/src/a.py"],
            pattern_id="hp0",
            weight=1,
            label="key_value",
        )
        recipe = hotpath_to_recipe(hp)
        self.assertIsInstance(recipe, MeshRecipe)
        self.assertEqual(recipe.name, "translated_hp0")
        self.assertIn("[project]", recipe.body)
        self.assertIn("[task:init]", recipe.body)
        self.assertIn("a.py", recipe.body)

    def test_multiple_sources_capped(self):
        sources = [f"/src/file{i}.py" for i in range(30)]
        hp = HotPath(
            source_files=sources,
            pattern_id="hp1",
            weight=30,
            label="multi",
        )
        recipe = hotpath_to_recipe(hp)
        # Should be capped by _MAX_FAMILY_INSTANCES (5 reader tasks)
        reader_count = recipe.body.count("reader | Scanning hot-path source")
        self.assertLessEqual(reader_count, _MAX_FAMILY_INSTANCES)

    def test_output_path_in_recipe(self):
        hp = HotPath(
            source_files=["/x.py"],
            pattern_id="hp2",
            weight=1,
            label="test",
        )
        recipe = hotpath_to_recipe(hp, output_dir="/custom/output")
        self.assertIn("/custom/output", recipe.output_path)

    def test_aggregation_task_included(self):
        hp = HotPath(
            source_files=["/a.py", "/b.py"],
            pattern_id="hp3",
            weight=2,
            label="agg",
        )
        recipe = hotpath_to_recipe(hp)
        self.assertIn("op = call", recipe.body)
        self.assertIn("write_file", recipe.body)


if __name__ == "__main__":
    unittest.main()
