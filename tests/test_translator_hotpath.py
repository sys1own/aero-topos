# -*- coding: utf-8 -*-
"""Unit tests for translator.hotpath_scanner."""

import os
import tempfile
import unittest

from translator.hotpath_scanner import (
    ScannedFile,
    HotPath,
    fingerprint,
    parse_kv_file,
    scan_directory,
    identify_hotpaths,
)


class TestFingerprint(unittest.TestCase):
    def test_consistent_fingerprint(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            fp1 = fingerprint(path)
            fp2 = fingerprint(path)
            self.assertEqual(fp1, fp2)
            self.assertEqual(len(fp1), 16)
        finally:
            os.unlink(path)

    def test_different_content_different_fingerprint(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("file A")
            path_a = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("file B")
            path_b = f.name
        try:
            self.assertNotEqual(fingerprint(path_a), fingerprint(path_b))
        finally:
            os.unlink(path_a)
            os.unlink(path_b)


class TestParseKvFile(unittest.TestCase):
    def test_extracts_kv_pairs(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("FOO=bar\nBAZ_1=hello world\ninvalid line\n")
            path = f.name
        try:
            result = parse_kv_file(path)
            self.assertEqual(result, {"FOO": "bar", "BAZ_1": "hello world"})
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            path = f.name
        try:
            result = parse_kv_file(path)
            self.assertEqual(result, {})
        finally:
            os.unlink(path)

    def test_ignores_lowercase_keys(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("lowercase=val\nUPPER=ok\n")
            path = f.name
        try:
            result = parse_kv_file(path)
            self.assertEqual(result, {"UPPER": "ok"})
        finally:
            os.unlink(path)


class TestScanDirectory(unittest.TestCase):
    def test_scans_files(self):
        with tempfile.TemporaryDirectory() as td:
            f1 = os.path.join(td, "a.txt")
            f2 = os.path.join(td, "b.txt")
            with open(f1, "w") as f:
                f.write("KEY=value1\n")
            with open(f2, "w") as f:
                f.write("KEY=value2\n")
            results = scan_directory(td)
            self.assertEqual(len(results), 2)
            self.assertIsInstance(results[0], ScannedFile)
            self.assertTrue(results[0].size > 0)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as td:
            results = scan_directory(td)
            self.assertEqual(results, [])


class TestIdentifyHotpaths(unittest.TestCase):
    def test_groups_by_schema(self):
        files = [
            ScannedFile(path="a.txt", size=10, fingerprint="aaa", fields={"KEY": "1"}),
            ScannedFile(path="b.txt", size=20, fingerprint="bbb", fields={"KEY": "2"}),
            ScannedFile(path="c.txt", size=30, fingerprint="ccc", fields={"OTHER": "x"}),
        ]
        hotpaths = identify_hotpaths(files, weight_threshold=2)
        self.assertEqual(len(hotpaths), 1)
        self.assertIsInstance(hotpaths[0], HotPath)
        self.assertEqual(hotpaths[0].weight, 2)
        self.assertIn("a.txt", hotpaths[0].source_files)
        self.assertIn("b.txt", hotpaths[0].source_files)

    def test_no_hotpaths_below_threshold(self):
        files = [
            ScannedFile(path="a.txt", size=10, fingerprint="aaa", fields={"A": "1"}),
            ScannedFile(path="b.txt", size=20, fingerprint="bbb", fields={"B": "2"}),
        ]
        hotpaths = identify_hotpaths(files, weight_threshold=2)
        self.assertEqual(hotpaths, [])

    def test_raw_label_for_empty_schema(self):
        files = [
            ScannedFile(path="a.txt", size=10, fingerprint="aaa", fields={}),
            ScannedFile(path="b.txt", size=20, fingerprint="bbb", fields={}),
        ]
        hotpaths = identify_hotpaths(files, weight_threshold=2)
        self.assertEqual(len(hotpaths), 1)
        self.assertEqual(hotpaths[0].label, "raw")


if __name__ == "__main__":
    unittest.main()
