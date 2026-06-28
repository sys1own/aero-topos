# -*- coding: utf-8 -*-
"""Comprehensive tests for the ``blueprint_lang`` block-DSL parser/validator.

Organised by layer:

* ``TestLexer``            -- tokenisation, strings/escapes, numbers, comments.
* ``TestParserValid``      -- well-formed documents parse into the right AST.
* ``TestParserErrors``     -- structural errors report the exact line/column.
* ``TestValidatorErrors``  -- semantic errors (schema, requires, cycles).
* ``TestErrorRendering``   -- the ``^`` pointer / ``line:column`` formatting.
* ``TestPublicApi``        -- check/load helpers, ``to_config``, detection.
* ``TestCli``              -- ``python -m blueprint_lang`` behaviour + exit codes.

Each broken-blueprint test asserts not just *that* parsing fails but *where*,
so the diagnostics stay accurate.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import blueprint_lang as bl
from blueprint_lang import (
    BlueprintSyntaxError,
    BlueprintValidationError,
    BoolValue,
    ListValue,
    NumberValue,
    StringValue,
    TokenKind,
)
from blueprint_lang import cli as bl_cli
from blueprint_lang.validator import Validator, _find_cycle


# Canonical valid example from the task description.
EXAMPLE = '''
project "my_universal_app" {
    version = "1.0.0"
}

target "core_engine" {
    language = "cpp"
    sources = ["src/core/**/*.cpp", "src/core/**/*.hpp"]
    flags = ["-O3", "-std=c++20"]
}

target "bindings" {
    language = "python"
    requires = ["core_engine"]
    sources = ["src/bindings/*.py"]
}
'''


def _project(body_targets: str) -> str:
    """A minimal valid project plus whatever target text is supplied."""
    return 'project "p" { version = "1.0.0" }\n' + body_targets


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


class TestLexer(unittest.TestCase):
    def test_basic_token_stream(self):
        toks = bl.tokenize('target "x" { k = 1 }')
        kinds = [t.kind for t in toks]
        self.assertEqual(
            kinds,
            [
                TokenKind.IDENT,
                TokenKind.STRING,
                TokenKind.LBRACE,
                TokenKind.IDENT,
                TokenKind.EQUALS,
                TokenKind.NUMBER,
                TokenKind.RBRACE,
                TokenKind.EOF,
            ],
        )

    def test_string_value_is_unquoted(self):
        toks = bl.tokenize('k = "hello world"')
        string_tok = toks[2]
        self.assertEqual(string_tok.kind, TokenKind.STRING)
        self.assertEqual(string_tok.value, "hello world")
        self.assertEqual(string_tok.text, '"hello world"')

    def test_string_escapes_decode(self):
        toks = bl.tokenize(r'k = "a\tb\nc\"d\\e\/f"')
        self.assertEqual(toks[2].value, 'a\tb\nc"d\\e/f')

    def test_numbers_int_and_float(self):
        toks = bl.tokenize("a = 42  b = -3  c = 0.5  d = 1.5e3")
        nums = [t.value for t in toks if t.kind == TokenKind.NUMBER]
        self.assertEqual(nums, [42, -3, 0.5, 1500.0])
        self.assertIsInstance(nums[0], int)
        self.assertIsInstance(nums[2], float)

    def test_booleans_are_keywords(self):
        toks = bl.tokenize("a = true  b = false")
        bools = [t for t in toks if t.kind == TokenKind.BOOL]
        self.assertEqual([t.value for t in bools], [True, False])

    def test_comments_and_blank_lines_skipped(self):
        src = "# a comment\n\n// another\ntarget \"x\" {}  # trailing\n"
        toks = bl.tokenize(src)
        kinds = [t.kind for t in toks]
        self.assertEqual(
            kinds,
            [TokenKind.IDENT, TokenKind.STRING, TokenKind.LBRACE, TokenKind.RBRACE, TokenKind.EOF],
        )

    def test_positions_are_one_based(self):
        toks = bl.tokenize('  k = "v"')
        ident = toks[0]
        self.assertEqual(ident.span.start.line, 1)
        self.assertEqual(ident.span.start.column, 3)  # after two spaces

    def test_position_tracks_newlines(self):
        toks = bl.tokenize('a = 1\nb = 2')
        second = toks[3]  # the 'b' identifier
        self.assertEqual(second.span.start.line, 2)
        self.assertEqual(second.span.start.column, 1)


# ---------------------------------------------------------------------------
# Parser -- valid documents
# ---------------------------------------------------------------------------


class TestParserValid(unittest.TestCase):
    def test_full_example_parses(self):
        bp = bl.parse_source(EXAMPLE, "blueprint.aero")
        self.assertEqual(len(bp.projects), 1)
        self.assertEqual(len(bp.targets), 2)
        self.assertEqual(bp.projects[0].name, "my_universal_app")
        self.assertEqual([t.name for t in bp.targets], ["core_engine", "bindings"])

    def test_field_values_have_correct_types(self):
        bp = bl.parse_source('t "x" { s = "hi" n = 3 f = 0.5 b = true xs = ["a", "b"] }')
        fields = bp.blocks[0].field_map()
        self.assertIsInstance(fields["s"].value, StringValue)
        self.assertIsInstance(fields["n"].value, NumberValue)
        self.assertIsInstance(fields["f"].value, NumberValue)
        self.assertIsInstance(fields["b"].value, BoolValue)
        self.assertIsInstance(fields["xs"].value, ListValue)
        self.assertEqual(fields["xs"].value.value, ["a", "b"])

    def test_trailing_comma_in_list_allowed(self):
        bp = bl.parse_source('t "x" { xs = ["a", "b", ] }')
        self.assertEqual(bp.blocks[0].get("xs").value.value, ["a", "b"])

    def test_empty_list_allowed_by_parser(self):
        bp = bl.parse_source('t "x" { xs = [] }')
        self.assertEqual(bp.blocks[0].get("xs").value.value, [])

    def test_empty_block_allowed_by_parser(self):
        bp = bl.parse_source('project "x" {}')
        self.assertEqual(bp.blocks[0].fields, [])

    def test_tabs_and_irregular_whitespace(self):
        src = 'target\t"x"\t{\n\t\tlanguage = "c"\n}'
        bp = bl.parse_source(src)
        self.assertEqual(bp.blocks[0].get("language").value.value, "c")


# ---------------------------------------------------------------------------
# Parser -- structural errors (assert exact line/column)
# ---------------------------------------------------------------------------


class TestParserErrors(unittest.TestCase):
    def _syntax_error(self, src: str) -> BlueprintSyntaxError:
        with self.assertRaises(BlueprintSyntaxError) as ctx:
            bl.parse_source(src, "blueprint.aero")
        return ctx.exception

    def test_unterminated_string_eof(self):
        err = self._syntax_error('project "p" {\n    version = "1.0.0\n}\n')
        self.assertIn("unterminated string", err.message)
        self.assertEqual(err.line, 2)
        self.assertEqual(err.column, 15)  # the opening quote

    def test_unterminated_string_points_at_open_quote(self):
        err = self._syntax_error('k = "abc')
        self.assertEqual((err.line, err.column), (1, 5))

    def test_invalid_escape_sequence(self):
        err = self._syntax_error(r'project "p" { version = "1.0\x0" }')
        self.assertIn("invalid escape", err.message)
        self.assertIn(r"\x", err.message)

    def test_missing_closing_brace_points_at_opening(self):
        err = self._syntax_error('project "p" {\n    version = "1.0.0"\n')
        self.assertIn("missing closing '}'", err.message)
        self.assertEqual((err.line, err.column), (1, 13))  # the '{'

    def test_missing_opening_brace(self):
        err = self._syntax_error('project "p"\n    version = "1"\n')
        self.assertIn("expected '{'", err.message)

    def test_missing_name_after_block_type(self):
        err = self._syntax_error('project {\n}\n')
        self.assertIn("quoted name", err.message)
        self.assertEqual(err.line, 1)

    def test_missing_equals(self):
        err = self._syntax_error('project "p" {\n    version "1.0.0"\n}\n')
        self.assertIn("expected '='", err.message)
        self.assertEqual((err.line, err.column), (2, 13))  # the string token

    def test_bareword_value_suggests_quotes(self):
        err = self._syntax_error('target "t" {\n    language = cpp\n}\n')
        self.assertIn("bareword", err.message)
        self.assertIn('"cpp"', err.hint)
        self.assertEqual((err.line, err.column), (2, 16))

    def test_missing_comma_in_list(self):
        err = self._syntax_error('t "x" { xs = ["a" "b"] }')
        self.assertIn("',' to separate", err.message)

    def test_unclosed_list_trailing_comma(self):
        err = self._syntax_error('t "x" {\n    xs = ["a",\n}\n')
        self.assertIn("missing closing ']'", err.message)

    def test_unexpected_character(self):
        err = self._syntax_error('project "p" {\n    version = @\n}\n')
        self.assertIn("unexpected character", err.message)
        self.assertEqual((err.line, err.column), (2, 15))

    def test_duplicate_key(self):
        err = self._syntax_error('project "p" {\n    version = "1"\n    version = "2"\n}\n')
        self.assertIn("duplicate key", err.message)
        self.assertEqual(err.line, 3)  # the second occurrence
        self.assertIn("line 2", err.hint)

    def test_top_level_junk(self):
        err = self._syntax_error('"floating string"')
        self.assertIn("block type", err.message)

    def test_eof_inside_block_expecting_value(self):
        err = self._syntax_error('project "p" {\n    version =')
        self.assertIn("end of file", err.message)


# ---------------------------------------------------------------------------
# Validator -- semantic errors
# ---------------------------------------------------------------------------


class TestValidatorErrors(unittest.TestCase):
    def _validation_error(self, src: str) -> BlueprintValidationError:
        with self.assertRaises(BlueprintValidationError) as ctx:
            bl.load_source(src, "blueprint.aero")
        return ctx.exception

    def test_valid_example_passes(self):
        # Should not raise.
        bp = bl.load_source(EXAMPLE, "blueprint.aero")
        self.assertEqual(len(bp.targets), 2)

    def test_empty_blueprint(self):
        err = self._validation_error("# only a comment\n")
        self.assertIn("empty", err.message)

    def test_missing_project(self):
        err = self._validation_error('target "t" { language = "c" sources = ["a"] }')
        self.assertIn("missing required 'project'", err.message)

    def test_two_projects(self):
        src = 'project "a" { version = "1" }\nproject "b" { version = "2" }\ntarget "t" { language = "c" sources = ["a"] }'
        err = self._validation_error(src)
        self.assertIn("duplicate 'project'", err.message)
        self.assertEqual(err.line, 2)

    def test_no_targets(self):
        err = self._validation_error('project "p" { version = "1" }')
        self.assertIn("no 'target' blocks", err.message)

    def test_unknown_block_type(self):
        err = self._validation_error('widget "w" { foo = "bar" }')
        self.assertIn("unknown block type 'widget'", err.message)
        self.assertEqual((err.line, err.column), (1, 1))

    def test_unknown_key(self):
        err = self._validation_error(
            _project('target "t" {\n    language = "c"\n    sources = ["a"]\n    nonsense = "x"\n}\n')
        )
        self.assertIn("unknown key 'nonsense'", err.message)
        self.assertEqual(err.line, 5)

    def test_missing_required_key(self):
        err = self._validation_error(_project('target "t" { sources = ["a"] }'))
        self.assertIn("missing required key 'language'", err.message)

    def test_wrong_type_string_expected(self):
        err = self._validation_error(_project('target "t" { language = ["c"] sources = ["a"] }'))
        self.assertIn("must be a string", err.message)

    def test_wrong_type_list_expected(self):
        err = self._validation_error(_project('target "t" { language = "c" sources = "a.py" }'))
        self.assertIn("must be a list of strings", err.message)

    def test_list_item_wrong_type(self):
        err = self._validation_error(_project('target "t" { language = "c" sources = ["a", 3] }'))
        self.assertIn("must be a string", err.message)

    def test_unsupported_language(self):
        err = self._validation_error(_project('target "t" { language = "cobol" sources = ["a"] }'))
        self.assertIn("unsupported language 'cobol'", err.message)

    def test_empty_sources(self):
        err = self._validation_error(_project('target "t" { language = "c" sources = [] }'))
        self.assertIn("no sources", err.message)

    def test_duplicate_target_name(self):
        src = _project(
            'target "t" { language = "c" sources = ["a"] }\n'
            'target "t" { language = "c" sources = ["b"] }\n'
        )
        err = self._validation_error(src)
        self.assertIn("duplicate target name 't'", err.message)
        self.assertEqual(err.line, 3)

    def test_requires_unknown_target(self):
        err = self._validation_error(
            _project('target "t" { language = "c" sources = ["a"] requires = ["ghost"] }')
        )
        self.assertIn("requires unknown target 'ghost'", err.message)

    def test_direct_cycle(self):
        src = _project(
            'target "a" { language = "c" sources = ["a"] requires = ["b"] }\n'
            'target "b" { language = "c" sources = ["b"] requires = ["a"] }\n'
        )
        err = self._validation_error(src)
        self.assertIn("cyclic target dependency", err.message)
        self.assertIn("->", err.message)

    def test_self_cycle(self):
        err = self._validation_error(
            _project('target "a" { language = "c" sources = ["a"] requires = ["a"] }')
        )
        self.assertIn("cyclic target dependency", err.message)

    def test_long_cycle(self):
        src = _project(
            'target "a" { language = "c" sources = ["s"] requires = ["b"] }\n'
            'target "b" { language = "c" sources = ["s"] requires = ["c"] }\n'
            'target "c" { language = "c" sources = ["s"] requires = ["a"] }\n'
        )
        err = self._validation_error(src)
        self.assertIn("cyclic target dependency", err.message)

    def test_acyclic_requires_passes(self):
        src = _project(
            'target "a" { language = "c" sources = ["s"] }\n'
            'target "b" { language = "c" sources = ["s"] requires = ["a"] }\n'
            'target "c" { language = "c" sources = ["s"] requires = ["a", "b"] }\n'
        )
        bp = bl.load_source(src, "blueprint.aero")  # should not raise
        self.assertEqual(len(bp.targets), 3)

    def test_collect_reports_multiple_errors(self):
        src = _project(
            'target "t" {\n'
            '    language = "cobol"\n'   # unsupported language
            '    sources = ["a"]\n'
            '    bogus = "x"\n'           # unknown key
            '}\n'
        )
        bp = bl.parse_source(src, "blueprint.aero")
        errors = Validator(bp).collect()
        messages = " | ".join(e.message for e in errors)
        self.assertIn("unknown key 'bogus'", messages)
        self.assertIn("unsupported language", messages)
        # Reported in source order (unknown key appears before language value? no:
        # language is line 2, bogus line 4 -> language first).
        self.assertLessEqual(errors[0].span.start.offset, errors[1].span.start.offset)


class TestFindCycle(unittest.TestCase):
    def test_acyclic(self):
        self.assertIsNone(_find_cycle({"a": ["b"], "b": ["c"], "c": []}))

    def test_direct(self):
        cycle = _find_cycle({"a": ["b"], "b": ["a"]})
        self.assertIsNotNone(cycle)
        self.assertEqual(cycle[0], cycle[-1])

    def test_self(self):
        self.assertEqual(_find_cycle({"a": ["a"]}), ["a", "a"])

    def test_ignores_dangling_edges(self):
        # 'ghost' is not a node -> not a cycle, just a dangling edge.
        self.assertIsNone(_find_cycle({"a": ["ghost"]}))


# ---------------------------------------------------------------------------
# Error rendering -- the user-facing ^ pointer
# ---------------------------------------------------------------------------


class TestErrorRendering(unittest.TestCase):
    def _render(self, src: str):
        rendered = bl.check_source(src, "blueprint.aero")
        self.assertIsNotNone(rendered)
        return rendered

    def test_contains_caret_and_location(self):
        rendered = self._render('target "t" {\n    language = cpp\n    sources = ["a"]\n}\n')
        self.assertIn("error:", rendered)
        self.assertIn("--> blueprint.aero:2:16", rendered)
        self.assertIn("^", rendered)
        self.assertIn("help:", rendered)

    def test_caret_aligns_under_column(self):
        rendered = self._render('project "p" {\n    version = @\n}\n')
        lines = rendered.splitlines()
        # Find the source line and the caret line beneath it.
        src_idx = next(i for i, ln in enumerate(lines) if ln.strip().endswith("version = @"))
        caret_line = lines[src_idx + 1]
        source_line = lines[src_idx]
        self.assertEqual(caret_line.index("^"), source_line.index("@"))

    def test_multi_char_token_underlined(self):
        rendered = self._render('project "p" { version = "1" }\ntarget "t" { language = "cobol" sources = ["a"] }')
        caret_run = [ln for ln in rendered.splitlines() if set(ln.strip()) == {"^"} or "^" in ln]
        # "cobol" (with quotes) is 7 characters -> 7 carets in the underline.
        self.assertTrue(any(line.count("^") == 7 for line in caret_run))

    def test_tab_indent_preserved_in_caret(self):
        # Tab-indented source: the caret line must keep the tab so it aligns.
        src = 'project "p" { version = "1" }\ntarget "t" {\n\tlanguage = cobol\n\tsources = ["a"]\n}\n'
        rendered = self._render(src)
        caret_line = next(ln for ln in rendered.splitlines() if "^" in ln and "-->" not in ln)
        self.assertIn("\t", caret_line)

    def test_spanless_error_renders_without_crash(self):
        rendered = self._render("# nothing here\n")
        self.assertIn("error:", rendered)
        self.assertIn("empty", rendered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TestPublicApi(unittest.TestCase):
    def test_check_source_returns_none_when_valid(self):
        self.assertIsNone(bl.check_source(EXAMPLE, "blueprint.aero"))

    def test_check_source_returns_message_when_invalid(self):
        msg = bl.check_source('project "p" {', "blueprint.aero")
        self.assertIsInstance(msg, str)
        self.assertIn("error:", msg)

    def test_to_config_shape(self):
        cfg = bl.load_source(EXAMPLE, "blueprint.aero").to_config()
        self.assertEqual(cfg["project"]["name"], "my_universal_app")
        self.assertEqual(cfg["project"]["version"], "1.0.0")
        self.assertIn("core_engine", cfg["targets"])
        self.assertEqual(cfg["targets"]["bindings"]["requires"], ["core_engine"])
        self.assertEqual(
            cfg["targets"]["core_engine"]["sources"],
            ["src/core/**/*.cpp", "src/core/**/*.hpp"],
        )

    def test_load_file_roundtrip(self):
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False, encoding="utf-8") as fh:
            fh.write(EXAMPLE)
            path = fh.name
        try:
            bp = bl.load_file(path)
            self.assertEqual(len(bp.targets), 2)
        finally:
            os.remove(path)

    def test_check_file_missing(self):
        msg = bl.check_file("/no/such/blueprint.aero")
        self.assertIsNotNone(msg)
        self.assertIn("cannot read", msg)

    def test_detection_dsl_vs_ini_vs_json(self):
        self.assertTrue(bl.looks_like_blueprint_dsl(EXAMPLE))
        self.assertTrue(bl.looks_like_blueprint_dsl('target "x" {\n}\n'))
        self.assertFalse(bl.looks_like_blueprint_dsl("[meta]\nname = 1\n"))
        self.assertFalse(bl.looks_like_blueprint_dsl('{"project": {}}'))
        self.assertFalse(bl.looks_like_blueprint_dsl("# just a comment\n"))

    def test_sample_file_is_valid(self):
        sample = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fixtures", "blueprint.aero.sample"
        )
        if os.path.exists(sample):
            self.assertIsNone(bl.check_file(sample))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli(unittest.TestCase):
    def _write(self, content: str) -> str:
        fh = tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False, encoding="utf-8")
        fh.write(content)
        fh.close()
        self.addCleanup(os.remove, fh.name)
        return fh.name

    def test_cli_valid_returns_zero(self):
        path = self._write(EXAMPLE)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = bl_cli.main([path])
        self.assertEqual(rc, 0)
        self.assertIn("OK", out.getvalue())

    def test_cli_invalid_returns_one(self):
        path = self._write('project "p" {\n    version = "1.0\n}\n')
        err = io.StringIO()
        with redirect_stderr(err):
            rc = bl_cli.main([path])
        self.assertEqual(rc, 1)
        self.assertIn("^", err.getvalue())
        self.assertIn("aborting", err.getvalue())

    def test_cli_check_verb_optional(self):
        path = self._write(EXAMPLE)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = bl_cli.main(["check", path])
        self.assertEqual(rc, 0)

    def test_cli_detects_legacy_ini(self):
        path = self._write("[meta]\nname = \"x\"\n")
        out = io.StringIO()
        with redirect_stdout(out):
            rc = bl_cli.main([path])
        self.assertEqual(rc, 0)
        self.assertIn("legacy", out.getvalue())


if __name__ == "__main__":
    unittest.main()
