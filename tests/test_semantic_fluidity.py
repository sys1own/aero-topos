"""Tests for the Domain-Agnostic Semantic Fluidity engine.

Organised by layer:

* ``TestSchema``               -- invariant dataclasses, ids, merge/finalize, validation.
* ``TestDomainInference``      -- keyword-scored prose domains + code:* short-circuit.
* ``TestDocumentLoader``       -- mixed-format directory walking + per-file error isolation.
* ``TestPdfText``               -- dependency-free PDF text extraction fallback.
* ``TestTextRuleExtractor``    -- cue-phrase/regex prose extraction.
* ``TestCodeRuleExtractor``    -- AST-based Python + regex-based C/C++ extraction.
* ``TestJsonRuleExtractor``    -- structured-passthrough + generic-flatten JSON modes.
* ``TestLLMAssistedExtractor`` -- extensible client interface + offline fallback.
* ``TestSystemGraph``          -- domain-namespaced nodes, references, cross-domain linking.
* ``TestContextIngestionEngine`` -- end-to-end ingestion across every supported format.
* ``TestInvariantsCli``        -- the ``main.py`` ``invariants`` subcommand.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import main
from src.semantic_fluidity import (
    AlgorithmicBoundary,
    ContextIngestionEngine,
    DocumentLoader,
    Equation,
    IngestedDocument,
    InvariantSchema,
    JSON_SCHEMA,
    LLMAssistedExtractor,
    LLMClient,
    NullLLMClient,
    StateVariable,
    SystemGraph,
    infer_domain,
    make_id,
    validate_invariant_document,
)
from src.semantic_fluidity.extractors import CodeRuleExtractor, JsonRuleExtractor, TextRuleExtractor
from src.semantic_fluidity import pdf_text


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema(unittest.TestCase):
    def test_make_id_namespaces_symbol_under_domain(self):
        self.assertEqual(make_id("genomics", "rate"), "genomics::rate")

    def test_state_variable_id_and_dict(self):
        var = StateVariable(domain="physics", symbol="v", description="velocity", bounds=(0.0, 100.0))
        self.assertEqual(var.id, "physics::v")
        d = var.to_dict()
        self.assertEqual(d["kind"], "state_variable")
        self.assertEqual(d["bounds"], [0.0, 100.0])

    def test_boundary_and_equation_ids(self):
        boundary = AlgorithmicBoundary(domain="d", symbol="b", description="x", expression="x <= 1", variables=["x"])
        equation = Equation(domain="d", symbol="e", expression="e = x", lhs="e", rhs="x", variables=["x"])
        self.assertEqual(boundary.id, "d::b")
        self.assertEqual(equation.id, "d::e")
        self.assertEqual(boundary.to_dict()["kind"], "boundary")
        self.assertEqual(equation.to_dict()["kind"], "equation")

    def test_domains_property_dedupes_in_order(self):
        schema = InvariantSchema(
            state_variables=[
                StateVariable(domain="b", symbol="x"),
                StateVariable(domain="a", symbol="y"),
                StateVariable(domain="b", symbol="z"),
            ]
        )
        self.assertEqual(schema.domains, ["b", "a"])

    def test_merge_extends_all_three_buckets(self):
        first = InvariantSchema(state_variables=[StateVariable(domain="d", symbol="x")])
        second = InvariantSchema(boundaries=[AlgorithmicBoundary(domain="d", symbol="b", description="", expression="x>0", variables=["x"])])
        first.merge(second)
        self.assertEqual(len(first.state_variables), 1)
        self.assertEqual(len(first.boundaries), 1)

    def test_finalize_disambiguates_colliding_symbols(self):
        schema = InvariantSchema(
            state_variables=[
                StateVariable(domain="d", symbol="x"),
                StateVariable(domain="d", symbol="x"),
                StateVariable(domain="d", symbol="x"),
                StateVariable(domain="other", symbol="x"),
            ]
        )
        schema.finalize()
        symbols = [v.symbol for v in schema.state_variables]
        self.assertEqual(symbols, ["x", "x#2", "x#3", "x"])
        ids = [v.id for v in schema.state_variables]
        self.assertEqual(len(ids), len(set(ids)))

    def test_finalize_disambiguates_buckets_independently(self):
        schema = InvariantSchema(
            state_variables=[StateVariable(domain="d", symbol="x")],
            boundaries=[AlgorithmicBoundary(domain="d", symbol="x", description="", expression="x>0", variables=["x"])],
        )
        schema.finalize()
        self.assertEqual(schema.state_variables[0].symbol, "x")
        self.assertEqual(schema.boundaries[0].symbol, "x")

    def test_to_dict_shape(self):
        schema = InvariantSchema(state_variables=[StateVariable(domain="d", symbol="x")])
        d = schema.to_dict()
        self.assertEqual(set(d.keys()), {"domains", "state_variables", "boundaries", "equations"})

    def test_json_schema_declares_required_top_level_keys(self):
        self.assertEqual(
            set(JSON_SCHEMA["required"]), {"domains", "state_variables", "boundaries", "equations"}
        )

    def test_validate_invariant_document_accepts_valid_document(self):
        schema = InvariantSchema(
            state_variables=[StateVariable(domain="d", symbol="x")],
            boundaries=[AlgorithmicBoundary(domain="d", symbol="b", description="", expression="x>0", variables=["x"])],
            equations=[Equation(domain="d", symbol="e", expression="e=x", lhs="e", rhs="x", variables=["x"])],
        )
        self.assertEqual(validate_invariant_document(schema.to_dict()), [])

    def test_validate_invariant_document_rejects_non_dict(self):
        self.assertIn("document must be a JSON object", validate_invariant_document([]))

    def test_validate_invariant_document_reports_missing_top_level_key(self):
        errors = validate_invariant_document({"domains": [], "state_variables": [], "boundaries": []})
        self.assertTrue(any("equations" in e for e in errors))

    def test_validate_invariant_document_reports_missing_item_field(self):
        doc = {
            "domains": ["d"],
            "state_variables": [{"kind": "state_variable", "domain": "d"}],  # missing id, symbol
            "boundaries": [],
            "equations": [],
        }
        errors = validate_invariant_document(doc)
        self.assertTrue(any("missing required field 'id'" in e for e in errors))
        self.assertTrue(any("missing required field 'symbol'" in e for e in errors))

    def test_validate_invariant_document_reports_wrong_kind(self):
        doc = {
            "domains": ["d"],
            "state_variables": [{"id": "d::x", "kind": "boundary", "domain": "d", "symbol": "x"}],
            "boundaries": [],
            "equations": [],
        }
        errors = validate_invariant_document(doc)
        self.assertTrue(any("expected 'state_variable'" in e for e in errors))


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------


class TestDomainInference(unittest.TestCase):
    def _doc(self, text: str, fmt: str = "text", name: str = "doc.txt") -> IngestedDocument:
        return IngestedDocument(path=Path(name), format=fmt, text=text)

    def test_code_documents_use_their_own_format_as_domain(self):
        doc = self._doc("MAX = 1", fmt="code:python", name="m.py")
        self.assertEqual(infer_domain(doc), "code:python")

    def test_genomics_keywords_win(self):
        doc = self._doc("The gene's DNA sequence determines the codon and chromosome mapping.")
        self.assertEqual(infer_domain(doc), "genomics")

    def test_medicine_keywords_win(self):
        doc = self._doc("The patient's dosage was adjusted after the clinical diagnosis by the physician.")
        self.assertEqual(infer_domain(doc), "medicine")

    def test_economics_keywords_win(self):
        doc = self._doc("Market equilibrium shifts with elasticity of demand and fiscal policy on inflation.")
        self.assertEqual(infer_domain(doc), "economics")

    def test_physics_keywords_win(self):
        doc = self._doc("The velocity and acceleration determine the force, momentum and kinetic energy.")
        self.assertEqual(infer_domain(doc), "physics")

    def test_game_engine_keywords_win(self):
        doc = self._doc("Every frame the render pipeline updates each entity collider in the game loop.")
        self.assertEqual(infer_domain(doc), "game_engine")

    def test_mathematics_keywords_win(self):
        doc = self._doc("The theorem's proof relies on the matrix eigenvalue and the integral of the polynomial.")
        self.assertEqual(infer_domain(doc), "mathematics")

    def test_default_domain_when_nothing_scores(self):
        doc = self._doc("Hello there, this is just a plain sentence.")
        self.assertEqual(infer_domain(doc), "general")


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


class TestDocumentLoader(unittest.TestCase):
    def test_load_directory_classifies_every_supported_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("hello")
            (root / "b.md").write_text("# hello")
            (root / "c.json").write_text("{}")
            (root / "d.py").write_text("x = 1\n")
            (root / "e.cpp").write_text("int x = 1;\n")
            (root / "unsupported.xyz").write_text("ignored")

            docs = DocumentLoader().load_directory(root)
            by_name = {doc.path.name: doc for doc in docs}
            self.assertEqual(by_name["a.txt"].format, "text")
            self.assertEqual(by_name["b.md"].format, "text")
            self.assertEqual(by_name["c.json"].format, "json")
            self.assertEqual(by_name["d.py"].format, "code:python")
            self.assertEqual(by_name["e.cpp"].format, "code:cpp")
            self.assertNotIn("unsupported.xyz", by_name)

    def test_load_directory_skips_ignored_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ignored = root / "__pycache__"
            ignored.mkdir()
            (ignored / "cached.py").write_text("x = 1\n")
            (root / "kept.py").write_text("y = 2\n")

            docs = DocumentLoader().load_directory(root)
            names = [doc.path.name for doc in docs]
            self.assertEqual(names, ["kept.py"])

    def test_load_directory_missing_root_returns_empty(self):
        docs = DocumentLoader().load_directory(Path("/nonexistent/does-not-exist-12345"))
        self.assertEqual(docs, [])

    def test_load_file_returns_none_for_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.xyz"
            path.write_text("data")
            self.assertIsNone(DocumentLoader().load_file(path))

    def test_load_file_reads_supported_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.json"
            path.write_text('{"a": 1}')
            doc = DocumentLoader().load_file(path)
            self.assertIsNotNone(doc)
            self.assertEqual(doc.format, "json")
            self.assertIsNone(doc.error)

    def test_per_file_error_is_isolated_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A directory whose name carries a recognised extension: reading it
            # as text raises IsADirectoryError, which must be captured on the
            # document rather than propagated.
            broken = Path(tmp) / "broken.py"
            broken.mkdir()
            doc = DocumentLoader().load_file(broken)
            self.assertIsNotNone(doc)
            self.assertIsNotNone(doc.error)
            self.assertEqual(doc.text, "")


# ---------------------------------------------------------------------------
# PDF text extraction (dependency-free fallback)
# ---------------------------------------------------------------------------


class TestPdfText(unittest.TestCase):
    def test_extract_text_from_plain_literal_stream(self):
        raw = b"<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td (Hello World) Tj ET\nendstream\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.pdf"
            path.write_bytes(raw)
            text = pdf_text.extract_text(path)
        self.assertIn("Hello World", text)

    def test_extract_text_inflates_flate_decode_stream(self):
        compressed = zlib.compress(b"BT (Compressed Text) Tj ET")
        raw = b"<< /Filter /FlateDecode >>\nstream\n" + compressed + b"\nendstream\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.pdf"
            path.write_bytes(raw)
            text = pdf_text.extract_text(path)
        self.assertIn("Compressed Text", text)

    def test_extract_text_handles_tj_array_operator(self):
        raw = b"<< /Length 10 >>\nstream\n[(Array) (Text)] TJ\nendstream\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.pdf"
            path.write_bytes(raw)
            text = pdf_text.extract_text(path)
        self.assertIn("Array", text)
        self.assertIn("Text", text)

    def test_extract_text_missing_file_returns_empty(self):
        self.assertEqual(pdf_text.extract_text(Path("/nonexistent/missing.pdf")), "")

    def test_unescape_handles_octal_and_backslash_escapes(self):
        self.assertEqual(pdf_text._unescape(rb"line1\nline2"), "line1\nline2")
        self.assertEqual(pdf_text._unescape(rb"\050paren\051"), "(paren)")


# ---------------------------------------------------------------------------
# Text rule extractor
# ---------------------------------------------------------------------------


class TestTextRuleExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = TextRuleExtractor()

    def _doc(self, text: str) -> IngestedDocument:
        return IngestedDocument(path=Path("notes.txt"), format="text", text=text)

    def test_empty_text_yields_empty_schema(self):
        schema = self.extractor.extract(self._doc(""), "general")
        self.assertEqual(schema.state_variables, [])
        self.assertEqual(schema.boundaries, [])
        self.assertEqual(schema.equations, [])

    def test_let_be_extracts_state_variable(self):
        doc = self._doc("Let x be the patient's heart rate.")
        schema = self.extractor.extract(doc, "medicine")
        self.assertEqual(len(schema.state_variables), 1)
        var = schema.state_variables[0]
        self.assertEqual(var.symbol, "x")
        self.assertEqual(var.description, "the patient's heart rate")

    def test_where_denotes_extracts_state_variable(self):
        doc = self._doc("where rate denotes the mutation frequency")
        schema = self.extractor.extract(doc, "genomics")
        self.assertEqual(schema.state_variables[0].symbol, "rate")

    def test_between_extracts_boundary(self):
        doc = self._doc("x must remain between 60 and 100 bpm.")
        schema = self.extractor.extract(doc, "medicine")
        self.assertEqual(len(schema.boundaries), 1)
        boundary = schema.boundaries[0]
        self.assertEqual(boundary.symbol, "x")
        self.assertEqual(boundary.expression, "60 <= x <= 100")

    def test_not_exceed_extracts_upper_bound(self):
        doc = self._doc("Dosage must not exceed 500 mg per day.")
        schema = self.extractor.extract(doc, "medicine")
        self.assertEqual(schema.boundaries[0].expression, "Dosage <= 500")

    def test_below_and_above_extract_strict_bounds(self):
        doc = self._doc("temp must remain below 90. pressure must remain above 10.")
        schema = self.extractor.extract(doc, "physics")
        expressions = {b.symbol: b.expression for b in schema.boundaries}
        self.assertEqual(expressions["temp"], "temp < 90")
        self.assertEqual(expressions["pressure"], "pressure > 10")

    def test_equation_with_decimal_rhs_is_not_truncated_at_decimal_point(self):
        doc = self._doc("The kinetic energy is given by E = 0.5 * m * v^2.")
        schema = self.extractor.extract(doc, "physics")
        self.assertEqual(len(schema.equations), 1)
        equation = schema.equations[0]
        self.assertEqual(equation.rhs, "0.5 * m * v^2")
        self.assertEqual(equation.expression, "E = 0.5 * m * v^2")
        self.assertEqual(equation.variables, ["m", "v"])

    def test_equation_variables_are_deduplicated(self):
        doc = self._doc("area = side * side")
        schema = self.extractor.extract(doc, "mathematics")
        self.assertEqual(schema.equations[0].variables, ["side"])

    def test_source_ref_tracks_line_number(self):
        doc = self._doc("line one\nline two\nLet y be the second-line variable.\n")
        schema = self.extractor.extract(doc, "general")
        self.assertEqual(schema.state_variables[0].source.line, 3)


# ---------------------------------------------------------------------------
# Code rule extractor
# ---------------------------------------------------------------------------


class TestCodeRuleExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = CodeRuleExtractor()

    def test_python_module_constant_becomes_state_variable(self):
        doc = IngestedDocument(path=Path("m.py"), format="code:python", text="MAX_RETRIES = 3\n")
        schema = self.extractor.extract(doc, "code:python")
        self.assertEqual(len(schema.state_variables), 1)
        var = schema.state_variables[0]
        self.assertEqual(var.symbol, "MAX_RETRIES")
        self.assertEqual(var.type_hint, "integer")
        self.assertEqual(var.bounds, (3.0, 3.0))

    def test_python_assert_becomes_boundary(self):
        doc = IngestedDocument(
            path=Path("m.py"), format="code:python", text="def f(x):\n    assert x > 0\n    return x\n"
        )
        schema = self.extractor.extract(doc, "code:python")
        self.assertEqual(len(schema.boundaries), 1)
        boundary = schema.boundaries[0]
        self.assertEqual(boundary.expression, "x > 0")
        self.assertIn("x", boundary.variables)

    def test_python_single_return_function_becomes_equation(self):
        doc = IngestedDocument(
            path=Path("m.py"),
            format="code:python",
            text="def area(radius):\n    return 3.14159 * radius * radius\n",
        )
        schema = self.extractor.extract(doc, "code:python")
        self.assertEqual(len(schema.equations), 1)
        equation = schema.equations[0]
        self.assertEqual(equation.symbol, "area")
        self.assertEqual(equation.rhs, "3.14159 * radius * radius")
        self.assertEqual(equation.variables, ["radius"])  # deduplicated

    def test_python_trivial_return_is_not_an_equation(self):
        doc = IngestedDocument(path=Path("m.py"), format="code:python", text="def f(x):\n    return x\n")
        schema = self.extractor.extract(doc, "code:python")
        self.assertEqual(schema.equations, [])

    def test_python_function_with_docstring_and_single_return_is_an_equation(self):
        doc = IngestedDocument(
            path=Path("m.py"),
            format="code:python",
            text='def area(r):\n    """Area of a circle."""\n    return 3.14159 * r * r\n',
        )
        schema = self.extractor.extract(doc, "code:python")
        self.assertEqual(len(schema.equations), 1)

    def test_python_syntax_error_yields_empty_schema(self):
        doc = IngestedDocument(path=Path("bad.py"), format="code:python", text="def (:\n")
        schema = self.extractor.extract(doc, "code:python")
        self.assertEqual(schema.state_variables, [])
        self.assertEqual(schema.boundaries, [])
        self.assertEqual(schema.equations, [])

    def test_cpp_define_becomes_state_variable(self):
        doc = IngestedDocument(path=Path("e.cpp"), format="code:cpp", text="#define MAX_ENTITIES 1024\n")
        schema = self.extractor.extract(doc, "code:cpp")
        self.assertEqual(schema.state_variables[0].symbol, "MAX_ENTITIES")
        self.assertEqual(schema.state_variables[0].bounds, (1024.0, 1024.0))
        self.assertEqual(schema.state_variables[0].type_hint, "integer")

    def test_cpp_const_float_becomes_state_variable_with_real_type(self):
        doc = IngestedDocument(path=Path("e.cpp"), format="code:cpp", text="const float GRAVITY = 9.8;\n")
        schema = self.extractor.extract(doc, "code:cpp")
        self.assertEqual(schema.state_variables[0].symbol, "GRAVITY")
        self.assertEqual(schema.state_variables[0].type_hint, "real")
        self.assertEqual(schema.state_variables[0].bounds, (9.8, 9.8))

    def test_cpp_string_define_becomes_string_state_variable(self):
        doc = IngestedDocument(path=Path("e.cpp"), format="code:cpp", text='#define VERSION "1.0.0"\n')
        schema = self.extractor.extract(doc, "code:cpp")
        self.assertEqual(schema.state_variables[0].type_hint, "string")

    def test_cpp_assert_becomes_boundary(self):
        doc = IngestedDocument(
            path=Path("e.cpp"),
            format="code:cpp",
            text="void tick() { assert(MAX_ENTITIES > 0); }\n",
        )
        schema = self.extractor.extract(doc, "code:cpp")
        self.assertEqual(len(schema.boundaries), 1)
        self.assertEqual(schema.boundaries[0].expression, "MAX_ENTITIES > 0")

    def test_cpp_const_identifier_rhs_becomes_equation(self):
        doc = IngestedDocument(path=Path("e.cpp"), format="code:cpp", text="const int DOUBLE_MAX = MAX_ENTITIES * 2;\n")
        schema = self.extractor.extract(doc, "code:cpp")
        self.assertEqual(len(schema.equations), 1)
        self.assertEqual(schema.equations[0].variables, ["MAX_ENTITIES"])


# ---------------------------------------------------------------------------
# JSON rule extractor
# ---------------------------------------------------------------------------


class TestJsonRuleExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = JsonRuleExtractor()

    def _doc(self, payload) -> IngestedDocument:
        return IngestedDocument(path=Path("data.json"), format="json", text=json.dumps(payload))

    def test_structured_state_variables_passthrough(self):
        doc = self._doc({"state_variables": [{"symbol": "x", "description": "demo", "bounds": [0, 1]}]})
        schema = self.extractor.extract(doc, "general")
        self.assertEqual(schema.state_variables[0].symbol, "x")
        self.assertEqual(schema.state_variables[0].bounds, (0, 1))

    def test_structured_constraints_alias_maps_to_boundaries(self):
        doc = self._doc({"constraints": [{"symbol": "x", "expression": "x > 0", "variables": ["x"]}]})
        schema = self.extractor.extract(doc, "general")
        self.assertEqual(schema.boundaries[0].expression, "x > 0")

    def test_structured_equations_passthrough(self):
        doc = self._doc({"equations": [{"lhs": "y", "rhs": "x * 2", "variables": ["x"]}]})
        schema = self.extractor.extract(doc, "general")
        self.assertEqual(schema.equations[0].symbol, "y")
        self.assertEqual(schema.equations[0].rhs, "x * 2")

    def test_generic_flatten_scalars_into_state_variables(self):
        doc = self._doc({"threshold": 42, "name": "demo", "nested": {"skip": True}, "list": [1, 2]})
        schema = self.extractor.extract(doc, "general")
        symbols = {v.symbol: v for v in schema.state_variables}
        self.assertEqual(symbols["threshold"].bounds, (42.0, 42.0))
        self.assertEqual(symbols["threshold"].type_hint, "integer")
        self.assertEqual(symbols["name"].type_hint, "string")
        self.assertNotIn("nested", symbols)
        self.assertNotIn("list", symbols)

    def test_invalid_json_yields_empty_schema(self):
        doc = IngestedDocument(path=Path("bad.json"), format="json", text="{not valid json")
        schema = self.extractor.extract(doc, "general")
        self.assertEqual(schema.state_variables, [])


# ---------------------------------------------------------------------------
# LLM-assisted extractor / extensibility interface
# ---------------------------------------------------------------------------


class TestLLMAssistedExtractor(unittest.TestCase):
    def test_null_llm_client_raises_on_complete(self):
        with self.assertRaises(RuntimeError):
            NullLLMClient().complete("prompt")

    def test_falls_back_to_offline_extractor_when_client_unconfigured(self):
        doc = IngestedDocument(path=Path("m.py"), format="code:python", text="MAX = 3\n")
        extractor = LLMAssistedExtractor(client=NullLLMClient(), fallback=CodeRuleExtractor())
        schema = extractor.extract(doc, "code:python")
        self.assertEqual(schema.state_variables[0].symbol, "MAX")

    def test_falls_back_to_offline_extractor_even_when_client_succeeds(self):
        class EchoClient(LLMClient):
            def complete(self, prompt: str) -> str:
                return "{}"

        doc = IngestedDocument(path=Path("m.py"), format="code:python", text="MAX = 3\n")
        extractor = LLMAssistedExtractor(client=EchoClient(), fallback=CodeRuleExtractor())
        schema = extractor.extract(doc, "code:python")
        self.assertEqual(schema.state_variables[0].symbol, "MAX")


# ---------------------------------------------------------------------------
# System graph
# ---------------------------------------------------------------------------


class TestSystemGraph(unittest.TestCase):
    def test_add_schema_creates_domain_and_defines_edges(self):
        schema = InvariantSchema(state_variables=[StateVariable(domain="physics", symbol="v")])
        graph = SystemGraph()
        graph.add_schema(schema)
        self.assertTrue(graph.graph.has_node("domain::physics"))
        self.assertTrue(graph.graph.has_node("physics::v"))
        self.assertEqual(graph.graph["domain::physics"]["physics::v"]["edge_type"], "defines")

    def test_boundary_references_same_domain_state_variable(self):
        schema = InvariantSchema(
            state_variables=[StateVariable(domain="physics", symbol="x")],
            boundaries=[
                AlgorithmicBoundary(domain="physics", symbol="b", description="", expression="x>0", variables=["x"])
            ],
        )
        graph = SystemGraph()
        graph.add_schema(schema)
        self.assertEqual(graph.graph["physics::b"]["physics::x"]["edge_type"], "references")

    def test_auto_link_shared_symbols_links_across_domains_without_merging(self):
        schema = InvariantSchema(
            state_variables=[
                StateVariable(domain="genomics", symbol="rate"),
                StateVariable(domain="game_engine", symbol="rate"),
            ]
        )
        graph = SystemGraph()
        graph.add_schema(schema)
        added = graph.auto_link_shared_symbols()
        self.assertEqual(added, 2)  # symmetric edge in both directions
        self.assertEqual(graph.graph["genomics::rate"]["game_engine::rate"]["edge_type"], "shared_symbol")
        self.assertEqual(graph.graph["game_engine::rate"]["genomics::rate"]["edge_type"], "shared_symbol")
        # Both nodes still exist independently -- this never merges definitions.
        self.assertEqual(graph.graph.nodes["genomics::rate"]["domain"], "genomics")
        self.assertEqual(graph.graph.nodes["game_engine::rate"]["domain"], "game_engine")
        self.assertNotEqual("genomics::rate", "game_engine::rate")

    def test_auto_link_shared_symbols_ignores_same_domain_duplicates(self):
        schema = InvariantSchema(
            state_variables=[
                StateVariable(domain="physics", symbol="x"),
                StateVariable(domain="physics", symbol="x#2"),
            ]
        )
        graph = SystemGraph()
        graph.add_schema(schema)
        added = graph.auto_link_shared_symbols()
        self.assertEqual(added, 0)

    def test_link_domains_adds_explicit_edge(self):
        schema = InvariantSchema(
            state_variables=[
                StateVariable(domain="genomics", symbol="rate"),
                StateVariable(domain="economics", symbol="rate"),
            ]
        )
        graph = SystemGraph()
        graph.add_schema(schema)
        graph.link_domains("genomics", "rate", "economics", "rate", relation="maps_to")
        self.assertEqual(graph.graph["genomics::rate"]["economics::rate"]["edge_type"], "maps_to")

    def test_link_domains_raises_keyerror_for_missing_node(self):
        graph = SystemGraph()
        graph.add_domain("genomics")
        with self.assertRaises(KeyError):
            graph.link_domains("genomics", "missing", "economics", "missing")

    def test_statistics_and_to_dict_shapes(self):
        schema = InvariantSchema(
            state_variables=[
                StateVariable(domain="genomics", symbol="rate"),
                StateVariable(domain="game_engine", symbol="rate"),
            ]
        )
        graph = SystemGraph()
        graph.add_schema(schema)
        graph.auto_link_shared_symbols()
        stats = graph.statistics()
        self.assertEqual(stats["domain_count"], 2)
        self.assertEqual(set(stats["domains"]), {"genomics", "game_engine"})
        self.assertGreaterEqual(stats["cross_domain_edges"], 2)

        as_dict = graph.to_dict()
        self.assertIn("nodes", as_dict)
        self.assertIn("edges", as_dict)
        node_ids = {node["id"] for node in as_dict["nodes"]}
        self.assertIn("genomics::rate", node_ids)
        self.assertIn("game_engine::rate", node_ids)


# ---------------------------------------------------------------------------
# ContextIngestionEngine (end-to-end)
# ---------------------------------------------------------------------------


class TestContextIngestionEngine(unittest.TestCase):
    def _build_mixed_directory(self, root: Path) -> None:
        (root / "genome.txt").write_text(
            "DNA sequences encode genes and mutations.\n"
            "Let rate be the mutation rate observed in the genome.\n"
        )
        (root / "game.txt").write_text(
            "The render pipeline updates every frame in the game loop entity system.\n"
            "Let rate be the frame render rate for the game loop.\n"
        )
        (root / "config.json").write_text(json.dumps({"max_retries": 3, "timeout": 1.5}))
        (root / "engine.cpp").write_text(
            "#define MAX_ENTITIES 1024\nconst float GRAVITY = 9.8;\n"
            "void tick() { assert(MAX_ENTITIES > 0); }\n"
        )
        (root / "model.py").write_text("MAX_RETRIES = 3\ndef area(radius):\n    return 3.14159 * radius * radius\n")

    def test_ingest_directory_segregates_domains_without_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_mixed_directory(root)
            engine = ContextIngestionEngine()
            schema = engine.ingest_directory(root)

        self.assertIn("genomics", schema.domains)
        self.assertIn("game_engine", schema.domains)
        self.assertIn("code:cpp", schema.domains)
        self.assertIn("code:python", schema.domains)

        ids = [v.id for v in schema.state_variables]
        self.assertIn("genomics::rate", ids)
        self.assertIn("game_engine::rate", ids)
        # Same bare symbol, two distinct namespaced ids -- no collision.
        self.assertEqual(len(ids), len(set(ids)))

    def test_ingest_directory_builds_cross_domain_graph_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_mixed_directory(root)
            engine = ContextIngestionEngine()
            engine.ingest_directory(root)
            stats = engine.last_graph.statistics()

        self.assertGreaterEqual(stats["cross_domain_edges"], 2)

    def test_to_compilation_inputs_round_trips_through_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_mixed_directory(root)
            engine = ContextIngestionEngine()
            schema = engine.ingest_directory(root)
            payload = engine.to_compilation_inputs(schema)

        reloaded = json.loads(json.dumps(payload))
        self.assertEqual(validate_invariant_document(reloaded["invariant_schema"]), [])
        self.assertIn("graph_statistics", reloaded)
        self.assertEqual(reloaded["ingestion_errors"], [])

    def test_ingest_and_export_writes_report_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_mixed_directory(root)
            output_path = root / "out" / "report.json"
            engine = ContextIngestionEngine()
            payload = engine.ingest_and_export(root, output_path)

            self.assertTrue(output_path.exists())
            on_disk = json.loads(output_path.read_text())
            self.assertEqual(on_disk["invariant_schema"]["domains"], payload["invariant_schema"]["domains"])

    def test_empty_directory_yields_empty_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = ContextIngestionEngine()
            schema = engine.ingest_directory(Path(tmp))
        self.assertEqual(schema.to_dict()["domains"], [])


# ---------------------------------------------------------------------------
# CLI integration (main.py "invariants" subcommand)
# ---------------------------------------------------------------------------


class TestInvariantsCli(unittest.TestCase):
    def test_parser_wires_invariants_subcommand(self):
        args = main.create_parser().parse_args(["invariants", "--source-dir", "ctx"])
        self.assertIs(args.handler, main.invariants_command)
        self.assertEqual(args.source_dir, "ctx")
        self.assertEqual(args.workspace, ".")
        self.assertIsNone(args.output)

    def test_invariants_command_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.txt").write_text("Let x be a demo variable.\n")
            out = io.StringIO()
            with redirect_stdout(out):
                rc = main.main(["invariants", "--source-dir", str(root), "--workspace", str(root)])
            self.assertEqual(rc, 0)
            self.assertIn("Semantic Fluidity Ingestion:", out.getvalue())
            self.assertTrue((root / ContextIngestionEngine.REPORT_NAME).exists())

    def test_invariants_command_reports_missing_source_dir(self):
        err = io.StringIO()
        with redirect_stderr(err):
            rc = main.main(["invariants", "--source-dir", "/nonexistent/missing-context-dir-12345"])
        self.assertEqual(rc, 1)
        self.assertIn("directory not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
