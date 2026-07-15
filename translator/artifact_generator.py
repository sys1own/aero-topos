"""Template-agnostic artifact generator for the Aero FFI bridge.

This module replaces the hard-coded, project-specific code generation in
``translator.ffi_codegen`` with a registry-driven, lightweight templating
system.  All domain-specific logic now lives in external ``templates/ffi/``
files and in the ``registry.json`` mapping; the generator only knows how to
render templates, detect name collisions, and guarantee idempotent output.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from translator.rust_ast import RustFn, safe_ident


class TemplateNotFoundError(Exception):
    """Raised when a requested template cannot be found in the search path."""


class NameConflictError(Exception):
    """Raised when a requested definition collides with an existing name."""


@dataclass
class GenerationReport:
    """Outcome of an artifact generation run."""

    files: Dict[str, str] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    emitted: List[str] = field(default_factory=list)


class ArtifactGenerator:
    """Generate Rust FFI artifacts from templates and a blueprint/registry.

    The generator searches for templates in this order:

    1. ``extra_template_dirs`` supplied by the caller / blueprint.
    2. Any ``artifact_generation.template_dirs`` entry in the blueprint.
    3. ``<repo-root>/templates/ffi`` (project/user overrides).
    4. ``<repo-root>/translator/templates/ffi`` (package defaults).
    """

    def __init__(
        self,
        blueprint: Optional[Dict[str, Any]] = None,
        extra_template_dirs: Optional[Iterable[Path]] = None,
        package_template_dir: Optional[Path] = None,
    ) -> None:
        self.blueprint: Dict[str, Any] = dict(blueprint or {})
        self.template_dirs: List[Path] = []

        if extra_template_dirs:
            for d in extra_template_dirs:
                p = Path(d)
                if p.is_dir():
                    self.template_dirs.append(p)

        # Allow the blueprint to point at project-local template directories.
        bp_dirs = self._blueprint_template_dirs()
        if bp_dirs:
            self.template_dirs.extend(bp_dirs)

        # Repo-root user/project overrides.
        repo_root = Path(__file__).parent.parent
        repo_templates = repo_root / "templates" / "ffi"
        if repo_templates.is_dir():
            self.template_dirs.append(repo_templates)

        # Package default templates.
        if package_template_dir is None:
            package_template_dir = repo_root / "translator" / "templates" / "ffi"
        if package_template_dir.is_dir():
            self.template_dirs.append(package_template_dir)

        self._registry: Dict[str, Dict[str, str]] = {}
        self._cache: Dict[str, str] = {}
        self._emitted: Set[str] = set()
        self._output_cache: Dict[str, str] = {}
        self._emitted_keys: Set[tuple[str, str]] = set()

        self._load_registries()

    # ------------------------------------------------------------------
    # Template discovery and rendering
    # ------------------------------------------------------------------

    def _blueprint_template_dirs(self) -> List[Path]:
        """Return project-local template directories declared in the blueprint."""
        ag = self.blueprint.get("artifact_generation")
        if not isinstance(ag, dict):
            return []
        raw = ag.get("template_dirs", [])
        if isinstance(raw, str):
            raw = [raw]
        dirs: List[Path] = []
        for d in raw:
            p = Path(d)
            if p.is_dir():
                dirs.append(p)
        return dirs

    @staticmethod
    def _deep_update(
        target: Dict[str, Any], source: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge *source* into *target* (in-place)."""
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                ArtifactGenerator._deep_update(target[key], value)
            else:
                target[key] = value
        return target

    def _load_registries(self) -> None:
        """Merge ``registry.json`` files found in the template search path.

        Registries are loaded from lowest-precedence directories first so that
        project-level and blueprint-level entries override package defaults.
        Nested entries are merged rather than replaced, so a user registry that
        only overrides ``default.wrapper`` does not erase ``default.legacy``.
        """
        for directory in reversed(self.template_dirs):
            path = directory / "registry.json"
            if not path.is_file():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._deep_update(self._registry, data)
            except (json.JSONDecodeError, OSError):
                continue

    def register(self, name: str, mapping: Dict[str, str]) -> None:
        """Register a name -> {wrapper, legacy, ...} template mapping at runtime."""
        self._registry[name] = mapping

    def resolve(self, template_name: str) -> Path:
        """Locate a template file by name, appending ``.rs`` when unqualified."""
        candidate = template_name
        if not candidate.endswith((".rs", ".json")):
            candidate = candidate + ".rs"
        for directory in self.template_dirs:
            path = directory / candidate
            if path.is_file():
                return path
        raise TemplateNotFoundError(
            f"Template '{template_name}' not found in any of {self.template_dirs}"
        )

    def load(self, template_name: str) -> str:
        """Load and cache a template file."""
        if template_name not in self._cache:
            path = self.resolve(template_name)
            self._cache[template_name] = path.read_text(encoding="utf-8")
        return self._cache[template_name]

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with :class:`string.Template` substitution."""
        template = self.load(template_name)
        return string.Template(template).substitute(context)

    def list_templates(self) -> List[str]:
        """List every template visible in the search path."""
        names: Set[str] = set()
        for directory in self.template_dirs:
            if not directory.is_dir():
                continue
            for p in directory.iterdir():
                if p.is_file():
                    names.add(p.name)
        return sorted(names)

    # ------------------------------------------------------------------
    # Naming, collisions, and idempotency
    # ------------------------------------------------------------------

    @staticmethod
    def existing_definitions(source: str) -> Set[str]:
        """Return the set of top-level Rust function names in *source*.

        Uses the Tree-sitter backed extractor when available, otherwise falls
        back to a conservative regex scan so collision checks work even when
        the Rust parser is unavailable.
        """
        try:
            from translator.rust_ast import extract_functions

            return {f.name for f in extract_functions(source)}
        except Exception:
            return set(re.findall(r"\b(?:pub\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)", source))

    def check_conflicts(
        self, requested_names: Iterable[str], source: str
    ) -> List[str]:
        """Return any *requested_names* that already exist in *source*."""
        existing = self.existing_definitions(source)
        return sorted({n for n in requested_names if n in existing})

    def unique_name(self, name: str, existing: Set[str], suffix: str = "_generated") -> str:
        """Return a unique symbol name, avoiding collisions and re-emission."""
        if name not in existing and name not in self._emitted:
            return name
        i = 1
        candidate = f"{name}{suffix}{i}"
        while candidate in existing or candidate in self._emitted:
            i += 1
            candidate = f"{name}{suffix}{i}"
        return candidate

    def _mark_emitted(self, name: str) -> None:
        self._emitted.add(name)

    # ------------------------------------------------------------------
    # Node rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _node_context(
        node: Any,
        *,
        aeroc_module: str = "",
        aero_node_cap: int = 4096,
        stub_expr: str = "",
    ) -> Dict[str, Any]:
        """Build a rendering context from an ``FfiNode``."""
        name = safe_ident(node.fn.name)
        ctx: Dict[str, Any] = {
            "name": name,
            "index": getattr(node, "index", 0),
            "hook": node.hook,
            "legacy": node.legacy,
            "signature": node.fn.signature.rstrip(),
            "return_type": node.fn.return_type,
            "fn_name": node.fn.name,
            "aeroc_module": aeroc_module,
            "AERO_NODE_CAP": aero_node_cap,
        }
        if stub_expr:
            ctx["stub_expr"] = stub_expr
        return ctx

    def _template_for(self, node_name: str, kind: str) -> str:
        """Resolve the template for *kind* (wrapper/legacy/...) for a node."""
        mapping = self._registry.get(node_name, {})
        if kind in mapping:
            return mapping[kind]
        default = self._registry.get("default", {})
        if kind in default:
            return default[kind]
        if kind == "wrapper":
            return "wrapper_generic"
        if kind == "legacy":
            return "legacy_generic"
        raise TemplateNotFoundError(f"No default template for kind '{kind}'")

    def wrapper(self, node: Any) -> str:
        """Render the wrapper (hot-path replacement) for *node*.

        Calling this with an identical node (same name + hook) more than once is
        idempotent: the cached output is returned and no duplicate definition is
        emitted.
        """
        name = safe_ident(node.fn.name)
        key = (name, node.hook)
        if key in self._emitted_keys:
            return self._output_cache[name]
        if name in self._emitted:
            raise NameConflictError(
                f"Wrapper for '{name}' was already emitted by this generator"
            )
        template = self._template_for(name, "wrapper")
        ctx = self._node_context(node)
        rendered = self.render(template, ctx)
        self._mark_emitted(name)
        self._emitted_keys.add(key)
        self._output_cache[name] = rendered
        return rendered

    def legacy(self, node: Any) -> str:
        """Render the legacy (verification stub) implementation for *node*."""
        name = safe_ident(node.fn.name)
        legacy_name = f"{name}_legacy"
        key = (legacy_name, node.hook)
        if key in self._emitted_keys:
            return self._output_cache[legacy_name]
        if legacy_name in self._emitted:
            raise NameConflictError(
                f"Legacy implementation '{legacy_name}' was already emitted"
            )
        template = self._template_for(name, "legacy")
        ctx = self._node_context(node)
        rendered = self.render(template, ctx)
        self._mark_emitted(legacy_name)
        self._emitted_keys.add(key)
        self._output_cache[legacy_name] = rendered
        return rendered

    def aero_ffi_header(self, aeroc_module: str, aero_node_cap: int = 4096) -> str:
        """Render the shared ``aero_ffi.rs`` header."""
        ctx = {
            "aeroc_module": aeroc_module,
            "AERO_NODE_CAP": aero_node_cap,
        }
        return self.render("aero_ffi_header", ctx)

    def aero_ffi_hook(
        self,
        node: Any,
        *,
        aeroc_module: str = "",
        aero_node_cap: int = 4096,
        stub_expr: str = "Ok(input.to_vec())",
    ) -> str:
        """Render one ``aero_execute_node{N}`` hook function."""
        ctx = self._node_context(
            node,
            aeroc_module=aeroc_module,
            aero_node_cap=aero_node_cap,
            stub_expr=stub_expr,
        )
        return self.render("aero_ffi_hook", ctx)

    # ------------------------------------------------------------------
    # High-level artifact generation
    # ------------------------------------------------------------------

    def generate_ffi_artifacts(
        self,
        source: str,
        nodes: Iterable[Any],
        aeroc_module: str,
        aero_node_cap: int = 4096,
    ) -> GenerationReport:
        """Generate a complete set of FFI artifacts from *source* and *nodes*.

        Returns a :class:`GenerationReport` containing the strings that would
        be written to ``lib.rs`` (wrappers), ``legacy.rs`` (verification stubs),
        and ``aero_ffi.rs`` (bridge module).  Before any rendering, the
        requested wrapper/legacy names are checked against the definitions
        already present in *source*; conflicts are reported and not written.
        """
        report = GenerationReport()
        node_list = list(nodes)

        requested: Set[str] = set()
        for node in node_list:
            name = safe_ident(node.fn.name)
            requested.add(name)
            requested.add(f"{name}_legacy")

        report.conflicts = self.check_conflicts(requested, source)
        if report.conflicts:
            return report

        wrappers: List[str] = []
        legacies: List[str] = []
        hooks: List[str] = []

        seen: Set[str] = set()
        for node in node_list:
            name = safe_ident(node.fn.name)
            if name in seen:
                continue
            seen.add(name)

            wrappers.append(self.wrapper(node))
            legacies.append(self.legacy(node))
            stub = getattr(node, "stub_expr", None) or "Ok(input.to_vec())"
            hooks.append(
                self.aero_ffi_hook(
                    node,
                    aeroc_module=aeroc_module,
                    aero_node_cap=aero_node_cap,
                    stub_expr=stub,
                )
            )

        header = self.aero_ffi_header(aeroc_module, aero_node_cap=aero_node_cap)
        report.files = {
            "lib.rs": "".join(wrappers),
            "legacy.rs": "".join(legacies),
            "aero_ffi.rs": header + "\n".join(hooks),
        }
        report.emitted = sorted(self._emitted)
        return report

    def write_file(self, path: Path, content: str) -> bool:
        """Idempotently write *content* to *path*, returning True if changed."""
        path = Path(path)
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing == content:
                return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True
