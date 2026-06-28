# -*- coding: utf-8 -*-
"""
Cargo manifest handling for Aero's Rust backend.

This module decides, for a Rust target, **whether to respect a user-provided
``Cargo.toml`` or synthesise a fresh one**, and where the crate actually lives
(so subdirectory crates like ``crates/foo/`` build correctly).

Behaviour
---------
* If a ``Cargo.toml`` already exists at the crate root -- either discovered next
  to the target's sources, or pointed at explicitly via ``manifest_path`` /
  ``root`` -- it is used **verbatim**.  Aero never rewrites a manifest a user
  committed, so builds that pin older dependency APIs keep working.
* If no manifest exists, one is synthesised.  Dependency versions can be pinned
  from the blueprint via a ``cargo.dependencies`` mapping
  (e.g. ``{"rug": "0.22"}`` or ``{"rug": {"version": "0.22", "features": [...]}}``).
* The resolved crate root is returned so the build runs ``cargo`` from the right
  directory and collects artefacts from *that* crate's ``target/`` directory.

No third-party TOML writer is required: existing manifests are read with the
stdlib :mod:`tomllib` (read-only, 3.11+), and synthesised manifests are rendered
by a tiny, well-tested emitter below.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("aero.cargo")

MANIFEST_NAME = "Cargo.toml"
DEFAULT_EDITION = "2021"
DEFAULT_VERSION = "0.1.0"

# Features that a ``pyo3`` dependency must always carry so modern structural
# (declarative) ``#[pymodule]`` macros compile without manual user edits.  The
# framework injects these implicitly whenever ``pyo3`` is a target dependency.
PYO3_REQUIRED_FEATURES = ("extension-module", "experimental-declarative-modules")


def ensure_pyo3_features(dependencies: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Guarantee a ``pyo3`` dependency declares the declarative-modules features.

    Whenever ``pyo3`` appears in *dependencies* the inline-table form is forced
    and :data:`PYO3_REQUIRED_FEATURES` are merged in (preserving any features the
    caller already pinned and their order).  A bare version string such as
    ``"0.21"`` is upgraded to ``{"version": "0.21", "features": [...]}``.

    The input mapping is never mutated; a normalised copy is returned.
    """
    if not dependencies or "pyo3" not in dependencies:
        return dict(dependencies or {})

    deps = dict(dependencies)
    spec = deps["pyo3"]
    if isinstance(spec, dict):
        new_spec = dict(spec)
        features = list(new_spec.get("features", []) or [])
    else:
        # Bare version string (or anything scalar) -> promote to a table.
        new_spec = {"version": str(spec)}
        features = []
    for feature in PYO3_REQUIRED_FEATURES:
        if feature not in features:
            features.append(feature)
    new_spec["features"] = features
    deps["pyo3"] = new_spec
    return deps


# ---------------------------------------------------------------------------
# Plan returned to the compiler backend
# ---------------------------------------------------------------------------


@dataclass
class CargoPlan:
    """How (and where) a Rust target should be built."""

    crate_root: Path
    manifest_path: Path
    crate_name: str
    used_existing: bool
    synthesized: bool
    dependencies: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def target_dir(self) -> Path:
        """The crate's own ``target/`` directory (where cargo writes artefacts)."""
        return self.crate_root / "target"

    def profile_dir(self, release: bool = False) -> Path:
        """The directory holding compiled artefacts for the selected profile."""
        return self.target_dir / ("release" if release else "debug")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crate_root": str(self.crate_root),
            "manifest_path": str(self.manifest_path),
            "crate_name": self.crate_name,
            "used_existing": self.used_existing,
            "synthesized": self.synthesized,
            "dependencies": dict(self.dependencies),
            "target_dir": str(self.target_dir),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Crate-root resolution (fixes subdirectory crates)
# ---------------------------------------------------------------------------


def _walk_up_for_manifest(start: Path, ceiling: Path) -> Optional[Path]:
    """Walk up from ``start`` to ``ceiling`` looking for a ``Cargo.toml``."""
    start = start.resolve()
    ceiling = ceiling.resolve()
    current = start if start.is_dir() else start.parent
    while True:
        candidate = current / MANIFEST_NAME
        if candidate.is_file():
            return candidate
        if current == ceiling or current.parent == current:
            return None
        if ceiling not in current.parents and current != ceiling:
            # Stepped outside the workspace -- stop.
            return None
        current = current.parent


def resolve_crate_root(
    workspace: Path,
    sources: Sequence[str] = (),
    manifest_path: Optional[str] = None,
    root: Optional[str] = None,
) -> Path:
    """Determine the directory that is (or will hold) the crate's manifest.

    Precedence:

    1. ``manifest_path`` -- a path to a ``Cargo.toml`` (its parent is the crate
       root) or to a directory containing one.
    2. ``root`` -- a subdirectory of the workspace that is the crate root.
    3. an existing ``Cargo.toml`` discovered by walking up from the target's
       sources (so ``crates/foo/src/lib.rs`` resolves to ``crates/foo``).
    4. the workspace root, as a last resort.
    """
    workspace = Path(workspace).resolve()

    if manifest_path:
        candidate = (workspace / manifest_path).resolve()
        if candidate.name == MANIFEST_NAME or candidate.suffix == ".toml":
            return candidate.parent
        return candidate

    if root:
        return (workspace / root).resolve()

    # Infer from the sources: prefer an existing manifest above any source.
    for source in sources:
        source_path = (workspace / source).resolve()
        found = _walk_up_for_manifest(source_path, workspace)
        if found is not None:
            return found.parent

    # No manifest discovered: use the common parent directory of the sources if
    # they all live under one subdirectory, else the workspace itself.
    source_dirs = {
        (workspace / s).resolve().parent for s in sources if s
    }
    if len(source_dirs) == 1:
        only = next(iter(source_dirs))
        # If sources sit in a conventional ``src/`` dir, the crate root is its parent.
        if only.name == "src":
            return only.parent
        return only
    return workspace


def find_existing_manifest(crate_root: Path) -> Optional[Path]:
    """Return ``crate_root/Cargo.toml`` if it exists, else ``None``."""
    candidate = Path(crate_root) / MANIFEST_NAME
    return candidate if candidate.is_file() else None


def read_manifest_package_name(manifest_path: Path) -> Optional[str]:
    """Best-effort read of ``[package].name`` from an existing manifest."""
    try:
        import tomllib

        with open(manifest_path, "rb") as fh:
            data = tomllib.load(fh)
        name = data.get("package", {}).get("name")
        return str(name) if name else None
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - tomllib errors, malformed manifest
        logger.debug("Could not read package name from %s: %s", manifest_path, exc)
        return None


# ---------------------------------------------------------------------------
# Manifest synthesis (TOML emitter)
# ---------------------------------------------------------------------------


def sanitize_crate_name(name: str) -> str:
    """Turn an arbitrary target name into a valid Cargo crate name."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip()).strip("_")
    if not cleaned:
        cleaned = "aero_crate"
    if cleaned[0].isdigit():
        cleaned = f"crate_{cleaned}"
    return cleaned.lower()


def _render_toml_scalar(value: Any) -> str:
    """Render a Python scalar/list as a TOML value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render_toml_scalar(item) for item in value) + "]"
    # default: string
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_dependency(name: str, spec: Any) -> str:
    """Render one ``[dependencies]`` entry.

    ``spec`` may be a bare version string (``"0.22"``) or an inline table
    (``{"version": "0.22", "features": ["std"], "optional": true}``).
    """
    if isinstance(spec, dict):
        inner = ", ".join(f"{key} = {_render_toml_scalar(val)}" for key, val in spec.items())
        return f"{name} = {{ {inner} }}"
    return f"{name} = {_render_toml_scalar(str(spec))}"


def render_manifest(
    crate_name: str,
    dependencies: Optional[Dict[str, Any]] = None,
    edition: str = DEFAULT_EDITION,
    version: str = DEFAULT_VERSION,
    crate_type: Optional[Sequence[str]] = None,
    header: Optional[str] = None,
) -> str:
    """Render a minimal, valid ``Cargo.toml`` as text.

    ``header`` overrides the default "synthesised, commit your own" comment --
    standalone-repo generators pass a header appropriate to a committed manifest.
    """
    # Implicitly inject the declarative-modules feature set whenever pyo3 is a
    # dependency, so generated manifests build against modern structural macros.
    dependencies = ensure_pyo3_features(dependencies or {})
    if header is None:
        header_lines = [
            "# Synthesised by Aero Universal. Commit a Cargo.toml to take full control;",
            "# Aero will then use it verbatim and never overwrite it.",
        ]
    else:
        header_lines = [line if line.startswith("#") or not line else f"# {line}" for line in header.splitlines()]
    lines = [
        *header_lines,
        "[package]",
        f'name = "{crate_name}"',
        f'version = "{version}"',
        f'edition = "{edition}"',
        "",
    ]
    if crate_type:
        lines.extend(["[lib]", f"crate-type = {_render_toml_scalar(list(crate_type))}", ""])
    lines.append("[dependencies]")
    for name, spec in dependencies.items():
        lines.append(_render_dependency(name, spec))
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def prepare_crate(
    workspace: Path,
    target_name: str,
    sources: Sequence[str] = (),
    cargo_options: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[str] = None,
    root: Optional[str] = None,
    write: bool = True,
) -> CargoPlan:
    """Resolve the crate root and either respect or synthesise its manifest.

    Returns a :class:`CargoPlan` describing where ``cargo`` should run and
    whether an existing manifest was honoured or a new one written.
    """
    workspace = Path(workspace).resolve()
    cargo_options = cargo_options or {}

    crate_root = resolve_crate_root(workspace, sources, manifest_path, root)

    # An explicit manifest_path may point straight at an existing file.
    explicit_manifest: Optional[Path] = None
    if manifest_path:
        candidate = (workspace / manifest_path).resolve()
        if candidate.is_file():
            explicit_manifest = candidate
            crate_root = candidate.parent

    existing = explicit_manifest or find_existing_manifest(crate_root)
    if existing is not None:
        crate_name = read_manifest_package_name(existing) or sanitize_crate_name(target_name)
        return CargoPlan(
            crate_root=crate_root,
            manifest_path=existing,
            crate_name=crate_name,
            used_existing=True,
            synthesized=False,
            dependencies={},
            notes=[f"using existing manifest at {existing}"],
        )

    # No manifest: synthesise one, honouring blueprint-pinned dependency versions.
    crate_name = sanitize_crate_name(cargo_options.get("package_name", target_name))
    dependencies = dict(cargo_options.get("dependencies", {}) or {})
    edition = str(cargo_options.get("edition", DEFAULT_EDITION))
    version = str(cargo_options.get("version", DEFAULT_VERSION))
    crate_type = cargo_options.get("crate_type")

    manifest = crate_root / MANIFEST_NAME
    notes = [f"synthesised manifest for crate '{crate_name}'"]
    if dependencies:
        notes.append("pinned dependencies: " + ", ".join(f"{k}={v}" for k, v in dependencies.items()))

    if write:
        crate_root.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            render_manifest(crate_name, dependencies, edition, version, crate_type),
            encoding="utf-8",
        )

    return CargoPlan(
        crate_root=crate_root,
        manifest_path=manifest,
        crate_name=crate_name,
        used_existing=False,
        synthesized=True,
        dependencies=dependencies,
        notes=notes,
    )


def _parse_dependency_list(entries: Sequence[str]) -> Dict[str, str]:
    """Parse ``["rug=0.22", "serde = 1.0"]`` into ``{"rug": "0.22", ...}``.

    This is the flat form the block-DSL / INI dialects use, since they cannot
    express the nested ``cargo = { dependencies = { ... } }`` table.
    """
    deps: Dict[str, str] = {}
    for entry in entries:
        if "=" not in str(entry):
            continue
        name, _, version = str(entry).partition("=")
        name = name.strip()
        if name:
            deps[name] = version.strip().strip('"')
    return deps


def extract_cargo_options(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the ``cargo`` option block out of a target's metadata dict.

    Accepts either a nested ``cargo`` object (JSON blueprint) or top-level
    ``cargo_dependencies`` (a convenience for the flatter dialects), where the
    latter may be a ``{name: version}`` mapping or a ``["name=version"]`` list.
    """
    options: Dict[str, Any] = {}
    cargo = meta.get("cargo")
    if isinstance(cargo, dict):
        options.update(cargo)

    flat = meta.get("cargo_dependencies")
    if isinstance(flat, dict):
        options.setdefault("dependencies", {}).update(flat)
    elif isinstance(flat, (list, tuple)):
        options.setdefault("dependencies", {}).update(_parse_dependency_list(flat))
    return options
