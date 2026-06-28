"""Loader for the AeroNova *living blueprint* schema.

The living blueprint is a TOML document that describes a project as an evolving
system rather than a one-shot build manifest.  This module turns that document
into typed dataclasses.  Every section is optional: a blueprint that omits a
section (or the file entirely) falls back to sensible defaults so that minimal
blueprints — like the one shipped at the repo root — load cleanly.

Sections understood here:

    [system]            high-level identity + build strategy
    [context_registry]  named contexts (legacy/modern/unrelated codebases)
    [abstractions]      legacy -> target rewrite rules
    [scaling]           auto-decomposition / complexity policy
    [context_bridges]   how two registered contexts talk to each other

TOML is parsed with the stdlib ``tomllib`` (Python 3.11+); on older
interpreters it transparently falls back to the third-party ``tomli`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
    import tomli as _toml  # type: ignore[no-redef]


# --- Scaling defaults (kept in sync with the shipped blueprint.aero) ---------
_DEFAULT_AUTO_SPLIT_THRESHOLD = 1500
_DEFAULT_MAX_MODULE_COMPLEXITY = 200
_DEFAULT_HIERARCHY_DEPTH = 4


@dataclass
class SystemConfig:
    """The ``[system]`` section: project identity and build strategy."""

    name: str = ""
    version: str = ""
    strategy: str = "microkernel"
    ephemeral_code: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SystemConfig":
        return cls(
            name=str(data.get("name", "")),
            version=str(data.get("version", "")),
            strategy=str(data.get("strategy", "microkernel")),
            ephemeral_code=bool(data.get("ephemeral_code", False)),
        )


@dataclass
class ContextEntry:
    """A single named context inside ``[context_registry]``."""

    name: str
    path: str = ""
    language: str = ""
    dialect: str = ""
    preserve_original_logic: bool = False
    integration_mode: str = ""
    target_output_language: str = ""

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> "ContextEntry":
        return cls(
            name=name,
            path=str(data.get("path", "")),
            language=str(data.get("language", "")),
            dialect=str(data.get("dialect", "")),
            preserve_original_logic=bool(data.get("preserve_original_logic", False)),
            integration_mode=str(data.get("integration_mode", "")),
            target_output_language=str(data.get("target_output_language", "")),
        )


@dataclass
class RewriteRule:
    """A legacy -> target rewrite rule from ``[[abstractions.rewrite]]``."""

    legacy_pattern: str = ""
    target_pattern: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RewriteRule":
        return cls(
            legacy_pattern=str(data.get("legacy_pattern", "")),
            target_pattern=str(data.get("target_pattern", "")),
        )


@dataclass
class Scaling:
    """The ``[scaling]`` section: auto-decomposition / complexity policy."""

    auto_split_threshold: int = _DEFAULT_AUTO_SPLIT_THRESHOLD
    max_module_complexity: int = _DEFAULT_MAX_MODULE_COMPLEXITY
    hierarchy_depth: int = _DEFAULT_HIERARCHY_DEPTH

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scaling":
        return cls(
            auto_split_threshold=int(
                data.get("auto_split_threshold", _DEFAULT_AUTO_SPLIT_THRESHOLD)
            ),
            max_module_complexity=int(
                data.get("max_module_complexity", _DEFAULT_MAX_MODULE_COMPLEXITY)
            ),
            hierarchy_depth=int(data.get("hierarchy_depth", _DEFAULT_HIERARCHY_DEPTH)),
        )


@dataclass
class ContextBridge:
    """A single entry in ``[[context_bridges]]`` connecting two contexts."""

    source: str = ""
    target: str = ""
    bridge_type: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextBridge":
        return cls(
            source=str(data.get("source", "")),
            target=str(data.get("target", "")),
            bridge_type=str(data.get("bridge_type", "")),
        )


@dataclass
class LivingBlueprint:
    """A fully-parsed living blueprint.

    Access the typed sections directly, e.g. ``bp.scaling.auto_split_threshold``
    or ``bp.context_registry["legacy_cobol"].language``.
    """

    system: SystemConfig = field(default_factory=SystemConfig)
    context_registry: Dict[str, ContextEntry] = field(default_factory=dict)
    abstractions: List[RewriteRule] = field(default_factory=list)
    scaling: Scaling = field(default_factory=Scaling)
    context_bridges: List[ContextBridge] = field(default_factory=list)

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LivingBlueprint":
        """Build a blueprint from an already-parsed TOML mapping.

        Unknown sections are ignored; missing sections use defaults.
        """

        system = SystemConfig.from_dict(_as_mapping(data.get("system")))

        registry: Dict[str, ContextEntry] = {}
        for name, entry in _as_mapping(data.get("context_registry")).items():
            if isinstance(entry, Mapping):  # skip stray scalars under the table
                registry[name] = ContextEntry.from_dict(name, entry)

        abstractions = [
            RewriteRule.from_dict(rule)
            for rule in _as_list(_as_mapping(data.get("abstractions")).get("rewrite"))
            if isinstance(rule, Mapping)
        ]

        scaling = Scaling.from_dict(_as_mapping(data.get("scaling")))

        bridges = [
            ContextBridge.from_dict(bridge)
            for bridge in _as_list(data.get("context_bridges"))
            if isinstance(bridge, Mapping)
        ]

        return cls(
            system=system,
            context_registry=registry,
            abstractions=abstractions,
            scaling=scaling,
            context_bridges=bridges,
        )

    @classmethod
    def from_str(cls, text: str) -> "LivingBlueprint":
        """Parse a living blueprint from a TOML string."""

        return cls.from_dict(_toml.loads(text))


def load_blueprint(path: Union[str, Path]) -> LivingBlueprint:
    """Load a living blueprint from ``path``.

    A missing file yields a default blueprint (the schema is entirely optional),
    so callers can treat "no blueprint" and "empty blueprint" identically.
    """

    blueprint_path = Path(path)
    if not blueprint_path.is_file():
        return LivingBlueprint()
    with blueprint_path.open("rb") as handle:
        data = _toml.load(handle)
    return LivingBlueprint.from_dict(data)


# --- helpers -----------------------------------------------------------------
def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]
