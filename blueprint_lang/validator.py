# -*- coding: utf-8 -*-
"""Semantic validation for a parsed :class:`~blueprint_lang.nodes.Blueprint`.

Parsing guarantees the document is *structurally* well formed (balanced braces,
quoted strings, ``key = value`` shape).  Validation enforces the *schema* and
the cross-block invariants the build engine relies on:

* exactly one ``project`` block, and at least one ``target`` block;
* only known keys, with the right value types, per block;
* required keys are present and ``sources`` is non-empty;
* ``language`` is one of the supported languages;
* target names are unique;
* every ``requires`` entry points at a real target; and
* the ``requires`` graph is acyclic.

Validation runs in two stages.  Stage A checks each block in isolation; if any
Stage A error is found we stop before Stage B (the cross-target ``requires`` and
cycle checks), because those assume well-typed ``requires`` lists.  Within a
stage we report the error that appears *earliest* in the file, so the user fixes
problems top-to-bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .errors import BlueprintValidationError
from .nodes import Block, Blueprint, ListValue, NumberValue, StringValue

# Languages the wider Aero tool understands (Python via stdlib ``ast``; the rest
# via tree-sitter -- see requirements.txt / ARCHITECTURE.md).
SUPPORTED_LANGUAGES = ("c", "cpp", "fortran", "python", "rust")

# Value-type tags used by the schema below.
_STRING = "string"
_NUMBER = "number"
_STRING_LIST = "string_list"
_BOOL = "boolean"


@dataclass(frozen=True)
class BlockSchema:
    """Allowed/required keys (and their expected types) for one block type."""

    fields: Dict[str, str]
    required: Tuple[str, ...] = ()

    def allowed_keys(self) -> List[str]:
        return sorted(self.fields)


SCHEMAS: Dict[str, BlockSchema] = {
    "project": BlockSchema(
        fields={
            "version": _STRING,
            "description": _STRING,
            "authors": _STRING_LIST,
        },
        required=("version",),
    ),
    "target": BlockSchema(
        fields={
            "language": _STRING,
            "sources": _STRING_LIST,
            "requires": _STRING_LIST,
            "flags": _STRING_LIST,
            "defines": _STRING_LIST,
            "output": _STRING,
            "optional": _BOOL,
            # Rust/Cargo: build a crate that lives in a subdirectory and/or has
            # its own committed manifest (see README "Rust / Cargo support").
            "manifest_path": _STRING,
            "root": _STRING,
            # Pin dependency versions for a synthesised manifest, as a flat list
            # of "name=version" entries, e.g. cargo_dependencies = ["rug=0.22"].
            "cargo_dependencies": _STRING_LIST,
            # Control RUSTFLAGS injection: optimization = "none" | "generic" |
            # "native" | "size", or rustflags = ["-C", "target-cpu=generic"].
            "optimization": _STRING,
            "rustflags": _STRING_LIST,
            # Polyglot code generation: when set, the emitter translates the
            # UAST into this target language instead of compiling in the source
            # language.  Must be one of the SUPPORTED_LANGUAGES.
            "target_output_language": _STRING,
        },
        required=("language", "sources"),
    ),
    "scaling": BlockSchema(
        fields={
            "auto_split_threshold": _NUMBER,
            "max_module_complexity": _NUMBER,
            "hierarchy_depth": _NUMBER,
        },
    ),
}


def _quote_join(items: List[str]) -> str:
    return ", ".join(f"'{i}'" for i in items)


def _check_value_type(block_type: str, fld, expected: str) -> Optional[BlueprintValidationError]:
    """Return a typed-value error for ``fld`` against ``expected``, or ``None``."""
    value = fld.value
    if expected == _STRING:
        if not isinstance(value, StringValue):
            return BlueprintValidationError(
                f"key '{fld.key}' in '{block_type}' must be a string",
                value.span,
                label=f"expected a quoted string, found {value.kind}",
            )
    elif expected == _BOOL:
        if value.kind != "boolean":
            return BlueprintValidationError(
                f"key '{fld.key}' in '{block_type}' must be a boolean (true/false)",
                value.span,
                label=f"expected true/false, found {value.kind}",
            )
    elif expected == _NUMBER:
        if not isinstance(value, NumberValue):
            return BlueprintValidationError(
                f"key '{fld.key}' in '{block_type}' must be a number",
                value.span,
                label=f"expected a number, found {value.kind}",
            )
    elif expected == _STRING_LIST:
        if not isinstance(value, ListValue):
            return BlueprintValidationError(
                f"key '{fld.key}' in '{block_type}' must be a list of strings",
                value.span,
                label=f"expected a [list], found {value.kind}",
                hint=f'wrap it in brackets, e.g. {fld.key} = ["..."]',
            )
        for item in value.items:
            if not isinstance(item, StringValue):
                return BlueprintValidationError(
                    f"every item in '{fld.key}' must be a string",
                    item.span,
                    label=f"this item is {item.kind}, not a string",
                )
    return None


class Validator:
    """Validate a :class:`Blueprint`, surfacing precise, positioned errors."""

    def __init__(self, blueprint: Blueprint) -> None:
        self.blueprint = blueprint

    # -- public API --------------------------------------------------------

    def collect(self) -> List[BlueprintValidationError]:
        """All errors from the first non-empty stage, sorted by source order."""
        stage_a = self._stage_a()
        if stage_a:
            return self._sorted(stage_a)
        return self._sorted(self._stage_b())

    def validate(self) -> None:
        """Raise the earliest error, or return ``None`` if the blueprint is valid."""
        errors = self.collect()
        if errors:
            raise errors[0]

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _sorted(errors: List[BlueprintValidationError]) -> List[BlueprintValidationError]:
        def key(err: BlueprintValidationError):
            if err.span is None:
                return (1, 0)
            return (0, err.span.start.offset)

        return sorted(errors, key=key)

    # -- stage A: document shape + per-block schema ------------------------

    def _stage_a(self) -> List[BlueprintValidationError]:
        errors: List[BlueprintValidationError] = []
        blocks = self.blueprint.blocks

        if not blocks:
            errors.append(
                BlueprintValidationError(
                    "blueprint is empty: define a 'project' block and at least one 'target'",
                    None,
                    hint='start with:  project "name" { version = "1.0.0" }',
                )
            )
            return errors

        projects = self.blueprint.projects
        targets = self.blueprint.targets

        # Unknown block types.
        for block in blocks:
            if block.type not in SCHEMAS:
                errors.append(
                    BlueprintValidationError(
                        f"unknown block type '{block.type}'",
                        block.type_span,
                        label="not a valid block",
                        hint=f"valid block types are {_quote_join(sorted(SCHEMAS))}",
                    )
                )

        # Exactly one project.
        if not projects:
            errors.append(
                BlueprintValidationError(
                    "missing required 'project' block",
                    None,
                    hint='add a block:  project "name" { version = "1.0.0" }',
                )
            )
        else:
            for extra in projects[1:]:
                errors.append(
                    BlueprintValidationError(
                        "duplicate 'project' block; a blueprint may define only one",
                        extra.type_span,
                        label="second 'project' here",
                        hint=f"the first 'project' is on line {projects[0].type_span.start.line}",
                    )
                )

        # At most one scaling block.
        scaling_blocks = [b for b in blocks if b.type == "scaling"]
        for extra in scaling_blocks[1:]:
            errors.append(
                BlueprintValidationError(
                    "duplicate 'scaling' block; a blueprint may define at most one",
                    extra.type_span,
                    label="second 'scaling' here",
                    hint=f"the first 'scaling' is on line {scaling_blocks[0].type_span.start.line}",
                )
            )

        # At least one target.
        if not targets:
            errors.append(
                BlueprintValidationError(
                    "no 'target' blocks defined; a blueprint needs at least one target",
                    None,
                    hint='add a block:  target "name" { language = "python" sources = ["..."] }',
                )
            )

        # Per-block schema checks (only for blocks we recognise).
        for block in blocks:
            schema = SCHEMAS.get(block.type)
            if schema is None:
                continue
            errors.extend(self._check_block(block, schema))

        # Duplicate target names.
        first_seen: Dict[str, Block] = {}
        for block in targets:
            if block.name in first_seen:
                errors.append(
                    BlueprintValidationError(
                        f"duplicate target name '{block.name}'",
                        block.name_span,
                        label="reused here",
                        hint="target names must be unique; the first was on line "
                        f"{first_seen[block.name].name_span.start.line}",
                    )
                )
            else:
                first_seen[block.name] = block

        return errors

    def _check_block(self, block: Block, schema: BlockSchema) -> List[BlueprintValidationError]:
        errors: List[BlueprintValidationError] = []

        if not block.name.strip():
            errors.append(
                BlueprintValidationError(
                    f"'{block.type}' block has an empty name",
                    block.name_span,
                    label="name cannot be blank",
                )
            )

        # Unknown keys + per-key type checks.
        for fld in block.fields:
            expected = schema.fields.get(fld.key)
            if expected is None:
                errors.append(
                    BlueprintValidationError(
                        f"unknown key '{fld.key}' in '{block.type}' block",
                        fld.key_span,
                        label="not a recognised key",
                        hint=f"allowed keys are {_quote_join(schema.allowed_keys())}",
                    )
                )
                continue
            type_error = _check_value_type(block.type, fld, expected)
            if type_error is not None:
                errors.append(type_error)

        # Missing required keys.
        present = {fld.key for fld in block.fields}
        for key in schema.required:
            if key not in present:
                errors.append(
                    BlueprintValidationError(
                        f"'{block.type}' block \"{block.name}\" is missing required key '{key}'",
                        block.name_span,
                        label=f"add '{key}' here",
                    )
                )

        # Target-specific value constraints.
        if block.type == "target":
            errors.extend(self._check_target_values(block))

        return errors

    def _check_target_values(self, block: Block) -> List[BlueprintValidationError]:
        errors: List[BlueprintValidationError] = []

        language = block.get("language")
        if language is not None and isinstance(language.value, StringValue):
            lang = language.value.value
            if lang not in SUPPORTED_LANGUAGES:
                errors.append(
                    BlueprintValidationError(
                        f"unsupported language '{lang}'",
                        language.value.span,
                        label="not a supported language",
                        hint=f"supported languages are {_quote_join(list(SUPPORTED_LANGUAGES))}",
                    )
                )

        sources = block.get("sources")
        if sources is not None and isinstance(sources.value, ListValue):
            if not sources.value.items:
                errors.append(
                    BlueprintValidationError(
                        f"target \"{block.name}\" declares no sources",
                        sources.value.span,
                        label="this list is empty",
                        hint="list at least one source file or glob, e.g. [\"src/**/*.py\"]",
                    )
                )

        target_output = block.get("target_output_language")
        if target_output is not None and isinstance(target_output.value, StringValue):
            out_lang = target_output.value.value
            if out_lang not in SUPPORTED_LANGUAGES:
                errors.append(
                    BlueprintValidationError(
                        f"unsupported target_output_language '{out_lang}'",
                        target_output.value.span,
                        label="not a supported output language",
                        hint=f"supported languages are {_quote_join(list(SUPPORTED_LANGUAGES))}",
                    )
                )

        return errors

    # -- stage B: cross-target requires + cycle detection ------------------

    def _stage_b(self) -> List[BlueprintValidationError]:
        errors: List[BlueprintValidationError] = []
        targets = self.blueprint.targets
        names = {b.name for b in targets}

        graph: Dict[str, List[str]] = {}
        for block in targets:
            deps: List[str] = []
            requires = block.get("requires")
            if requires is not None and isinstance(requires.value, ListValue):
                for item in requires.value.items:
                    if not isinstance(item, StringValue):
                        continue
                    dep = item.value
                    deps.append(dep)
                    if dep not in names:
                        errors.append(
                            BlueprintValidationError(
                                f"target \"{block.name}\" requires unknown target '{dep}'",
                                item.span,
                                label="no target with this name",
                                hint="check the spelling, or define a target with this name",
                            )
                        )
            graph[block.name] = deps

        if errors:
            # Don't run cycle detection over a graph with dangling edges.
            return errors

        cycle = _find_cycle(graph)
        if cycle:
            block_by_name = {b.name: b for b in targets}
            anchor = block_by_name.get(cycle[0])
            span = anchor.name_span if anchor is not None else None
            errors.append(
                BlueprintValidationError(
                    "cyclic target dependency: " + " -> ".join(cycle),
                    span,
                    label="this target is part of a dependency cycle",
                    hint="targets cannot (transitively) require themselves; break the loop",
                )
            )
        return errors


def _find_cycle(graph: Dict[str, List[str]]) -> Optional[List[str]]:
    """Return one cycle as a node list (e.g. ``[a, b, a]``), or ``None``."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {node: WHITE for node in graph}
    stack: List[str] = []

    def dfs(node: str) -> Optional[List[str]]:
        color[node] = GRAY
        stack.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in color:
                continue  # dangling edge -- reported separately
            if color[neighbour] == GRAY:
                start = stack.index(neighbour)
                return stack[start:] + [neighbour]
            if color[neighbour] == WHITE:
                found = dfs(neighbour)
                if found:
                    return found
        color[node] = BLACK
        stack.pop()
        return None

    for node in graph:
        if color[node] == WHITE:
            found = dfs(node)
            if found:
                return found
    return None


def validate(blueprint: Blueprint) -> None:
    """Validate ``blueprint``; raise the earliest :class:`BlueprintValidationError`."""
    Validator(blueprint).validate()
