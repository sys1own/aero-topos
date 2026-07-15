"""Standardized zero-allocation FFI bridge generator for Rust targets.

This module is the public front-end for the template-agnostic
:class:`translator.artifact_generator.ArtifactGenerator`.  It still emits the
same three artifacts:

* ``aero_ffi.rs`` — a thread-safe bridge module exposing one
  ``aero_execute_node{N}`` hook per deactivated hot path;
* ``legacy.rs`` — preserved original implementations used for differential
  verification;
* The per-hot-path replacement wrapper that is spliced into the target file.

All domain-specific marshalling code now lives in external
``templates/ffi/*.rs`` files and is selected through ``templates/ffi/registry.json``
(or a project-supplied registry), so the generator itself contains no physics
constants, matrix definitions, or project-specific function bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from translator.artifact_generator import ArtifactGenerator
from translator.rust_ast import RustFn, safe_ident

# Stack scratch capacity for the zero-allocation FFI call frame.
AERO_NODE_CAP = 4096

_default_generator: Optional[ArtifactGenerator] = None


def _get_generator(
    blueprint: Optional[Dict[str, Any]] = None,
    extra_template_dirs: Optional[List[str]] = None,
) -> ArtifactGenerator:
    """Return the shared, lazily-created default generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = ArtifactGenerator(
            blueprint=blueprint,
            extra_template_dirs=extra_template_dirs,
        )
    return _default_generator


@dataclass
class FfiNode:
    """A hot path bound to a stable AeroVM bytecode node index."""

    index: int
    fn: RustFn
    hook: str            # aero_execute_node{index}
    legacy: str          # {name}_legacy
    stub_expr: str = ""  # non-linked (verification) branch expression


def assign_nodes(hot_fns: List[RustFn]) -> List[FfiNode]:
    """Assign each hot path a deterministic ``aero_execute_node{N}`` hook."""
    nodes: List[FfiNode] = []
    for i, fn in enumerate(hot_fns):
        ident = safe_ident(fn.name)
        legacy = f"{ident}_legacy"
        nodes.append(FfiNode(
            index=i,
            fn=fn,
            hook=f"aero_execute_node{i}",
            legacy=legacy,
            stub_expr=f"Ok(super::legacy::{legacy}(input))",
        ))
    return nodes


# ---------------------------------------------------------------------------
# Public artifact generation API
# ---------------------------------------------------------------------------

def generate_wrapper_fn(
    node: FfiNode,
    blueprint: Optional[Dict[str, Any]] = None,
    extra_template_dirs: Optional[List[str]] = None,
) -> str:
    """Generate the replacement wrapper (verbatim modern standard signature).

    The wrapper always uses the modern Aero-Calculus FFI signature:
    ``pub fn $name(input: &[f64]) -> Vec<f64>`` and delegates to the
    matching ``aero_ffi::aero_execute_node{N}`` hook.
    """
    gen = _get_generator(blueprint=blueprint, extra_template_dirs=extra_template_dirs)
    return gen.wrapper(node)


def generate_aero_ffi_module(
    nodes: List[FfiNode],
    aeroc_module: str,
    blueprint: Optional[Dict[str, Any]] = None,
    extra_template_dirs: Optional[List[str]] = None,
) -> str:
    """Generate the thread-safe, zero-allocation ``aero_ffi.rs`` bridge."""
    gen = _get_generator(blueprint=blueprint, extra_template_dirs=extra_template_dirs)
    header = gen.aero_ffi_header(aeroc_module, aero_node_cap=AERO_NODE_CAP)

    hooks: List[str] = []
    seen: set[str] = set()
    for node in nodes:
        name = safe_ident(node.fn.name)
        if name in seen:
            continue
        seen.add(name)
        stub = node.stub_expr or "Ok(input.to_vec())"
        hooks.append(
            gen.aero_ffi_hook(
                node,
                aeroc_module=aeroc_module,
                aero_node_cap=AERO_NODE_CAP,
                stub_expr=stub,
            )
        )

    return header + "\n".join(hooks)


def generate_legacy_dispatch(
    nodes: List[FfiNode],
    blueprint: Optional[Dict[str, Any]] = None,
    extra_template_dirs: Optional[List[str]] = None,
) -> str:
    """Generate ``legacy.rs`` with preserved implementations for verification."""
    gen = _get_generator(blueprint=blueprint, extra_template_dirs=extra_template_dirs)
    code = "//! Legacy implementations preserved for differential verification.\n"
    code += "//! Each accepts a flat f64 slice and returns Vec<f64>.\n"
    code += "#![allow(dead_code)]\n\n"

    seen: set[str] = set()
    for node in nodes:
        name = safe_ident(node.fn.name)
        if name in seen:
            continue
        seen.add(name)
        code += gen.legacy(node) + "\n"

    return code


def generate_ffi_artifacts(
    source: str,
    nodes: List[FfiNode],
    aeroc_module: str,
    blueprint: Optional[Dict[str, Any]] = None,
    extra_template_dirs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a complete FFI artifact set with collision checks.

    The returned dictionary has three keys:

    * ``files`` — mapping ``{filename: content}`` for ``lib.rs``, ``legacy.rs``
      and ``aero_ffi.rs``.
    * ``conflicts`` — list of requested definition names that already exist in
      *source* (the collision-avoidance guard).
    * ``emitted`` — list of definition names that were rendered.

    If ``conflicts`` is non-empty, ``files`` will be empty: the generator does
    not emit partial, duplicate definitions.
    """
    gen = ArtifactGenerator(
        blueprint=blueprint,
        extra_template_dirs=extra_template_dirs,
    )
    report = gen.generate_ffi_artifacts(
        source,
        nodes,
        aeroc_module,
        aero_node_cap=AERO_NODE_CAP,
    )
    return {
        "files": report.files,
        "conflicts": report.conflicts,
        "emitted": report.emitted,
    }


# ---------------------------------------------------------------------------
# Standalone single-function handle (kept for blueprint_manager compatibility)
# ---------------------------------------------------------------------------

def generate_single_handle(
    function_name: str,
    aeroc_module: str,
    param_types: Optional[List[tuple[str, str]]] = None,
    return_type: str = "Vec<f64>",
    blueprint: Optional[Dict[str, Any]] = None,
    extra_template_dirs: Optional[List[str]] = None,
) -> str:
    """Generate a standalone FFI handle module for a single function.

    Uses the same standardized ``extern "C"`` raw-pointer mapping as the
    multi-node bridge, so all FFI codegen shares one implementation.
    """
    ident = safe_ident(function_name)
    fn = RustFn(
        name=function_name,
        start_byte=0,
        end_byte=0,
        start_line=0,
        end_line=0,
        signature=f"pub fn {ident}",
        return_type=return_type,
    )
    # The standalone handle is self-contained (no sibling legacy module), so its
    # verification stub is a direct passthrough rather than a legacy delegation.
    node = FfiNode(
        index=0,
        fn=fn,
        hook=f"{ident}_invoke",
        legacy=f"{ident}_legacy",
        stub_expr="Ok(input.to_vec())",
    )
    return generate_aero_ffi_module(
        [node],
        aeroc_module,
        blueprint=blueprint,
        extra_template_dirs=extra_template_dirs,
    )
