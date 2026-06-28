# -*- coding: utf-8 -*-
"""
Legacy universal compiler driver — now delegates to the out-of-tree scaffold pipeline.

The previous implementation wrote Cargo.toml, ``target/`` and build artefacts
directly into the tool tree (hard-coded ``src/lib.rs``).  All compilation is
now routed through :class:`src.scaffold.pipeline.ScaffoldBuildPipeline` so
artefacts stay in the user-defined ``distribution_directory``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.scaffold.pipeline import ScaffoldBuildPipeline


def run_universal_compiler(
    workspace_path: str,
    target_name: str,
    hardware_recipe: Optional[Dict[str, Any]] = None,
    *,
    source_entry: Optional[str] = None,
    distribution_directory: Optional[str] = None,
    compatibility_shims: Optional[list] = None,
    language: Optional[str] = None,
) -> bool:
    """Run an isolated scaffold build for a Rust/pyo3 target.

    ``workspace_path`` is retained for backward compatibility but is **not**
    used as the compile root.  The source is resolved from ``source_entry``
    (or, as a last resort, ``{workspace_path}/../lib.rs`` and similar paths —
    never a hard-coded ``src/lib.rs`` inside the tool tree).
    """
    _ = hardware_recipe  # reserved for future RUSTFLAGS injection via blueprint

    resolved_source = source_entry
    if not resolved_source:
        candidates = [
            Path(workspace_path).parent / "lib.rs",
            Path("/content/lib.rs"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                resolved_source = str(candidate)
                break

    if not resolved_source:
        print(
            "[Error] No source_entry configured and no external lib.rs found. "
            "Set scaffold.source_entry in blueprint.aero."
        )
        return False

    dist = distribution_directory or str(
        Path(workspace_path).parent / f"{target_name}_repository"
    )

    context: Dict[str, Any] = {
        "frameworks": {"language": language or "rust"},
        "scaffold": {
            "source_entry": resolved_source,
            "auto_layout": True,
            "distribution_directory": dist,
            "name": target_name,
            "compatibility_shims": list(compatibility_shims or [
                "rug_v1_30_patch",
                "pyo3_usize_alignment",
            ]),
            "dependencies": {},
        }
    }

    print(f"\n[Aero Universal] Isolated scaffold build for target: '{target_name}'")
    print(f"[Aero Universal] source_entry={resolved_source}")
    print(f"[Aero Universal] distribution_directory={dist}")

    pipeline = ScaffoldBuildPipeline(verbose=True)
    try:
        result = pipeline.run(context, build=True)
    except Exception as exc:
        print(f"\n[Aero Build Failure] {exc}")
        return False

    if result.succeeded:
        artifact_root = Path(result.scaffold.workspace) / "target" / "release"
        print(f"\n[Success] Build complete — artefacts in {artifact_root}")
    else:
        print("\n[Aero Build Failure] cargo build did not succeed")
    return result.succeeded
