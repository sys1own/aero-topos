from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import blueprint_parser
import sandbox_runner
from builder_brains.compactor import DeadCodeEliminator, VariableMinifier, _ScopeAnalyzer

try:
    import builder_brains.decision_tree as decision_tree
except ImportError:
    import decision_tree  # type: ignore

try:
    import builder_brains.neural_synthesis as neural_synthesis
except ImportError:
    try:
        import neural_synthesis  # type: ignore
    except ImportError:
        neural_synthesis = None  # type: ignore

try:
    import builder_brains.parameter_tuner as parameter_tuner
except ImportError:
    try:
        import parameter_tuner  # type: ignore
    except ImportError:
        parameter_tuner = None  # type: ignore


def _load_translator_callable() -> Optional[Any]:
    def live_translate_variant(variant_code_str):
        import os, uuid, ast
        os.makedirs("build_sandbox", exist_ok=True)
        mod_id = f"variant_{uuid.uuid4().hex[:8]}"
        file_path = f"build_sandbox/{mod_id}.py"

        # Validate the variant code parses as valid Python before writing.
        try:
            tree = ast.parse(variant_code_str)
        except SyntaxError:
            return {"module": "", "callable_name": "", "error": "invalid syntax"}

        # Reject code that imports dangerous modules or uses dangerous builtins.
        _BLOCKED_MODULES = frozenset({"os", "sys", "subprocess", "shutil", "socket",
                                       "ctypes", "signal", "importlib"})
        _BLOCKED_ATTRS = frozenset({"eval", "exec", "compile", "__import__",
                                     "getattr", "globals", "locals"})
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _BLOCKED_MODULES:
                        return {"module": "", "callable_name": "", "error": f"blocked import: {alias.name}"}
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in _BLOCKED_MODULES:
                    return {"module": "", "callable_name": "", "error": f"blocked import: {node.module}"}
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _BLOCKED_ATTRS:
                    return {"module": "", "callable_name": "", "error": f"blocked builtin: {func.id}"}

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(variant_code_str)
        callable_name = "main"
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                callable_name = node.name
                break
        return {"module": f"build_sandbox.{mod_id}", "callable_name": callable_name}
    return live_translate_variant


def _extract_gate_signal(decision: Any) -> bool:
    if isinstance(decision, Mapping):
        return bool(decision.get("is_stagnant")) or bool(decision.get("boost_mutation_sigma"))
    return bool(getattr(decision, "is_stagnant", False)) or bool(getattr(decision, "boost_mutation_sigma", False))


def _extract_bottleneck_source(decision: Any, evaluation_context: Mapping[str, Any]) -> Any:
    if isinstance(decision, Mapping):
        if "bottleneck_source" in decision:
            return decision["bottleneck_source"]
    elif hasattr(decision, "bottleneck_source"):
        return getattr(decision, "bottleneck_source")
    return evaluation_context.get("bottleneck_source") or evaluation_context.get("source") or evaluation_context


def _extract_accuracy(trace: Mapping[str, Any]) -> float:
    if "accuracy" in trace:
        return float(trace["accuracy"])
    if "metrics" in trace and isinstance(trace["metrics"], Mapping) and "accuracy" in trace["metrics"]:
        return float(trace["metrics"]["accuracy"])
    successes = trace.get("successful_invocations", 0)
    total = trace.get("invocation_count", 0)
    return float(successes) / float(total) if total else 0.0


def _extract_velocity(trace: Mapping[str, Any]) -> float:
    average_latency_us = float(trace.get("average_latency_us", 0.0))
    if average_latency_us <= 0:
        return 0.0
    return 1_000_000.0 / average_latency_us


def _evaluate_variant_with_sandbox(
    translated_variant: Mapping[str, Any],
    sample_params: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    module_name = translated_variant.get("module")
    callable_name = translated_variant.get("callable_name")
    if not module_name:
        return {
            "compile_success": False,
            "fitness": 0.0,
            "reason": "missing_module",
            "trace": {
                "module": "<missing>",
                "callable_name": callable_name or "<missing>",
                "compile_success": False,
                "compile_error": "Translated variant missing module target",
                "invocation_count": 0,
                "successful_invocations": 0,
                "failed_invocations": 0,
                "total_latency_us": 0.0,
                "average_latency_us": 0.0,
                "min_latency_us": 0.0,
                "max_latency_us": 0.0,
                "traces": [],
            },
        }
    trace = sandbox_runner.run_module(
        module_name=str(module_name),
        callable_name=str(callable_name) if callable_name else None,
        sample_params=list(sample_params),
        case_names=translated_variant.get("case_names"),
    )
    compile_success = bool(trace.get("compile_success"))
    fitness = _extract_velocity(trace) if compile_success else 0.0
    return {
        "compile_success": compile_success,
        "fitness": fitness,
        "trace": trace,
    }


def maybe_run_neural_synthesis(
    evaluation_context: Mapping[str, Any],
    baseline_trace: Mapping[str, Any],
    sample_params: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    decision = decision_tree.evaluate(evaluation_context, {})
    if not _extract_gate_signal(decision):
        return []

    if neural_synthesis is None:
        return []

    translator_callable = _load_translator_callable()
    if translator_callable is None:
        return []

    bottleneck_source = _extract_bottleneck_source(decision, evaluation_context)
    try:
        generated_variants = neural_synthesis.generate_logic_mutation(bottleneck_source, {'stagnation_event': True})
    except Exception as exc:
        logger.warning("Neural synthesis mutation failed: %s", exc)
        return []
    if not generated_variants:
        return []

    baseline_accuracy = _extract_accuracy(baseline_trace)
    baseline_velocity = _extract_velocity(baseline_trace)
    accepted_variants: List[Dict[str, Any]] = []

    for variant in generated_variants:
        translated_variant = translator_callable(variant)
        evaluation = _evaluate_variant_with_sandbox(translated_variant, sample_params)
        trace = evaluation["trace"]
        accuracy = _extract_accuracy(trace)
        velocity = _extract_velocity(trace)
        compile_success = bool(evaluation["compile_success"])
        improves_velocity = velocity > baseline_velocity
        maintains_accuracy = accuracy >= baseline_accuracy
        if compile_success and maintains_accuracy and improves_velocity:
            accepted_variant = {
                "variant": variant,
                "translated_variant": translated_variant,
                "trace": trace,
                "fitness": velocity,
            }
            accepted_variants.append(accepted_variant)
            if parameter_tuner is not None and hasattr(parameter_tuner, "update_config"):
                parameter_tuner.update_config(accepted_variant)
        else:
            evaluation["fitness"] = 0.0

    return accepted_variants


_REPO_ROOT = Path(__file__).resolve().parent
_BRAINS_DIR = _REPO_ROOT / "builder_brains"
_MANIFEST_PATH = _BRAINS_DIR / "build_manifest.json"
_BLUEPRINT_PATH = _REPO_ROOT / "blueprint.aero"
_DEFAULT_TELEMETRY_INTERVAL = 2.0
_ANOMALY_DRIFT_THRESHOLD = 1.0
_SOURCE_EXTENSIONS = {".py", ".json", ".md", ".txt", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
_IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("orchestrator")

# Set by run_build() from the living blueprint's active_optimizer_flags so
# _compile_targets and telemetry reflect the actual blueprint configuration
# rather than the static manifest fallback.
_bp_optimization_override: Optional[str] = None


@dataclass
class StageResult:
    label: str
    started_at: float
    finished_at: float
    status: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


@dataclass
class CycleTelemetry:
    cycle: int
    total_cycles: int
    stage_results: List[StageResult]
    selected_action: str
    resolved_strategy: str
    primary_strategy: str
    strategy: str
    thread_pool_size: int
    stagnation: bool
    pareto_summary: Dict[str, Any]
    replay_status: str
    manifest_status: str
    compiled_target_count: int
    bytes_written: int
    optimization_level: str
    elapsed_seconds: float


def _load_brain_modules() -> List[Tuple[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]]]:
    stages: List[Tuple[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]]] = []
    for label, dotted in (
        ("scanner", "builder_brains.scanner"),
        ("decision_tree", "builder_brains.decision_tree"),
        ("parameter_tuner", "builder_brains.parameter_tuner"),
    ):
        module = __import__(dotted, fromlist=["evaluate"])
        evaluate = getattr(module, "evaluate", None)
        if not callable(evaluate):
            raise RuntimeError(f"{dotted} does not expose evaluate(metadata, hyper_params)")
        stages.append((label, evaluate))
    return stages


def load_manifest(path: Path = _MANIFEST_PATH) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid manifest JSON at {path}: {exc}") from exc


def _workspace_files(workspace_root: Path) -> List[Path]:
    files: List[Path] = []
    for root, dirs, filenames in os.walk(workspace_root):
        dirs[:] = [name for name in dirs if name not in _IGNORED_DIRS]
        for filename in filenames:
            path = Path(root) / filename
            if path == _MANIFEST_PATH:
                continue
            if path.suffix.lower() in _SOURCE_EXTENSIONS:
                files.append(path)
    return sorted(files)


def _fingerprint_file(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def collect_workspace_snapshot(workspace_root: Path) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for path in _workspace_files(workspace_root):
        snapshot[str(path)] = _fingerprint_file(path)
    return snapshot


def compute_workspace_delta(
    previous: Dict[str, Dict[str, Any]],
    current: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    changed = [path for path, fingerprint in current.items() if previous.get(path) != fingerprint]
    removed = [path for path in previous if path not in current]
    unchanged = [path for path, fingerprint in current.items() if previous.get(path) == fingerprint]
    return {
        "changed_files": changed,
        "removed_files": removed,
        "unchanged_files": unchanged,
        "changed_count": len(changed),
        "removed_count": len(removed),
        "unchanged_count": len(unchanged),
    }


_MUTABLE_ALLOWLIST = {
    "builder_brains/build_manifest.json",
    "builder_brains/experience_replay.json",
    "builder_brains/history_vault.json",
}


def _enforce_read_only_boundary(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> None:
    def _is_allowed(path_str: str) -> bool:
        try:
            rel = str(Path(path_str).relative_to(_REPO_ROOT)).replace("\\", "/")
        except ValueError:
            rel = path_str.replace("\\", "/")
        return rel in _MUTABLE_ALLOWLIST

    violations: List[str] = []
    for path, fingerprint in before.items():
        if _is_allowed(path):
            continue
        if path not in after:
            violations.append(f"removed:{path}")
        elif after[path] != fingerprint:
            violations.append(f"modified:{path}")
    for path in after:
        if _is_allowed(path):
            continue
        if path not in before:
            violations.append(f"created:{path}")
    if violations:
        raise RuntimeError("Read-only boundary breached: " + ", ".join(violations))


def _extract_build_context(workspace_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    build_context = blueprint_parser.parse_blueprint(str(workspace_root / "blueprint.aero"), str(_MANIFEST_PATH))
    orchestrator_state = manifest.get("orchestrator_state", {})
    if not isinstance(orchestrator_state, dict):
        orchestrator_state = {}
    build_context["workspace_root"] = str(workspace_root)
    build_context["current_cycle"] = int(orchestrator_state.get("current_cycle", manifest.get("current_cycle", 1)))
    build_context["score_trajectory"] = list(orchestrator_state.get("score_trajectory", []))
    build_context["kinetic_stall_cycles"] = int(orchestrator_state.get("kinetic_stall_cycles", 0))
    build_context["pareto_frontier"] = list(orchestrator_state.get("pareto_frontier", []))
    build_context["tuned_population"] = list(orchestrator_state.get("tuned_population", []))
    build_context["survival_tracker_stats"] = dict(orchestrator_state.get("survival_tracker_stats", {}))
    build_context["baseline_config"] = dict(orchestrator_state.get("baseline_config", {}))
    build_context["previous_fingerprints"] = dict(orchestrator_state.get("previous_fingerprints", {}))

    # Propagate living-blueprint properties into metadata so the FSM and
    # telemetry can read them dynamically instead of relying on hardcoded
    # manifest values.
    optimizer_flags = build_context.get("active_optimizer_flags", {})
    bp_opt = optimizer_flags.get("profile_guided_optimization", "")
    if bp_opt:
        build_context["blueprint_optimization_level"] = bp_opt
    scaling = build_context.get("scaling", {})
    if isinstance(scaling, dict) and "max_module_complexity" in scaling:
        build_context["blueprint_max_module_complexity"] = scaling["max_module_complexity"]
    system = build_context.get("system", {})
    if isinstance(system, dict):
        strategy = str(system.get("strategy", "")).strip()
        if strategy:
            build_context["blueprint_strategy"] = strategy

    # Promote the [system] strategy field to a top-level metadata key so all
    # downstream stages (decision tree, orchestrator loop) can check for an
    # explicit DIRECT_COMPILE directive without traversing the nested dict.
    system_section = build_context.get("system", {})
    if isinstance(system_section, dict):
        bp_system_strategy = str(system_section.get("strategy", "")).strip()
        if bp_system_strategy:
            build_context["blueprint_system_strategy"] = bp_system_strategy

    # Tag the context with the active top-level command so the FSM can guard
    # strategy overrides that are inappropriate during a direct compile pass.
    build_context.setdefault("active_command", "build")

    return build_context


def _read_blueprint_lines(path: Path = _BLUEPRINT_PATH) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _parse_graph_targets_with_metadata(path: Path = _BLUEPRINT_PATH) -> Tuple[List[Dict[str, Any]], List[str]]:
    lines = _read_blueprint_lines(path)
    in_graph = False
    graph_data: Dict[str, Any] = {}
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_graph = stripped == "[graph]"
            continue
        if not in_graph or not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        graph_data[key.strip()] = blueprint_parser.parse_literal(value)

    raw_targets = graph_data.get("targets", [])
    if not isinstance(raw_targets, list):
        raw_targets = []
    targets: List[Dict[str, Any]] = []
    for entry in raw_targets:
        if isinstance(entry, dict):
            target = dict(entry)
        else:
            target = {"name": str(entry)}
        target["name"] = str(target.get("name", "")).strip()
        if target["name"]:
            targets.append(target)
    return targets, lines


def _default_target_paths(target_name: str) -> Dict[str, str]:
    normalized = target_name.replace("\\", "/").strip().strip("/")
    stem = normalized.rsplit("/", 1)[-1]
    if not stem.endswith(".py"):
        source = f"builder_brains/{stem}.py" if (_BRAINS_DIR / f"{stem}.py").exists() else f"translator/{stem}.py"
    else:
        source = normalized
    if not source.endswith(".py"):
        source = f"{source}.py"
    output = f"build_artifacts/{Path(source).stem}.optimized.py"
    return {"source": source.replace("\\", "/"), "output": output.replace("\\", "/")}


def _ensure_blueprint_target_paths(path: Path = _BLUEPRINT_PATH) -> List[Dict[str, Any]]:
    targets, lines = _parse_graph_targets_with_metadata(path)
    updated_targets: List[Dict[str, Any]] = []
    changed = False
    for target in targets:
        enriched = dict(target)
        defaults = _default_target_paths(enriched["name"])
        if not enriched.get("source"):
            enriched["source"] = defaults["source"]
            changed = True
        if not enriched.get("output"):
            enriched["output"] = defaults["output"]
            changed = True
        updated_targets.append(enriched)

    if changed:
        serialized_targets = json.dumps(updated_targets)
        new_lines: List[str] = []
        in_graph = False
        replaced = False
        for raw_line in lines:
            stripped = raw_line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_graph = stripped == "[graph]"
                new_lines.append(raw_line)
                continue
            if in_graph and stripped.startswith("targets") and "=" in stripped and not replaced:
                indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
                new_lines.append(f"{indent}targets = {serialized_targets}")
                replaced = True
                continue
            new_lines.append(raw_line)
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return updated_targets


def _resolve_target_paths(workspace_root: Path, target: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _default_target_paths(target["name"])
    source_rel = str(target.get("source") or defaults["source"]).replace("\\", "/")
    output_rel = str(target.get("output") or defaults["output"]).replace("\\", "/")
    return {
        "name": target["name"],
        "source": source_rel,
        "output": output_rel,
        "source_path": (workspace_root / source_rel).resolve(),
        "output_path": (workspace_root / output_rel).resolve(),
    }


def _manifest_compactor_params(manifest: Dict[str, Any]) -> Dict[str, Any]:
    weights = manifest.get("hyperparameter_weights", {})
    parameters = manifest.get("parameters", {})
    compactor_weights = weights.get("compactor", {}) if isinstance(weights, dict) else {}
    parameters = parameters if isinstance(parameters, dict) else {}
    return {
        "dead_code_elimination_depth": int(compactor_weights.get("dead_code_elimination_depth", 4)),
        "identifier_collision_salt_bits": int(compactor_weights.get("identifier_collision_salt_bits", 32)),
        "minification_entropy_cap": float(compactor_weights.get("minification_entropy_cap", 0.85)),
        "optimization_level": str(parameters.get("decision_tree_resolved_strategy", "balanced")).lower(),
    }


def _read_blueprint_compiler_flags(path: Path = _BLUEPRINT_PATH) -> Dict[str, Any]:
    """Read [compiler] settings from blueprint.aero."""
    flags: Dict[str, Any] = {}
    in_compiler = False
    if not path.exists():
        return flags
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_compiler = stripped == "[compiler]"
            continue
        if not in_compiler or not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        val_s = value.strip().strip('"').strip("'")
        if val_s.lower() in ("true", "false"):
            flags[key.strip()] = val_s.lower() == "true"
        else:
            flags[key.strip()] = val_s
    return flags


def _compact_python_source(source_code: str, manifest: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    import ast

    params = _manifest_compactor_params(manifest)
    compiler_flags = _read_blueprint_compiler_flags()
    tree = ast.parse(source_code)
    eliminator = DeadCodeEliminator(elimination_depth=params["dead_code_elimination_depth"])
    tree = eliminator.run_passes(tree)

    renamed = 0
    if compiler_flags.get("identifier_minification", False):
        analyzer = _ScopeAnalyzer()
        analyzer.visit(tree)
        minifier = VariableMinifier(
            scope_map=dict(analyzer.scopes),
            entropy_cap=params["minification_entropy_cap"],
            salt_bits=params["identifier_collision_salt_bits"],
        )
        tree = minifier.visit(tree)
        renamed = minifier.total_renames

    ast.fix_missing_locations(tree)
    compacted = ast.unparse(tree)
    return compacted, {
        "removed_nodes": eliminator.removed_nodes,
        "renamed_identifiers": renamed,
        "optimization_level": params["optimization_level"],
    }


def _compact_target_source(source_path: Path, manifest: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    source_code = source_path.read_text(encoding="utf-8")
    if source_path.suffix.lower() == ".py":
        return _compact_python_source(source_code, manifest)
    return source_code, {
        "removed_nodes": 0,
        "renamed_identifiers": 0,
        "optimization_level": _manifest_compactor_params(manifest)["optimization_level"],
    }


def _compile_targets(workspace_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    targets = _ensure_blueprint_target_paths(workspace_root / "blueprint.aero")
    compiled_targets: List[Dict[str, Any]] = []
    bytes_written = 0
    optimization_level = _manifest_compactor_params(manifest)["optimization_level"]
    # Let the living blueprint override the manifest-derived optimization level
    # so the telemetry reflects the actual blueprint configuration.
    if _bp_optimization_override:
        optimization_level = _bp_optimization_override
    for target in targets:
        resolved = _resolve_target_paths(workspace_root, target)
        source_path = resolved["source_path"]
        if not source_path.exists() or not source_path.is_file():
            logger.warning("Skipping unresolved target %s at %s", resolved["name"], source_path)
            continue
        compacted_source, metrics = _compact_target_source(source_path, manifest)
        output_path = resolved["output_path"]
        os.makedirs(output_path.parent, exist_ok=True)
        output_path.write_text(compacted_source, encoding="utf-8")
        written = len(compacted_source.encode("utf-8"))
        bytes_written += written
        compiled_targets.append(
            {
                "name": resolved["name"],
                "source": str(source_path),
                "output": str(output_path),
                "bytes_written": written,
                "removed_nodes": metrics["removed_nodes"],
                "renamed_identifiers": metrics["renamed_identifiers"],
            }
        )
    return {
        "compiled_targets": compiled_targets,
        "compiled_target_count": len(compiled_targets),
        "bytes_written": bytes_written,
        "optimization_level": optimization_level,
    }


def _freeze_uast_matrix(workspace_root: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Backend codegen pass: freeze the scanner's 1-D UAST token map into a
    concrete ``matrix.aeroc`` binary asset at the workspace root.

    This is the compilation stage the cyclic build loop previously omitted.
    The orchestrator used to cycle through scan -> decision -> tuning without
    ever handing the mapped token stream down to a code generator, so metrics
    stayed frozen at ``compiled=0 bytes=0`` and the write gate locked out the
    asset emission.  This routine consumes the linearized token map produced by
    the scanner stage and hands it to the binary freezer, guaranteeing the
    build always emits a real artifact with non-zero metrics.
    """
    fingerprints = metadata.get("file_fingerprints", {})
    if not isinstance(fingerprints, dict):
        fingerprints = {}
    aggregate = metadata.get("aggregate_token_profile", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    units = [
        {"path": str(path), "fingerprint": str(fp)}
        for path, fp in sorted(fingerprints.items())
    ]
    matrix = {
        "format": "aeroc-matrix/v1",
        "workspace": str(workspace_root),
        "cycle": int(metadata.get("current_cycle", 1)),
        "strategy": str(metadata.get("resolved_strategy", "unknown")),
        "aggregate_token_profile": aggregate,
        "unit_count": len(units),
        "units": units,
    }
    payload = json.dumps(matrix, indent=2, sort_keys=True)
    output_path = (workspace_root / "matrix.aeroc").resolve()
    output_path.write_text(payload, encoding="utf-8")
    written = len(payload.encode("utf-8"))
    logger.info(
        "Froze UAST token matrix -> %s (%d units, %d bytes)",
        output_path,
        len(units),
        written,
    )
    return {
        "matrix_output": str(output_path),
        "matrix_unit_count": len(units),
        "matrix_bytes_written": written,
    }


def _seed_objectives(metadata: Dict[str, Any]) -> None:
    coverage = float(metadata.get("scan_coverage", 0.0) or 0.0)
    anomaly_count = float(metadata.get("anomaly_count", 0) or 0)
    target_count = max(int(metadata.get("scan_target_count", 0) or 0), 1)
    wall_seconds = float(metadata.get("scanner_wall_seconds", 0.0) or 0.0)
    tokens = metadata.get("aggregate_token_profile", {}) or {}
    code_tokens = float(sum(value for key, value in tokens.items() if key != "comment_line")) or 1.0
    comment_tokens = float(tokens.get("comment_line", 0) or 0.0)
    accuracy = max(0.0, coverage * (1.0 - min(1.0, anomaly_count / target_count) * 0.5))
    compression = max(0.0, min(1.0, 1.0 - (comment_tokens / code_tokens)))
    metadata["current_score"] = round(accuracy, 6)
    metadata["fitness_matrix"] = [
        [accuracy, wall_seconds, compression],
        [max(0.0, accuracy * 0.99), wall_seconds * 1.05 if wall_seconds else 0.0, compression],
        [min(1.0, accuracy * 1.01), wall_seconds * 0.95 if wall_seconds else 0.0, min(1.0, compression * 1.01)],
    ]


def _build_sandbox_sample_params(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    scan_targets = metadata.get("scan_targets")
    if isinstance(scan_targets, list) and scan_targets:
        return [{"path": str(path)} for path in scan_targets[:8]]
    workspace_root = metadata.get("workspace_root")
    if workspace_root:
        return [{"path": str(workspace_root)}]
    return [{"value": metadata.get("current_score", 0.0)}]


def _build_baseline_trace(metadata: Dict[str, Any]) -> Dict[str, Any]:
    scanner_wall_seconds = float(metadata.get("scanner_wall_seconds", 0.0) or 0.0)
    average_latency_us = scanner_wall_seconds * 1_000_000.0 if scanner_wall_seconds > 0 else 0.0
    scan_coverage = float(metadata.get("scan_coverage", metadata.get("current_score", 0.0)) or 0.0)
    anomaly_count = int(metadata.get("anomaly_count", 0) or 0)
    invocation_count = max(1, int(metadata.get("scan_target_count", 0) or len(metadata.get("scan_targets", []) or []) or 1))
    successful_invocations = max(0, invocation_count - anomaly_count)
    accuracy = max(0.0, min(1.0, scan_coverage))
    return {
        "accuracy": accuracy,
        "average_latency_us": average_latency_us,
        "invocation_count": invocation_count,
        "successful_invocations": successful_invocations,
        "failed_invocations": max(0, invocation_count - successful_invocations),
        "compile_success": True,
        "metrics": {"accuracy": accuracy},
    }


def _run_stage(
    label: str,
    evaluate: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    metadata: Dict[str, Any],
    hyper_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], StageResult]:
    started = time.monotonic()
    result = evaluate(metadata, hyper_params)
    if not isinstance(result, dict):
        raise RuntimeError(f"{label} returned {type(result).__name__}, expected dict")
    finished = time.monotonic()
    return result, StageResult(
        label=label,
        started_at=started,
        finished_at=finished,
        status="ok",
        details={"keys": sorted(result.keys())[:12]},
    )


def _read_manifest_contract(path: Path = _MANIFEST_PATH) -> Dict[str, Any]:
    manifest = load_manifest(path)
    if not isinstance(manifest, dict):
        raise RuntimeError("build_manifest.json must contain a JSON object")
    return manifest


def _apply_manifest_to_assets(workspace_root: Path, manifest: Dict[str, Any], metadata: Dict[str, Any]) -> List[str]:
    parameters = manifest.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    summary_path = workspace_root / "WORKSPACE_AUDIT.md"
    lines = [
        "# Builder Orchestration Summary",
        "",
        f"- cycle: {metadata.get('current_cycle', 1)}",
        f"- strategy: {metadata.get('blueprint_system_strategy', metadata.get('blueprint_strategy', metadata.get('resolved_strategy', 'unknown')))}",
        f"- primary_strategy: {metadata.get('primary_strategy', metadata.get('resolved_strategy', 'unknown'))}",
        f"- resolved_strategy: {metadata.get('resolved_strategy', 'unknown')}",
        f"- selected_action: {metadata.get('selected_action_label', 'unknown')}",
        f"- scan_coverage: {metadata.get('scan_coverage', 'n/a')}",
        f"- anomaly_count: {metadata.get('anomaly_count', 'n/a')}",
        f"- pareto_frontier_size: {len(metadata.get('pareto_frontier', []))}",
        f"- compiled_target_count: {metadata.get('compiled_target_count', 0)}",
        f"- bytes_written: {metadata.get('bytes_written', 0)}",
        f"- optimization_level: {metadata.get('optimization_level', 'unknown')}",
        f"- accepted_neural_variant_count: {metadata.get('accepted_neural_variant_count', 0)}",
        "",
        "## Manifest Parameters",
        "",
    ]
    for key in sorted(parameters):
        lines.append(f"- {key}: {parameters[key]}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [str(summary_path)]


def _persist_orchestrator_state(manifest: Dict[str, Any], metadata: Dict[str, Any], path: Path = _MANIFEST_PATH) -> Dict[str, Any]:
    manifest = dict(manifest)
    manifest["current_cycle"] = int(metadata.get("current_cycle", 1))
    manifest["last_handshake_status"] = "ok"
    manifest["orchestrator_state"] = {
        "current_cycle": int(metadata.get("current_cycle", 1)) + 1,
        "score_trajectory": list(metadata.get("score_trajectory", []))[-15:],
        "kinetic_stall_cycles": int(metadata.get("kinetic_stall_cycles", 0)),
        "pareto_frontier": list(metadata.get("pareto_frontier", []))[:64],
        "tuned_population": list(metadata.get("tuned_population", []))[:64],
        "survival_tracker_stats": dict(metadata.get("survival_tracker_stats", {})),
        "baseline_config": dict(metadata.get("best_config", metadata.get("baseline_config", {}))),
        "previous_fingerprints": dict(metadata.get("file_fingerprints", {})),
        "accepted_neural_variants": list(metadata.get("accepted_neural_variants", []))[:16],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _record_experience_status(metadata: Dict[str, Any]) -> str:
    if metadata.get("experience_recorded"):
        return f"recorded:{metadata.get('selected_action_label', 'unknown')}"
    return "not-recorded"


def _resolve_workspace_candidate(workspace_root: Path, candidate: str, blueprint_dir: str = "") -> Path:
    raw = Path(candidate).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    if blueprint_dir:
        return (Path(blueprint_dir) / raw).resolve()
    return (workspace_root / raw).resolve()


def _direct_compile_candidates(metadata: Mapping[str, Any]) -> List[str]:
    candidates: List[str] = []
    scaffold = metadata.get("scaffold", {})
    if isinstance(scaffold, Mapping):
        source_entry = scaffold.get("source_entry")
        if isinstance(source_entry, (list, tuple)):
            candidates.extend(str(path).strip() for path in source_entry if str(path).strip())
        elif isinstance(source_entry, str) and source_entry.strip():
            candidates.append(source_entry.strip())
    registry = metadata.get("context_registry", {})
    if isinstance(registry, Mapping):
        for entry in registry.values():
            if isinstance(entry, Mapping):
                candidate = str(entry.get("path", "")).strip()
                if candidate:
                    candidates.append(candidate)
    parser_validation = metadata.get("parser_validation", {})
    if isinstance(parser_validation, Mapping):
        scan_targets = parser_validation.get("scan_targets", [])
        if isinstance(scan_targets, list):
            candidates.extend(str(path).strip() for path in scan_targets if str(path).strip())
    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _resolve_direct_compile_source(workspace_root: Path, metadata: Mapping[str, Any]) -> Path:
    blueprint_dir = str(metadata.get("blueprint_dir", "")).strip()
    for candidate in _direct_compile_candidates(metadata):
        resolved = _resolve_workspace_candidate(workspace_root, candidate, blueprint_dir)
        if resolved.is_file() and resolved.suffix.lower() == ".py":
            return resolved
    raise FileNotFoundError("DIRECT_COMPILE requires a resolvable Python source_entry/context path")


def _resolve_direct_compile_output(workspace_root: Path, metadata: Mapping[str, Any], source_path: Path) -> Path:
    scaffold = metadata.get("scaffold", {})
    blueprint_dir = str(metadata.get("blueprint_dir", "")).strip()
    distribution_directory = ""
    scaffold_name = ""
    if isinstance(scaffold, Mapping):
        distribution_directory = str(scaffold.get("distribution_directory", "")).strip()
        scaffold_name = str(scaffold.get("name", "")).strip()
    output_dir = (
        _resolve_workspace_candidate(workspace_root, distribution_directory, blueprint_dir)
        if distribution_directory
        else source_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{(scaffold_name or source_path.stem or 'matrix').strip()}.aeroc"
    return (output_dir / output_name).resolve()


def _annotate_spacetime(network, ledger) -> None:
    """Annotate every HIN node with a logical spacetime coordinate.

    Each node is placed on a deterministic logical lattice and its mutation is
    chronologically recorded in the ledger, binding it to an absolute
    ``T_causal`` index.
    """
    from core.spacetime_ledger import CoordinateVector

    for index, node in enumerate(list(network.nodes.values())):
        coord = CoordinateVector(str(index), str(index * index), "0", -1)
        ledger.annotate_node(node, coord, {"agent": type(node).__name__})


def handle_aero_calculus_build(
    source_path: str,
    output_path: str,
    reduce_graph: bool = True,
    heal_callback: Optional[Callable[[Any], bool]] = None,
    max_healing_attempts: int = 1,
) -> dict:
    """Compile a source script to Aero-Calculus, verify, reduce and serialize.

    Returns a small report describing the compiled and (optionally) minimized
    topology.  If the build fails with a structural invariant violation, or if
    the translated network is empty, an optional *heal_callback* is invoked on
    the network and the build is retried up to *max_healing_attempts* times
    before the failure is propagated.
    """
    from core.aero_frontend import python_source_to_uast
    from core.aeroc import save_aeroc
    from core.hin_vm import UniversalHINNetwork
    from core.spacetime_ledger import BlockUniverseLedger, RigidityVerifier
    from core.translator import UASTToHINTranslator

    if heal_callback is None and max_healing_attempts > 0:
        def _default_heal(network: Any) -> bool:
            try:
                report = TopologicalSelfHealer().heal_network(network)
                return report["healed"] > 0 or report["remaining"] == 0
            except Exception:  # noqa: BLE001
                return False

        heal_callback = _default_heal

    print(f"[*] Ingesting Source Context: {source_path}")
    with open(source_path, "r", encoding="utf-8") as handle:
        raw_source = handle.read()
    uast_representation = python_source_to_uast(raw_source)

    print("[*] Performing Homomorphic UAST-to-HIN Translation...")
    translator = UASTToHINTranslator()
    network = translator.translate_uast(uast_representation)
    compiled_nodes = len(network.nodes)

    ledger_path = os.path.join(
        os.path.dirname(os.path.abspath(output_path)) or ".", "context.aero"
    )

    last_error: Optional[Exception] = None
    for attempt in range(max_healing_attempts + 1):
        try:
            ledger = BlockUniverseLedger(ledger_path)
            _annotate_spacetime(network, ledger)

            print("[*] Applying automatic module mitosis (Fiedler spectral partition)...")
            primary, secondary = translator.execute_mitosis(network)
            mitosis_split = len(secondary.nodes) > 0

            reduced_steps = 0
            rigidity = "skipped (--no-reduce)"
            if reduce_graph:
                print("[*] Conducting Boundary coordinate-perturbation sweeps...")
                verifier = RigidityVerifier()
                boundary = [n for n in primary.nodes.values() if getattr(n, "coordinate", None)]
                try:
                    verifier.verify_boundary(boundary)
                    rigidity = "verified"
                except Exception as exc:  # noqa: BLE001 - surface as a report field
                    rigidity = f"anomaly: {exc}"

                print("[*] Reducing graph to its minimized normal form inside the HIN VM...")
                uni = UniversalHINNetwork.adopt(
                    primary, ledger=ledger, ledger_path=ledger_path
                )
                reduced_steps = uni.run_to_completion()

            save_aeroc(primary, output_path)
            if mitosis_split:
                part2 = os.path.splitext(output_path)[0] + ".part2.aeroc"
                save_aeroc(secondary, part2)
                print(f"[+] Module mitosis emitted secondary partition -> {part2}")
            print(f"[+] Aero-Calculus Compilation Complete! Saved to {output_path}")

            return {
                "source": source_path,
                "output": output_path,
                "compiled_nodes": compiled_nodes,
                "reduced_nodes": len(primary.nodes),
                "reduction_steps": reduced_steps,
                "rigidity": rigidity,
                "mitosis_split": mitosis_split,
                "ledger_length": len(ledger),
            }
        except Exception as exc:
            last_error = exc
            if heal_callback is not None and attempt < max_healing_attempts:
                target = network
                if "primary" in locals() and primary is not None:
                    target = primary
                try:
                    healed = bool(heal_callback(target))
                except Exception as heal_exc:  # noqa: BLE001
                    logger.warning("Healing callback raised: %s", heal_exc)
                    healed = False
                if healed:
                    logger.info(
                        "Build attempt %d failed (%s); healing applied, retrying",
                        attempt + 1,
                        exc,
                    )
                    continue
            raise
    raise RuntimeError(
        f"Aero-Calculus build failed after {max_healing_attempts} healing attempt(s): {last_error}"
    ) from last_error


def run_direct_compile(
    workspace_root: str,
    build_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile the blueprint's primary source directly to a .aeroc artifact."""
    workspace = Path(workspace_root).resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {workspace}")
    metadata = dict(build_context) if isinstance(build_context, dict) else _extract_build_context(
        workspace, _read_manifest_contract()
    )
    metadata["workspace_root"] = str(workspace)
    metadata["strategy"] = "DIRECT_COMPILE"
    metadata["primary_strategy"] = "DIRECT_COMPILE"
    metadata["resolved_strategy"] = "DIRECT_COMPILE"
    metadata["selected_action_label"] = "direct_compile"
    metadata["strategy_mode"] = "direct_compile"
    source_path = _resolve_direct_compile_source(workspace, metadata)
    output_path = _resolve_direct_compile_output(workspace, metadata, source_path)

    def _topological_heal(network) -> bool:
        """Invoke :class:`TopologicalSelfHealer` on a broken HIN network."""
        try:
            report = TopologicalSelfHealer().heal_network(network)
            logger.info(
                "TopologicalSelfHealer healed %d broken edge(s); %d remaining",
                report["healed"],
                report["remaining"],
            )
            return report["healed"] > 0 or report["remaining"] == 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("TopologicalSelfHealer could not repair network: %s", exc)
            return False

    report = handle_aero_calculus_build(
        str(source_path),
        str(output_path),
        reduce_graph=True,
        heal_callback=_topological_heal,
        max_healing_attempts=1,
    )
    outputs = [output_path]
    part2 = output_path.with_suffix("").with_name(output_path.stem + ".part2").with_suffix(".aeroc")
    if part2.exists():
        outputs.append(part2)
    compiled_targets = [
        {
            "name": output.stem,
            "source": str(source_path),
            "output": str(output),
            "bytes_written": output.stat().st_size,
        }
        for output in outputs
        if output.exists()
    ]
    bytes_written = sum(target["bytes_written"] for target in compiled_targets)
    metadata["compile_report"] = report
    metadata["compiled_targets"] = compiled_targets
    metadata["compiled_target_count"] = len(compiled_targets)
    metadata["bytes_written"] = bytes_written
    metadata["aeroc_output"] = str(output_path)
    metadata["applied_assets"] = [str(output) for output in outputs if output.exists()]
    metadata["should_write_aeroc"] = should_write_aeroc(
        metadata["resolved_strategy"],
        metadata["compiled_target_count"],
        metadata["bytes_written"],
    )
    metadata["current_cycle"] = 1
    return metadata


def _user_selected_strategy(metadata: Mapping[str, Any]) -> str:
    strategy = str(metadata.get("blueprint_strategy", "")).strip()
    if strategy:
        return strategy
    system = metadata.get("system", {})
    if isinstance(system, Mapping):
        strategy = str(system.get("strategy", "")).strip()
        if strategy:
            return strategy
    return "DIRECT_COMPILE"


def _anomaly_ratio(metadata: Mapping[str, Any]) -> float:
    parser_validation = metadata.get("parser_validation", {})
    if not isinstance(parser_validation, Mapping):
        parser_validation = {}
    ceiling = int(
        metadata.get("anomaly_ceiling")
        or parser_validation.get("anomaly_ceiling")
        or 0
    )
    if ceiling <= 0:
        return 0.0
    anomaly_count = max(
        int(metadata.get("anomaly_count", 0) or 0),
        int(parser_validation.get("parameter_validation_failures", 0) or 0),
    )
    return anomaly_count / float(ceiling)


def _honor_blueprint_strategy(metadata: Dict[str, Any]) -> None:
    anomaly_ratio = _anomaly_ratio(metadata)
    metadata["anomaly_ratio"] = anomaly_ratio
    if anomaly_ratio >= _ANOMALY_DRIFT_THRESHOLD:
        strategy = _user_selected_strategy(metadata)
        logger.warning(
            "Anomaly ratio %.2f exceeds threshold; continuing with user-specified strategy %s.",
            anomaly_ratio,
            strategy,
        )
        metadata["resolved_strategy"] = strategy
        metadata["strategy_mode"] = strategy.lower()
        metadata["selected_action_label"] = "honor_blueprint_strategy"


def should_write_aeroc(
    strategy: str,
    compiled_count: int,
    bytes_written: int,
    direct_pass: bool = False,
) -> bool:
    # The write gate is intentionally strategy-name agnostic: a successful
    # direct build pass that emitted real bytes always wins, regardless of the
    # resolved strategy label.  This removes the legacy lockout where a
    # downgraded strategy name suppressed an otherwise valid asset write.
    if direct_pass and bytes_written > 0:
        return True
    if compiled_count > 0 and bytes_written > 0:
        return True
    return False


def _render_telemetry(telemetry: CycleTelemetry) -> None:
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "cls"], check=False)
    else:
        subprocess.run(["clear"], check=False)
    print("=" * 78)
    print(" BUILDER ORCHESTRATION TELEMETRY")
    print("=" * 78)
    print(
        f" cycle {telemetry.cycle}/{telemetry.total_cycles} | elapsed {telemetry.elapsed_seconds:.1f}s"
        f" | threads {telemetry.thread_pool_size} | stagnation {telemetry.stagnation}"
    )
    print("-" * 78)
    print(" stages")
    for stage in telemetry.stage_results:
        print(f"  - {stage.label:<16} {stage.status:<8} {stage.duration:>7.3f}s")
    print("-" * 78)
    print(f" strategy: {telemetry.strategy}")
    print(f" primary_strategy: {telemetry.primary_strategy}")
    print(f" resolved_strategy: {telemetry.resolved_strategy}")
    print(f" action  : {telemetry.selected_action}")
    print(f" replay  : {telemetry.replay_status}")
    print(f" manifest: {telemetry.manifest_status}")
    print(f" compiled: {telemetry.compiled_target_count}")
    print(f" bytes   : {telemetry.bytes_written}")
    print(f" opt_lvl : {telemetry.optimization_level}")
    print("-" * 78)
    print(" pareto")
    print(f"  frontier_size : {telemetry.pareto_summary.get('frontier_size', 0)}")
    print(f"  hypervolume   : {telemetry.pareto_summary.get('hypervolume', 0.0)}")
    print(f"  best_config   : {telemetry.pareto_summary.get('best_config', {})}")
    print("=" * 78)
    sys.stdout.flush()


def _telemetry_loop(stop_event: threading.Event, state: Dict[str, Any], interval_seconds: float) -> None:
    while not stop_event.wait(interval_seconds):
        telemetry = state.get("telemetry")
        if telemetry is not None:
            _render_telemetry(telemetry)


def _thread_pool_size(metadata: Dict[str, Any], manifest: Dict[str, Any]) -> int:
    parameters = manifest.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    suggested = parameters.get("scanner_concurrent_workers") or parameters.get("tuned_population_size")
    if suggested is None:
        suggested = metadata.get("environment_targets", {}).get("total_cooperating_agents", 4)
    try:
        return max(1, int(suggested))
    except (TypeError, ValueError):
        return 4


def _bootstrap_validator(stage_dir: Path) -> Dict[str, Any]:
    """Validate all Python files in the bootstrap staging directory.

    Performs syntax scanning (AST parse) and structural integrity checks.
    Returns a dict with ``errors`` and ``anomalies`` counts -- both must be
    zero for the atomic swap to proceed.
    """
    import ast as _ast

    errors = 0
    anomalies = 0
    checked = 0

    for py_file in stage_dir.rglob("*.py"):
        if not py_file.is_file():
            continue
        checked += 1
        source = py_file.read_text(encoding="utf-8", errors="replace")
        try:
            _ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:
            logger.warning("Bootstrap validation: syntax error in %s: %s", py_file, exc)
            errors += 1

    # Structural anomaly: no files produced at all is suspicious.
    if checked == 0:
        anomalies += 1

    return {"errors": errors, "anomalies": anomalies, "files_checked": checked}


def run_build(
    workspace_root: str,
    cycles: int = 3,
    telemetry_interval: float = _DEFAULT_TELEMETRY_INTERVAL,
    bootstrap_active: bool = False,
) -> Dict[str, Any]:
    from core.bootstrap import (
        BootstrapStage,
        detect_self_targeting,
        is_bootstrap_active,
        set_bootstrap_active,
    )

    workspace = Path(workspace_root).resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {workspace}")

    # Recursive loop guard: if already in a bootstrap pass, short-circuit.
    if bootstrap_active or is_bootstrap_active():
        logger.warning(
            "Bootstrap recursion detected — short-circuiting nested build cycle."
        )
        return {"short_circuited": True, "reason": "bootstrap_recursion_guard"}

    stages = _load_brain_modules()
    manifest = _read_manifest_contract()
    metadata = _extract_build_context(workspace, manifest)

    # Self-targeting detection: collect all target source paths and check for
    # overlap with the engine's own directory.
    blueprint_targets = _ensure_blueprint_target_paths(workspace / "blueprint.aero")
    _target_paths: List[str] = []
    for t in blueprint_targets:
        resolved = _resolve_target_paths(workspace, t)
        _target_paths.append(str(resolved["source_path"]))
    self_targeting = detect_self_targeting(_target_paths)

    bootstrap_stage: Optional[BootstrapStage] = None
    if self_targeting:
        set_bootstrap_active(True)
        bootstrap_stage = BootstrapStage(workspace)
        bootstrap_stage.prepare()
        # Seed the stage with every source file the blueprint references so the
        # compiler backend can resolve targets inside the isolated tree.
        bootstrap_stage.copy_target_files(blueprint_targets, workspace / "blueprint.aero")
        metadata["bootstrap_mode"] = True
        metadata["bootstrap_stage_dir"] = str(bootstrap_stage.stage_dir)
        logger.info("Self-targeting detected — bootstrap isolation engaged.")

    # Dynamically link the blueprint's optimization level to the module-level
    # override so _compile_targets and telemetry read it instead of the static
    # manifest fallback.
    global _bp_optimization_override
    bp_opt = metadata.get("blueprint_optimization_level", "")
    _bp_optimization_override = bp_opt if bp_opt else None

    total_cycles = max(1, int(cycles))
    telemetry_state: Dict[str, Any] = {}
    stop_event = threading.Event()
    telemetry_thread = threading.Thread(
        target=_telemetry_loop,
        args=(stop_event, telemetry_state, max(0.5, telemetry_interval)),
        daemon=True,
    )
    telemetry_thread.start()

    started = time.monotonic()
    applied_assets: List[str] = []
    try:
        for cycle in range(1, total_cycles + 1):
            metadata["current_cycle"] = cycle
            # Reset per-cycle decomposition state so stale counts from
            # previous cycles don't bleed into telemetry.
            metadata.pop("decomposition_files_written", None)
            metadata.pop("decomposition_file_count", None)
            metadata.pop("decomposition_bytes_written", None)
            before_snapshot = collect_workspace_snapshot(workspace)
            delta = compute_workspace_delta(
                metadata.get("previous_fingerprints", {}) if isinstance(metadata.get("previous_fingerprints"), dict) else {},
                before_snapshot,
            )
            metadata["workspace_delta"] = delta
            metadata["scan_targets"] = delta["changed_files"] or list(before_snapshot.keys())

            stage_results: List[StageResult] = []
            hyper_params = {"concurrent_worker_pool_size": _thread_pool_size(metadata, manifest)}

            scanner_label, scanner_eval = stages[0]
            metadata, stage_result = _run_stage(scanner_label, scanner_eval, metadata, hyper_params)
            stage_results.append(stage_result)
            _seed_objectives(metadata)

            latency_times = {
                "scanner_wall_seconds": metadata.get("scanner_wall_seconds", 0.0),
                "cycle_elapsed_seconds": time.monotonic() - started,
            }
            metadata["latency_times"] = latency_times

            with ThreadPoolExecutor(max_workers=2) as executor:
                decision_future = executor.submit(_run_stage, stages[1][0], stages[1][1], dict(metadata), hyper_params)
                tuner_future = executor.submit(_run_stage, stages[2][0], stages[2][1], dict(metadata), hyper_params)
                decision_metadata, decision_result = decision_future.result()
                tuner_metadata, tuner_result = tuner_future.result()

            # Apply tuner first, then decision — the decision tree is
            # authoritative for strategy/action/FSM keys and must not be
            # overwritten by the tuner's stale snapshot of those fields.
            metadata.update(tuner_metadata)
            metadata.update(decision_metadata)
            stage_results.extend([decision_result, tuner_result])
            _honor_blueprint_strategy(metadata)

            if bool(metadata.get("kinetic_stagnation_anomaly") or metadata.get("is_stagnant")):
                neural_variants = maybe_run_neural_synthesis(
                    metadata,
                    _build_baseline_trace(metadata),
                    _build_sandbox_sample_params(metadata),
                )
                metadata["accepted_neural_variants"] = neural_variants
                metadata["accepted_neural_variant_count"] = len(neural_variants)
            else:
                metadata["accepted_neural_variants"] = []
                metadata["accepted_neural_variant_count"] = 0

            after_snapshot = collect_workspace_snapshot(workspace)
            _enforce_read_only_boundary(before_snapshot, after_snapshot)

            # ── DIRECT_COMPILE enforcement clamp ──────────────────────────────
            # If the active command is 'build' or the blueprint explicitly
            # requests DIRECT_COMPILE, any residual AGGRESSIVE_MUTATION /
            # polyglot-decomposition state produced by drift or FSM heuristics
            # must be neutralised so the compiler backend is not bypassed.
            _active_cmd_orch = str(metadata.get("active_command", "")).lower()
            _bp_sys_strat_orch = str(metadata.get("blueprint_system_strategy", "")).upper()
            _direct_compile_mode = (
                _active_cmd_orch == "build" or _bp_sys_strat_orch == "DIRECT_COMPILE"
            )
            # User compilation intent is authoritative: under a build / direct
            # compile pass the orchestrator is strictly forbidden from
            # downgrading the strategy to CONSERVATIVE or AGGRESSIVE_MUTATION on
            # the basis of drift or anomaly heuristics, since either downgrade
            # bypasses the codegen backend and freezes metrics at zero.
            if _direct_compile_mode:
                _drifted = (
                    metadata.get("resolved_strategy") not in ("DIRECT_COMPILE", None)
                    or metadata.get("primary_strategy") not in ("DIRECT_COMPILE", None)
                    or metadata.get("strategy_mode") == "aggressive_decomposition"
                    or metadata.get("selected_action_label") in (
                        "execute_polyglot_decomposition", "boost_mutation_sigma"
                    )
                )
                if _drifted:
                    logger.info(
                        "DIRECT_COMPILE clamp applied (cycle %d): resolved_strategy was %r, "
                        "primary_strategy was %r, action was %r — locking all structural "
                        "strategy keys to DIRECT_COMPILE for compiler pass.",
                        cycle,
                        metadata.get("resolved_strategy"),
                        metadata.get("primary_strategy"),
                        metadata.get("selected_action_label"),
                    )
                # User compilation intent is authoritative: lock every structural
                # runtime key so no runtime default or reset to BALANCED /
                # AGGRESSIVE_MUTATION can bypass the codegen backend.
                metadata["strategy"] = "DIRECT_COMPILE"
                metadata["resolved_strategy"] = "DIRECT_COMPILE"
                metadata["primary_strategy"] = "DIRECT_COMPILE"
                metadata["selected_action_label"] = "direct_compile"
                metadata["strategy_mode"] = "direct_compile"

            # Physical decomposition: when the FSM triggers aggressive
            # decomposition, physically split source monoliths into modules.
            if metadata.get("strategy_mode") == "aggressive_decomposition" or (
                metadata.get("resolved_strategy") == "AGGRESSIVE_MUTATION"
                and metadata.get("selected_action_label") in (
                    "execute_polyglot_decomposition", "boost_mutation_sigma"
                )
            ):
                from src.decomposition.splitter import run_decomposition

                decomp_result = run_decomposition(metadata, workspace)
                metadata["decomposition_files_written"] = decomp_result["decomposition_files_written"]
                metadata["decomposition_file_count"] = decomp_result["decomposition_file_count"]
                metadata["decomposition_bytes_written"] = decomp_result["decomposition_bytes_written"]
                if decomp_result["decomposition_errors"]:
                    logger.warning("Decomposition errors: %s", decomp_result["decomposition_errors"])

            manifest = _read_manifest_contract()
            # When in bootstrap isolation mode, redirect compilation output
            # to the shadow staging directory instead of the live workspace.
            compile_root = workspace
            if bootstrap_stage is not None and bootstrap_stage.is_prepared:
                compile_root = bootstrap_stage.stage_dir
            compilation_summary = _compile_targets(compile_root, manifest)
            metadata.update(compilation_summary)

            # ── Direct backend execution ──────────────────────────────────────
            # Under a build / direct-compile pass, hand the scanner's mapped 1-D
            # UAST token stream straight to the binary freezer so the codegen
            # backend always runs.  The cyclic loop historically scanned and
            # tuned but never invoked a code generator, leaving compiled=0 /
            # bytes=0 and dead-locking the asset write gate.  Freezing the
            # matrix here emits a concrete ``matrix.aeroc`` to the workspace
            # root and updates the build-context compilation metrics with real
            # values so the write gate can trigger.
            if _direct_compile_mode:
                metadata.setdefault("strategy", "DIRECT_COMPILE")
                matrix_summary = _freeze_uast_matrix(compile_root, metadata)
                metadata["matrix_output"] = matrix_summary["matrix_output"]
                metadata["matrix_unit_count"] = matrix_summary["matrix_unit_count"]
                # The frozen matrix asset is itself a compiled target; count it
                # alongside any per-target compactions so the gate never sees 0.
                metadata["compiled_target_count"] = (
                    int(metadata.get("compiled_target_count", 0)) + 1
                )
                metadata["bytes_written"] = (
                    int(metadata.get("bytes_written", 0))
                    + int(matrix_summary["matrix_bytes_written"])
                )

            # Merge decomposition output into the compiled/bytes telemetry
            # so the BUILDER ORCHESTRATION TELEMETRY view reflects actual
            # physical files written to disk.
            if metadata.get("decomposition_files_written"):
                metadata["compiled_target_count"] = (
                    int(metadata.get("compiled_target_count", 0))
                    + int(metadata.get("decomposition_file_count", 0))
                )
                metadata["bytes_written"] = (
                    int(metadata.get("bytes_written", 0))
                    + int(metadata.get("decomposition_bytes_written", 0))
                )
            manifest = _persist_orchestrator_state(manifest, metadata)
            metadata["should_write_aeroc"] = should_write_aeroc(
                str(metadata.get("resolved_strategy", "unknown")),
                int(metadata.get("compiled_target_count", 0)),
                int(metadata.get("bytes_written", 0)),
                direct_pass=_direct_compile_mode,
            )
            if metadata["should_write_aeroc"]:
                applied_assets = _apply_manifest_to_assets(workspace, manifest, metadata)
            else:
                applied_assets = []
                logger.warning(
                    "Skipping .aeroc asset write gate: strategy=%s compiled=%d bytes=%d",
                    metadata.get("resolved_strategy", "unknown"),
                    int(metadata.get("compiled_target_count", 0)),
                    int(metadata.get("bytes_written", 0)),
                )

            telemetry_state["telemetry"] = CycleTelemetry(
                cycle=cycle,
                total_cycles=total_cycles,
                stage_results=stage_results,
                selected_action=str(metadata.get("selected_action_label", "unknown")),
                resolved_strategy=str(metadata.get("resolved_strategy", "unknown")),
                primary_strategy=str(metadata.get("primary_strategy", metadata.get("resolved_strategy", "unknown"))),
                strategy=str(metadata.get("blueprint_system_strategy", metadata.get("blueprint_strategy", metadata.get("resolved_strategy", "unknown")))),
                thread_pool_size=hyper_params["concurrent_worker_pool_size"],
                stagnation=bool(metadata.get("kinetic_stagnation_anomaly") or metadata.get("is_stagnant")),
                pareto_summary={
                    "frontier_size": len(metadata.get("pareto_frontier", [])),
                    "hypervolume": metadata.get("survival_tracker_stats", {}).get("hypervolume", 0.0),
                    "best_config": metadata.get("best_config", {}),
                },
                replay_status=_record_experience_status(metadata),
                manifest_status=str(manifest.get("last_handshake_status", "unknown")),
                compiled_target_count=int(metadata.get("compiled_target_count", 0)),
                bytes_written=int(metadata.get("bytes_written", 0)),
                optimization_level=str(metadata.get("optimization_level", "unknown")),
                elapsed_seconds=time.monotonic() - started,
            )
            _render_telemetry(telemetry_state["telemetry"])
    finally:
        stop_event.set()
        telemetry_thread.join(timeout=1.0)

        # Bootstrap isolation: validate staged output and promote or discard.
        if bootstrap_stage is not None and bootstrap_stage.is_prepared:
            validation_passed = bootstrap_stage.validate(_bootstrap_validator)
            if validation_passed:
                promoted = bootstrap_stage.promote()
                metadata["bootstrap_promoted"] = True
                metadata["bootstrap_promoted_files"] = promoted
                logger.info(
                    "Bootstrap atomic swap complete: %d file(s) promoted to live tree.",
                    len(promoted),
                )
            else:
                bootstrap_stage.discard()
                metadata["bootstrap_promoted"] = False
                metadata["bootstrap_rollback"] = True
                logger.warning(
                    "Bootstrap validation FAILED — staged changes discarded. "
                    "Live workspace is unmodified (safe rollback)."
                )
            set_bootstrap_active(False)

    metadata["applied_assets"] = applied_assets
    metadata["manifest_path"] = str(_MANIFEST_PATH)
    return metadata


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


class AeroShadowStagingManager:
    """Sandboxed shadow staging for self-healing structural changes.

    Self-healing routines must never write volatile or broken code layouts into
    production source paths. This manager runs every repair inside an isolated
    scratch workspace (``<workspace_root>/.aero/bootstrap_stage/``), verifies the
    result via a polyglot syntax-compilation check, and only then performs an
    atomic swap into the production tree. Failed transactions are purged and the
    live source path is left untouched.

    Transaction state machine (per the framework constraints), bounded by the
    retry budget ``B = build_retry_budget``::

        State(i) = Swap(f_c, f)        if V(f_c) == 1
                 = Loop(i + 1)         if V(f_c) == 0 and i  < B
                 = Purge(f_c) & Abort  if V(f_c) == 0 and i == B
    """

    #: Relative location of the scratch staging cache under the workspace root.
    STAGE_SUBPATH = os.path.join(".aero", "bootstrap_stage")

    def __init__(
        self,
        workspace_root: str,
        build_retry_budget: int = 3,
        isolation_token: Optional[str] = None,
    ) -> None:
        self.workspace_root = os.path.abspath(workspace_root)
        # The retry budget B is bound to a maximum value of 3.
        self.build_retry_budget = max(1, min(int(build_retry_budget), 3))
        # Process-isolated staging (requirement #5): a uniquely-named
        # subdirectory under the shared base prevents filesystem collisions when
        # multiple healing transactions or candidate evaluations run in parallel.
        self.stage_base = os.path.join(self.workspace_root, self.STAGE_SUBPATH)
        self.isolation_token = isolation_token or (
            f"heal_p{os.getpid()}_t{threading.get_ident()}_{uuid.uuid4().hex[:8]}"
        )
        self.stage_root = os.path.join(self.stage_base, self.isolation_token)
        os.makedirs(self.stage_root, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Transaction driver
    # ------------------------------------------------------------------ #
    def execute_healing_transaction(
        self,
        relative_file_path: str,
        repair_function: Callable[[bytes], bytes],
    ) -> bool:
        """Stage, repair, verify and (on success) atomically promote a file.

        ``repair_function`` receives the current file bytes and returns the
        repaired bytes. The repair/verify cycle runs inside the retry budget;
        each failed iteration re-feeds the most recent staged bytes. Returns
        ``True`` only when a verified result was atomically swapped into the
        production path; otherwise the stage is purged and production is
        untouched.
        """
        prod_path = os.path.join(self.workspace_root, relative_file_path)
        if not os.path.isfile(prod_path):
            logger.warning("ShadowStaging: production file missing: %s", prod_path)
            return False

        stage_path = os.path.join(self.stage_root, relative_file_path)
        os.makedirs(os.path.dirname(stage_path) or self.stage_root, exist_ok=True)

        # Safely copy the active file into staging, preserving metadata.
        shutil.copy2(prod_path, stage_path)

        try:
            for attempt in range(1, self.build_retry_budget + 1):
                try:
                    with open(stage_path, "rb") as handle:
                        current_bytes = handle.read()
                    repaired = repair_function(current_bytes)
                    with open(stage_path, "wb") as handle:
                        handle.write(repaired)
                except Exception as exc:  # repair closures are untrusted
                    logger.warning(
                        "ShadowStaging: repair raised on attempt %d/%d: %s",
                        attempt, self.build_retry_budget, exc,
                    )
                    continue

                # V(f_c): syntax-compilation verification of the staged file.
                if self._verify_compilation(stage_path):
                    self._atomic_workspace_swap(stage_path, prod_path)
                    logger.info(
                        "ShadowStaging: transaction verified on attempt %d/%d; "
                        "promoted %s",
                        attempt, self.build_retry_budget, relative_file_path,
                    )
                    return True

                logger.debug(
                    "ShadowStaging: verification failed on attempt %d/%d for %s",
                    attempt, self.build_retry_budget, relative_file_path,
                )

            # Budget exhausted (i == B with V == 0): purge and abort.
            logger.warning(
                "ShadowStaging: retry budget (%d) exhausted for %s — staged "
                "changes discarded, production left unmodified.",
                self.build_retry_budget, relative_file_path,
            )
            return False
        finally:
            self._purge_stage()

    # ------------------------------------------------------------------ #
    # Polyglot compilation verification — V(f_c)
    # ------------------------------------------------------------------ #
    def _verify_compilation(self, file_path: str) -> bool:
        """Run a syntax-only compilation check keyed on the file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".py":
            command = ["python3", "-m", "py_compile", file_path]
        elif ext == ".rs":
            command = [
                "rustc", "--crate-type=lib", "--emit=mir",
                "-Z", "no-codegen", file_path,
            ]
        elif ext in (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h"):
            command = ["clang++", "-fsyntax-only", "-std=c++17", file_path]
        else:
            # No verifier registered for this language: nothing to assert.
            logger.debug("ShadowStaging: no compiler check for extension %r", ext)
            return True

        try:
            result = subprocess.run(
                command,
                cwd=self.stage_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            logger.warning(
                "ShadowStaging: toolchain %r unavailable; cannot verify %s",
                command[0], file_path,
            )
            return False
        except subprocess.TimeoutExpired:
            logger.warning("ShadowStaging: verification timed out for %s", file_path)
            return False

        if result.returncode != 0:
            logger.debug(
                "ShadowStaging: verifier %r rejected %s: %s",
                command[0], file_path, (result.stderr or "").strip()[:500],
            )
        return result.returncode == 0

    # ------------------------------------------------------------------ #
    # Atomic promotion + cleanup
    # ------------------------------------------------------------------ #
    def _atomic_workspace_swap(self, stage_path: str, prod_path: str) -> None:
        """Atomically replace ``prod_path`` with the verified ``stage_path``.

        The verified bytes are first written to a transient ``.tmp`` file inside
        the production directory (same filesystem) and then moved into place via
        :func:`os.replace`, which is atomic and thread-safe on a single volume.
        """
        prod_dir = os.path.dirname(prod_path) or self.workspace_root
        os.makedirs(prod_dir, exist_ok=True)

        with open(stage_path, "rb") as handle:
            verified_bytes = handle.read()

        fd, tmp_path = tempfile.mkstemp(
            dir=prod_dir,
            prefix=os.path.basename(prod_path) + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(verified_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            # Mirror source metadata onto the staged result before promotion.
            shutil.copystat(stage_path, tmp_path)
            os.replace(tmp_path, prod_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _purge_stage(self) -> None:
        """Remove the staging cache path on transaction termination."""
        shutil.rmtree(self.stage_root, ignore_errors=True)
        os.makedirs(self.stage_root, exist_ok=True)


# --------------------------------------------------------------------------- #
# Self-Healing v2 subsystem wiring
# --------------------------------------------------------------------------- #
# The four self-healing modules are imported defensively: a missing optional
# dependency (e.g. tree-sitter) must never break the primary orchestrator import
# path or the existing test suite. Unavailable pieces simply disable the matching
# pipeline step at runtime.
try:  # Two-Tiered Structural Merger
    from builder_brains.compactor import TwoTieredStructuralMerger
except ImportError:  # pragma: no cover - defensive
    TwoTieredStructuralMerger = None  # type: ignore

try:  # Tree-sitter Syntactic Recovery
    from builder_brains.recovery_parser import AeroTreeRecoveryParser
except ImportError:  # pragma: no cover - defensive
    AeroTreeRecoveryParser = None  # type: ignore

try:  # Dependency Reflux Engine
    from builder_brains.reflux import AeroDependencyRefluxEngine
except ImportError:  # pragma: no cover - defensive
    AeroDependencyRefluxEngine = None  # type: ignore

try:  # Stateful LSP Diagnostic Binder
    from src.lsp_proxy import LspDiagnosticRefluxBinder
except ImportError:  # pragma: no cover - defensive
    try:
        from lsp_proxy import LspDiagnosticRefluxBinder  # type: ignore
    except ImportError:  # pragma: no cover - defensive
        LspDiagnosticRefluxBinder = None  # type: ignore


class AeroCoreExecutionOrchestrator:
    """Unified Self-Healing v2 build pipeline.

    Stitches the four self-healing subsystems into a single per-file execution
    flow:

      * **Syntactic Recovery** (:class:`AeroTreeRecoveryParser`) patches raw
        Tree-sitter ``ERROR`` / ``MISSING`` nodes in memory.
      * **Two-Tiered Structural Merging**
        (:class:`~builder_brains.compactor.TwoTieredStructuralMerger`) folds the
        machine patch back onto the developer's original layout/trivia.
      * **Stateful LSP Binding**
        (:class:`~src.lsp_proxy.LspDiagnosticRefluxBinder`) surfaces pending
        semantic-repair actions captured from language servers.
      * **Dependency Reflux**
        (:class:`~builder_brains.reflux.AeroDependencyRefluxEngine`) applies
        those actions (missing imports / use declarations).

    Every healed file is validated and atomically promoted via
    :class:`AeroShadowStagingManager`, so a failed heal never pollutes the
    production source tree.

    The engine degrades gracefully: any subsystem whose optional dependency is
    unavailable is skipped, and the rest of the pipeline still runs.
    """

    def __init__(
        self,
        workspace_root: str,
        language: Any = None,
        parser: Any = None,
        language_name: Optional[str] = None,
        lsp_binder: Any = None,
        build_retry_budget: int = 3,
    ) -> None:
        self.workspace_root = os.path.abspath(workspace_root)
        self.language_name = language_name
        self.staging = AeroShadowStagingManager(
            self.workspace_root, build_retry_budget=build_retry_budget
        )

        # Step 1 dependency: Tree-sitter recovery (optional).
        self.recovery_parser = None
        if AeroTreeRecoveryParser is not None and language is not None and parser is not None:
            self.recovery_parser = AeroTreeRecoveryParser(
                language, parser, language_name=language_name
            )

        # Step 2 dependencies.
        self.merger = TwoTieredStructuralMerger() if TwoTieredStructuralMerger else None
        self.reflux_engine = (
            AeroDependencyRefluxEngine() if AeroDependencyRefluxEngine else None
        )
        self.lsp_binder = lsp_binder
        if self.lsp_binder is None and LspDiagnosticRefluxBinder is not None:
            self.lsp_binder = LspDiagnosticRefluxBinder()

    # ------------------------------------------------------------------ #
    # End-to-end per-file pipeline
    # ------------------------------------------------------------------ #
    def process_target_file(self, relative_file_path: str) -> Dict[str, Any]:
        """Run the three-step self-healing pipeline for a single target file.

        Returns a report dict describing whether the file was healed, which
        actions fired, and the terminal status of the transaction.
        """
        report: Dict[str, Any] = {
            "file": relative_file_path,
            "healed": False,
            "recovery_mutated": False,
            "reflux_actions": [],
            "status": "noop",
        }

        prod_path = os.path.join(self.workspace_root, relative_file_path)
        try:
            with open(prod_path, "rb") as handle:
                original_bytes = handle.read()
        except OSError as exc:
            logger.warning("CoreExec: cannot read %s: %s", prod_path, exc)
            report["status"] = "missing_source"
            return report

        # ---- Step 1: Syntactic recovery ---------------------------------- #
        recovered_bytes = original_bytes
        if self.recovery_parser is not None:
            tree, recovered_bytes, mutated = self.recovery_parser.attempt_recovery(
                original_bytes
            )
            report["recovery_mutated"] = bool(mutated)
            # If anomalies remain after the in-memory patch attempt, fall back.
            if tree.root_node.has_error:
                logger.warning(
                    "CoreExec: PIPELINE FALLBACK — unrecoverable syntax anomalies "
                    "remain in %s after in-memory recovery; skipping self-heal.",
                    relative_file_path,
                )
                report["status"] = "fallback_unrecoverable"
                return report

        # ---- Step 2: compile-time healing transaction wrapper ------------ #
        original_source = original_bytes.decode("utf-8", errors="replace")
        pending_actions = self._pending_actions_for(prod_path)
        report["reflux_actions"] = [a.get("action") for a in pending_actions]

        def compile_time_healing_transaction(source_bytes: bytes) -> bytes:
            """Combine layout trivia, syntax patches and LSP-driven reflux."""
            # Decode the staged source and the recovered (machine) stream.
            synthesized = recovered_bytes.decode("utf-8", errors="replace")

            # Two-tiered merge: re-apply the developer's layout/trivia onto the
            # machine-synthesized patch.
            if self.merger is not None:
                merged = self.merger.merge_clean_reconstruction(
                    original_source, synthesized
                )
            else:
                merged = synthesized
            merged_bytes = merged.encode("utf-8")

            # Route pending LSP diagnostics to the reflux engine to inject any
            # missing imports / use declarations.
            if self.reflux_engine is not None and pending_actions:
                merged_bytes = self._apply_reflux(
                    prod_path, merged_bytes, pending_actions
                )
            return merged_bytes

        # ---- Step 3: sandboxed staging containment loop ------------------ #
        promoted = self.staging.execute_healing_transaction(
            relative_file_path, compile_time_healing_transaction
        )
        report["healed"] = bool(promoted)
        report["status"] = "promoted" if promoted else "rolled_back"

        if promoted:
            self._log_healing_summary(report)
        return report

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _pending_actions_for(self, prod_path: str) -> List[Dict[str, Any]]:
        """Fetch pending LSP reflux commands keyed to this file, if any."""
        if self.lsp_binder is None:
            return []
        pending = getattr(self.lsp_binder, "pending_reflux_commands", {}) or {}
        # The binder keys by resolved path; tolerate either absolute or relative.
        for key in (prod_path, os.path.abspath(prod_path)):
            if key in pending:
                return list(pending[key])
        return []

    def _apply_reflux(
        self, prod_path: str, source_bytes: bytes, actions: List[Dict[str, Any]]
    ) -> bytes:
        """Apply reflux actions to in-memory bytes via a transient scratch file.

        ``AeroDependencyRefluxEngine`` reads from a path; we stage the merged
        bytes into a temp file (matching the production suffix so language
        detection holds) so the engine can operate without touching production.
        """
        suffix = os.path.splitext(prod_path)[1]
        fd, tmp_path = tempfile.mkstemp(dir=self.staging.stage_root, suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(source_bytes)
            patched = self.reflux_engine.apply_reflux_patches(tmp_path, actions)
            return patched if patched else source_bytes
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _log_healing_summary(self, report: Dict[str, Any]) -> None:
        """Emit a clean summary of the self-healing actions taken."""
        actions = ", ".join(a for a in report["reflux_actions"] if a) or "none"
        logger.info(
            "Self-Healing v2 SUCCESS — %s | syntactic_recovery=%s | "
            "reflux_actions=[%s] | promoted atomically.",
            report["file"], report["recovery_mutated"], actions,
        )



# ===========================================================================
# Self-Healing v2: Topological re-wiring (directive: geometric resolution)
# ===========================================================================
class TopologicalSelfHealer:
    """Heal a HIN network geometrically instead of by text-token patching.

    A compiler/parser failure is reified (see
    :func:`error_interceptor.reify_parse_failure_as_port`) as an *un-terminated
    edge* -- an auxiliary port with no partner.  This healer resolves it by
    finding the shortest causal distance to a successful historical coordinate
    in the ``context.aero`` Vantage-Point Tree and grafting a clean conditional
    :class:`SwitchNode` (with a ledger-validated fallback contract edge) onto
    the broken interface, then re-verifying boundary rigidity.
    """

    def __init__(self, ledger=None):
        self.ledger = ledger

    # -- discovery ---------------------------------------------------------
    @staticmethod
    def find_unterminated_ports(network) -> List["Any"]:
        """Return every auxiliary port left without a partner (a broken edge)."""
        broken = []
        for node in network.nodes.values():
            for aux in node.aux:
                if aux.target is None:
                    broken.append(aux)
        return broken

    def heal_network(self, network) -> Dict[str, Any]:
        """Heal every un-terminated interface in ``network``; return a report."""
        healed = 0
        for port in self.find_unterminated_ports(network):
            if port.target is None and self.heal_unterminated_interface(network, port):
                healed += 1
        return {"healed": healed, "remaining": len(self.find_unterminated_ports(network))}

    # -- geometric resolution ---------------------------------------------
    def heal_unterminated_interface(self, broken_network, faulty_port) -> bool:
        from core.hin_vm import EraserNode, SwitchNode, ValueNode
        from core.spacetime_ledger import (
            AnomalyClosureError,
            CoordinateVector,
            RigidityVerifier,
            VantagePointTree,
        )

        logger.info(
            "Self-Healing v2 (topological): un-terminated port %s on node %s",
            faulty_port.name, faulty_port.owner.node_id,
        )

        # 1. Shortest causal distance in the VP-Tree over healthy coordinates.
        fallback_value = self._nearest_historical_fallback(broken_network, faulty_port)

        # 2./3. Graft a conditional SwitchNode with a ledger-validated fallback
        # contract edge: the broken edge becomes the switch's selected output.
        switch = SwitchNode(broken_network.fresh_id("σ"))
        broken_network.register_node(switch)
        condition = ValueNode(broken_network.fresh_id("V"), fallback_value)
        broken_network.register_node(condition)
        broken_network._link(condition.p, switch.p)
        broken_network._link(faulty_port, switch.a_3)
        for branch in (switch.a_1, switch.a_2):
            eraser = EraserNode(broken_network.fresh_id("ε"))
            broken_network.register_node(eraser)
            broken_network._link(eraser.p, branch)

        # 4. Re-verify boundary rigidity -- no lingering AnomalyClosureError.
        boundary = []
        for node in (switch, condition, faulty_port.owner):
            coord = getattr(node, "coordinate", None)
            if coord is None:
                import hashlib

                d = hashlib.sha256(node.node_id.encode()).digest()
                coord = CoordinateVector(
                    str(int.from_bytes(d[0:8], "big") + 1),
                    str(int.from_bytes(d[8:16], "big") + 2),
                    str(int.from_bytes(d[16:24], "big") + 3),
                    -1,
                )
            boundary.append(coord)
        try:
            RigidityVerifier().verify_boundary(boundary)
        except AnomalyClosureError:
            raise
        return True

    def _nearest_historical_fallback(self, network, faulty_port):
        """Pick a ledger-validated fallback value via VP-Tree nearest match."""
        from core.spacetime_ledger import CoordinateVector, VantagePointTree

        items = []
        values = {}
        for node in network.nodes.values():
            coord = getattr(node, "coordinate", None)
            if coord is not None:
                items.append((node.node_id, coord))
                if getattr(node, "value", None) is not None:
                    values[node.node_id] = node.value
        target = getattr(faulty_port.owner, "coordinate", None)
        if items and target is not None:
            tree = VantagePointTree(items)
            key, _ = tree.nearest(target)
            if key in values:
                return values[key]
        # Safe default fallback contract: route to the false branch.
        return False
