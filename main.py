from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import orchestrator
from core.verify_dependencies import ContractViolationError, VerifyDependencies

logger = logging.getLogger("aero.main")

_BLUEPRINT_CONFIG = Path(__file__).resolve().parent / "tests" / "fixtures" / "blueprint_config.json"


def _load_blueprint_config(path: Optional[str] = None) -> dict:
    """Load the optional JSON build-configuration overlay.

    The overlay tunes default CLI behaviour (cycle counts, telemetry cadence)
    without touching the blueprint itself.  A missing or malformed file is not
    fatal -- the engine simply falls back to its built-in defaults.
    """
    config_path = Path(path) if path else _BLUEPRINT_CONFIG
    if not config_path.is_file():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable blueprint config %s: %s", config_path, exc)
        return {}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def plan_command(args: argparse.Namespace) -> int:
    """Render the build DAG for a blueprint without executing it."""
    # Aero-Calculus mode: render the physical HIN port topology of a .aeroc.
    if getattr(args, "aeroc", None):
        if not os.path.isfile(args.aeroc):
            print(f"error: .aeroc not found: {args.aeroc}", file=sys.stderr)
            return 1
        from core.aeroc import load_aeroc

        network = load_aeroc(args.aeroc)
        print("\n".join(_render_aeroc_topology(network)))
        return 0

    blueprint_path = args.blueprint or "blueprint.aero"
    if not os.path.isfile(blueprint_path):
        print(f"error: blueprint not found: {blueprint_path}", file=sys.stderr)
        return 1

    with open(blueprint_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    import blueprint_lang

    if blueprint_lang.looks_like_blueprint_dsl(content):
        from build_graph import blueprint_to_dag

        try:
            blueprint = blueprint_lang.load_source(content, blueprint_path)
            graph = blueprint_to_dag(blueprint)
        except Exception as exc:  # noqa: BLE001 - surface validation errors cleanly
            print(f"error: invalid blueprint: {exc}", file=sys.stderr)
            return 1
        print(graph.render_tree())
        return 0

    # Legacy INI/JSON/TOML blueprints: lower through the stable parser and show
    # a flat plan derived from the resolved compilation targets.
    from blueprint_parser import parse_blueprint

    context = parse_blueprint(blueprint_path)
    targets = context.get("compilation_targets") or []
    dependencies = context.get("dependency_matrix") or {}

    print("Build Plan (legacy INI/JSON)")
    print("")
    if not targets:
        print("  (no targets declared)")
    for index, name in enumerate(targets):
        connector = "└──" if index == len(targets) - 1 else "├──"
        deps = dependencies.get(name) or []
        suffix = f"  requires: {', '.join(deps)}" if deps else ""
        print(f"{connector} {name}{suffix}")
    print("")
    print(f"{len(targets)} targets")
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build_command(args: argparse.Namespace) -> int:
    """Run a build: the isolated scaffold pipeline, the core engine, or a cNrGA evolution pass."""
    # Aero-Calculus native target: compile a source script to a .aeroc graph.
    if getattr(args, "source", None):
        return aero_build_command(args)

    workspace = args.workspace or "."
    blueprint_path = args.blueprint or os.path.join(workspace, "blueprint.aero")
    self_host_path = os.path.join(workspace, "self_host.aero")
    workspace_blueprint = os.path.join(workspace, "blueprint.aero")

    from blueprint_parser import parse_blueprint
    from src.scaffold.pipeline import ScaffoldBuildPipeline, should_run_scaffold_pipeline

    if not os.path.isfile(blueprint_path):
        print(f"error: blueprint not found: {blueprint_path}", file=sys.stderr)
        return 1

    # If the requested blueprint lives outside the workspace, mirror it so the
    # orchestrator's hardcoded workspace/blueprint.aero path sees the right file.
    if os.path.abspath(blueprint_path) != os.path.abspath(workspace_blueprint):
        try:
            shutil.copy(blueprint_path, workspace_blueprint)
        except OSError as exc:
            print(f"error: cannot copy blueprint to workspace: {exc}", file=sys.stderr)
            return 1

    # --cycles > 0 triggers the cNrGA evolution loop on self_host.aero before the
    # final build, removing the need for a separate ``python evolve.py`` step.
    config = _load_blueprint_config(args.config)
    evolve_generations = args.cycles if args.cycles is not None and args.cycles > 0 else 0
    if evolve_generations > 0:
        import evolve

        if not os.path.isfile(self_host_path):
            shutil.copy(blueprint_path, self_host_path)
        evolve.execute_evolution_loop(workspace, evolve_generations)
        # Promote the evolved blueprint to the active build blueprint.
        blueprint_path = self_host_path
        shutil.copy(self_host_path, workspace_blueprint)
        build_cycles = int(config.get("cycles", 3))
    else:
        build_cycles = args.cycles if args.cycles is not None else int(config.get("cycles", 3))

    context = parse_blueprint(blueprint_path)
    try:
        VerifyDependencies(context).verify()
    except ContractViolationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    system = context.get("system", {}) if isinstance(context.get("system"), dict) else {}
    strategy = str(system.get("strategy", "")).strip().upper()

    if strategy == "DIRECT_COMPILE":
        try:
            summary = orchestrator.run_direct_compile(workspace, build_context=context)
        except Exception as exc:  # noqa: BLE001 - never leak raw tracebacks to the user
            print(f"error: direct compile failed: {exc}", file=sys.stderr)
            return 1
        compiled_target_count = int(summary.get("compiled_target_count", 0))
        bytes_written = int(summary.get("bytes_written", 0))
        if compiled_target_count <= 0 or bytes_written <= 0:
            print("error: direct compile produced no .aeroc output", file=sys.stderr)
            return 1
        print("Direct compile complete")
        print(f"  {'compiled':<17}: {compiled_target_count}")
        print(f"  {'bytes':<17}: {bytes_written}")
        print(f"  {'aeroc':<17}: {summary.get('aeroc_output')}")
        return 0

    if should_run_scaffold_pipeline(context):
        blueprint_dir = Path(os.path.dirname(os.path.abspath(blueprint_path)))
        pipeline = ScaffoldBuildPipeline(logger=print, verbose=True)
        result = pipeline.run(
            context,
            blueprint_dir=blueprint_dir,
            build=not args.no_scaffold_build,
        )
        if result.succeeded:
            print("Isolated scaffold build complete")
            print(f"  {'language':<17}: {result.language}")
            print(f"  {'workspace':<17}: {result.scaffold.workspace}")
            return 0
        print("Isolated scaffold build failed", file=sys.stderr)
        return 1

    # Core self-evolving build cycle.
    try:
        summary = orchestrator.run_build(workspace, cycles=build_cycles)
    except Exception as exc:  # noqa: BLE001 - never leak raw tracebacks to the user
        print(f"error: build failed: {exc}", file=sys.stderr)
        return 1

    if isinstance(summary, dict) and summary.get("short_circuited"):
        print(f"Build short-circuited: {summary.get('reason')}")
        return 0
    print("Build complete")
    return 0


# ---------------------------------------------------------------------------
# heal
# ---------------------------------------------------------------------------


def _python_build_fn(path: Path):
    """A ``build_fn`` for the self-healing loop backed by ``compile()``.

    Returns an empty list when the file parses cleanly, otherwise a single
    structured :class:`Diagnostic` describing the syntax error.
    """
    from core.toolchain.self_healing import Diagnostic

    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        return []
    except SyntaxError as exc:
        return [
            Diagnostic(
                message=exc.msg or "syntax error",
                file=str(path),
                severity="error",
                line=exc.lineno or 1,
                column=(exc.offset or 1),
                source="python-compile",
            )
        ]


def heal_command(args: argparse.Namespace) -> int:
    """Run self-healing: topological re-wiring on a graph, or source repair."""
    # Aero-Calculus mode: geometrically re-wire un-terminated HIN edges.
    if getattr(args, "aeroc", None):
        return aero_heal_command(args)

    if not getattr(args, "path", None):
        print("error: heal requires --path or --aeroc", file=sys.stderr)
        return 1
    target = Path(args.path)
    if not target.is_file():
        print(f"error: file not found: {target}", file=sys.stderr)
        return 1

    from core.toolchain.self_healing import detect_language, heal_module

    language = detect_language(target)
    if language != "python":
        # Non-Python languages need a real compiler-backed build_fn; without a
        # toolchain invocation here we can only verify and report.
        print(f"[Heal] {target}  language={language or 'unknown'} (verify-only)")
        return 0

    report = heal_module(target, _python_build_fn, language="python")
    if report.success:
        if report.applied:
            print(f"[Heal] {target}: repaired via {', '.join(report.applied)}")
        else:
            print(f"[Heal] {target}: already clean")
        return 0

    print(f"[Heal] {target}: unresolved after {report.attempts} attempt(s)", file=sys.stderr)
    for diag in report.final_diagnostics:
        print(f"  {diag.file}:{diag.line}:{diag.column}: {diag.message}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# scaffold
# ---------------------------------------------------------------------------


def scaffold_command(args: argparse.Namespace) -> int:
    """Generate a standalone, out-of-tree repository from a single source entry."""
    from src.scaffold import ScaffoldEngine, SourceEntryNotFound
    from src.scaffold.source_resolver import infer_language

    if not args.no_build:
        # Infer the target language from the source path so we can check the
        # relevant toolchain before any scaffolding or compilation begins.
        src_path = Path(args.source_entry)
        language = infer_language(src_path)
        if language == "unknown":
            language = "rust"
        try:
            VerifyDependencies.verify_language(language)
        except ContractViolationError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    dist = Path(args.distribution_directory) if args.distribution_directory else None
    engine = ScaffoldEngine(logger=print, verbose=True)
    try:
        result = engine.scaffold(
            source_entry=args.source_entry,
            name=args.name,
            distribution_directory=dist,
            build=not args.no_build,
        )
    except SourceEntryNotFound as exc:
        print(f"error: source entry not found: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: not found: {exc}", file=sys.stderr)
        return 1

    print(f"Standalone repository generated at {result.workspace}")
    if result.build is not None and not result.build.get("succeeded", True):
        print("  (note: post-generation build step did not succeed)", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# infer  (zero-config Invisible Configuration Layer)
# ---------------------------------------------------------------------------


def infer_command(args: argparse.Namespace) -> int:
    """Infer a full build DAG from a lean blueprint + the project file tree."""
    workspace = Path(args.workspace or ".")
    blueprint_path = workspace / "blueprint.aero"
    if not blueprint_path.is_file():
        print(f"error: blueprint not found: {blueprint_path}", file=sys.stderr)
        return 1

    content = blueprint_path.read_text(encoding="utf-8")

    from src.invisible_config import InvisibleConfigEngine
    from src.invisible_config.lean_parser import looks_like_lean_blueprint

    if not looks_like_lean_blueprint(content):
        print(
            f"error: {blueprint_path} is not an ultra-lean blueprint; "
            "'infer' requires the zero-config dialect.",
            file=sys.stderr,
        )
        return 1

    dag = InvisibleConfigEngine(workspace).infer_from_source(content)
    payload = dag.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("zero-config build inference (Invisible Configuration Layer)")
    print(f"  project: {payload['project']}")
    print(f"  optimize: {payload['optimize']}")
    print(f"  execution order: {' -> '.join(payload['execution_order']) or '(none)'}")
    print("  targets:")
    for target in payload["targets"]:
        reason = target.get("language_reason")
        suffix = f"  ({reason})" if reason else ""
        print(f"    - {target['name']} [{target['language']}]{suffix}")
    if payload["ffi_boundaries"]:
        print("  ffi boundaries:")
        for boundary in payload["ffi_boundaries"]:
            print(
                f"    - {boundary.get('provider_language')} -> "
                f"{boundary.get('consumer_language')} via {boundary['mechanism']}"
            )
    return 0


# ---------------------------------------------------------------------------
# decompose  (complexity-driven module splitting + DAG write-back)
# ---------------------------------------------------------------------------


def decompose_command(args: argparse.Namespace) -> int:
    """Analyse the workspace, build the dependency DAG, and persist it."""
    workspace = Path(args.workspace or ".")
    if not workspace.is_dir():
        print(f"error: workspace not found: {workspace}", file=sys.stderr)
        return 1

    from core.analysis import InferenceEngine

    engine = InferenceEngine(workspace)
    sources = [
        p for p in workspace.rglob("*.py")
        if ".aero" not in p.parts and "__pycache__" not in p.parts
    ]
    if not sources:
        print("No Python sources found to analyse.")
        return 0

    analyses = engine.analyze_paths([str(p) for p in sorted(sources)])
    dag = engine.build_dag(analyses)

    blueprint_path = workspace / "blueprint.aero"
    engine.write_dag_to_blueprint(blueprint_path, dag)
    print(f"Decomposition DAG written to {blueprint_path} ({len(dag)} nodes)")
    return 0


# ---------------------------------------------------------------------------
# invariants  (Semantic Fluidity context ingestion)
# ---------------------------------------------------------------------------


def invariants_command(args: argparse.Namespace) -> int:
    """Ingest unstructured context into a typed invariant schema."""
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"error: context directory not found: {source_dir}", file=sys.stderr)
        return 1

    from src.semantic_fluidity.engine import ContextIngestionEngine

    engine = ContextIngestionEngine()
    output = Path(args.output) if args.output else None
    report = engine.ingest_and_export(source_dir, output)

    print("Semantic Fluidity Ingestion:")
    variables = report.get("variables") if isinstance(report, dict) else None
    if variables is not None:
        print(f"  extracted {len(variables)} invariant variable(s)")
    print(f"  schema report written under {source_dir}")
    return 0


# ---------------------------------------------------------------------------
# polymorphize  (autonomous hardware polymerization)
# ---------------------------------------------------------------------------


def _format_topology(topology) -> List[str]:
    return [
        f"  arch: {topology.arch or 'unknown'}",
        f"  cores: {topology.physical_cores} physical / {topology.logical_cores} logical",
        f"  best SIMD: {topology.best_simd()}  (vector width {topology.vector_width_bytes()}B)",
        f"  cache line: {topology.cache_line_bytes()}B",
        f"  memory bandwidth class: {topology.memory_bandwidth_class}",
    ]


def polymorphize_command(args: argparse.Namespace) -> int:
    """Probe the host and rewrite generated source for its exact topology."""
    from src.polymorphization import PolymorphizationEngine

    engine = PolymorphizationEngine()

    if args.profile_only:
        topology = engine.profile_host()
        print("Hardware Topology:")
        for line in _format_topology(topology):
            print(line)
        return 0

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"error: source directory not found: {source_dir}", file=sys.stderr)
        return 1

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    report = engine.polymerize_tree(source_dir, cache_dir)

    print("Autonomous Hardware-Polymerization:")
    for line in _format_topology(engine.last_topology):
        print(line)
    rewrite = report.get("rewrite", {})
    rewritten = rewrite.get("rewritten_files", rewrite.get("files", []))
    print(f"  rewrote {len(rewritten) if hasattr(rewritten, '__len__') else rewritten} file(s) into the polymorph cache")
    return 0


# ---------------------------------------------------------------------------
# ingest  (AST registry ingestion)
# ---------------------------------------------------------------------------


def ingest_command(args: argparse.Namespace) -> int:
    """Ingest source into the AST registry and register it in the blueprint."""
    workspace = Path(args.workspace or ".")
    blueprint_path = workspace / "blueprint.aero"

    if args.list:
        from src.blueprint import load_blueprint

        print("Ingested contexts:")
        if blueprint_path.is_file():
            blueprint = load_blueprint(blueprint_path)
            registry = getattr(blueprint, "context_registry", {}) or {}
            for name in sorted(registry):
                print(f"  - {name}")
        return 0

    if not args.path:
        print("error: ingest requires --path or --list", file=sys.stderr)
        return 1

    target = Path(args.path)
    if not target.exists():
        print(f"error: path not found: {target}", file=sys.stderr)
        return 1

    from src.registry import ingest_context
    from src.registry.ingest import IngestError

    base = target if target.is_dir() else target.parent
    context_name = base.resolve().name or "context"
    db_path = workspace / ".aero" / "registry.db"
    try:
        result = ingest_context(
            context_name,
            target,
            db_path=db_path,
            blueprint_path=blueprint_path,
        )
    except IngestError as exc:
        print(f"error: ingest failed: {exc}", file=sys.stderr)
        return 1

    ingested = getattr(result, "ingested", None)
    count = len(ingested) if ingested is not None else "?"
    print(f"Ingested context '{context_name}' ({count} file(s)) into the registry.")
    return 0


# ---------------------------------------------------------------------------
# commit-overlay  (preserve manual edits across regeneration)
# ---------------------------------------------------------------------------


def commit_overlay_command(args: argparse.Namespace) -> int:
    """Capture manual edits to a generated file as a reusable overlay patch."""
    from src.overlay import OverlayManager
    from src.overlay.manager import OverlayError

    manager = OverlayManager(args.workspace or ".")
    try:
        patch = manager.commit_overlay(args.file)
    except OverlayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if patch is None:
        print("No edits to commit (file matches its pristine baseline).")
        return 0
    print(f"Committed overlay for {args.file}")
    return 0


# ---------------------------------------------------------------------------
# Aero-Calculus pipeline (HIN VM / translator / spacetime ledger)
# ---------------------------------------------------------------------------


def handle_aero_calculus_build(
    source_path: str, output_path: str, reduce_graph: bool = True
) -> dict:
    """Thin wrapper around :func:`orchestrator.handle_aero_calculus_build`.

    Keeps the public entry point in ``main`` while the implementation (and the
    retry-with-healing hook) lives in ``orchestrator``.
    """
    return orchestrator.handle_aero_calculus_build(
        source_path, output_path, reduce_graph=reduce_graph
    )


def _render_aeroc_topology(network) -> List[str]:
    """Render the physical HIN port-connection topology of a network."""
    lines: List[str] = []
    lines.append(f"Aero-Calculus HIN topology ({len(network.nodes)} nodes)")
    lines.append("")
    if network.active_pairs:
        lines.append("Active pairs (p ⋈ p):")
        for a, b in network.active_pairs:
            lines.append(f"  {a.node_id} ⋈ {b.node_id}")
        lines.append("")
    lines.append("Physical port connections:")
    seen = set()
    for node in network.nodes.values():
        for port in node.ports():
            target = port.target
            if target is None:
                continue
            key = frozenset((id(port), id(target)))
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"  {node.node_id}.{port.name} ── {target.owner.node_id}.{target.name}"
            )
    return lines


def aero_build_command(args: argparse.Namespace) -> int:
    """``build --source`` entry point: compile a script to ``.aeroc``."""
    source = args.source
    if not os.path.isfile(source):
        print(f"error: source not found: {source}", file=sys.stderr)
        return 1
    try:
        # The Aero-Calculus pipeline is driven by Python code that relies on
        # the core runtime packages; verify those before doing any work.
        VerifyDependencies({"context_registry": {"source": {"language": "python"}}}).verify()
    except ContractViolationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    output = args.aeroc_out or (os.path.splitext(source)[0] + ".aeroc")
    try:
        report = handle_aero_calculus_build(
            source, output, reduce_graph=not args.no_reduce
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: aero-calculus build failed: {exc}", file=sys.stderr)
        return 1
    print("")
    print(f"  compiled nodes : {report['compiled_nodes']}")
    print(f"  reduced nodes  : {report['reduced_nodes']}")
    print(f"  reduction steps: {report['reduction_steps']}")
    print(f"  rigidity       : {report['rigidity']}")
    return 0


def init_command(args: argparse.Namespace) -> int:
    """Manually set up a project architecture (workspace + living blueprint)."""
    from core.environment_bootstrap import RuntimeEnvironmentBootstrapper

    try:
        report = RuntimeEnvironmentBootstrapper.init_workspace(args.workspace or ".")
    except Exception as exc:  # noqa: BLE001
        print(f"error: workspace init failed: {exc}", file=sys.stderr)
        return 1
    print(f"  root             : {report['root']}")
    print(f"  blueprint        : {'seeded' if report['blueprint_created'] else 'preserved'}")
    print(f"  context.aero     : {'seeded' if report['ledger_created'] else 'present/skip'}")
    return 0


def audit_command(args: argparse.Namespace) -> int:
    """Run the pre-flight test integrity sweep and self-heal core logic bugs."""
    from core.test_auditor import PreFlightTestAuditor

    auditor = PreFlightTestAuditor(test_dir=args.test_dir, max_rounds=args.max_rounds)
    ok = auditor.run_suite_and_heal()
    if auditor.patched_files:
        print(f"[+] Self-healing patched {len(auditor.patched_files)} file(s).")
    return 0 if ok else 1


def aero_heal_command(args: argparse.Namespace) -> int:
    """Geometrically re-wire un-terminated edges in a compiled ``.aeroc``."""
    if not os.path.isfile(args.aeroc):
        print(f"error: .aeroc not found: {args.aeroc}", file=sys.stderr)
        return 1
    from core.aeroc import load_aeroc, save_aeroc
    from orchestrator import TopologicalSelfHealer

    network = load_aeroc(args.aeroc)
    healer = TopologicalSelfHealer()
    broken = healer.find_unterminated_ports(network)
    print(f"[Heal] Topological self-healing: {len(broken)} un-terminated edge(s)")
    try:
        report = healer.heal_network(network)
    except Exception as exc:  # noqa: BLE001
        print(f"error: self-healing failed: {exc}", file=sys.stderr)
        return 1
    network.validate_conservation()
    out = args.output or args.aeroc
    save_aeroc(network, out)
    print(f"[Heal] re-wired {report['healed']} edge(s); remaining={report['remaining']}")
    return 0


def evolve_command(args: argparse.Namespace) -> int:
    """Type-safe graph-rewriting evolution over a compiled ``.aeroc`` file."""
    import evolve as evolve_engine

    if not os.path.isfile(args.aeroc):
        print(f"error: .aeroc not found: {args.aeroc}", file=sys.stderr)
        return 1
    try:
        report = evolve_engine.evolve_aeroc(
            args.aeroc,
            generations=args.generations,
            output_path=args.output,
            mutation_rate=args.mutation_rate,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: evolution failed: {exc}", file=sys.stderr)
        return 1
    print("Aero-Calculus evolution complete")
    print(f"  generations    : {report['generations']}")
    print(f"  start nodes     : {report['start_nodes']}")
    print(f"  final nodes     : {report['final_nodes']}")
    print(f"  saved           : {report['output']}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aero",
        description="Aero Future -- the infinite build orchestration engine.",
    )
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build", help="Build the workspace from a blueprint.")
    p_build.add_argument("--workspace", default=None, help="Workspace root (default: .).")
    p_build.add_argument("--blueprint", default=None, help="Path to the blueprint file.")
    p_build.add_argument("--config", default=None, help="Optional JSON config overlay.")
    p_build.add_argument("--cycles", type=int, default=None, help="Evolution cycles to run.")
    p_build.add_argument(
        "--no-scaffold-build",
        action="store_true",
        help="Generate the scaffold repo but skip the cargo/python build step.",
    )
    p_build.add_argument(
        "--no-polymorph",
        action="store_true",
        help="Skip the hardware-polymorphization rewrite stage.",
    )
    p_build.add_argument(
        "--source",
        default=None,
        help="Compile a source script to an Aero-Calculus .aeroc graph and run it.",
    )
    p_build.add_argument(
        "--aeroc-out",
        dest="aeroc_out",
        default=None,
        help="Output path for the compiled .aeroc (default: <source>.aeroc).",
    )
    p_build.add_argument(
        "--no-reduce",
        action="store_true",
        help="Skip HIN-VM graph reduction; serialize the un-reduced topology.",
    )
    p_build.set_defaults(handler=build_command)

    p_plan = sub.add_parser("plan", help="Render the build DAG without executing it.")
    p_plan.add_argument("--blueprint", default=None, help="Path to the blueprint file.")
    p_plan.add_argument(
        "--aeroc",
        default=None,
        help="Render the physical HIN port topology of a compiled .aeroc file.",
    )
    p_plan.set_defaults(handler=plan_command)

    p_evolve = sub.add_parser(
        "evolve", help="Type-safe graph-rewriting evolution over a .aeroc graph."
    )
    p_evolve.add_argument("--aeroc", required=True, help="Compiled .aeroc graph to evolve.")
    p_evolve.add_argument(
        "--generations", type=int, default=8, help="Evolution generations to run."
    )
    p_evolve.add_argument(
        "--mutation-rate",
        dest="mutation_rate",
        type=float,
        default=0.1,
        help="SHX type-safe mutation rate per node (default: 0.1).",
    )
    p_evolve.add_argument(
        "--output", default=None, help="Output path (default: overwrite input)."
    )
    p_evolve.set_defaults(handler=evolve_command)

    p_heal = sub.add_parser("heal", help="Run self-healing on a source file or .aeroc graph.")
    p_heal.add_argument("--path", default=None, help="Source file to heal.")
    p_heal.add_argument(
        "--aeroc",
        default=None,
        help="Topologically re-wire un-terminated edges in a compiled .aeroc.",
    )
    p_heal.add_argument(
        "--output", default=None, help="Output path for the healed .aeroc (default: in place)."
    )
    p_heal.set_defaults(handler=heal_command)

    p_scaffold = sub.add_parser(
        "scaffold", help="Generate a standalone repository from one source entry."
    )
    p_scaffold.add_argument("--source-entry", required=True, help="Source file to scaffold from.")
    p_scaffold.add_argument("--name", default=None, help="Generated project name.")
    p_scaffold.add_argument(
        "--distribution-directory", default=None, help="Output directory for the repo."
    )
    p_scaffold.add_argument(
        "--no-build", action="store_true", help="Skip the build step after generation."
    )
    p_scaffold.set_defaults(handler=scaffold_command)

    p_infer = sub.add_parser("infer", help="Infer a build DAG from a lean blueprint.")
    p_infer.add_argument("--workspace", default=".", help="Project root (default: .).")
    p_infer.add_argument("--json", action="store_true", help="Emit the inferred DAG as JSON.")
    p_infer.set_defaults(handler=infer_command)

    p_decompose = sub.add_parser(
        "decompose", help="Analyse the workspace and write its dependency DAG."
    )
    p_decompose.add_argument("--workspace", default=".", help="Project root (default: .).")
    p_decompose.set_defaults(handler=decompose_command)

    p_invariants = sub.add_parser(
        "invariants", help="Ingest unstructured context into a typed invariant schema."
    )
    p_invariants.add_argument("--source-dir", required=True, help="Directory of context files.")
    p_invariants.add_argument("--workspace", default=".", help="Project root (default: .).")
    p_invariants.add_argument("--output", default=None, help="Schema report output path.")
    p_invariants.set_defaults(handler=invariants_command)

    p_poly = sub.add_parser(
        "polymorphize", help="Rewrite generated source for the host hardware topology."
    )
    p_poly.add_argument("--source-dir", default="build_artifacts", help="Generated source dir.")
    p_poly.add_argument("--cache-dir", default=None, help="Ephemeral polymorph cache dir.")
    p_poly.add_argument(
        "--profile-only", action="store_true", help="Only print the host hardware topology."
    )
    p_poly.set_defaults(handler=polymorphize_command)

    p_ingest = sub.add_parser("ingest", help="Ingest source into the AST registry.")
    p_ingest.add_argument("--workspace", default=".", help="Project root (default: .).")
    p_ingest.add_argument("--path", default=None, help="File or directory to ingest.")
    p_ingest.add_argument("--list", action="store_true", help="List ingested contexts.")
    p_ingest.set_defaults(handler=ingest_command)

    p_overlay = sub.add_parser(
        "commit-overlay", help="Capture manual edits to a generated file as an overlay."
    )
    p_overlay.add_argument("file", help="The generated file whose edits to preserve.")
    p_overlay.add_argument("--workspace", default=".", help="Project root (default: .).")
    p_overlay.set_defaults(handler=commit_overlay_command)

    p_init = sub.add_parser(
        "init", help="Initialize a project architecture (workspace + living blueprint)."
    )
    p_init.add_argument("--workspace", default=".", help="Project root (default: .).")
    p_init.set_defaults(handler=init_command)

    p_audit = sub.add_parser(
        "audit", help="Run the pre-flight test sweep and self-heal core logic bugs."
    )
    p_audit.add_argument("--test-dir", dest="test_dir", default="tests", help="Test directory.")
    p_audit.add_argument(
        "--max-rounds", dest="max_rounds", type=int, default=3,
        help="Maximum self-healing patch/re-run rounds.",
    )
    p_audit.set_defaults(handler=audit_command)

    return parser


_BOOTSTRAP_DONE = False


def _maybe_bootstrap() -> None:
    """No-op pre-flight guard.

    The legacy auto-bootstrapper has been replaced by the Environment Contract.
    Dependency and toolchain checks are performed by the active command handler
    against the parsed blueprint, so no automatic installation or workspace
    seeding happens here.  The guard still short-circuits under test runners.
    """
    global _BOOTSTRAP_DONE
    if _BOOTSTRAP_DONE:
        return
    if os.environ.get("AERO_DISABLE_BOOTSTRAP") or os.environ.get("AERO_AUDIT_ACTIVE"):
        return
    if "unittest" in sys.modules or "pytest" in sys.modules:
        return
    _BOOTSTRAP_DONE = True


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Force environment + workspace stability before any command runs.
    _maybe_bootstrap()

    args = create_parser().parse_args(list(argv) if argv is not None else None)
    handler = getattr(args, "handler", None)
    if handler is None:
        create_parser().print_help(sys.stderr)
        return 1
    return int(handler(args))


def cli_entry(argv: Optional[Sequence[str]] = None) -> int:
    """Hardened process entry point: optional pre-flight audit, dispatch.

    All errors are trapped into elegant, actionable guidance instead of bare
    Python stack traces.  Environment Contract violations are reported first so
    the operator knows exactly which dependency is missing.
    """
    try:
        # Opt-in full pre-flight self-healing audit (AERO_PREFLIGHT=1).  Kept
        # opt-in so routine commands stay zero-friction; the `audit` subcommand
        # runs it explicitly.
        if os.environ.get("AERO_PREFLIGHT", "").strip().lower() in ("1", "true", "yes"):
            from core.test_auditor import PreFlightTestAuditor

            if not PreFlightTestAuditor().run_suite_and_heal():
                print(
                    "[-] Critical Stop: pre-flight self-healing could not fix "
                    "core architecture errors. Re-run with AERO_PREFLIGHT=0 to "
                    "bypass, or inspect the reported source files.",
                    file=sys.stderr,
                )
                return 1

        return main(argv)
    except ContractViolationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[-] Interrupted by user.", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - elegant user-facing guidance
        print("==============================================================================", file=sys.stderr)
        print(" AERO FUTURE — UNEXPECTED ERROR", file=sys.stderr)
        print("==============================================================================", file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  Try:  python main.py init        (re-seed the workspace)", file=sys.stderr)
        print("  Or:   AERO_PREFLIGHT=1 python main.py audit   (self-heal core bugs)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli_entry())
