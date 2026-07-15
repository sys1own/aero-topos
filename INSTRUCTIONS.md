# Aero Topos — Universal Reference Instructions

This document is the canonical operator guide for the `aero-topos` engine. It covers the system's architecture, every CLI command, the complete `blueprint.aero` schema, the standard autonomous workflow, and advanced troubleshooting for the self-healing HIN-VM pipeline.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [CLI Reference](#2-cli-reference)
3. [Blueprint Schema](#3-blueprint-schema-the-universal-reference)
4. [Pipeline Orchestration — The Golden Path](#4-pipeline-orchestration--the-golden-path)
5. [Advanced Troubleshooting](#5-advanced-troubleshooting)
6. [Developer / Researcher Workflow](#6-developer--researcher-workflow)
7. [See Also](#7-see-also)

---

## 1. Architecture Overview

Aero Topos is a self-hosting, topology-first compilation and optimization substrate. It treats program logic as a geometric object rather than a sequence of text instructions. The core workflow is:

```
source code  →  UAST  →  HIN graph  →  boundary verification  →  reduction  →  .aeroc
```

### 1.1 Core concepts

| Concept | What it is | Where it lives |
|---|---|---|
| **Aero-Calculus** | The intermediate language/representation of a program as a *Hierarchical Interaction Net* (HIN). | `core/hin_vm.py`, `core/translator.py` |
| **UAST** | Universal Abstract Syntax Tree. A normalized, language-agnostic linear AST produced from Python (and eventually other languages). | `core/aero_frontend.py` |
| **HIN / HIN-VM** | Hierarchical Interaction Net Virtual Machine. Programs are graphs of typed nodes connected by ports; reduction happens on *active pairs* (two principal ports wired together). | `core/hin_vm.py` |
| **MELL** | Multiplicative-Exponential Linear Logic type system used on HIN ports (`I`, `Tensor`, `Implication`, `Bang`). | `core/hin_vm.py` (`MELLType`) |
| **Spacetime Ledger** | Append-only block-universe ledger (`context.aero`) that records every graph mutation and compile metric with a causal `T_causal` index. | `core/spacetime_ledger.py` |
| **Rigidity Verifier** | Runs coordinate-perturbation sweeps over module boundaries to prove algebraic invariants survive geometric transport. | `core/spacetime_ledger.py` (`RigidityVerifier`) |
| **Topological Self-Healer** | Geometrically rewires unterminated HIN edges using `Switch`, `Value` and `Eraser` nodes, then re-verifies boundary rigidity. | `orchestrator.py` (`TopologicalSelfHealer`) |
| **ScaffoldEngine** | Generates out-of-tree Rust or Python repositories from a source entry and optionally builds/validates them. | `src/scaffold/engine.py` |
| **cNrGA loop** | Continuous Non-dominated Ranking Genetic Algorithm that evolves blueprint parameters and/or source files. | `evolve.py` |

### 1.2 Execution philosophy

1. **Homomorphic translation.** Source is lowered to UAST, then UAST is translated to a HIN graph where *every variable scope is resolved to a direct port-to-port wire*.
2. **Module mitosis.** If a graph exceeds `scaling.auto_split_threshold`, the translator bisects it along the Fiedler vector of the graph Laplacian and reifies crossing edges as `BoundaryPortNode` pairs.
3. **Boundary sweeps + reduction.** `RigidityVerifier` perturbs boundary coordinates and checks eigenvalue persistence; `UniversalHINNetwork` then reduces the graph by firing active-pair rewrite rules.
4. **Serialization.** The reduced graph is written as JSON to `.aeroc`; secondary partitions become `.part2.aeroc`.
5. **Self-healing.** If compilation hits a 0-node result, an invariant error, or a rigidity anomaly, `TopologicalSelfHealer` is invoked automatically and the build is retried.

---

## 2. CLI Reference

The primary entry point is `main.py`. Pre-flight bootstrapping is automatic unless `AERO_DISABLE_BOOTSTRAP=1` is set.

```bash
python main.py <command> [options]
```

### 2.1 `build`

Compile, optimize, and/or scaffold a workspace.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--workspace` | string | `.` | Project root. |
| `--blueprint` | string | `workspace/blueprint.aero` | Path to the blueprint. |
| `--config` | string | `tests/fixtures/blueprint_config.json` | Optional JSON config overlay. |
| `--cycles` | int | `None` | If `> 0`, runs the cNrGA evolution loop for this many generations before the final build. If `<= 0` or omitted, used as build-cycle count only. |
| `--no-scaffold-build` | bool | `False` | Generate scaffold repo but skip the `cargo`/`compileall` build step. |
| `--no-polymorph` | bool | `False` | Skip the hardware-polymorphization rewrite stage. |
| `--source` | string | `None` | Compile a source script directly to an `.aeroc` graph. |
| `--aeroc-out` | string | `<source>.aeroc` | Output path when using `--source`. |
| `--no-reduce` | bool | `False` | Skip HIN-VM graph reduction. |

**Examples**

```bash
# Standard workspace build
python main.py build --workspace . --blueprint blueprint.aero

# Direct source compile
python main.py build --source src/app_logic.py --aeroc-out app.aeroc --no-reduce

# Build with autonomous evolution (3 generations)
python main.py build --workspace . --blueprint blueprint.aero --cycles 3
```

### 2.2 `plan`

Render the build DAG without executing it.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--blueprint` | string | `blueprint.aero` | Blueprint to plan. |
| `--aeroc` | string | `None` | Render the physical HIN port topology of a compiled `.aeroc`. |

### 2.3 `evolve`

Type-safe graph-rewriting evolution over a compiled `.aeroc` file.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--aeroc` | string | **required** | Compiled `.aeroc` graph to evolve. |
| `--generations` | int | `8` | Maximum generations to run. |
| `--mutation-rate` | float | `0.1` | SHX type-safe mutation rate per node. |
| `--output` | string | overwrite input | Output path for the evolved graph. |

### 2.4 `heal`

Repair source code or HIN graphs.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--path` | string | `None` | Python source file to syntax-repair. |
| `--aeroc` | string | `None` | Compiled `.aeroc` to topologically re-wire. |
| `--output` | string | in place | Output path for the healed `.aeroc`. |

**Examples**

```bash
# Re-wire unterminated edges in an .aeroc
python main.py heal --aeroc graph.aeroc --output graph_healed.aeroc

# Fix a Python syntax error via self-healing
python main.py heal --path src/app_logic.py
```

### 2.5 `scaffold`

Generate a standalone, out-of-tree repository from a single source entry.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--source-entry` | string | **required** | Source file to scaffold from. |
| `--name` | string | stem | Generated project name. |
| `--distribution-directory` | string | `None` | Output directory (temp dir if omitted). |
| `--no-build` | bool | `False` | Skip post-generation build/validation. |

### 2.6 `infer`

Infer a full build DAG from a *lean* blueprint.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--workspace` | string | `.` | Project root. |
| `--json` | bool | `False` | Emit the inferred DAG as JSON. |

### 2.7 `decompose`

Analyze Python sources, build a dependency DAG, and write it back to `blueprint.aero`.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--workspace` | string | `.` | Project root. |

### 2.8 `invariants`

Ingest unstructured context files into a typed invariant schema.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--source-dir` | string | **required** | Directory of context files. |
| `--workspace` | string | `.` | Project root. |
| `--output` | string | `None` | Optional schema report output path. |

### 2.9 `polymorphize`

Probe host hardware and rewrite generated source for its exact topology.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--source-dir` | string | `build_artifacts` | Generated source directory. |
| `--cache-dir` | string | `None` | Ephemeral polymorph cache directory. |
| `--profile-only` | bool | `False` | Only print hardware topology. |

### 2.10 `ingest`

Register source files/directories in the AST registry and blueprint.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--workspace` | string | `.` | Project root. |
| `--path` | string | `None` | File or directory to ingest. |
| `--list` | bool | `False` | List already ingested contexts. |

### 2.11 `commit-overlay`

Capture manual edits to a generated file as a reusable overlay patch.

| Argument | Type | Meaning |
|---|---|---|
| `file` | string | The generated file whose edits to preserve. |

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--workspace` | string | `.` | Project root. |

### 2.12 `init`

Initialize a fresh workspace with a default living `blueprint.aero` and `context.aero`.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--workspace` | string | `.` | Project root. |

### 2.13 `audit`

Run the pre-flight test sweep and self-heal core logic bugs.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--test-dir` | string | `tests` | Test directory. |
| `--max-rounds` | int | `3` | Maximum self-healing patch/re-run rounds. |

---

## 3. Blueprint Schema — The Universal Reference

A `blueprint.aero` is a declarative document that controls compilation, evolution, scaffold generation, hardware targeting, and distributed execution. The parser accepts several dialects:

| Dialect | Trigger | Notes |
|---|---|---|
| **TOML native (living blueprint)** | first non-comment line is `[system]` or `[context_registry.*]` | Recommended for new projects. |
| **Legacy INI** | first section is `[graph]`, `[compiler]`, or `[cortex]` | Backward-compatible. |
| **Ultra-lean** | detected by `src/invisible_config/lean_parser.py` | Entirely inferred from the file tree. |
| **JSON** | first non-whitespace character is `{` | Strict schema with required sections. |
| **Block DSL** | detected by `blueprint_lang` | Internal DSL format. |

### 3.1 Required sections

Only `[system]` is strictly required for a TOML native blueprint. Legacy INI blueprints may satisfy the parser with the "old trio" (`[graph]`, `[compiler]`, `[cortex]`) or with `[system]`.

### 3.2 `[system]` — project identity and strategy

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | `""` | Project name. |
| `version` | string | `""` | Project version. |
| `strategy` | string | `"microkernel"` / `"DIRECT_COMPILE"` (legacy) | Build strategy. Use `"DIRECT_COMPILE"` to route source straight to the Aero-Calculus compiler. |
| `ephemeral_code` | bool | `False` | Whether generated code is allowed to be transient/overwritten. |

### 3.3 `[context_registry.<name>]` — register a source context

Each sub-table registers a named module that the engine can compile, link, or scaffold.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `path` | string | `""` | Relative or absolute source path. |
| `language` | string | `""` | Source language (`python`, `rust`, `c`, `cpp`, `fortran`, `shell`, ...). |
| `dialect` | string | `""` | Optional dialect marker. |
| `preserve_original_logic` | bool | `False` | Whether to keep the original source unmodified. |
| `integration_mode` | string | `""` | How this context is integrated (`in_process`, `ffi`, `kernel`, ...). |
| `target_output_language` | string | `""` | Target language for transpilation. |
| `compile_target` | string | `""` | Optional compile target hint, e.g. `aero-calculus`. |

### 3.4 `[scaling]` — auto-decomposition thresholds

| Key | Type | Default | Meaning |
|---|---|---|---|
| `auto_split_threshold` | int | `1500` | Node count above which module mitosis splits the HIN graph. |
| `max_module_complexity` | int | `200` | Hard ceiling on per-module cyclomatic complexity. |
| `hierarchy_depth` | int | `4` | Maximum recursion depth for hierarchical decomposition. |

### 3.5 `[graph]` — dependency DAG

| Key | Type | Default | Meaning |
|---|---|---|---|
| `targets` | list of strings or tables | `[]` | Compilation targets. A table form supports `name`, `source`, `output`, `language`, etc. |
| `dependencies` | dict | `{}` | Mapping `target_name → list[dependency_name]`. Cycles are rejected. |
| `entrypoint` | string | `"orchestrator"` | Root node of the execution DAG. |
| `workspace_mode` | string | `"incremental"` | `"incremental"` or `"fallback_manifest"`. |
| `allow_partial_graph` | bool | `False` | Allow build to proceed with unresolved targets. |
| `boundaries` | list of strings | `[]` | Contexts that act as FFI/module boundaries. |

### 3.6 `[compiler]` — compilation optimizer flags

| Key | Type | Default | Meaning |
|---|---|---|---|
| `toolchain` | string | `""` | Active toolchain (`python3`, `rustc`, etc.). |
| `optimization_level` | string | `"O3"` | Optimization level passed to the backend compiler. |
| `profile_guided_optimization` | string | `"enabled_strict"` | PGO mode. |
| `tier_shifting_hotness_threshold` | int | `100` | Hotness threshold for tier shifting. |
| `hotspot_loop_unroll_depth` | int | `32` | Loop unroll depth for hotspots. |
| `aot_boundary_check_elimination` | bool | `True` | Eliminate redundant boundary checks. |
| `vector_intrinsics_auto_generation` | bool | `True` | Auto-generate vector intrinsics. |
| `pipeline_budget_seconds` | float | `120.0` | Time budget for the whole pipeline. |
| `max_memory_mb` | int | `2048` | Memory ceiling in MiB. |
| `dead_code_elimination_passes` | int | `0` | Number of dead-code passes. |
| `identifier_minification` | bool | `False` | Minify identifiers. |
| `scope_aware_alpha_renaming` | bool | `False` | Rename variables to avoid shadowing. |
| `strip_comments_and_dunders` | bool | `False` | Strip comments and dunder fields. |
| `enforce_strict_typing_checks` | bool | `False` | Require strict type assertions. |

Additional keys are preserved verbatim and may be consumed by specific language routers.

### 3.7 `[cortex]` — runtime / consensus configuration

| Key | Type | Default | Meaning |
|---|---|---|---|
| `consensus_protocol` | string | `"raft_driven_mutation_lock"` | Mutation consensus protocol. |
| `mutation_entropy_clamp_threshold` | float | `0.05` | Maximum allowed mutation entropy. |
| `total_cooperating_agents` | int | `8` | Number of cooperating optimizer agents. |
| `heuristic_exploration_depth` | int | `3` | Search depth for heuristic exploration. |
| `execution_mode` | string | `"lock_free_polling_wheel_realtime"` | Runtime scheduling mode. |
| `core_affinity_mask` | string | `"0xFFFF"` | CPU affinity bitmask. |
| `numa_node_locality_binding` | bool | `True` | Bind memory allocations to local NUMA nodes. |
| `inter_core_ring_buffer_capacity` | int | `262144` | Ring-buffer capacity for inter-core comms. |
| `target_accuracy_floor` | float | `0.995` | Minimum acceptable accuracy for evolution. |
| `cycles` | int | `11` | Default number of build cycles. |
| `max_processing_latency_limit_us` | int/float | `20000` | Latency ceiling in microseconds. |
| `target_compression_ratio_ceiling` | float | `0.2` | Maximum allowed output compression ratio. |
| `kinetic_stagnation_window_size` | int | `2` | Stagnation detection window. |
| `experience_replay_save_threshold` | float | `0.0001` | Threshold for replay/telemetry retention. |
| `optimization_strategy_priority` | string | `"AGGRESSIVE_HYPER_MUTATION"` | Strategy selector for the optimizer.

### 3.8 `[cortex.nsga2]` — genetic optimizer tuning

| Key | Type | Default | Meaning |
|---|---|---|---|
| `population_size` | int | `28` | NSGA-II population size. |
| `mutation_rate` | float | `0.101` | Per-gene mutation rate. |
| `crossover_rate` | float | `0.678` | Crossover rate. |

### 3.9 `[precision_shield]` — floating-point hardening

| Key | Type | Default | Allowed | Meaning |
|---|---|---|---|---|
| `floating_point_contract` | string | `"disallow"` | `allow`, `disallow` | Allow/reject FP contraction. |
| `fast_math_override` | bool | `False` | bool | Block `-ffast-math` / equivalent. |
| `ieee_compliance` | string | `"strict"` | `strict`, `relaxed` | IEEE-754 compliance level. |
| `default_float` | string | `"double"` | `double`, `quad`, `arbitrary` | Default FP width. |
| `arbitrary_precision_bits` | int | `128` | `> 0` | Bits for arbitrary precision. |
| `per_zone_overrides` | dict | `{}` | object | Per-function/zone FP overrides. |
| `auto_detect_need` | bool | `False` | bool | Auto-enable precision shield. |

### 3.10 `[libraries]` — external math/HPC libraries

| Key | Type | Default | Allowed | Meaning |
|---|---|---|---|---|
| `blas` | string | `"none"` | `auto`, `mkl`, `openblas`, `none` | BLAS backend. |
| `lapack` | string | `"none"` | `auto`, `mkl`, `openblas`, `none` | LAPACK backend. |
| `mpi` | bool | `False` | bool | Enable MPI. |
| `mpi_flavor` | string | `None` | `openmpi`, `mpich`, `null` | MPI distribution. |
| `cuda` | string | `"none"` | `auto`, `cuda`, `none` | CUDA library request. |

### 3.11 `[distributed]` — multi-node execution

| Key | Type | Default | Allowed | Meaning |
|---|---|---|---|---|
| `enabled` | bool | `False` | bool | Enable distributed execution. |
| `worker_nodes` | list of strings | `[]` | list | Hostnames or addresses of workers. |
| `cache_sharing` | string | `"nfs"` | `nfs`, `redis`, `s3` | Shared cache backend. |

### 3.12 `[gpu]` — GPU kernel offloading

| Key | Type | Default | Allowed | Meaning |
|---|---|---|---|---|
| `enabled` | bool | `False` | bool | Enable GPU compilation. |
| `backend` | string | `"cuda"` | `cuda`, `hip`, `opencl` | GPU API. |
| `kernel_sources` | list of strings | `[]` | list | Paths to hand-written kernel files. |

### 3.13 `[physics]` — physical-dimension validation

| Key | Type | Default | Meaning |
|---|---|---|---|
| `dimensions` | list of strings | `[]` | Declared physical dimensions, e.g. `["length", "time"]`. |
| `symbolic_validation` | bool | `False` | Enable symbolic dimension checking. |

### 3.14 `[hpc]` — batch scheduler integration

| Key | Type | Default | Allowed | Meaning |
|---|---|---|---|---|
| `scheduler` | string | `"none"` | `slurm`, `pbs`, `none` | Batch scheduler. |
| `queue` | string | `"cpu"` | string | Queue/partition name. |
| `nodes` | int | `1` | `> 0` | Number of nodes. |
| `tasks_per_node` | int | `1` | `> 0` | Tasks per node. |
| `walltime` | string | `"01:00:00"` | HH:MM:SS | Wall-clock limit. |
| `environment` | dict | `{}` | object | Environment variables for the job. |
| `build_on_login_node` | bool | `True` | bool | Build before submitting. |
| `post_build_run` | bool | `False` | bool | Submit/run after build. |

### 3.15 `[runtime]` — feedback and benchmarking

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enable_feedback` | bool | `False` | Feed runtime metrics back into the optimizer. |
| `benchmark_command` | string | `""` | Shell command to run for benchmarking. |
| `metrics_to_collect` | list of strings | `["wall_time"]` | Metrics to record. |
| `accuracy_reference` | string | `""` | Path/file to compare against for accuracy. |
| `feedback_weight` | float | `0.3` | Weight of feedback in scoring (0–1). |

### 3.16 `[frameworks]` — language / framework selection

| Key | Type | Default | Allowed | Meaning |
|---|---|---|---|---|
| `language` | string | `""` | `rust`, `python`, `""` | Primary target framework. |

Additional keys may declare framework-specific objects; each must be a JSON object and is preserved verbatim.

### 3.17 `[analysis]` — source analysis toggles

| Key | Type | Default | Meaning |
|---|---|---|---|
| `ast_scanning` | string | `"pass_through"` | AST scanning mode. |
| `dead_code_elimination` | bool | `False` | Remove unused functions/variables. |
| `static_import_pruning` | bool | `False` | Remove unused imports. |
| `macro_expansion` | string | `"pass_through"` | Macro expansion mode. |

### 3.18 `[validation]` — test harness

| Key | Type | Default | Meaning |
|---|---|---|---|
| `suite` | string | `""` | Path to test suite/script. |
| `tolerance` | float | `1e-8` | Numerical tolerance. |
| `test_cases` | list of strings | `[]` | Additional test-case paths. |
| `execution_command` | string | `""` | Command used to execute the suite. |
| `validation_required` | bool | `True` | Fail build if validation fails. |
| `generate_test_shims` | bool | `False` | Auto-generate test stubs during scaffold. |

### 3.19 `[context]` — external context sources

| Key | Type | Default | Meaning |
|---|---|---|---|
| `sources` | list of objects | `[]` | Named context sources for semantic-fluidity ingestion. |

Each source object may contain:

| Key | Type | Meaning |
|---|---|---|
| `path` | string | Source path. |
| `language` | string | Source language. |
| `purpose` | string | Role of this context (`input`, `reference`, `constraint`, ...). |
| `repair_rules` | list | Rules applied during ingestion (`fix_imports`, `add_stub`, ...). |
| `target_mapping` | dict | Map of symbols to canonical blueprint targets. |

### 3.20 `[scaffold]` — out-of-tree repository generation

| Key | Type | Default | Meaning |
|---|---|---|---|
| `source_entry` | string or list | `""` | Source path(s) to scaffold from. |
| `auto_layout` | bool | `False` | Automatically choose a layout. |
| `distribution_directory` | string | `""` | Output directory for the generated repo. |
| `name` | string | `""` | Generated project name. |
| `compatibility_shims` | list of strings | `[]` | Rust-only compatibility shims. |
| `dependencies` | dict | `{}` | Extra dependencies to inject. |
| `decomposition_mode` | string | `""` | `"modular_package"` or `""`. |
| `module_mapping` | dict | `{}` | For `modular_package`: filename → list of symbol names to extract. |

### 3.21 `[evolution]` — cNrGA configuration

#### 3.21.1 `[evolution.genome.<name>]` — tunable parameters

Each sub-table declares one gene in the evolutionary genome.

| Key | Type | Meaning |
|---|---|---|
| `path` | string | Dotted TOML location where the optimized value is written, e.g. `cortex.nsga2.mutation_rate`. |
| `type` | string | `int`, `float`, or `str`. |
| `min` | number | Lower bound. |
| `max` | number | Upper bound. |
| `step` | number | Mutation step scale. |
| `default` | any | Default value. |
| `choices` | list | Allowed categorical values. |

#### 3.21.2 `[evolution.source_mutation]` — source-code evolution

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `False` | Mutate source files during evolution. |
| `mutation_rate` | float | `0.8` | Probability of mutating a file. |
| `target_files` | list | `["*.py", "builder_brains/*.py"]` | Glob patterns for mutation targets. |
| `rules` | list | `["insert_function_stub", "add_docstring"]` | Mutation rule names. |

#### 3.21.3 `[evolution]` top-level keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rollback_threshold` | float | `2.0` | Speed-gain percentage below which a generation is rolled back. |

### 3.22 `[abstractions]` — legacy → target rewrite rules

Use the array-of-tables syntax:

```toml
[[abstractions.rewrite]]
legacy_pattern = "legacy_func"
target_pattern = "new_func"
```

### 3.23 `[context_bridges]` — cross-context bindings

```toml
[[context_bridges]]
source = "legacy_c"
target = "rust_core"
bridge_type = "ffi"
```

### 3.24 `[dag]` — legacy dependency alias

A flat mapping from each source file stem to the list of dependencies it imports. Used as a pre-DAG shorthand in legacy blueprints.

```ini
[dag]
"translator_core_orchestrator_py" = []
"scanner_py" = ["translator_core_orchestrator_py"]
```

### 3.25 `[meta]` — legacy project metadata

| Key | Type | Default | Meaning |
|---|---|---|---|
| `project_name` | string | `""` | Human-readable project name. |
| `target_version` | string | `""` | Target version string. |
| `execution_mode` | string | `""` | Legacy execution-mode hint. |

This section is tolerated by the parser but is **not** propagated into the runtime `build_context`; use `[system]` for active project identity.

### 3.26 Example blueprints

#### Minimal TOML native blueprint

```toml
[system]
name = "my_project"
strategy = "DIRECT_COMPILE"

[context_registry.main]
path = "./src/app_logic.py"
language = "python"
preserve_original_logic = false

[scaling]
auto_split_threshold = 120
max_module_complexity = 12
```

#### Blueprint with evolution genome

```toml
[system]
strategy = "DIRECT_COMPILE"

[context_registry.main]
path = "./src/main.py"
language = "python"

[evolution.genome.mutation_rate]
path = "cortex.nsga2.mutation_rate"
type = "float"
min = 0.05
max = 0.25
step = 0.001
default = 0.1

[evolution.rollback_threshold]
rollback_threshold = 2.0
```

#### Legacy INI-style blueprint

```ini
[system]
strategy = DIRECT_COMPILE

[graph]
targets = ["core_orchestrator"]
dependencies = {}

[compiler]
profile_guided_optimization = enabled_strict
max_memory_mb = 4096

[cortex]
total_cooperating_agents = 8
```

---

## 4. Pipeline Orchestration — The Golden Path

The recommended autonomous workflow progresses from a lean declaration to a hardened native artifact.

### 4.1 Step 0 — initialize a workspace

```bash
python main.py init --workspace ./my_project
cd my_project
```

This seeds `blueprint.aero`, `src/app_logic.py`, and `context.aero`.

### 4.2 Step 1 — write the blueprint

For a Python project:

```toml
[system]
name = "my_project"
strategy = "DIRECT_COMPILE"

[context_registry.main]
path = "./src/app_logic.py"
language = "python"

[scaling]
auto_split_threshold = 120
```

For a Rust project that needs an out-of-tree crate:

```toml
[system]
name = "my_project"
strategy = "DIRECT_COMPILE"

[frameworks]
language = "rust"

[scaffold]
source_entry = "./src/lib.rs"
distribution_directory = "./dist"
name = "my_crate"
```

### 4.3 Step 2 — infer from lean intent (optional)

If you prefer a near-empty blueprint, use the ultra-lean dialect and:

```bash
python main.py infer --workspace .
```

This scans the workspace and produces a full build DAG.

### 4.4 Step 3 — build

```bash
# Plain build
python main.py build --workspace .

# Build with autonomous evolution
python main.py build --workspace . --cycles 5

# Direct source compile
python main.py build --source src/app_logic.py --aeroc-out app.aeroc
```

### 4.5 Step 4 — decompose when the monolith grows

```bash
python main.py decompose --workspace .
```

This rewrites `blueprint.aero` with a `[graph]` dependency DAG derived from the Python source tree.

### 4.6 Step 5 — scaffold and harden

For a Rust project:

```bash
python main.py scaffold \
  --source-entry src/lib.rs \
  --distribution-directory ./dist \
  --name my_crate
```

For a Python project with modular decomposition:

```toml
[scaffold]
source_entry = "./src/monolith.py"
distribution_directory = "./dist"
decomposition_mode = "modular_package"
module_mapping = { monolith = ["module_a", "module_b"] }
```

Then:

```bash
python main.py build --workspace . --blueprint blueprint.aero
```

### 4.7 Build strategy routing

`main.build_command` resolves the blueprint and routes as follows:

1. If `--source` is given, compile to `.aeroc` via `aero_build_command`.
2. If `system.strategy == "DIRECT_COMPILE"`, call `orchestrator.run_direct_compile`.
3. If `[scaffold]` is present, run the `ScaffoldBuildPipeline`.
4. Otherwise, run the core self-evolving build cycle `orchestrator.run_build`.

---

## 5. Advanced Troubleshooting

### 5.1 Quarantine and bootstrap isolation

When a build targets files inside the running engine tree, the **Shadow Bootstrapper** engages:

1. `detect_self_targeting()` checks whether any target path resolves inside `sys.path[0]`.
2. `BootstrapStage` creates `.aero/bootstrap_stage/<token>/`.
3. `copy_target_files()` copies every blueprint source and `blueprint.aero` into the stage.
4. The build runs against the stage.
5. `BootstrapStage.validate()` runs the validation suite in the stage.
6. If validation reports `0 errors, 0 anomalies`, `promote()` atomically swaps the stage into the live tree.
7. If validation fails, `discard()` removes the stage and the live tree is untouched.

### 5.2 HIN-VM quarantine triggers

A quarantine (0-node compilation or `InvariantError`) is raised when:

- `UASTToHINTranslator.translate_uast()` emits an empty graph.
- `HINNetwork.validate_conservation()` finds an unbound auxiliary port or asymmetric edge.
- `RigidityVerifier.verify_boundary()` detects an off-shell coordinate shift (`AnomalyClosureError`).
- `UniversalHINNetwork.reduce_step()` hits an active-pair rewrite that violates a rigidity invariant.

### 5.3 Healing a topological anomaly

The engine tries to heal automatically. To invoke healing manually:

```bash
python main.py heal --aeroc graph.aeroc --output graph_fixed.aeroc
```

Steps performed:

1. `load_aeroc()` deserializes the HIN.
2. `TopologicalSelfHealer.find_unterminated_ports()` scans for auxiliary ports with `target is None`.
3. For each broken port, `heal_unterminated_interface()` inserts a structural node:
   - `ValueNode` for a missing constant/value input.
   - `SwitchNode` for a conditional/discarded wire.
   - `EraserNode` for a dead-code output.
4. The repaired boundary is re-verified by `RigidityVerifier`.
5. `save_aeroc()` writes the healed graph.

To also heal source files:

```bash
python main.py heal --path src/app_logic.py
```

### 5.4 Audit self-healing

```bash
# Standard audit (up to 3 rounds)
python main.py audit

# Deep audit
python main.py audit --max-rounds 5
```

`PreFlightTestAuditor`:

- Runs the test suite in a subprocess.
- Classifies failures as **Category A** (missing dependency → triggers bootstrap provisioning) or **Category B** (core logic bug → patched in place, e.g. path-literal normalization).
- Re-runs tests up to `max_rounds` or until convergence.

### 5.5 Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `0 error(s), 1 anomaly(ies)` during bootstrap | A target source was not copied into `bootstrap_stage`. | Ensure `BootstrapStage.copy_target_files()` is called; check `core/bootstrap/isolation.py`. |
| `AnomalyClosureError` during reduction | Boundary rigidity failed under perturbation. | Run `heal --aeroc`; increase `scaling.auto_split_threshold` to reduce fragmentation; disable reduction with `--no-reduce`. |
| `ModuleNotFoundError` in audit | Missing optional dependency. | Set `AERO_DISABLE_BOOTSTRAP=0` or install the package manually. |
| `compiled: 0` | Source produced empty UAST or HIN. | Check that the source uses supported constructs; use `--no-reduce` to inspect raw graph. |
| `InterfaceChangedError` in evolution | Crossover mutated a wave boundary. | Use `BoundaryAwareMutator` with adaptor insertion (enabled by `evolve_aeroc`). |

---

## 6. Developer / Researcher Workflow

### 6.1 Module mitosis

Module mitosis splits an oversized HIN graph using spectral partitioning.

```python
from core.translator import UASTToHINTranslator
from core.aero_frontend import python_source_to_uast
from core.aeroc import save_aeroc

source = open("src/monolith.py").read()
uast = python_source_to_uast(source)

translator = UASTToHINTranslator(auto_split_threshold=120)
primary, secondary = translator.execute_mitosis(
    translator.translate(uast)
)

save_aeroc(primary, "monolith.aeroc")
if secondary.nodes:
    save_aeroc(secondary, "monolith.part2.aeroc")
```

The split is computed from the Fiedler vector of the graph Laplacian `L = D - A`. Edges crossing the cut are severed and capped with `BoundaryPortNode` pairs, forming the explicit API contract `∂Ω`.

### 6.2 Coordinate-perturbation sweeps

```python
from core.spacetime_ledger import RigidityVerifier
from core.aeroc import load_aeroc

network = load_aeroc("monolith.aeroc")
boundary = [n for n in network.nodes.values() if getattr(n, "coordinate", None)]

verifier = RigidityVerifier()
verifier.verify_boundary(boundary)
```

The verifier groups boundary nodes into waves of at most `max_boundary=8`, builds a high-precision Decimal distance matrix, and checks that eigenvalue signatures persist under a `10⁻¹²²`-scale perturbation.

### 6.3 Geometric re-wiring of the holographic boundary ($\gamma^{-1}$)

The holographic boundary is repaired by applying the inverse constructor/destructor pair:

```python
from core.aeroc import load_aeroc, save_aeroc
from orchestrator import TopologicalSelfHealer

network = load_aeroc("broken.aeroc")
healer = TopologicalSelfHealer()
report = healer.heal_network(network)

print(f"healed={report['healed']} remaining={report['remaining']}")
save_aeroc(network, "healed.aeroc")
```

Each unterminated port is terminated by one of the structural nodes (`ValueNode` `V`, `SwitchNode` `σ`, `EraserNode` `ε`), then the network is re-validated. This is the geometric analogue of `γ⁻¹` (destructor/application) annihilating a dangling wire and re-introducing a well-typed term.

### 6.4 Verifying structural invariants

```python
from core.aeroc import load_aeroc
from core.hin_graph import HINGraph
from core.invariants import InvariantVerifier
from core.wavefront_scheduler import WavefrontScheduler

network = load_aeroc("graph.aeroc")
graph = HINGraph.from_hin_network(network)

verifier = InvariantVerifier()
waves = WavefrontScheduler().compute_wavefronts(graph)
verifier.verify_all(graph, waves=waves)
```

`verify_all` checks:

1. **Edge conservation** — every node's actual degree equals `expected_arity` and port types match.
2. **Interface signatures** — boundary ports crossing each wave match the stored signature.
3. **Spectral stability** — the graph Laplacian's `λ₂` has not collapsed below the threshold.

### 6.5 Extending the blueprint schema

New top-level sections are preserved by the parser; to make them functional, update `blueprint_parser.normalize_optional_sections()` or consume them in `orchestrator.run_build`.

---

## 7. See Also

- `README.md` — high-level project pitch and quick start.
- `ARCHITECTURE.md` — deep architectural breakdown and data-flow diagrams.
- `WORKSPACE_AUDIT.md` — audit and self-healing procedures.
- `core/hin_vm.py` — HIN node and reduction semantics.
- `core/translator.py` — UAST → HIN translation and module mitosis.
- `evolve.py` — cNrGA evolution loop and genome schema.
- `orchestrator.py` — build orchestration and `TopologicalSelfHealer`.
- `src/scaffold/engine.py` — repository generation and `.part2.aeroc` bundling.
