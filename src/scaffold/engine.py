# -*- coding: utf-8 -*-
"""
``ScaffoldEngine`` -- the zero-config, out-of-tree repository generator.

End-to-end flow is routed by :mod:`src.scaffold.language_router`:

* **rust**  → Cargo layout + optional cargo build + Rust diagnostic recovery
* **python** → native Python layout + compileall/py_compile validation (no Cargo)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.scaffold.decomposition import (
    DecompositionResult,
    ModularDecomposer,
    merge_source_asts,
)
from src.scaffold.language_router import (
    cargo_bypass_warning,
    is_native_crate_language,
    is_python,
    resolve_target_language,
)
from src.scaffold.python_repo_generator import (
    PythonGeneratedRepo,
    build_python_spec,
    generate_python_repo,
    infer_import_dependencies,
)
from src.scaffold.pre_write_validator import PreWriteValidator, ValidationError
from src.scaffold.python_validator import PythonValidationRunner
from src.scaffold.recovery import DiagnosticRecoveryRunner, RecoveryResult
from src.scaffold.repo_generator import GeneratedRepo, build_spec, generate_repo
from src.scaffold.rust_shield import RustSemanticShield, ShieldReport
from src.scaffold.source_resolver import (
    SourceEntry,
    SourceEntryNotFound,
    copy_into_workspace,
    resolve_source_entry,
)
from src.scaffold.test_matrix import generate_test_matrix
from src.scaffold.workspace import OutOfTreeWorkspace

Logger = Callable[[str], None]


@dataclass
class ScaffoldResult:
    """The outcome of a scaffolding run."""

    source: Dict[str, Any]
    repo: Dict[str, Any]
    shield: Dict[str, Any]
    workspace: str
    out_of_tree: bool
    language: str = "rust"
    build: Optional[Dict[str, Any]] = None
    merge: Optional[Dict[str, Any]] = None
    messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "repo": self.repo,
            "shield": self.shield,
            "workspace": self.workspace,
            "out_of_tree": self.out_of_tree,
            "language": self.language,
            "build": self.build,
            "merge": self.merge,
        }


class ScaffoldEngine:
    """Generate a standalone repo from a source entry, fully out-of-tree."""

    def __init__(self, logger: Optional[Logger] = None, verbose: bool = False) -> None:
        self._logger = logger
        self.verbose = verbose
        self.shield = RustSemanticShield()
        self._python_validator = PythonValidationRunner()

    def _log(self, message: str) -> None:
        if self.verbose and self._logger is not None:
            self._logger(message)

    def _finalize_workspace(
        self,
        workspace: OutOfTreeWorkspace,
        language: str,
        build_info: Optional[Dict[str, Any]],
        repo_dict: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Run delegated validation and promote the staging workspace on success.

        Returns ``(committed, build_info, workspace_path)``.  When validation fails
        the staging workspace is cleaned up and ``build_info`` carries the external
        command's output so the failure is reported against the generated code, not
        the orchestration logic.
        """
        workspace_path = str(workspace.root)
        if build_info is None:
            build_info = {}
        try:
            result = self._pre_write_validator.validate(
                workspace_path, language=language
            )
            build_info["pre_write_validation"] = {
                "succeeded": True,
                "command": " ".join(result.command) if result.command else "",
                "output": result.output,
                "return_code": result.return_code,
            }
            self._log("pre-write validation succeeded")
        except ValidationError as exc:
            build_info["succeeded"] = False
            build_info["pre_write_validation"] = {
                "succeeded": False,
                "error": str(exc),
                "output": exc.output,
            }
            self._log(f"pre-write validation failed: {exc}")
            workspace.cleanup()
            return False, build_info, workspace_path

        workspace.commit()
        workspace_path = str(workspace.root)
        repo_dict["root"] = workspace_path
        return True, build_info, workspace_path

    # ------------------------------------------------------------------

    def scaffold(
        self,
        source_entry: Union[str, List[str]],
        name: Optional[str] = None,
        base_dir: Optional[Path] = None,
        distribution_directory: Optional[Path] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        compatibility_shims: Optional[List[str]] = None,
        build: bool = False,
        keep: Optional[bool] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        module_mapping: Optional[Dict[str, List[str]]] = None,
        decomposition_mode: Optional[str] = None,
        prune_imports: bool = False,
        generate_tests: bool = False,
        merge_active: bool = False,
    ) -> ScaffoldResult:
        context = context or {}
        self._pre_write_validator = PreWriteValidator(context)

        # Multi-file source ingestion matrix: a single path or a list of paths.
        raw_entries = source_entry if isinstance(source_entry, (list, tuple)) else [source_entry]
        raw_entries = [str(p).strip() for p in raw_entries if str(p).strip()]
        if not raw_entries:
            raise SourceEntryNotFound("no source_entry path(s) provided")
        entries = [resolve_source_entry(p, base_dir=base_dir) for p in raw_entries]
        entry = entries[0]

        # Aero-Calculus artifact packaging: if the source entry is a compiled
        # .aeroc graph (or one of multiple entries is), route to the artifact
        # packager so any .part2.aeroc partitions are included automatically.
        if entry.path.suffix.lower() == ".aeroc":
            return self._scaffold_aeroc(
                entry=entry,
                name=name,
                distribution_directory=distribution_directory,
                keep=keep,
                generate_tests=generate_tests,
            )

        target_language = language or resolve_target_language(context, source_entry=entry)
        if len(entries) > 1:
            self._log(
                f"ingest matrix -> {len(entries)} source files: "
                f"{', '.join(e.path.name for e in entries)}"
            )
        self._log(
            f"language router -> {target_language!r}  "
            f"(source={entry.path.name}, resolved={entry.language})"
        )
        self._log(f"resolved source_entry -> {entry.path}  (language: {entry.language})")

        if is_python(target_language) and decomposition_mode == "modular_package" and module_mapping:
            self._log("decomposition router -> 'modular_package' (AST node extraction)")
            return self._scaffold_python_modular(
                entries=entries,
                name=name,
                distribution_directory=distribution_directory,
                dependencies=dependencies,
                module_mapping=module_mapping,
                build=build,
                keep=keep,
                prune_imports=prune_imports,
                generate_tests=generate_tests,
            )
        if len(entries) > 1:
            self._log(
                "ingest matrix: multiple files only merge for the python "
                "'modular_package' path; using the first entry for this target"
            )
        if is_python(target_language):
            return self._scaffold_python(
                entry=entry,
                name=name,
                distribution_directory=distribution_directory,
                dependencies=dependencies,
                build=build,
                keep=keep,
                generate_tests=generate_tests,
            )
        return self._scaffold_rust(
            entry=entry,
            name=name,
            distribution_directory=distribution_directory,
            dependencies=dependencies,
            compatibility_shims=compatibility_shims,
            build=build,
            keep=keep,
            generate_tests=generate_tests,
            merge_active=merge_active,
        )

    # ------------------------------------------------------------------
    # Aero-Calculus artifact packaging
    # ------------------------------------------------------------------

    def _collect_aeroc_artifacts(
        self,
        entry: SourceEntry,
        workspace_root: Path,
        repo_dict: Dict[str, Any],
    ) -> List[str]:
        """Copy a primary .aeroc and any .part2.aeroc partition into the workspace.

        The relative paths are appended to ``repo_dict['files']`` so the generated
        build artifact manifest lists them automatically.
        """
        artifact_dir = workspace_root / "build_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        collected: List[str] = []

        def _copy(src: Path) -> None:
            if not src.is_file():
                return
            dest = artifact_dir / src.name
            shutil.copy2(src, dest)
            rel = str(dest.relative_to(workspace_root))
            collected.append(rel)
            repo_dict.setdefault("files", []).append(rel)

        if entry.path.suffix.lower() == ".aeroc":
            primary = entry.path
        else:
            primary = entry.path.with_suffix(".aeroc")
        part2 = primary.parent / (primary.stem + ".part2.aeroc")
        _copy(primary)
        _copy(part2)
        if collected:
            repo_dict.setdefault("aeroc_artifacts", []).extend(collected)
            self._log(f"aeroc artifacts: {collected}")
        return collected

    def _scaffold_aeroc(
        self,
        *,
        entry: SourceEntry,
        name: Optional[str],
        distribution_directory: Optional[Path],
        keep: Optional[bool],
        generate_tests: bool = False,
    ) -> ScaffoldResult:
        """Package an .aeroc graph (and its .part2 partition) into a standalone artifact workspace."""
        workspace = OutOfTreeWorkspace(distribution_directory=distribution_directory, keep=keep)
        workspace.create()
        self._log(
            f"workspace: {workspace.root}  "
            f"({'temporary, auto-cleaned' if workspace.is_temporary else 'distribution directory'})"
        )

        repo_dict: Dict[str, Any] = {"root": str(workspace.root), "files": []}
        self._collect_aeroc_artifacts(entry, workspace.root, repo_dict)

        build_info: Optional[Dict[str, Any]] = None
        if generate_tests:
            # No language-specific test harness for raw .aeroc artifacts.
            build_info = {"tests": "skipped (no test harness for .aeroc artifacts)"}

        committed, build_info, workspace_path = self._finalize_workspace(
            workspace, "aeroc", build_info, repo_dict
        )
        if not committed and distribution_directory is not None:
            self._log("scaffold: .aeroc artifact staging failed validation; not promoted")

        return ScaffoldResult(
            source=entry.to_dict(),
            repo=repo_dict,
            shield={"anchors": [], "applied": [], "changed": False, "skipped": "aeroc-artifact"},
            workspace=workspace_path,
            out_of_tree=True,
            language="aeroc",
            build=build_info,
        )

    # ------------------------------------------------------------------
    # Rust path
    # ------------------------------------------------------------------

    def _scaffold_rust(
        self,
        *,
        entry: SourceEntry,
        name: Optional[str],
        distribution_directory: Optional[Path],
        dependencies: Optional[Dict[str, Any]],
        compatibility_shims: Optional[List[str]],
        build: bool,
        keep: Optional[bool],
        generate_tests: bool = False,
        merge_active: bool = False,
    ) -> ScaffoldResult:
        source_text = entry.read_text()
        shield_report = self._shield_rust(entry, source_text, compatibility_shims=compatibility_shims)

        workspace = OutOfTreeWorkspace(distribution_directory=distribution_directory, keep=keep)
        workspace.create()
        self._log(
            f"workspace: {workspace.root}  "
            f"({'temporary, auto-cleaned' if workspace.is_temporary else 'distribution directory'})"
        )

        spec = build_spec(name or entry.stem, shield_report.source, dependencies=dependencies)
        self._log(
            f"crate '{spec.name}'  deps={list(spec.dependencies) or '(none)'}  "
            f"crate-type={spec.crate_type}  pymodule={spec.python_module or '-'}"
        )
        repo = generate_repo(spec, workspace.root)
        for written in repo.files:
            self._log(f"  + {written}")
        copy_into_workspace(entry, workspace.root / "src" / "lib.rs", content=spec.source)

        repo_dict = repo.to_dict()
        if generate_tests:
            matrix = generate_test_matrix(
                "rust", workspace.root, crate=spec.name, logger=self._logger if self.verbose else None
            )
            repo_dict.setdefault("files", [])
            repo_dict["files"] = list(repo_dict["files"]) + matrix.files
            repo_dict["test_matrix"] = matrix.to_dict()

        build_info: Optional[Dict[str, Any]] = None
        merge_info: Optional[Dict[str, Any]] = None
        # Strict Rust routing gate: the low-level cargo build path is reserved
        # for genuine Rust crate targets.  Only run the compilation sequence when
        # the resolved source language is 'rust'; any other target (c/cpp/
        # fortran/shell/unknown that fell through to this path) bypasses cargo
        # cleanly instead of triggering spurious Cargo errors.
        resolved = entry.language
        if build and not is_native_crate_language(resolved):
            warning = cargo_bypass_warning(resolved)
            self._log(warning)
            build_info = {
                "succeeded": True,
                "bypassed": True,
                "language": resolved,
                "warning": warning,
                "attempts": [],
            }
        elif build:
            build_info = self._build_rust_with_recovery(repo).to_dict()
            build_info["language"] = "rust"
            if merge_active:
                merge_info = self._merge_active(
                    workspace.root, spec, succeeded=bool(build_info.get("succeeded"))
                )
        elif merge_active:
            merge_info = {
                "merged": False,
                "reason": "--merge-active requires --build (nothing was compiled)",
            }
            self._log("merge: skipped — --merge-active requires a successful --build")

        self._collect_aeroc_artifacts(entry, workspace.root, repo_dict)
        committed, build_info, workspace_path = self._finalize_workspace(
            workspace, "rust", build_info, repo_dict
        )
        if not committed and distribution_directory is not None:
            self._log("scaffold: rust workspace failed validation; not promoted")

        return ScaffoldResult(
            source=entry.to_dict(),
            repo=repo_dict,
            shield=shield_report.to_dict(),
            workspace=workspace_path,
            out_of_tree=True,
            language="rust",
            build=build_info,
            merge=merge_info,
        )

    def _merge_active(self, workspace_root: Path, spec: Any, *, succeeded: bool) -> Dict[str, Any]:
        """Merge the verified cdylib into the live runtime extension layer."""
        from src.scaffold.active_merge import merge_active as _merge

        if not succeeded:
            self._log("merge: skipped — out-of-tree build did not succeed")
            return {"merged": False, "reason": "out-of-tree build did not succeed"}

        module_name = spec.python_module or spec.name
        result = _merge(workspace_root, spec.name, module_name)
        if result.merged:
            self._log(f"merge: {' / '.join(result.notes)}")
            live = "now live in-process" if result.loaded else "staged for next start"
            self._log(f"merge: '{module_name}' -> {result.destination} ({live})")
        else:
            self._log(f"merge: failed — {result.reason}")
        return result.to_dict()

    def _shield_rust(
        self,
        entry: SourceEntry,
        source_text: str,
        compatibility_shims: Optional[List[str]] = None,
    ) -> ShieldReport:
        if entry.language != "rust":
            return ShieldReport(source=source_text)
        report = self.shield.apply(source_text, compatibility_shims=compatibility_shims)
        if report.anchors:
            self._log(f"shield: detected anchors {sorted(report.anchors)}")
        for fix in report.applied:
            self._log(f"shield: applied {fix}")
        if report.anchors and not report.applied:
            self._log("shield: source already compatible; no fixes needed")
        return report

    def _build_rust_with_recovery(self, repo: GeneratedRepo) -> RecoveryResult:
        """Build the generated crate from its own root, recovering on failure."""
        from src.build.compilers import RustCompiler

        compiler = RustCompiler()
        if compiler.discover() is None:
            self._log("build: no cargo/rustc toolchain found; skipping compile")
            return RecoveryResult(succeeded=False, final_output="no rust toolchain")

        crate_root = repo.root

        def _run_cargo() -> tuple:
            result = compiler.compile(
                target_name=repo.spec.name if repo.spec else "crate",
                sources=["src/lib.rs"],
                workdir=crate_root,
                options={"root": "."},
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            return result.success, output, result.return_code

        self._log(f"build: cargo build (cwd={crate_root}); target/ stays out-of-tree")
        runner = DiagnosticRecoveryRunner(self.shield, max_retries=1)
        recovery = runner.run(crate_root, _run_cargo)
        for attempt in recovery.attempts:
            status = "ok" if attempt.succeeded else f"failed (code {attempt.return_code})"
            extra = f"; corrections: {', '.join(attempt.corrections)}" if attempt.corrections else ""
            self._log(f"build: attempt {attempt.attempt} {status}{extra}")
        if recovery.recovered:
            self._log("build: recovered after auto-correction")
        return recovery

    # ------------------------------------------------------------------
    # Python path
    # ------------------------------------------------------------------

    def _scaffold_python(
        self,
        *,
        entry: SourceEntry,
        name: Optional[str],
        distribution_directory: Optional[Path],
        dependencies: Optional[Dict[str, Any]],
        build: bool,
        keep: Optional[bool],
        generate_tests: bool = False,
    ) -> ScaffoldResult:
        source_text = entry.read_text()
        self._log("shield: skipped (Rust-specific shields not applied to Python targets)")

        workspace = OutOfTreeWorkspace(distribution_directory=distribution_directory, keep=keep)
        workspace.create()
        self._log(
            f"workspace: {workspace.root}  "
            f"({'temporary, auto-cleaned' if workspace.is_temporary else 'distribution directory'})"
        )

        dep_list: Optional[List[str]] = None
        if dependencies:
            dep_list = [str(v) if not isinstance(v, str) else v for v in dependencies.values()]

        spec = build_python_spec(
            name or entry.stem,
            source_text,
            entry_filename=entry.name,
            dependencies=dep_list,
        )
        self._log(
            f"project '{spec.name}'  entry={spec.entry_filename}  "
            f"deps={spec.dependencies or '(none)'}"
        )
        repo = generate_python_repo(spec, workspace.root)
        for written in repo.files:
            self._log(f"  + {written}")

        repo_dict = repo.to_dict()
        if generate_tests:
            module = spec.entry_filename
            module = module[:-3] if module.endswith(".py") else module
            matrix = generate_test_matrix(
                "python",
                workspace.root,
                package="",
                modules=[module],
                logger=self._logger if self.verbose else None,
            )
            repo_dict["files"] = list(repo_dict.get("files", [])) + matrix.files
            repo_dict["test_matrix"] = matrix.to_dict()

        build_info: Optional[Dict[str, Any]] = None
        if build:
            build_info = self._validate_python(repo).to_dict()

        self._collect_aeroc_artifacts(entry, workspace.root, repo_dict)
        committed, build_info, workspace_path = self._finalize_workspace(
            workspace, "python", build_info, repo_dict
        )
        if not committed and distribution_directory is not None:
            self._log("scaffold: python workspace failed validation; not promoted")

        return ScaffoldResult(
            source=entry.to_dict(),
            repo=repo_dict,
            shield={"anchors": [], "applied": [], "changed": False, "skipped": "python-target"},
            workspace=workspace_path,
            out_of_tree=True,
            language="python",
            build=build_info,
        )

    def _scaffold_python_modular(
        self,
        *,
        entries: List[SourceEntry],
        name: Optional[str],
        distribution_directory: Optional[Path],
        dependencies: Optional[Dict[str, Any]],
        module_mapping: Dict[str, List[str]],
        build: bool,
        keep: Optional[bool],
        prune_imports: bool = False,
        generate_tests: bool = False,
    ) -> ScaffoldResult:
        entry = entries[0]
        self._log("shield: skipped (Rust-specific shields not applied to Python targets)")

        # Multi-file ingestion matrix: merge several ASTs into one schema first.
        if len(entries) > 1:
            merged = merge_source_asts(
                [(e.path.name, e.read_text()) for e in entries],
                logger=self._logger if self.verbose else None,
            )
            source_text = merged.source
            orchestrator_name = "main.py"
            self._log(
                f"ingest matrix: merged {len(entries)} files -> "
                f"{len(merged.definitions)} top-level definition(s)"
            )
        else:
            source_text = entry.read_text()
            orchestrator_name = entry.name

        workspace = OutOfTreeWorkspace(distribution_directory=distribution_directory, keep=keep)
        workspace.create()
        self._log(
            f"workspace: {workspace.root}  "
            f"({'temporary, auto-cleaned' if workspace.is_temporary else 'distribution directory'})"
        )
        if prune_imports:
            self._log("optimize: static import pruning enabled (analysis.static_import_pruning)")

        decomposer = ModularDecomposer(
            logger=self._logger, verbose=self.verbose, prune_imports=prune_imports
        )
        decomposition: DecompositionResult = decomposer.decompose(
            source_text,
            module_mapping,
            source_filename=orchestrator_name,
            dest_dir=workspace.root,
        )
        for written in decomposition.files:
            self._log(f"  + {written}")

        if generate_tests:
            matrix = generate_test_matrix(
                "python",
                workspace.root,
                package="",
                modules=[m.filename for m in decomposition.modules],
                logger=self._logger if self.verbose else None,
            )
            decomposition.files.extend(matrix.files)

        dep_overrides: Optional[List[str]] = None
        if dependencies:
            dep_overrides = [str(v) for v in dependencies.values()]
        deps = infer_import_dependencies(source_text, dep_overrides)

        spec = {
            "name": name or entry.stem,
            "version": "0.1.0",
            "entry_filename": decomposition.orchestrator,
            "dependencies": deps,
            "language": "python",
            "decomposition_mode": "modular_package",
        }
        repo: Dict[str, Any] = {
            "root": str(workspace.root),
            "files": list(decomposition.files),
            "spec": spec,
            "language": "python",
            "decomposition": decomposition.to_dict(),
        }

        build_info: Optional[Dict[str, Any]] = None
        if build:
            self._log(
                f"validate: python bytecode check (cwd={workspace.root}); cargo skipped"
            )
            result = self._python_validator.validate_workspace(workspace.root)
            attempt = result.attempts[0] if result.attempts else None
            if attempt and attempt.succeeded:
                self._log("validate: compileall/py_compile ok")
            elif attempt:
                for err in attempt.errors:
                    self._log(f"validate: {err}")
            build_info = result.to_dict()

        self._collect_aeroc_artifacts(entry, workspace.root, repo)
        committed, build_info, workspace_path = self._finalize_workspace(
            workspace, "python", build_info, repo
        )
        if not committed and distribution_directory is not None:
            self._log("scaffold: python modular workspace failed validation; not promoted")

        return ScaffoldResult(
            source=entry.to_dict(),
            repo=repo,
            shield={"anchors": [], "applied": [], "changed": False, "skipped": "python-target"},
            workspace=workspace_path,
            out_of_tree=True,
            language="python",
            build=build_info,
        )

    def _validate_python(self, repo: PythonGeneratedRepo) -> Any:
        self._log(f"validate: python bytecode check (cwd={repo.root}); cargo skipped")
        result = self._python_validator.validate_workspace(repo.root)
        attempt = result.attempts[0] if result.attempts else None
        if attempt and attempt.succeeded:
            self._log("validate: compileall/py_compile ok")
        elif attempt:
            for err in attempt.errors:
                self._log(f"validate: {err}")
        return result
