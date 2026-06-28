"""CMake/Meson-style host toolchain introspection.

A deterministic, dependency-free discovery engine that locates the compilers,
linkers and runtimes available on the host, extracts their version and target
architecture, validates them with a minimal sanity-compile, and caches the
successful configuration locally.  No network, no cloud services.

Languages are resolved from the source extensions tracked in the context
registry (via :data:`core.parser.universal.LANGUAGE_BY_EXTENSION`), so the engine
only probes for toolchains the project actually needs.

Environment overrides honoured for portable/relocatable execution:
``CC``, ``CXX``, ``FC``, ``RUSTC``, ``PYTHON`` (binary selection) and
``CFLAGS`` / ``CXXFLAGS`` / ``LDFLAGS`` (compile/link flags).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from core.parser.universal import LANGUAGE_BY_EXTENSION

_PathLike = Union[str, Path]

_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?)")


@dataclass
class ToolchainSpec:
    """Static description of how to discover/validate a language's toolchain."""

    language: str
    kind: str                       # "compiler" | "runtime"
    binary_env: Optional[str]       # env var that overrides the binary (CC/CXX/...)
    candidates: Sequence[str]
    version_args: Sequence[str]
    arch_args: Optional[Sequence[str]]
    flags_env: Optional[str] = None  # CFLAGS / CXXFLAGS
    source_ext: str = ""
    sanity_source: str = ""
    needs_compile: bool = True       # compilers compile+run; runtimes just run


_SPECS: Dict[str, ToolchainSpec] = {
    "c": ToolchainSpec(
        language="c", kind="compiler", binary_env="CC",
        candidates=("cc", "gcc", "clang"),
        version_args=("--version",), arch_args=("-dumpmachine",), flags_env="CFLAGS",
        source_ext=".c", sanity_source="int main(void){return 0;}\n",
    ),
    "cpp": ToolchainSpec(
        language="cpp", kind="compiler", binary_env="CXX",
        candidates=("c++", "g++", "clang++"),
        version_args=("--version",), arch_args=("-dumpmachine",), flags_env="CXXFLAGS",
        source_ext=".cpp", sanity_source="int main(){return 0;}\n",
    ),
    "rust": ToolchainSpec(
        language="rust", kind="compiler", binary_env="RUSTC",
        candidates=("rustc",),
        version_args=("--version",), arch_args=("-vV",), flags_env=None,
        source_ext=".rs", sanity_source="fn main(){}\n",
    ),
    "fortran": ToolchainSpec(
        language="fortran", kind="compiler", binary_env="FC",
        candidates=("gfortran", "flang"),
        version_args=("--version",), arch_args=("-dumpmachine",), flags_env="FFLAGS",
        source_ext=".f90", sanity_source="program t\nend program t\n",
    ),
    "python": ToolchainSpec(
        language="python", kind="runtime", binary_env="PYTHON",
        candidates=("python3", "python"),
        version_args=("--version",), arch_args=None, flags_env=None,
        source_ext=".py", sanity_source="print(0)\n", needs_compile=False,
    ),
}

# Linker candidates, probed independently of language.
_LINKER_CANDIDATES = ("ld", "ld.lld", "lld", "ld.gold", "mold", "link")


@dataclass
class Toolchain:
    language: str
    kind: str
    binary: str                     # the invoked name (may come from CC/CXX)
    path: str
    version: Optional[str] = None
    version_raw: Optional[str] = None
    target: Optional[str] = None    # architectural signature (e.g. x86_64-linux-gnu)
    extra_flags: List[str] = field(default_factory=list)   # from CC="gcc -m32"
    compile_flags: List[str] = field(default_factory=list)  # from CFLAGS/CXXFLAGS
    link_flags: List[str] = field(default_factory=list)     # from LDFLAGS
    sane: Optional[bool] = None
    sanity_detail: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class IntrospectionError(Exception):
    pass


class ToolchainIntrospector:
    def __init__(
        self,
        workspace: _PathLike = ".",
        *,
        cache_dir: Optional[_PathLike] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.env = dict(os.environ if env is None else env)
        self.cache_dir = Path(cache_dir) if cache_dir else self.workspace / ".aero" / "toolchain"
        self.timeout = timeout

    # -- low-level execution --------------------------------------------------
    def _run(self, cmd: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[float] = None):
        try:
            proc = subprocess.run(
                list(cmd),
                cwd=str(cwd) if cwd else None,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout or self.timeout,
                text=True,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except (subprocess.TimeoutExpired, OSError) as exc:
            return 1, "", str(exc)

    # -- binary resolution ----------------------------------------------------
    def _resolve_binary(self, spec: ToolchainSpec):
        """Return (binary_name, path, extra_flags) honouring env overrides."""
        override = self.env.get(spec.binary_env) if spec.binary_env else None
        if override:
            parts = shlex.split(override)
            if parts:
                name = parts[0]
                path = shutil.which(name) or (name if Path(name).is_file() else None)
                if path:
                    return name, path, parts[1:]
        for candidate in spec.candidates:
            path = shutil.which(candidate)
            if path:
                return candidate, path, []
        return None, None, []

    def _flags_from_env(self, var: Optional[str]) -> List[str]:
        if not var:
            return []
        return shlex.split(self.env.get(var, ""))

    # -- version / architecture extraction ------------------------------------
    def _extract_version(self, path: str, args: Sequence[str]):
        rc, out, err = self._run([path, *args])
        text = (out or err).strip()
        first_line = text.splitlines()[0] if text else ""
        match = _VERSION_RE.search(first_line) or _VERSION_RE.search(text)
        return (match.group(1) if match else None), first_line

    def _extract_arch(self, spec: ToolchainSpec, path: str) -> Optional[str]:
        if not spec.arch_args:
            return None
        rc, out, err = self._run([path, *spec.arch_args])
        if rc != 0:
            return None
        text = (out or err)
        if spec.language == "rust":
            for line in text.splitlines():
                if line.startswith("host:"):
                    return line.split(":", 1)[1].strip()
            return None
        return text.strip().splitlines()[0] if text.strip() else None

    # -- discovery ------------------------------------------------------------
    def discover(self, language: str) -> Optional[Toolchain]:
        spec = _SPECS.get(language)
        if spec is None:
            return None
        name, path, extra = self._resolve_binary(spec)
        if not path:
            return None
        version, version_raw = self._extract_version(path, spec.version_args)
        target = self._extract_arch(spec, path)
        return Toolchain(
            language=language,
            kind=spec.kind,
            binary=name,
            path=path,
            version=version,
            version_raw=version_raw,
            target=target,
            extra_flags=list(extra),
            compile_flags=self._flags_from_env(spec.flags_env),
            link_flags=self._flags_from_env("LDFLAGS"),
        )

    def discover_linker(self) -> Optional[Toolchain]:
        override = self.env.get("LD")
        candidates = ([override] if override else []) + list(_LINKER_CANDIDATES)
        for candidate in candidates:
            path = shutil.which(candidate)
            if path:
                version, version_raw = self._extract_version(path, ("--version",))
                return Toolchain(
                    language="", kind="linker", binary=candidate, path=path,
                    version=version, version_raw=version_raw,
                    link_flags=self._flags_from_env("LDFLAGS"),
                )
        return None

    def languages_from_extensions(self, extensions: Sequence[str]) -> List[str]:
        langs = []
        for ext in extensions:
            lang = LANGUAGE_BY_EXTENSION.get(ext.lower())
            if lang and lang in _SPECS and lang not in langs:
                langs.append(lang)
        return sorted(langs)

    def languages_from_registry(self, blueprint_path: _PathLike) -> List[str]:
        """Resolve the set of languages declared in ``[context_registry]``."""
        try:
            from src.blueprint import load_blueprint

            registry = load_blueprint(blueprint_path).context_registry
        except Exception:
            return []
        langs = []
        for entry in registry.values():
            lang = entry.language
            if lang in _SPECS and lang not in langs:
                langs.append(lang)
        return sorted(langs)

    def discover_all(self, languages: Sequence[str]) -> Dict[str, Toolchain]:
        found: Dict[str, Toolchain] = {}
        for lang in languages:
            tc = self.discover(lang)
            if tc:
                found[lang] = tc
        return found

    # -- sanity validation ----------------------------------------------------
    def sanity_check(self, toolchain: Toolchain) -> bool:
        """Build (and run) a minimal program to confirm the toolchain works."""
        spec = _SPECS.get(toolchain.language)
        if spec is None:
            toolchain.sane = False
            toolchain.sanity_detail = "no spec"
            return False

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / ("sanity" + spec.source_ext)
            src.write_text(spec.sanity_source, encoding="utf-8")

            if not spec.needs_compile:
                rc, out, err = self._run([toolchain.path, *toolchain.extra_flags, str(src)], cwd=tmp_path)
                ok = rc == 0
                toolchain.sane = ok
                toolchain.sanity_detail = (err or out).strip()[:200] if not ok else "ok"
                if ok:
                    self._cache_toolchain(toolchain)
                return ok

            out_bin = tmp_path / ("sanity.exe" if os.name == "nt" else "sanity.out")
            if toolchain.language == "rust":
                cmd = [toolchain.path, *toolchain.extra_flags, str(src), "-o", str(out_bin)]
                cmd += toolchain.link_flags
            else:
                cmd = [toolchain.path, *toolchain.extra_flags, *toolchain.compile_flags,
                       str(src), "-o", str(out_bin), *toolchain.link_flags]
            rc, out, err = self._run(cmd, cwd=tmp_path)
            if rc != 0:
                toolchain.sane = False
                toolchain.sanity_detail = (err or out).strip()[:200]
                return False

            run_rc, _, run_err = self._run([str(out_bin)], cwd=tmp_path)
            ok = run_rc == 0
            toolchain.sane = ok
            toolchain.sanity_detail = "ok" if ok else (run_err.strip()[:200] or f"exit {run_rc}")
            if ok:
                self._cache_toolchain(toolchain)
            return ok

    # -- local cache ----------------------------------------------------------
    def _cache_path(self, language: str) -> Path:
        return self.cache_dir / f"{language or 'linker'}.json"

    def _cache_toolchain(self, toolchain: Toolchain) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(toolchain.language).write_text(
            json.dumps(toolchain.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def cached_toolchain(self, language: str) -> Optional[Toolchain]:
        path = self._cache_path(language)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Toolchain(**data)

    # -- high-level entry point ----------------------------------------------
    def introspect(
        self,
        languages: Optional[Sequence[str]] = None,
        *,
        blueprint_path: Optional[_PathLike] = None,
        validate: bool = True,
    ) -> Dict[str, Toolchain]:
        """Discover (and optionally validate) toolchains for the given languages.

        If ``languages`` is omitted, they are resolved from the blueprint's
        ``[context_registry]`` (when ``blueprint_path`` is given) or default to
        every spec.
        """
        if languages is None:
            if blueprint_path is not None:
                languages = self.languages_from_registry(blueprint_path) or list(_SPECS)
            else:
                languages = list(_SPECS)

        result = self.discover_all(languages)
        if validate:
            for tc in result.values():
                self.sanity_check(tc)
        return result
