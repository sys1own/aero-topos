# -*- coding: utf-8 -*-
"""
Root-cause analysis for Rust compilation failures.

The raw compiler output for a version mismatch -- e.g. calling ``neg_mut`` on a
type from a dependency whose API changed -- is cryptic: ``error[E0599]: no method
named `neg_mut` found for struct `Integer` in the current scope``.  Nothing in
that message tells the user *why* (a version mismatch) or *which version is
actually in use*.

:func:`analyze_rust_error` scans compiler ``stderr`` for these "method not found"
errors, attributes the receiver type to a dependency crate where possible, looks
up the **actual resolved version** from ``Cargo.lock`` (falling back to the
declared version), and produces concrete, human-readable suggestions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# error[E0599]: no method named `neg_mut` found for struct `Integer` in the current scope
# Also matches: "found for reference `&rug::Integer`", "found for type `Foo`", etc.
_NO_METHOD_RE = re.compile(
    r"no method named `(?P<method>[^`]+)` found for [\w ]+ `(?P<recv>[^`]+)`",
)
# error[E0432]/E0433: unresolved import / failed to resolve -- often a renamed or
# removed item across versions.
_UNRESOLVED_RE = re.compile(r"unresolved import `(?P<path>[^`]+)`")


@dataclass
class RustErrorDiagnosis:
    """A root-cause hypothesis for a Rust build failure."""

    kind: str  # "method_not_found" | "unresolved_import"
    method: str = ""
    receiver_type: str = ""
    crate: str = ""
    resolved_version: Optional[str] = None
    declared_version: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "method": self.method,
            "receiver_type": self.receiver_type,
            "crate": self.crate,
            "resolved_version": self.resolved_version,
            "declared_version": self.declared_version,
            "suggestions": list(self.suggestions),
        }

    def render(self) -> List[str]:
        """Render the diagnosis as a list of presentable lines."""
        lines: List[str] = []
        if self.kind == "method_not_found":
            head = f"method `{self.method}` not found"
            if self.receiver_type:
                head += f" on type `{self.receiver_type}`"
            lines.append(head)
        elif self.kind == "unresolved_import":
            lines.append(f"unresolved import `{self.receiver_type}`")
        lines.extend(self.suggestions)
        return lines


# ---------------------------------------------------------------------------
# Cargo.lock / version resolution
# ---------------------------------------------------------------------------


def read_locked_versions(crate_root: Optional[Path]) -> Dict[str, str]:
    """Read ``{crate: version}`` from a ``Cargo.lock`` at ``crate_root``."""
    if crate_root is None:
        return {}
    lock = Path(crate_root) / "Cargo.lock"
    if not lock.is_file():
        return {}
    try:
        import tomllib

        with open(lock, "rb") as fh:
            data = tomllib.load(fh)
    except Exception:  # noqa: BLE001 - malformed/locked file, best-effort only
        return {}
    versions: Dict[str, str] = {}
    for package in data.get("package", []) or []:
        name = package.get("name")
        version = package.get("version")
        if name and version:
            versions[str(name)] = str(version)
    return versions


def _normalise_receiver(recv: str) -> str:
    """Strip references / ``mut`` / generics from a receiver type string."""
    cleaned = recv.strip()
    cleaned = cleaned.lstrip("&")
    cleaned = re.sub(r"^\s*mut\s+", "", cleaned)
    # Drop generic parameters: Integer<...> -> Integer
    cleaned = re.sub(r"<.*$", "", cleaned)
    return cleaned.strip()


def _crate_from_type(receiver: str, dependencies: Sequence[str]) -> Optional[str]:
    """Best-effort: which dependency crate does this receiver type come from?"""
    receiver = _normalise_receiver(receiver)
    if "::" in receiver:
        head = receiver.split("::", 1)[0]
        # Cargo crate names use '-' but the path form uses '_'.
        candidates = {head, head.replace("_", "-")}
        for dep in dependencies:
            if dep in candidates or dep.replace("-", "_") == head:
                return dep
        return head  # a path-qualified type still names its crate
    # Bare type (e.g. `Integer`): if exactly one dependency is declared, blame it.
    deps = list(dependencies)
    if len(deps) == 1:
        return deps[0]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_rust_error(
    stderr: str,
    dependencies: Optional[Dict[str, object]] = None,
    crate_root: Optional[Path] = None,
) -> Optional[RustErrorDiagnosis]:
    """Analyse Rust ``stderr`` and return a root-cause diagnosis, or ``None``.

    ``dependencies`` is the target's declared ``{name: version}`` mapping (used
    to attribute a bare type to a crate and to report the declared version);
    ``crate_root`` is where ``Cargo.lock`` lives (for the *resolved* version).
    """
    if not stderr:
        return None
    declared = {str(k): str(v) for k, v in (dependencies or {}).items()}
    locked = read_locked_versions(crate_root)

    match = _NO_METHOD_RE.search(stderr)
    if match:
        method = match.group("method")
        receiver = _normalise_receiver(match.group("recv"))
        crate = _crate_from_type(match.group("recv"), list(declared.keys())) or ""
        resolved = locked.get(crate) or locked.get(crate.replace("-", "_"))
        declared_version = declared.get(crate) or declared.get(crate.replace("-", "_"))
        return RustErrorDiagnosis(
            kind="method_not_found",
            method=method,
            receiver_type=receiver,
            crate=crate,
            resolved_version=resolved,
            declared_version=declared_version,
            suggestions=_method_suggestions(method, receiver, crate, resolved, declared_version),
        )

    unresolved = _UNRESOLVED_RE.search(stderr)
    if unresolved:
        path = unresolved.group("path")
        crate = path.split("::", 1)[0] if "::" in path else path
        resolved = locked.get(crate) or locked.get(crate.replace("-", "_"))
        declared_version = declared.get(crate)
        return RustErrorDiagnosis(
            kind="unresolved_import",
            receiver_type=path,
            crate=crate,
            resolved_version=resolved,
            declared_version=declared_version,
            suggestions=_import_suggestions(path, crate, resolved, declared_version),
        )

    return None


def _version_phrase(crate: str, resolved: Optional[str], declared: Optional[str]) -> str:
    if resolved and declared and resolved != declared:
        return f"crate `{crate}` in use: {resolved} (resolved) — declared as \"{declared}\""
    if resolved:
        return f"crate `{crate}` in use: version {resolved} (from Cargo.lock)"
    if declared:
        return f"crate `{crate}` declared as \"{declared}\" (not yet resolved; no Cargo.lock)"
    return f"could not determine the version of crate `{crate}` in use"


def _method_suggestions(
    method: str, receiver: str, crate: str, resolved: Optional[str], declared: Optional[str]
) -> List[str]:
    suggestions = [
        f"likely cause: a version mismatch — `{method}` is not part of the API of "
        f"`{receiver or 'the receiver type'}` in the version currently being built.",
    ]
    if crate:
        suggestions.append(_version_phrase(crate, resolved, declared))
        suggestions.append(
            f"check whether `{method}` exists in that version of `{crate}`; if not, pin a "
            f"compatible version, e.g. cargo_dependencies = [\"{crate}=<version>\"] "
            f"(or a cargo.dependencies entry), then rebuild."
        )
    else:
        suggestions.append(
            "check the version of the dependency that defines this type against the API you expect; "
            "pin a compatible version via cargo_dependencies / cargo.dependencies."
        )
    suggestions.append("run `aero build --debug` to see the exact manifest, cargo command and versions in use.")
    return suggestions


def _import_suggestions(path: str, crate: str, resolved: Optional[str], declared: Optional[str]) -> List[str]:
    return [
        f"likely cause: `{path}` does not exist in the version of `{crate}` being built "
        "(items are often moved or renamed between versions).",
        _version_phrase(crate, resolved, declared),
        f"pin a compatible version of `{crate}` (cargo_dependencies / cargo.dependencies) and rebuild.",
    ]
