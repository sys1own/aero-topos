# -*- coding: utf-8 -*-
"""
Static import pruning — lightweight, AST-driven dead-import elimination.

When :class:`src.scaffold.decomposition.ModularDecomposer` splits a monolith into
per-file modules, it duplicates the monolith's entire top-level import block into
every generated file.  Most of those imports are dead weight in any single file.

:func:`prune_dead_imports` performs a pure-``ast`` analysis of one module and
removes the imports whose bound names are never referenced in that module's body:

1. Collect every name bound by ``ast.Import`` / ``ast.ImportFrom`` (honouring
   ``as`` aliases; ``import os.path`` binds ``os``; ``from sys import argv`` binds
   ``argv``).
2. Walk the rest of the module and collect every used token from ``ast.Name``
   nodes (attribute roots like ``os.path`` surface as ``Name('os')``).
3. Intersect: an import alias absent from the use map is dead and is stripped.

Defensive safeguards keep the pass from breaking dynamic code:

* ``from __future__`` imports and ``*`` star-imports are never pruned.
* A bound name that also appears as an identifier *inside a string literal* is
  treated as used (it may be resolved dynamically) and kept.
* If the module touches ``sys.modules`` / ``importlib.import_module`` / builtins
  like ``__import__`` / ``eval`` / ``exec`` / ``globals`` / ``vars``, pruning is
  suppressed for the whole module — dynamic lookups can reference *any* import by
  string, so it is unsafe to remove any of them.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Sequence, Set

# Word-like tokens inside string literals (for the dynamic-lookup safeguard).
_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")

# Builtins whose presence implies string-driven, dynamic name/module resolution.
_DYNAMIC_SENTINELS = frozenset({"__import__", "eval", "exec", "globals", "vars"})


@dataclass
class PruneOutcome:
    """Result of a single-module import-pruning pass."""

    kept_imports: List[ast.stmt] = field(default_factory=list)
    pruned: List[str] = field(default_factory=list)
    skipped_dynamic: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.pruned)


def _bound_name(alias: ast.alias) -> str:
    """The local name an ``alias`` binds into the module namespace."""
    if alias.asname:
        return alias.asname
    return alias.name.split(".")[0]


def _has_dynamic_lookup(tree: ast.AST) -> bool:
    """True when the module performs string-driven import/name resolution."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            # sys.modules[...]
            if (
                node.attr == "modules"
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
            ):
                return True
            # importlib.import_module(...)
            if node.attr == "import_module":
                return True
        elif isinstance(node, ast.Name) and node.id in _DYNAMIC_SENTINELS:
            return True
    return False


def _used_names(tree: ast.AST) -> Set[str]:
    """Every name referenced via ``ast.Name`` anywhere in the tree."""
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _string_tokens(tree: ast.AST) -> Set[str]:
    """Identifier-like tokens found inside string literals (safeguard set)."""
    tokens: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            tokens.update(_IDENTIFIER_RE.findall(node.value))
    return tokens


def prune_dead_imports(module_ast: ast.Module) -> PruneOutcome:
    """Strip imports whose bound names are unused in ``module_ast``.

    Returns a :class:`PruneOutcome` carrying the import statements to keep (with
    unused aliases removed, possibly dropping a statement entirely) and the list
    of pruned binding names.  The input AST is never mutated.
    """
    body: Sequence[ast.stmt] = getattr(module_ast, "body", []) or []
    top_imports = [n for n in body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not top_imports:
        return PruneOutcome(kept_imports=[], pruned=[], skipped_dynamic=False)

    # Dynamic-lookup safeguard: keep everything, untouched.
    if _has_dynamic_lookup(module_ast):
        return PruneOutcome(kept_imports=list(top_imports), pruned=[], skipped_dynamic=True)

    safe_used = _used_names(module_ast) | _string_tokens(module_ast)

    kept: List[ast.stmt] = []
    pruned: List[str] = []

    for node in top_imports:
        # Structural / opaque imports are always preserved.
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            kept.append(node)
            continue
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            kept.append(node)
            continue

        keep_aliases = [a for a in node.names if _bound_name(a) in safe_used]
        dead_aliases = [a for a in node.names if _bound_name(a) not in safe_used]
        pruned.extend(_bound_name(a) for a in dead_aliases)

        if not keep_aliases:
            continue  # whole statement is dead — drop it
        if len(keep_aliases) == len(node.names):
            kept.append(node)  # nothing dead on this line — keep verbatim
            continue

        # Partially-dead line: rebuild it with only the used aliases.
        if isinstance(node, ast.Import):
            rebuilt: ast.stmt = ast.Import(names=keep_aliases)
        else:
            rebuilt = ast.ImportFrom(module=node.module, names=keep_aliases, level=node.level)
        ast.copy_location(rebuilt, node)
        kept.append(rebuilt)

    return PruneOutcome(kept_imports=kept, pruned=pruned, skipped_dynamic=False)


def render_imports(nodes: Sequence[ast.stmt]) -> List[str]:
    """Render kept import statements back to canonical one-line source."""
    return [ast.unparse(node) for node in nodes]
