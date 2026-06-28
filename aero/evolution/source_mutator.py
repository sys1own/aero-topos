"""Safe, reversible source mutation for the self-evolution loop.

The mutator applies small, syntactically-safe edits to engine source files so
the evolution loop can test whether structural changes improve fitness.  Every
edit is:

* **non-destructive** -- it only inserts lines, never drops existing code;
* **idempotent** -- a marker prevents re-applying the same rule twice;
* **reversible** -- :meth:`rollback` restores the exact pre-mutation bytes.

(The historical implementation truncated files at the first function when
adding a docstring; that data-loss bug is fixed here.)
"""

import os
import glob
import random
from typing import List, Dict, Any

_STUB_MARKER = "# Aero Future Mutation"
_DOCSTRING_MARKER = "Aero Future Docstring"
_IMPORT_MARKER = "# Aero Future Import"


class SourceMutator:
    def __init__(self, target_files: List[str], rules: List[str], mutation_rate: float = 0.5):
        self.target_files = target_files
        self.rules = rules
        self.mutation_rate = mutation_rate
        self._backups: Dict[str, str] = {}

    # -- individual rules (each returns the new source, or None if no-op) ---

    def _rule_insert_function_stub(self, source: str) -> str:
        if _STUB_MARKER in source:
            return source
        stub = (
            f"\n\n{_STUB_MARKER}\n"
            "def _aero_future_identity(data):\n"
            '    """Evolution-inserted pass-through; benign and side-effect free."""\n'
            "    return data\n"
        )
        return source.rstrip() + "\n" + stub

    def _rule_insert_import(self, source: str) -> str:
        if _IMPORT_MARKER in source:
            return source
        lines = source.splitlines()
        # Insert after the existing leading import block / module docstring.
        insert_at = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) or stripped.startswith("#"):
                insert_at = i + 1
            elif stripped == "" and insert_at:
                insert_at = i + 1
            elif insert_at:
                break
        lines.insert(insert_at, f"{_IMPORT_MARKER}\nimport math as _aero_future_math  # noqa: F401")
        return "\n".join(lines) + ("\n" if source.endswith("\n") else "")

    def _rule_add_docstring(self, source: str) -> str:
        """Insert a docstring into the first function that lacks one.

        Crucially, this preserves *every* line of the file -- the docstring is
        inserted in place rather than truncating the remainder.
        """
        if _DOCSTRING_MARKER in source:
            return source
        lines = source.splitlines()
        out: List[str] = []
        inserted = False
        i = 0
        while i < len(lines):
            line = lines[i]
            out.append(line)
            if not inserted and line.strip().startswith("def ") and line.rstrip().endswith(":"):
                indent = len(line) - len(line.lstrip())
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                already_doc = next_line.strip().startswith(('"""', "'''", '"', "'"))
                if not already_doc:
                    out.append(" " * (indent + 4) + f'"""{_DOCSTRING_MARKER}."""')
                    inserted = True
            i += 1
        if not inserted:
            return source
        return "\n".join(out) + ("\n" if source.endswith("\n") else "")

    def _apply_rule(self, rule: str, source: str) -> str:
        if rule == "insert_function_stub":
            return self._rule_insert_function_stub(source)
        if rule == "insert_import":
            return self._rule_insert_import(source)
        if rule == "add_docstring":
            return self._rule_add_docstring(source)
        return source

    # -- orchestration -----------------------------------------------------

    def mutate(self, workspace: str) -> Dict[str, Any]:
        mutated_files: List[str] = []
        for pattern in self.target_files:
            full_pattern = os.path.join(workspace, pattern)
            for fpath in glob.glob(full_pattern, recursive=True):
                if not os.path.isfile(fpath):
                    continue
                if random.random() >= self.mutation_rate:
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8") as handle:
                        source = handle.read()
                    rule = random.choice(self.rules) if self.rules else None
                    if rule is None:
                        continue
                    new_source = self._apply_rule(rule, source)
                    if new_source == source:
                        continue
                    # Only snapshot the first time we touch a file this round.
                    self._backups.setdefault(fpath, source)
                    with open(fpath, "w", encoding="utf-8") as handle:
                        handle.write(new_source)
                    mutated_files.append(fpath)
                except OSError as exc:
                    print(f"Error mutating {fpath}: {exc}")
        return {"mutated_files": mutated_files}

    def rollback(self) -> None:
        for fpath, content in self._backups.items():
            try:
                with open(fpath, "w", encoding="utf-8") as handle:
                    handle.write(content)
            except OSError as exc:
                print(f"Error rolling back {fpath}: {exc}")
        self._backups.clear()
