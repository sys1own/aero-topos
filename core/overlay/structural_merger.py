"""Language-agnostic, two-tiered structural 3-way AST merge engine.

Replaces line-by-line unified-diff overlays with a structural merge built on the
universal Tree-sitter engine, implementing the ast-grep / Mergiraf ("Weave")
concepts:

* **Tier 1 -- coarse-grained semantic entity merge.**  Files are split into
  semantic entities (imports, classes, functions, ...) separated by layout
  blocks.  Each entity gets an identity tuple ``(scope, construct_type, name)``.
  An entity changed only in *Left* (the user's manual edit) is applied cleanly
  onto *Right* (the freshly generated output); an entity changed in both is
  routed to Tier 2.  Imports are treated commutatively (order-independent) so
  re-ordering never causes a false conflict.
* **Tier 2 -- fine-grained node merge.**  Conflicting entities are aligned down
  to Tree-sitter named children via identity signatures (top-down) and content
  hashes; non-overlapping concurrent edits are woven together by splicing the
  user's child-level changes onto the Right frame, while overlapping edits that
  target the same node along conflicting paths are flagged as conflicts.  PCS
  (Parent-Child-Successor) triples are provided for anomaly detection.
* **Reconstruction.**  The Right layout (interstitial spacing between entities)
  is re-embedded; multi-line leaf comments are kept on their own lines.
* **Safety verification.**  Before a merge is accepted it must parse cleanly,
  stay structurally sound (no duplicated entity ids), and -- when a build
  function is supplied -- compile in isolation.  On failure the change is
  rejected, the original file restored, and the collision flagged in
  ``blueprint.aero``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from core.parser.universal import CANONICAL_KIND, _load_language, detect_language, parse_source

_PathLike = Union[str, Path]

_COMMENT_TYPES = {"comment", "line_comment", "block_comment", "doc_comment"}

EntityId = Tuple[str, str, str]  # (scope, construct_type, name)


# ---------------------------------------------------------------------------
# Tree-sitter helpers
# ---------------------------------------------------------------------------
def _parse(text: str, language: str):
    tree = __import__("tree_sitter").Parser(_load_language(language)).parse(text.encode("utf-8"))
    return tree


def _node_text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _entity_name(node, src: bytes) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(name_node, src)
    queue = list(node.children)
    while queue:
        n = queue.pop(0)
        if n.type in ("identifier", "type_identifier", "field_identifier"):
            return _node_text(n, src)
        queue.extend(n.children)
    return ""


def _normalize_import(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _signature(node, src: bytes) -> str:
    """Identity signature for alignment: type + name, or type + content hash."""
    name = None
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        name = _node_text(name_node, src)
    if name:
        return f"{node.type}#{name}"
    if not node.named_children or node.type in _COMMENT_TYPES:
        return f"{node.type}@{_content_hash(_node_text(node, src))}"
    return f"{node.type}@{_content_hash(_node_text(node, src))}"


# ---------------------------------------------------------------------------
# Tier 1 entity model
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    id: EntityId
    construct: str          # import | function | type | comment | other
    name: str
    text: str
    start_byte: int
    end_byte: int


@dataclass
class EntityFile:
    language: str
    source: str
    entities: List[Entity]
    prefix: str             # layout before the first entity
    gaps: List[str]         # gap[i] = layout after entity i (last = trailing)

    def by_id(self) -> Dict[EntityId, Entity]:
        return {e.id: e for e in self.entities}


def _classify(node, language: str, src: bytes) -> Tuple[str, str]:
    kind = CANONICAL_KIND.get(language, {}).get(node.type)
    if kind == "import_declaration":
        return "import", _normalize_import(_node_text(node, src))
    if kind == "function_declaration":
        return "function", _entity_name(node, src)
    if kind == "type_declaration":
        return "type", _entity_name(node, src)
    if node.type in _COMMENT_TYPES:
        return "comment", _content_hash(_node_text(node, src))
    return "other", _content_hash(_node_text(node, src))


def extract_entities(text: str, language: str, scope: str = "") -> EntityFile:
    """Split *text* into top-level semantic entities with layout captured."""
    src = text.encode("utf-8")
    root = _parse(text, language).root_node
    top = list(root.named_children)

    entities: List[Entity] = []
    for node in top:
        construct, name = _classify(node, language, src)
        entities.append(Entity(
            id=(scope, construct, name),
            construct=construct,
            name=name,
            text=_node_text(node, src),
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        ))

    if not entities:
        return EntityFile(language, text, [], prefix=text, gaps=[])

    prefix = text[: _byte_to_char(text, entities[0].start_byte)]
    gaps: List[str] = []
    for i, ent in enumerate(entities):
        start_char = _byte_to_char(text, ent.end_byte)
        if i + 1 < len(entities):
            end_char = _byte_to_char(text, entities[i + 1].start_byte)
            gaps.append(text[start_char:end_char])
        else:
            gaps.append(text[start_char:])
    return EntityFile(language, text, entities, prefix=prefix, gaps=gaps)


def _byte_to_char(text: str, byte_offset: int) -> int:
    return len(text.encode("utf-8")[:byte_offset].decode("utf-8", "replace"))


# ---------------------------------------------------------------------------
# PCS (Parent-Child-Successor) triples
# ---------------------------------------------------------------------------
def pcs_triples(text: str, language: str) -> List[Tuple[str, str, str]]:
    """Decompose the tree into Parent-Child-Successor triples by signature."""
    src = text.encode("utf-8")
    root = _parse(text, language).root_node
    triples: List[Tuple[str, str, str]] = []

    def visit(node):
        kids = list(node.named_children)
        psig = _signature(node, src)
        prev = "⊢"  # start sentinel
        for kid in kids:
            ksig = _signature(kid, src)
            triples.append((psig, prev, ksig))
            prev = ksig
        triples.append((psig, prev, "⊣"))  # end sentinel
        for kid in kids:
            visit(kid)

    visit(root)
    return triples


def detect_pcs_conflicts(base: str, left: str, right: str, language: str) -> List[Tuple[str, str, str]]:
    """Flag PCS triples whose (parent, child) is given conflicting successors."""
    b = set(pcs_triples(base, language))
    left_new = set(pcs_triples(left, language)) - b
    right_new = set(pcs_triples(right, language)) - b
    left_succ = {(p, c): s for (p, c, s) in left_new}
    conflicts = []
    for (p, c, s) in right_new:
        if (p, c) in left_succ and left_succ[(p, c)] != s:
            conflicts.append((p, c, s))
    return conflicts


# ---------------------------------------------------------------------------
# Tier 2: fine-grained node merge (splice user edits onto the Right frame)
# ---------------------------------------------------------------------------
def _named_children(node):
    return list(node.named_children)


def _align_sig(node, src: bytes) -> str:
    """Content-independent structural identity for *alignment* (not equality).

    Named constructs align by (type, name); everything else aligns by type, so
    same-role nodes (a function ``block``, the k-th ``expression_statement``)
    pair up across versions and content differences are resolved by recursion.
    """
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return f"{node.type}#{_node_text(name_node, src)}"
    return node.type


def _match_map(base_sigs: Sequence[str], other_sigs: Sequence[str]) -> Dict[int, int]:
    import difflib

    sm = difflib.SequenceMatcher(a=list(base_sigs), b=list(other_sigs), autojunk=False)
    mapping: Dict[int, int] = {}
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    return mapping


def _align(bk, ok, sb: bytes, so: bytes) -> Dict[int, int]:
    """Align base children to other children: exact-content anchors first, then
    match leftover (modified) nodes by type within the anchored window.

    Returns ``{base_index: other_index}``.  This two-pass scheme prevents the
    ambiguity that pure type-sequence matching suffers from when statements of
    the same type are inserted/removed.
    """
    b_keys = [f"{c.type}@{_content_hash(_node_text(c, sb))}" for c in bk]
    o_keys = [f"{c.type}@{_content_hash(_node_text(c, so))}" for c in ok]
    result = _match_map(b_keys, o_keys)  # exact-content anchors
    used = set(result.values())

    for bi in range(len(bk)):
        if bi in result:
            continue
        lo = -1
        for k in range(bi - 1, -1, -1):
            if k in result:
                lo = result[k]
                break
        hi = len(ok)
        for k in range(bi + 1, len(bk)):
            if k in result:
                hi = result[k]
                break
        for oj in range(lo + 1, hi):
            if oj in used:
                continue
            if ok[oj].type == bk[bi].type:
                result[bi] = oj
                used.add(oj)
                break
    return result


def _merge_entity(base_text: str, left_text: str, right_text: str, language: str,
                  conflicts: List[str], depth: int = 0) -> str:
    """3-way merge two concurrently-edited entity bodies; weave onto Right."""
    if left_text == base_text:
        return right_text
    if right_text == base_text:
        return left_text
    if left_text == right_text:
        return left_text
    if depth > 8:
        conflicts.append("max merge depth exceeded")
        return right_text

    b_root = _parse(base_text, language).root_node
    l_root = _parse(left_text, language).root_node
    r_root = _parse(right_text, language).root_node
    # Descend through single-child wrappers (module -> the actual entity node).
    b_node = b_root.named_children[0] if len(b_root.named_children) == 1 else b_root
    l_node = l_root.named_children[0] if len(l_root.named_children) == 1 else l_root
    r_node = r_root.named_children[0] if len(r_root.named_children) == 1 else r_root

    return _merge_node(b_node, l_node, r_node,
                       base_text.encode(), left_text.encode(), right_text.encode(),
                       conflicts, depth)


def _merge_node(b, l, r, sb: bytes, sl: bytes, sr: bytes,
                conflicts: List[str], depth: int) -> str:
    tb, tl, tr = _node_text(b, sb), _node_text(l, sl), _node_text(r, sr)
    if tl == tb:
        return tr
    if tr == tb:
        return tl
    if tl == tr:
        return tl

    bk, lk, rk = _named_children(b), _named_children(l), _named_children(r)
    if not bk or not lk or not rk:
        conflicts.append(f"leaf-level conflict in <{b.type}>")
        return tr  # keep generated; safety layer decides

    # Structural alignment: map base children onto left and right children
    # (exact-content anchors first, then modified nodes by type).
    ml = _align(bk, lk, sb, sl)   # base index -> left index
    mr = _align(bk, rk, sb, sr)   # base index -> right index
    matched_left = set(ml.values())

    edits: List[Tuple[int, int, str]] = []   # (start, end, replacement) in right bytes
    for i, bc in enumerate(bk):
        lj = ml.get(i)
        rj = mr.get(i)
        left_child = lk[lj] if lj is not None else None
        right_child = rk[rj] if rj is not None else None
        base_child_text = _node_text(bc, sb)
        left_text = _node_text(left_child, sl) if left_child is not None else None
        right_text = _node_text(right_child, sr) if right_child is not None else None

        if left_text is None:  # user deleted this child
            if right_child is not None and right_text == base_child_text:
                edits.append((right_child.start_byte, right_child.end_byte, ""))
            elif right_child is not None and right_text != base_child_text:
                conflicts.append(f"delete/modify conflict on <{bc.type}>")
            continue

        if left_text == base_child_text:
            continue  # user did not touch this child; keep right's version

        if right_child is None:
            conflicts.append(f"modify/delete conflict on <{bc.type}>")
            continue
        if right_text == base_child_text:
            edits.append((right_child.start_byte, right_child.end_byte, left_text))  # user-only
        elif left_text == right_text:
            continue  # both made the identical change
        else:
            merged = _merge_node(bc, left_child, right_child, sb, sl, sr, conflicts, depth + 1)
            edits.append((right_child.start_byte, right_child.end_byte, merged))

    # Children the user added (left indices not matched to any base child) and
    # that the generator did not also add.
    right_texts = {_node_text(c, sr) for c in rk}
    for j, lc in enumerate(lk):
        if j in matched_left:
            continue
        added = _node_text(lc, sl)
        if added in right_texts:
            continue  # both added the same thing
        anchor = _insertion_anchor_left(j, ml, mr, rk)
        indent = _indent_of(sr, anchor) if anchor is not None else ""
        insert_at = anchor.end_byte if anchor is not None else (rk[0].start_byte if rk else r.end_byte)
        edits.append((insert_at, insert_at, "\n" + indent + added))

    return _apply_byte_edits(sr, r, edits)


def _insertion_anchor_left(left_idx: int, ml: Dict[int, int], mr: Dict[int, int], rk):
    """Right-side node after which a user-added left child should be inserted."""
    inverse_ml = {lj: bi for bi, lj in ml.items()}
    for k in range(left_idx - 1, -1, -1):
        base_i = inverse_ml.get(k)
        if base_i is not None and base_i in mr:
            return rk[mr[base_i]]
    return None


def _indent_of(src: bytes, node) -> str:
    line_start = src.rfind(b"\n", 0, node.start_byte) + 1
    raw = src[line_start:node.start_byte]
    return re.match(rb"[ \t]*", raw).group(0).decode("utf-8", "replace")


def _apply_byte_edits(src: bytes, frame_node, edits: List[Tuple[int, int, str]]) -> str:
    """Apply edits (absolute byte coords, within frame_node) and return only the
    frame node's resulting text."""
    text = src
    for start, end, repl in sorted(edits, key=lambda e: e[0], reverse=True):
        text = text[:start] + repl.encode("utf-8") + text[end:]
    delta = sum(len(repl.encode("utf-8")) - (end - start) for start, end, repl in edits)
    return text[frame_node.start_byte: frame_node.end_byte + delta].decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Tier 1 orchestration + reconstruction
# ---------------------------------------------------------------------------
@dataclass
class MergeResult:
    text: str
    conflicts: List[str] = field(default_factory=list)
    tier1_clean: int = 0     # entities resolved purely at Tier 1
    tier2_merged: int = 0    # entities woven at Tier 2
    success: bool = True

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)


def three_way_merge(base: str, left: str, right: str, language: str) -> MergeResult:
    """Tier-1 entity merge, routing concurrent edits to the Tier-2 node merge."""
    base_f = extract_entities(base, language)
    left_f = extract_entities(left, language)
    right_f = extract_entities(right, language)

    b_by, l_by, r_by = base_f.by_id(), left_f.by_id(), right_f.by_id()
    all_ids = set(b_by) | set(l_by) | set(r_by)

    result = MergeResult(text="")
    decisions: Dict[EntityId, Optional[str]] = {}  # id -> text or None(=delete)

    for eid in all_ids:
        b = b_by.get(eid)
        l = l_by.get(eid)
        r = r_by.get(eid)

        if b and l and r:
            if l.text == b.text and r.text == b.text:
                decisions[eid] = r.text
            elif l.text != b.text and r.text == b.text:
                decisions[eid] = l.text; result.tier1_clean += 1   # user-only change
            elif l.text == b.text and r.text != b.text:
                decisions[eid] = r.text
            elif l.text == r.text:
                decisions[eid] = l.text
            else:  # concurrent edit -> Tier 2
                sub_conflicts: List[str] = []
                merged = _merge_entity(b.text, l.text, r.text, language, sub_conflicts)
                if sub_conflicts:
                    result.conflicts.extend(f"{eid[1]} '{eid[2]}': {c}" for c in sub_conflicts)
                    decisions[eid] = r.text
                else:
                    decisions[eid] = merged; result.tier2_merged += 1
        elif b and l and not r:   # right deleted
            decisions[eid] = None if l.text == b.text else _conflict(result, eid, "modify/delete", l.text)
        elif b and r and not l:   # left deleted
            if r.text == b.text:
                decisions[eid] = None
            else:
                result.conflicts.append(f"{eid[1]} '{eid[2]}': delete/modify")
                decisions[eid] = r.text
        elif not b and l and r:   # added by both
            decisions[eid] = l.text if l.text == r.text else _merge_added(result, eid, l, r, language)
        elif not b and l and not r:
            decisions[eid] = l.text   # user-added
        elif not b and r and not l:
            decisions[eid] = r.text   # generator-added
        elif b and not l and not r:
            decisions[eid] = None     # deleted by both

    result.text = _reconstruct(base_f, left_f, right_f, decisions)
    result.success = not result.has_conflicts
    return result


def _conflict(result: MergeResult, eid: EntityId, kind: str, fallback: str) -> str:
    result.conflicts.append(f"{eid[1]} '{eid[2]}': {kind}")
    return fallback


def _merge_added(result: MergeResult, eid: EntityId, l: Entity, r: Entity, language: str) -> str:
    sub: List[str] = []
    merged = _merge_entity("", l.text, r.text, language, sub)
    if sub:
        result.conflicts.append(f"{eid[1]} '{eid[2]}': add/add")
        return r.text
    result.tier2_merged += 1
    return merged


def _reconstruct(base_f, left_f, right_f, decisions) -> str:
    """Re-embed Right's layout; weave user-only additions; dedup imports."""
    right_ids = {e.id for e in right_f.entities}
    base_ids = {e.id for e in base_f.entities}

    out: List[str] = [right_f.prefix] if right_f.entities else []

    # Track where the import block ends to place user-added imports sensibly.
    last_import_marker = len(out)

    for i, ent in enumerate(right_f.entities):
        decided = decisions.get(ent.id, ent.text)
        gap = right_f.gaps[i] if i < len(right_f.gaps) else "\n"
        if decided is None:
            continue
        out.append(decided)
        out.append(gap)
        if ent.construct == "import":
            last_import_marker = len(out)

    # User-only additions (in left, not in base, not in right): preserve them.
    import_adds: List[str] = []
    other_adds: List[str] = []
    for ent in left_f.entities:
        if ent.id in base_ids or ent.id in right_ids:
            continue
        if decisions.get(ent.id) is None:
            continue
        block = decisions.get(ent.id, ent.text)
        if ent.construct == "import":
            import_adds.append(block)
        else:
            other_adds.append(block)

    if import_adds:
        injected = "".join(b + "\n" for b in import_adds)
        out.insert(last_import_marker, injected)

    text = "".join(out)
    for block in other_adds:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + block + "\n"

    if not right_f.entities:
        # No Right entities at all: fall back to user's content.
        text = left_f.source
    return _split_multiline_comment_lines(text)


def _split_multiline_comment_lines(text: str) -> str:
    """Ensure trailing CRs are normalised so comment leaves stay line-aligned."""
    return text.replace("\r\n", "\n")


# ---------------------------------------------------------------------------
# Safety verification
# ---------------------------------------------------------------------------
@dataclass
class VerifyResult:
    ok: bool
    reason: Optional[str] = None
    diagnostics: List = field(default_factory=list)


def verify_merge(text: str, language: str, build_fn: Optional[Callable] = None,
                 path: Optional[_PathLike] = None) -> VerifyResult:
    """I(U_patched) and Verify(U_patched): parse, structure, optional compile."""
    uast = parse_source(text, language)
    flags = uast["metadata"]["parser_flags"]
    if flags.get("fallback"):
        return VerifyResult(False, f"parser fallback: {flags.get('fallback_reason')}")
    if flags.get("has_error") or flags.get("error_nodes") or flags.get("missing_nodes"):
        return VerifyResult(False, "patched source has syntax errors")

    # Structural soundness: no duplicated top-level entity ids (botched weave).
    ef = extract_entities(text, language)
    seen = set()
    for ent in ef.entities:
        if ent.construct in ("function", "type") and ent.id in seen:
            return VerifyResult(False, f"duplicate {ent.construct} '{ent.name}' after merge")
        seen.add(ent.id)

    if build_fn is not None and path is not None:
        diags = build_fn(Path(path))
        if diags:
            return VerifyResult(False, "patched module failed isolated compile", diags)
    return VerifyResult(True)


# ---------------------------------------------------------------------------
# Disk orchestrator with reject/restore/flag
# ---------------------------------------------------------------------------
@dataclass
class StructuralMergeOutcome:
    path: str
    accepted: bool
    merge: MergeResult
    verify: Optional[VerifyResult] = None
    reason: Optional[str] = None


class StructuralMerger:
    def __init__(self, workspace: _PathLike = ".") -> None:
        self.workspace = Path(workspace).resolve()

    def merge_file(
        self,
        path: _PathLike,
        base_text: str,
        right_text: str,
        *,
        language: Optional[str] = None,
        build_fn: Optional[Callable] = None,
        blueprint_path: Optional[_PathLike] = None,
    ) -> StructuralMergeOutcome:
        """Merge ``base`` (pristine), the file's current text (Left) and ``right``.

        On success the merged text is written; on conflict/verification failure
        the original file is left untouched and the collision is flagged in
        ``blueprint.aero``.
        """
        file_path = Path(path).resolve()
        language = language or detect_language(file_path)
        if language is None:
            raise ValueError(f"Unsupported file extension: {file_path.suffix!r}")

        left_text = file_path.read_text(encoding="utf-8")
        merge = three_way_merge(base_text, left_text, right_text, language)

        if merge.has_conflicts:
            self._flag(blueprint_path, file_path, "; ".join(merge.conflicts[:5]))
            return StructuralMergeOutcome(file_path.as_posix(), False, merge,
                                          reason="structural conflict")

        # Write to a temp sibling for isolated verification, then swap on success.
        verify = verify_merge(merge.text, language, build_fn=None)
        if not verify.ok:
            self._flag(blueprint_path, file_path, verify.reason or "verification failed")
            return StructuralMergeOutcome(file_path.as_posix(), False, merge, verify,
                                          reason=verify.reason)

        if build_fn is not None:
            tmp = file_path.with_suffix(file_path.suffix + ".merge_check")
            tmp.write_text(merge.text, encoding="utf-8")
            try:
                cverify = verify_merge(merge.text, language, build_fn=build_fn, path=tmp)
            finally:
                tmp.unlink(missing_ok=True)
            if not cverify.ok:
                self._flag(blueprint_path, file_path, cverify.reason or "compile failed")
                return StructuralMergeOutcome(file_path.as_posix(), False, merge, cverify,
                                              reason=cverify.reason)
            verify = cverify

        file_path.write_text(merge.text, encoding="utf-8")
        return StructuralMergeOutcome(file_path.as_posix(), True, merge, verify)

    def _flag(self, blueprint_path: Optional[_PathLike], file_path: Path, reason: str) -> None:
        if blueprint_path is None:
            return
        rel = self._rel(file_path)
        _flag_structural_collision(blueprint_path, rel, reason)

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.workspace).as_posix()
        except ValueError:
            return path.as_posix()


# ---------------------------------------------------------------------------
# blueprint.aero collision flagging
# ---------------------------------------------------------------------------
def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _strip_table(text: str, name: str) -> str:
    out: List[str] = []
    skip = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            skip = (stripped == f"[{name}]" or stripped.startswith(f"[{name}.")
                    or stripped.startswith(f"[[{name}"))
        if not skip:
            out.append(line)
    result = "\n".join(out).rstrip("\n")
    return result + "\n" if result else ""


def _flag_structural_collision(blueprint_path: _PathLike, module: str, reason: str) -> None:
    bp_path = Path(blueprint_path)
    text = bp_path.read_text(encoding="utf-8") if bp_path.is_file() else ""
    existing: Dict[str, str] = {}
    try:
        from src.blueprint.loader import _toml

        if text:
            existing = {k: v for k, v in (_toml.loads(text).get("merge_collisions") or {}).items()
                        if isinstance(v, str)}
    except Exception:
        pass
    existing[module] = f"rejected: {reason}"
    text = _strip_table(text, "merge_collisions")
    lines = ["[merge_collisions]"] + [f"{_toml_str(k)} = {_toml_str(existing[k])}" for k in sorted(existing)]
    block = "\n".join(lines) + "\n"
    if text and not text.endswith("\n"):
        text += "\n"
    if text and not text.endswith("\n\n"):
        text += "\n"
    text += block
    bp_path.parent.mkdir(parents=True, exist_ok=True)
    bp_path.write_text(text, encoding="utf-8")
