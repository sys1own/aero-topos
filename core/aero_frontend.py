"""Normalize Python source into the translator's UAST schema.

The :mod:`core.translator` frontend consumes a small, normalized UAST dialect
(``module`` / ``function_declaration`` / ``binding`` / ``reference`` /
``literal`` / ``if`` / ``call``).  This adapter lowers real Python source --
via the stdlib :mod:`ast` -- into that dialect so ``.py`` files can be compiled
straight to Aero-Calculus HIN graphs.

The lowering is intentionally a representative subset: it captures functions,
single-target assignments, name references, literals, ``if`` expressions and
calls -- enough to exercise the full compile -> verify -> reduce -> serialize
pipeline end to end.  Constructs outside the subset are skipped gracefully.
"""

from __future__ import annotations

import ast
from typing import List, Optional


def python_source_to_uast(source: str) -> dict:
    """Parse Python ``source`` and return a normalized UAST ``module`` dict."""
    tree = ast.parse(source)
    children: List[dict] = []
    for stmt in tree.body:
        node = _lower_stmt(stmt)
        if node is not None:
            children.append(node)
    return {"type": "module", "children": children}


# ---------------------------------------------------------------------------
# statements
# ---------------------------------------------------------------------------
def _lower_stmt(stmt: ast.stmt) -> Optional[dict]:
    if isinstance(stmt, ast.FunctionDef):
        params = [a.arg for a in stmt.args.args]
        body = [n for n in (_lower_stmt(s) for s in stmt.body) if n is not None]
        return {
            "type": "function_declaration",
            "name": stmt.name,
            "params": params,
            "param": params[0] if params else None,
            "body": body,
        }
    if isinstance(stmt, ast.Assign) and stmt.targets:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return {
                "type": "binding",
                "name": target.id,
                "value": _lower_expr(stmt.value),
            }
    if isinstance(stmt, ast.Return):
        # A return statement surfaces its value as the block's result.
        return _lower_expr(stmt.value) if stmt.value is not None else None
    if isinstance(stmt, ast.If):
        return _lower_if(stmt)
    if isinstance(stmt, ast.Expr):
        return _lower_expr(stmt.value)
    return None


def _lower_if(stmt: ast.If) -> dict:
    then_body = [n for n in (_lower_stmt(s) for s in stmt.body) if n is not None]
    else_body = [n for n in (_lower_stmt(s) for s in stmt.orelse) if n is not None]
    return {
        "type": "if",
        "condition": _lower_expr(stmt.test),
        "then": then_body[-1] if then_body else None,
        "else": else_body[-1] if else_body else None,
    }


# ---------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------
def _lower_expr(expr: Optional[ast.expr]) -> Optional[dict]:
    if expr is None:
        return None
    if isinstance(expr, ast.Constant):
        return {"type": "literal", "value": expr.value}
    if isinstance(expr, ast.Name):
        return {"type": "reference", "name": expr.id}
    if isinstance(expr, ast.Call):
        arg = expr.args[0] if expr.args else None
        return {
            "type": "call",
            "function": _lower_expr(expr.func),
            "argument": _lower_expr(arg),
        }
    if isinstance(expr, ast.IfExp):
        return {
            "type": "if",
            "condition": _lower_expr(expr.test),
            "then": _lower_expr(expr.body),
            "else": _lower_expr(expr.orelse),
        }
    if isinstance(expr, ast.BinOp):
        # Model a binary op as a call consuming both operands (left then right);
        # the right operand becomes a second reference fork where needed.
        return {
            "type": "call",
            "function": _lower_expr(expr.left),
            "argument": _lower_expr(expr.right),
        }
    if isinstance(expr, ast.Attribute):
        return _lower_expr(expr.value)
    return None


__all__ = ["python_source_to_uast"]
