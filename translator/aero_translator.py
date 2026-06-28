import ast
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("translator.aero_translator")
a = 25
b = 5

@dataclass
class c:
    op: str
    detail: str = ''
    label: str = ''
    family: str = 'translated'

@dataclass
class d:
    name: str
    source_file: str
    ops: list[c] = field(default_factory=list)
    translatable: bool = True
    reject_reason: str = ''

@dataclass
class e:
    source_file: str
    functions: list[d] = field(default_factory=list)
    recipe_body: str = ''
    node_count: int = 0

class f(ast.NodeVisitor):

    def __init__(o, p, q):
        o.p = p
        o.q = q
        o.r = []
        o.s = 0
        o.t = []

    def visit_For(o, n):
        o.s += 1
        u = ast.dump(n.target) if hasattr(n, 'target') else 'iter'
        v = 'compute' if o.s >= 2 else 'iterator'
        o.r.append(c(op='print', detail=f"text = '-- {v} | {o.p}: loop depth {o.s} over {o._safe_name(n.target)} --'", label=f'loop-depth-{o.s}', family=v))
        o.generic_visit(n)
        o.s -= 1
    visit_While = visit_For

    def visit_Call(o, n):
        w = o._call_name(n)
        if w in ('print', 'logging', 'log', 'console'):
            o.r.append(c(op='print', detail=f"text = '-- logger | {o.p}: log output via {w} --'", label='log-output', family='logger'))
        elif w in ('open', 'write', 'write_file', 'save', 'dump'):
            x = o._args_preview(n)
            o.r.append(c(op='call', detail=f"fn = write_file\nargs = 'build_sandbox/mesh_outputs/{o.p}_{w}.dat', 'translated'\nfamily = io", label=f'write-{w}', family='io'))
        elif w in ('append', 'extend', 'insert', 'push', 'sort', 'split', 'join', 'replace', 'strip', 'filter', 'map', 'reduce', 'find', 'index', 'count'):
            o.r.append(c(op='print', detail=f"text = '-- transform | {o.p}: array/string op {w} --'", label=f'transform-{w}', family='transform'))
        else:
            o.r.append(c(op='print', detail=f"text = '-- worker | {o.p}: call {w} --'", label=f'call-{w}', family='worker'))
        o.generic_visit(n)

    def visit_AugAssign(o, n):
        y = type(n.op).__name__
        o.r.append(c(op='print', detail=f"text = '-- compute | {o.p}: augmented assign ({y}) --'", label=f'augassign-{y}', family='compute'))
        o.generic_visit(n)

    def visit_Return(o, n):
        o.r.append(c(op='call', detail=f"fn = write_file\nargs = 'build_sandbox/mesh_outputs/{o.p}_return.dat', 'result'\nfamily = output", label='return-value', family='output'))
        o.generic_visit(n)

    def visit_FunctionDef(o, n):
        o.generic_visit(n)

    def _call_name(o, n):
        if isinstance(n.func, ast.Name):
            return n.func.id
        if isinstance(n.func, ast.Attribute):
            return n.func.attr
        return 'unknown'

    def _safe_name(o, n):
        if isinstance(n, ast.Name):
            return n.id
        return 'expr'

    def _args_preview(o, n):
        z = []
        for arg in n.args[:2]:
            if isinstance(arg, ast.Constant):
                z.append(repr(arg.value))
            elif isinstance(arg, ast.Name):
                z.append(arg.id)
        return ', '.join(z) if z else '...'

def translate_function(p, q, r):
    try:
        s = ast.parse(p, filename=r)
    except SyntaxError as e:
        return d(name=q, source_file=r, translatable=False, reject_reason=f'SyntaxError: {e}')
    t = None
    for n in ast.walk(s):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.name == q:
                t = n
                break
    if t is None:
        return d(name=q, source_file=r, translatable=False, reject_reason=f"Function '{q}' not found")
    u = f(q, r)
    u.visit(t)
    if not u.r:
        return d(name=q, source_file=r, translatable=False, reject_reason='No translatable constructs found')
    return d(name=q, source_file=r, ops=u.r, translatable=True)

def translate_file(v, w=None):
    with open(v, 'r', encoding='utf-8') as x:
        y = x.read()
    try:
        z = ast.parse(y, filename=v)
    except SyntaxError as exc:
        logger.debug("Skipping %s due to syntax error: %s", v, exc)
        return e(source_file=v)
    if w is None:
        w = [n.name for n in ast.walk(z) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    aa = e(source_file=v)
    for bb in w:
        cc = translate_function(y, bb, v)
        aa.functions.append(cc)
    return aa

def translated_to_recipe(dd, ee='', ff='build_sandbox/recipes'):
    if not ee:
        gg = os.path.splitext(os.path.basename(dd.source_file))[0]
        ee = f'translated_{gg}'
    hh = f'{ff}/{ee}.aeroc'
    ii = f'[project]\nname = {ee}\noutput = {hh}\n\n'
    jj = []
    kk = 0
    ll = None
    mm = {}
    jj.append(f'[task:init]\nop = print\ntext = "-- Initializing translated pipeline for {ee} --"\n')
    ll = 'init'
    kk += 1
    for nn in dd.functions:
        if not nn.translatable:
            continue
        for oo in nn.ops:
            if kk >= a - 1:
                break
            mm.setdefault(oo.family, 0)
            if mm[oo.family] >= b:
                continue
            mm[oo.family] += 1
            pp = f'node{kk}'
            qq = f'[task:{pp}]\nop = {oo.op}\n{oo.detail}\nneeds = {ll}\n'
            jj.append(qq)
            ll = pp
            kk += 1
    if kk < a and ll:
        pp = f'node{kk}'
        jj.append(f"[task:{pp}]\nop = call\nfn = write_file\nargs = 'build_sandbox/mesh_outputs/{ee}_final.dat', 'complete'\nneeds = {ll}\n")
        kk += 1
    rr = ii + '\n'.join(jj)
    dd.recipe_body = rr
    dd.node_count = kk
    return rr