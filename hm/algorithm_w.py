"""Reference implementation: textbook Algorithm W with explicit substitutions.

This module shares nothing with `infer.py` except the AST and the prelude
signatures.  Types are immutable tuples, substitutions are dictionaries,
and generalisation computes the free variables of the whole environment,
exactly as in Damas and Milner (1982).  It is slower and has terser error
messages, and that is fine: its job is to be an independent oracle.  The
test suite checks that both algorithms agree, up to renaming of type
variables, on every hand written program and on thousands of random ones.
"""
from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, List, Tuple

from . import syntax as S
from .prelude import NATIVES, PRELUDE_SOURCE, parse_signature
from .types import TCon, TVar, show

# A type is ('v', name) or ('c', name, (args...)).
WType = tuple
Subst = Dict[str, WType]
Scheme = Tuple[FrozenSet[str], WType]

INT, BOOL, STRING, UNIT = ("c", "int", ()), ("c", "bool", ()), ("c", "string", ()), ("c", "unit", ())


def arrow(a: WType, b: WType) -> WType:
    return ("c", "->", (a, b))


def lst(a: WType) -> WType:
    return ("c", "list", (a,))


class WTypeError(Exception):
    pass


def apply(s: Subst, t: WType) -> WType:
    if t[0] == "v":
        r = s.get(t[1])
        return t if r is None else apply(s, r)
    if not t[2]:
        return t
    return ("c", t[1], tuple(apply(s, a) for a in t[2]))


def ftv(t: WType) -> FrozenSet[str]:
    if t[0] == "v":
        return frozenset([t[1]])
    out: FrozenSet[str] = frozenset()
    for a in t[2]:
        out |= ftv(a)
    return out


def ftv_env(s: Subst, env: Dict[str, Scheme]) -> FrozenSet[str]:
    out: FrozenSet[str] = frozenset()
    for qs, t in env.values():
        out |= ftv(apply(s, t)) - qs
    return out


class AlgorithmW:
    def __init__(self, with_prelude: bool = True):
        self._fresh = itertools.count()
        self.env: Dict[str, Scheme] = {}
        for name, (sig, _, _) in NATIVES.items():
            t = self._from_sig(parse_signature(sig))
            self.env[name] = (ftv(t), t)
        if with_prelude:
            self.check_program(S.parse_program(PRELUDE_SOURCE))

    def fresh(self) -> WType:
        return ("v", f"t{next(self._fresh)}")

    def _from_sig(self, n) -> WType:
        if n[0] == "var":
            return ("v", "sig_" + n[1])
        return ("c", n[1], tuple(self._from_sig(a) for a in n[2]))

    # -- unification: returns s extended with the most general unifier
    def unify(self, s: Subst, a: WType, b: WType) -> Subst:
        a, b = apply(s, a), apply(s, b)
        if a == b:
            return s
        if a[0] == "v":
            return self._bind(s, a[1], b)
        if b[0] == "v":
            return self._bind(s, b[1], a)
        if a[1] != b[1] or len(a[2]) != len(b[2]):
            raise WTypeError(f"cannot unify {a[1]} with {b[1]}")
        for x, y in zip(a[2], b[2]):
            s = self.unify(s, x, y)
        return s

    def _bind(self, s: Subst, v: str, t: WType) -> Subst:
        if v in ftv(t):
            raise WTypeError("occurs check failed")
        out = {k: apply({v: t}, x) for k, x in s.items()}   # keep s idempotent
        out[v] = t
        return out

    def instantiate(self, sch: Scheme) -> WType:
        qs, t = sch
        return apply({q: self.fresh() for q in qs}, t)

    def generalise(self, s: Subst, env: Dict[str, Scheme], t: WType) -> Scheme:
        t = apply(s, t)
        qs = ftv(t) - ftv_env(s, env)
        renaming = {q: self.fresh() for q in qs}
        t = apply(renaming, t)
        return frozenset(r[1] for r in renaming.values()), t

    # -- inference: threads a substitution, returns (s', t)
    def infer(self, s: Subst, env: Dict[str, Scheme], e: S.Expr) -> Tuple[Subst, WType]:
        if isinstance(e, S.Lit):
            v = e.value
            if v is None:
                return s, UNIT
            if isinstance(v, bool):
                return s, BOOL
            return s, (INT if isinstance(v, int) else STRING)
        if isinstance(e, S.Var):
            if e.name not in env:
                raise WTypeError(f"unbound {e.name}")
            return s, self.instantiate(env[e.name])
        if isinstance(e, S.Lam):
            a = self.fresh()
            inner = env if e.param == "_" else {**env, e.param: (frozenset(), a)}
            s, b = self.infer(s, inner, e.body)
            return s, arrow(a, b)
        if isinstance(e, S.App):
            s, f = self.infer(s, env, e.fn)
            s, x = self.infer(s, env, e.arg)
            r = self.fresh()
            s = self.unify(s, f, arrow(x, r))
            return s, r
        if isinstance(e, S.Let):
            s, sch = self.binding(s, env, e.name, e.rec, e.value)
            inner = env if e.name == "_" else {**env, e.name: sch}
            return self.infer(s, inner, e.body)
        if isinstance(e, S.If):
            s, c = self.infer(s, env, e.cond)
            s = self.unify(s, c, BOOL)
            s, a = self.infer(s, env, e.then)
            s, b = self.infer(s, env, e.other)
            return self.unify(s, a, b), a
        if isinstance(e, S.Tuple_):
            ts = []
            for x in e.items:
                s, t = self.infer(s, env, x)
                ts.append(t)
            return s, ("c", "*", tuple(ts))
        if isinstance(e, S.Nil):
            return s, lst(self.fresh())
        if isinstance(e, S.Cons):
            s, h = self.infer(s, env, e.head)
            s, t = self.infer(s, env, e.tail)
            return self.unify(s, t, lst(h)), lst(h)
        if isinstance(e, S.Seq):
            s, a = self.infer(s, env, e.first)
            s = self.unify(s, a, UNIT)
            return self.infer(s, env, e.second)
        if isinstance(e, S.Match):
            s, scrut = self.infer(s, env, e.scrutinee)
            result = self.fresh()
            for pat, body in e.arms:
                binds: Dict[str, Scheme] = {}
                s, pt = self.pattern(s, pat, binds)
                s = self.unify(s, pt, scrut)
                s, bt = self.infer(s, {**env, **binds}, body)
                s = self.unify(s, bt, result)
            return s, result
        raise TypeError(e)

    def pattern(self, s: Subst, p: S.Pat, binds: Dict[str, Scheme]) -> Tuple[Subst, WType]:
        if isinstance(p, S.PWild):
            return s, self.fresh()
        if isinstance(p, S.PVar):
            if p.name in binds:
                raise WTypeError(f"{p.name} bound twice")
            t = self.fresh()
            binds[p.name] = (frozenset(), t)
            return s, t
        if isinstance(p, S.PLit):
            return self.infer(s, {}, S.Lit(p.value))
        if isinstance(p, S.PTuple):
            ts = []
            for x in p.items:
                s, t = self.pattern(s, x, binds)
                ts.append(t)
            return s, ("c", "*", tuple(ts))
        if isinstance(p, S.PNil):
            return s, lst(self.fresh())
        if isinstance(p, S.PCons):
            s, h = self.pattern(s, p.head, binds)
            s, t = self.pattern(s, p.tail, binds)
            return self.unify(s, t, lst(h)), lst(h)
        raise TypeError(p)

    def binding(self, s: Subst, env, name: str, rec: bool, value: S.Expr) -> Tuple[Subst, Scheme]:
        if rec:
            me = self.fresh()
            s, t = self.infer(s, {**env, name: (frozenset(), me)}, value)
            s = self.unify(s, me, t)
        else:
            s, t = self.infer(s, env, value)
        return s, self.generalise(s, env, t)

    # -- top level
    def infer_expr(self, e: S.Expr) -> WType:
        s, t = self.infer({}, self.env, e)
        return apply(s, t)

    def check_program(self, items: List[S.Item]) -> List[Tuple[str, WType]]:
        out = []
        for it in items:
            if isinstance(it, S.Decl):
                s, sch = self.binding({}, self.env, it.name, it.rec, it.value)
                if it.name != "_":
                    self.env = {**self.env, it.name: sch}
                out.append((it.name, sch[1]))
            else:
                out.append(("-", self.infer_expr(it.expr)))
        return out


def to_display(t: WType) -> str:
    """Pretty print through the shared printer so names are canonical."""
    vars_: Dict[str, TVar] = {}

    def go(t: WType):
        if t[0] == "v":
            if t[1] not in vars_:
                vars_[t[1]] = TVar(0)
            return vars_[t[1]]
        return TCon(t[1], [go(a) for a in t[2]])

    return show(go(t))
