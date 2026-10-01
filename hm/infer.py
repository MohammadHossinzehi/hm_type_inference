"""Hindley Milner inference, Algorithm J with level based generalisation.

Entry points:

    Inferencer().check_program(items)  -> [(name, type)], warnings collected
    Inferencer().infer_expr(expr)      -> principal type of one expression

The inferencer walks the AST once.  Each sub expression gets a type that may
contain mutable type variables; constraints are solved eagerly by
destructive unification.  `let` is where polymorphism comes from: the right
hand side is inferred one level deeper, and afterwards every variable still
at that deeper level is generalised.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import syntax as S
from .patterns import check_match
from .prelude import NATIVES, PRELUDE_SOURCE, parse_signature
from .types import (BOOL, GENERIC, INT, STRING, UNIT, Namer, TCon, TVar, Type,
                    arrow, list_of, repr_, show, tuple_of)


class HMTypeError(Exception):
    def __init__(self, pos: Tuple[int, int], msg: str):
        self.pos = pos
        self.msg = msg
        super().__init__(f"line {pos[0]}, col {pos[1]}: {msg}")


class _Clash(Exception):
    def __init__(self, a: Type, b: Type):
        self.a, self.b = a, b


class _Occurs(Exception):
    def __init__(self, v: TVar, t: Type):
        self.v, self.t = v, t


# ---------------------------------------------------------------- unification

def _occurs_and_adjust(v: TVar, t: Type) -> None:
    """Fail if v occurs in t; otherwise lower the levels of t's variables.

    Lowering is what keeps generalisation sound: once a variable is linked
    into something visible at v's level, nothing inside it may be
    generalised at a deeper let.
    """
    t = repr_(t)
    if t is v:
        raise _Occurs(v, t)
    if isinstance(t, TVar):
        if t.level > v.level:
            t.level = v.level
        return
    for a in t.args:
        try:
            _occurs_and_adjust(v, a)
        except _Occurs as e:
            raise _Occurs(v, t) from e


def unify(a: Type, b: Type) -> None:
    a, b = repr_(a), repr_(b)
    if a is b:
        return
    if isinstance(a, TVar):
        _occurs_and_adjust(a, b)
        a.link = b
        return
    if isinstance(b, TVar):
        _occurs_and_adjust(b, a)
        b.link = a
        return
    assert isinstance(a, TCon) and isinstance(b, TCon)
    if a.name != b.name or len(a.args) != len(b.args):
        raise _Clash(a, b)
    for x, y in zip(a.args, b.args):
        unify(x, y)


# ---------------------------------------------------------------- schemes

def generalise(t: Type, level: int) -> None:
    t = repr_(t)
    if isinstance(t, TVar):
        if t.level > level:
            t.level = GENERIC
        return
    for a in t.args:
        generalise(a, level)


def instantiate(t: Type, level: int) -> Type:
    mapping: Dict[int, TVar] = {}

    def go(t: Type) -> Type:
        t = repr_(t)
        if isinstance(t, TVar):
            if t.level == GENERIC:
                if t.id not in mapping:
                    mapping[t.id] = TVar(level)
                return mapping[t.id]
            return t
        if not t.args:
            return t
        return TCon(t.name, [go(a) for a in t.args])

    return go(t)


def from_signature(sig: str) -> Type:
    tree = parse_signature(sig)
    vars_: Dict[str, TVar] = {}

    def go(n) -> Type:
        if n[0] == "var":
            if n[1] not in vars_:
                vars_[n[1]] = TVar(GENERIC)
            return vars_[n[1]]
        return TCon(n[1], [go(a) for a in n[2]])

    return go(tree)


# ---------------------------------------------------------------- inferencer

Env = Dict[str, Type]


class Inferencer:
    def __init__(self, with_prelude: bool = True):
        self.env: Env = {name: from_signature(sig) for name, (sig, _, _) in NATIVES.items()}
        self.warnings: List[Tuple[Tuple[int, int], str]] = []
        if with_prelude:
            self.check_program(S.parse_program(PRELUDE_SOURCE))
            self.warnings.clear()

    # -- error helpers
    def expect(self, actual: Type, expected: Type, pos, what: str = "expression") -> None:
        try:
            unify(actual, expected)
        except (_Clash, _Occurs) as err:
            namer = Namer()
            a, e = show(actual, namer), show(expected, namer)
            if what == "pattern":
                msg = (f"This pattern matches values of type {a} but a pattern was "
                       f"expected which matches values of type {e}")
            else:
                msg = f"This expression has type {a} but an expression was expected of type {e}"
            if isinstance(err, _Occurs):
                msg += (f"\n  The type variable {show(err.v, namer)} occurs inside "
                        f"{show(err.t, namer)} (this would be an infinite type)")
            else:
                ca, cb = show(err.a, namer), show(err.b, namer)
                if (ca, cb) not in ((a, e), (e, a)):
                    msg += f"\n  Type {ca} is not compatible with type {cb}"
            raise HMTypeError(pos, msg) from None

    # -- expressions
    def infer(self, env: Env, e: S.Expr, level: int) -> Type:
        if isinstance(e, S.Lit):
            v = e.value
            if v is None:
                return UNIT
            if isinstance(v, bool):
                return BOOL
            if isinstance(v, int):
                return INT
            return STRING

        if isinstance(e, S.Var):
            if e.name not in env:
                raise HMTypeError(e.pos, f"Unbound value {e.name}")
            return instantiate(env[e.name], level)

        if isinstance(e, S.Lam):
            param = TVar(level)
            inner = env if e.param == "_" else {**env, e.param: param}
            return arrow(param, self.infer(inner, e.body, level))

        if isinstance(e, S.App):
            fn_t = repr_(self.infer(env, e.fn, level))
            if isinstance(fn_t, TCon) and fn_t.name != "->":
                raise HMTypeError(e.fn.pos, f"This expression has type {show(fn_t)}\n"
                                            f"  It is not a function, it cannot be applied.")
            if isinstance(fn_t, TVar):
                a, r = TVar(level), TVar(level)
                unify(fn_t, arrow(a, r))
                fn_t = repr_(fn_t)
            param, result = fn_t.args
            arg_t = self.infer(env, e.arg, level)
            self.expect(arg_t, param, e.arg.pos)
            return result

        if isinstance(e, S.Let):
            t = self.infer_binding(env, e.name, e.rec, e.value, level)
            inner = env if e.name == "_" else {**env, e.name: t}
            return self.infer(inner, e.body, level)

        if isinstance(e, S.If):
            self.expect(self.infer(env, e.cond, level), BOOL, e.cond.pos)
            t = self.infer(env, e.then, level)
            self.expect(self.infer(env, e.other, level), t, e.other.pos)
            return t

        if isinstance(e, S.Tuple_):
            return tuple_of([self.infer(env, x, level) for x in e.items])

        if isinstance(e, S.Nil):
            return list_of(TVar(level))

        if isinstance(e, S.Cons):
            h = self.infer(env, e.head, level)
            self.expect(self.infer(env, e.tail, level), list_of(h), e.tail.pos)
            return list_of(h)

        if isinstance(e, S.Seq):
            first = self.infer(env, e.first, level)
            try:
                unify(first, UNIT)
            except (_Clash, _Occurs):
                raise HMTypeError(e.first.pos, f"This expression has type {show(first)} "
                                               f"but is used as a statement; it should have type unit") from None
            return self.infer(env, e.second, level)

        if isinstance(e, S.Match):
            scrut = self.infer(env, e.scrutinee, level)
            result = TVar(level)
            for pat, body in e.arms:
                binds: Dict[str, Type] = {}
                self.expect(self.infer_pattern(pat, level, binds), scrut, pat.pos, "pattern")
                self.expect(self.infer(env, body, level) if not binds else
                            self.infer({**env, **binds}, body, level), result, body.pos)
            example, redundant = check_match([p for p, _ in e.arms])
            for i in redundant:
                self.warnings.append((e.arms[i][0].pos, "this match case is unused."))
            if example is not None:
                self.warnings.append((e.pos, "this pattern matching is not exhaustive.\n"
                                             f"  Here is an example of a case that is not matched: {example}"))
            return result

        raise TypeError(f"unknown expression {e!r}")

    def infer_binding(self, env: Env, name: str, rec: bool, value: S.Expr, level: int) -> Type:
        if rec:
            self_t = TVar(level + 1)
            t = self.infer({**env, name: self_t}, value, level + 1)
            self.expect(t, self_t, value.pos)
        else:
            t = self.infer(env, value, level + 1)
        generalise(t, level)
        return t

    def infer_pattern(self, p: S.Pat, level: int, binds: Dict[str, Type]) -> Type:
        if isinstance(p, S.PWild):
            return TVar(level)
        if isinstance(p, S.PVar):
            if p.name in binds:
                raise HMTypeError(p.pos, f"Variable {p.name} is bound several times in this matching")
            binds[p.name] = TVar(level)
            return binds[p.name]
        if isinstance(p, S.PLit):
            return self.infer({}, S.Lit(p.value), level)
        if isinstance(p, S.PTuple):
            return tuple_of([self.infer_pattern(x, level, binds) for x in p.items])
        if isinstance(p, S.PNil):
            return list_of(TVar(level))
        if isinstance(p, S.PCons):
            h = self.infer_pattern(p.head, level, binds)
            self.expect(self.infer_pattern(p.tail, level, binds), list_of(h), p.tail.pos, "pattern")
            return list_of(h)
        raise TypeError(f"unknown pattern {p!r}")

    # -- top level
    def infer_expr(self, e: S.Expr) -> Type:
        t = self.infer(self.env, e, 1)
        generalise(t, 0)
        return t

    def check_item(self, item: S.Item) -> Tuple[str, Type]:
        if isinstance(item, S.Decl):
            t = self.infer_binding(self.env, item.name, item.rec, item.value, 0)
            if item.name != "_":
                self.env = {**self.env, item.name: t}
            return item.name, t
        return "-", self.infer_expr(item.expr)

    def check_program(self, items: List[S.Item]) -> List[Tuple[str, Type]]:
        return [self.check_item(it) for it in items]


def type_of(src: str) -> str:
    """Convenience: the principal type of an expression, pretty printed."""
    return show(Inferencer().infer_expr(S.parse_expr(src)))
