"""A small environment passing evaluator for type checked programs.

The evaluator is deliberately paranoid: whenever it meets a value of the
wrong shape (applying a non function, `if` on a non boolean, matching a
list pattern against a tuple) it raises `StuckError` rather than guessing.
For well typed programs that must never happen, and the fuzz tests rely on
this to check the type system's soundness empirically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import syntax as S
from .prelude import PRELUDE_SOURCE, native_values
from .values import (NIL, Builtin, Closure, ConsCell, Env, MLRuntimeError,
                     StuckError, is_int)


class Interpreter:
    def __init__(self, with_prelude: bool = True, fuel: Optional[int] = None):
        env = Env()
        for name, v in native_values().items():
            env = env.extend(name, v)
        self.env = env
        self.fuel = fuel                       # optional step budget for fuzzing
        if with_prelude:
            for item in S.parse_program(PRELUDE_SOURCE):
                self.run_item(item)

    def tick(self) -> None:
        if self.fuel is not None:
            self.fuel -= 1
            if self.fuel < 0:
                raise MLRuntimeError("Out_of_fuel")

    def apply(self, f: Any, arg: Any) -> Any:
        if isinstance(f, Closure):
            env = f.env if f.param == "_" else f.env.extend(f.param, arg)
            return self.eval(f.body, env)
        if isinstance(f, Builtin):
            return f.apply(arg)
        raise StuckError(f"cannot apply a non function value")

    def eval(self, e: S.Expr, env: Env) -> Any:
        self.tick()
        if isinstance(e, S.Lit):
            return e.value
        if isinstance(e, S.Var):
            return env.lookup(e.name)
        if isinstance(e, S.Lam):
            return Closure(e.param, e.body, env)
        if isinstance(e, S.App):
            f = self.eval(e.fn, env)
            return self.apply(f, self.eval(e.arg, env))
        if isinstance(e, S.Let):
            v = self.eval_binding(e.name, e.rec, e.value, env)
            return self.eval(e.body, env if e.name == "_" else env.extend(e.name, v))
        if isinstance(e, S.If):
            c = self.eval(e.cond, env)
            if c is True:
                return self.eval(e.then, env)
            if c is False:
                return self.eval(e.other, env)
            raise StuckError("if condition is not a boolean")
        if isinstance(e, S.Tuple_):
            return tuple(self.eval(x, env) for x in e.items)
        if isinstance(e, S.Nil):
            return NIL
        if isinstance(e, S.Cons):
            h = self.eval(e.head, env)
            t = self.eval(e.tail, env)
            if t is not NIL and not isinstance(t, ConsCell):
                raise StuckError("cons onto a non list")
            return ConsCell(h, t)
        if isinstance(e, S.Seq):
            if self.eval(e.first, env) is not None:
                raise StuckError("statement did not evaluate to unit")
            return self.eval(e.second, env)
        if isinstance(e, S.Match):
            v = self.eval(e.scrutinee, env)
            for pat, body in e.arms:
                binds: Dict[str, Any] = {}
                if self.match(pat, v, binds):
                    for k, x in binds.items():
                        env = env.extend(k, x)
                    return self.eval(body, env)
            line, col = e.pos
            raise MLRuntimeError(f"Match_failure (line {line}, col {col})")
        raise TypeError(e)

    def eval_binding(self, name: str, rec: bool, value: S.Expr, env: Env) -> Any:
        if rec:
            assert isinstance(value, S.Lam)
            clo = Closure(value.param, value.body, None)
            clo.env = env.extend(name, clo)          # tie the recursive knot
            return clo
        return self.eval(value, env)

    def match(self, p: S.Pat, v: Any, binds: Dict[str, Any]) -> bool:
        if isinstance(p, S.PWild):
            return True
        if isinstance(p, S.PVar):
            binds[p.name] = v
            return True
        if isinstance(p, S.PLit):
            lit = p.value
            if type(lit) is not type(v):
                raise StuckError("literal pattern against a value of another type")
            return lit == v
        if isinstance(p, S.PTuple):
            if not isinstance(v, tuple) or len(v) != len(p.items):
                raise StuckError("tuple pattern against a non tuple")
            return all(self.match(q, x, binds) for q, x in zip(p.items, v))
        if isinstance(p, S.PNil):
            if v is NIL:
                return True
            if isinstance(v, ConsCell):
                return False
            raise StuckError("list pattern against a non list")
        if isinstance(p, S.PCons):
            if isinstance(v, ConsCell):
                return self.match(p.head, v.head, binds) and self.match(p.tail, v.tail, binds)
            if v is NIL:
                return False
            raise StuckError("list pattern against a non list")
        raise TypeError(p)

    def run_item(self, item: S.Item) -> Any:
        if isinstance(item, S.Decl):
            v = self.eval_binding(item.name, item.rec, item.value, self.env)
            if item.name != "_":
                self.env = self.env.extend(item.name, v)
            return v
        return self.eval(item.expr, self.env)

    def eval_expr(self, e: S.Expr) -> Any:
        return self.eval(e, self.env)
