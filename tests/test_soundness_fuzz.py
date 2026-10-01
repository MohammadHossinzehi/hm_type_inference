"""Property tests on random programs.

1. Soundness ("well typed programs do not go wrong"): if Algorithm J accepts
   a program, evaluating it never gets stuck, and the value it produces
   inhabits the inferred type.  Allowed failures are the ones the language
   defines: division by zero, failwith, a non exhaustive match, and the
   step budget running out.
2. Agreement: Algorithm J and Algorithm W accept exactly the same random
   programs and infer the same principal type up to variable renaming.

The generator is deliberately type unaware: it builds well scoped but
otherwise arbitrary terms, so roughly a fifth of them type check and the
rest exercise the error paths.
"""
import random

import pytest

from hm import syntax as S
from hm.algorithm_w import AlgorithmW, WTypeError, to_display
from hm.infer import HMTypeError, Inferencer
from hm.interp import Interpreter
from hm.types import TCon, TVar, repr_, show
from hm.values import NIL, Builtin, Closure, ConsCell, MLRuntimeError, StuckError

GLOBALS = ["map", "filter", "fold_left", "length", "rev", "fst", "snd", "id", "compose",
           "not", "+", "-", "*", "/", "=", "<", "^", "@", "zip", "hd", "string_of_int"]


class Gen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.n = 0

    def name(self) -> str:
        self.n += 1
        return f"v{self.n}"

    def expr(self, depth: int, scope) -> S.Expr:
        r = self.rng
        if depth <= 0 or r.random() < 0.15:
            return self.leaf(scope)
        k = r.randrange(11)
        d = depth - 1
        if k == 0:
            x = self.name()
            return S.Lam(x, self.expr(d, scope + [x]))
        if k in (1, 2):
            return S.App(self.expr(d, scope), self.expr(d, scope))
        if k == 3:
            x = self.name()
            return S.Let(x, False, self.expr(d, scope), self.expr(d, scope + [x]))
        if k == 4:
            return S.If(self.expr(d, scope), self.expr(d, scope), self.expr(d, scope))
        if k == 5:
            return S.Tuple_([self.expr(d, scope) for _ in range(r.choice([2, 2, 3]))])
        if k == 6:
            return S.Cons(self.expr(d, scope), self.expr(d, scope) if r.random() < 0.5 else S.Nil())
        if k == 7:
            h, t = self.name(), self.name()
            return S.Match(self.expr(d, scope), [
                (S.PNil(), self.expr(d, scope)),
                (S.PCons(S.PVar(h), S.PVar(t)), self.expr(d, scope + [h, t])),
            ])
        if k == 8:
            a, b = self.name(), self.name()
            return S.Match(self.expr(d, scope), [(S.PTuple([S.PVar(a), S.PVar(b)]), self.expr(d, scope + [a, b]))])
        if k == 9:
            op = r.choice(["+", "-", "*", "/", "=", "<", "^", "@"])
            return S.App(S.App(S.Var(op), self.expr(d, scope)), self.expr(d, scope))
        x = self.name()
        return S.Let(x, False, S.Lam(self.name(), self.expr(d, scope)), self.expr(d, scope + [x]))

    def leaf(self, scope) -> S.Expr:
        r = self.rng
        k = r.random()
        if scope and k < 0.45:
            return S.Var(r.choice(scope))
        if k < 0.6:
            return S.Var(r.choice(GLOBALS))
        return r.choice([S.Lit(r.randint(-3, 5)), S.Lit(r.random() < 0.5), S.Lit("s"),
                         S.Lit(None), S.Nil()])


def inhabits(v, t) -> bool:
    """Does value v belong to type t?  Type variables accept anything."""
    t = repr_(t)
    if isinstance(t, TVar):
        return True
    n = t.name
    if n == "int":
        return type(v) is int
    if n == "bool":
        return v is True or v is False
    if n == "string":
        return isinstance(v, str)
    if n == "unit":
        return v is None
    if n == "->":
        return isinstance(v, (Closure, Builtin))
    if n == "*":
        return isinstance(v, tuple) and len(v) == len(t.args) and all(inhabits(x, a) for x, a in zip(v, t.args))
    if n == "list":
        while isinstance(v, ConsCell):
            if not inhabits(v.head, t.args[0]):
                return False
            v = v.tail
        return v is NIL
    return False


SEEDS = range(4000)


@pytest.fixture(scope="module")
def programs():
    out = []
    for seed in SEEDS:
        g = Gen(random.Random(seed))
        out.append(g.expr(5, []))
    return out


def test_well_typed_programs_do_not_go_wrong(programs):
    inf = Inferencer()
    typed = evaluated = 0
    for e in programs:
        try:
            t = inf.infer_expr(e)
        except HMTypeError:
            continue
        typed += 1
        interp = Interpreter(fuel=20_000)
        try:
            v = interp.eval_expr(e)
        except MLRuntimeError:
            continue                       # a failure the language permits
        except StuckError as err:          # pragma: no cover - would be a soundness bug
            pytest.fail(f"well typed program got stuck: {err}\n{e!r}\n: {show(t)}")
        evaluated += 1
        assert inhabits(v, t), f"value does not inhabit {show(t)}: {e!r}"
    # Make sure the generator is actually exercising the interesting path.
    assert typed >= 400, typed
    assert evaluated >= 300, evaluated


def test_algorithm_j_and_w_agree(programs):
    inf, w = Inferencer(), AlgorithmW()
    both_ok = 0
    for e in programs:
        try:
            a = show(inf.infer_expr(e))
        except HMTypeError:
            a = None
        try:
            b = to_display(w.infer_expr(e))
        except WTypeError:
            b = None
        assert a == b, f"J says {a}, W says {b} for {e!r}"
        both_ok += a is not None
    assert both_ok >= 400


def test_ill_typed_programs_do_get_stuck_sometimes(programs):
    """Sanity check of the oracle: without the type checker, programs go wrong."""
    inf = Inferencer()
    stuck = 0
    for e in programs[:1500]:
        try:
            inf.infer_expr(e)
            continue
        except HMTypeError:
            pass
        try:
            Interpreter(fuel=5_000).eval_expr(e)
        except StuckError:
            stuck += 1
        except (MLRuntimeError, RecursionError):
            pass
    assert stuck > 100
