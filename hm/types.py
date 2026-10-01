"""Type representation used by the level based inferencer (Algorithm J).

Type variables are mutable cells.  Unification links a variable to another
type instead of building substitutions, and `repr_` follows those links with
path compression, which is the classic union find trick that makes
Algorithm J run in near linear time.

Every unbound variable carries a *level*: the let nesting depth at which it
was created.  When the inferencer leaves a `let` right hand side, any
variable whose level is deeper than the surrounding context cannot be
mentioned by the environment, so it can be generalised.  Variables that
have been generalised get the level GENERIC and are copied fresh at every
use site.  This is the same scheme OCaml uses (due to Didier Remy) and it
avoids the expensive "free variables of the environment" scan that the
textbook Algorithm W performs at every let.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional

GENERIC = 1 << 30


class Type:
    pass


class TVar(Type):
    __slots__ = ("id", "level", "link")
    _ids = itertools.count()

    def __init__(self, level: int):
        self.id = next(TVar._ids)
        self.level = level
        self.link: Optional[Type] = None

    def __repr__(self) -> str:
        return f"TVar({self.id}, lvl={self.level}{', ->' + repr(self.link) if self.link else ''})"


class TCon(Type):
    """A type constructor applied to arguments: int, 'a list, a -> b, a * b."""
    __slots__ = ("name", "args")

    def __init__(self, name: str, args: Optional[List[Type]] = None):
        self.name = name
        self.args = args or []

    def __repr__(self) -> str:
        return f"TCon({self.name!r}, {self.args!r})"


INT = TCon("int")
BOOL = TCon("bool")
STRING = TCon("string")
UNIT = TCon("unit")


def arrow(a: Type, b: Type) -> TCon:
    return TCon("->", [a, b])


def tuple_of(items: List[Type]) -> TCon:
    return TCon("*", list(items))


def list_of(t: Type) -> TCon:
    return TCon("list", [t])


def repr_(t: Type) -> Type:
    """Follow variable links to the representative, compressing the path."""
    if isinstance(t, TVar) and t.link is not None:
        root = repr_(t.link)
        t.link = root
        return root
    return t


# ---------------------------------------------------------------- printing

class Namer:
    """Assigns 'a, 'b, ... to variables in order of first appearance.

    Sharing one Namer between two printed types (as error messages do) keeps
    variable names consistent between them.
    """

    def __init__(self) -> None:
        self.names: Dict[int, str] = {}

    def name(self, v: TVar) -> str:
        if v.id not in self.names:
            k = len(self.names)
            letters = ""
            while True:
                letters = chr(ord("a") + k % 26) + letters
                k = k // 26 - 1
                if k < 0:
                    break
            self.names[v.id] = "'" + letters
        return self.names[v.id]


PREC_ARROW, PREC_TUPLE, PREC_APP = 0, 1, 2


def show(t: Type, namer: Optional[Namer] = None) -> str:
    namer = namer or Namer()

    def go(t: Type, prec: int) -> str:
        t = repr_(t)
        if isinstance(t, TVar):
            return namer.name(t)
        assert isinstance(t, TCon)
        if t.name == "->":
            s = f"{go(t.args[0], PREC_TUPLE)} -> {go(t.args[1], PREC_ARROW)}"
            return f"({s})" if prec > PREC_ARROW else s
        if t.name == "*":
            s = " * ".join(go(a, PREC_APP) for a in t.args)
            return f"({s})" if prec > PREC_TUPLE else s
        if not t.args:
            return t.name
        if len(t.args) == 1:
            return f"{go(t.args[0], PREC_APP)} {t.name}"
        return f"({', '.join(go(a, PREC_ARROW) for a in t.args)}) {t.name}"

    return go(t, PREC_ARROW)


def free_vars(t: Type) -> List[TVar]:
    out: List[TVar] = []
    seen = set()

    def go(t: Type) -> None:
        t = repr_(t)
        if isinstance(t, TVar):
            if t.id not in seen:
                seen.add(t.id)
                out.append(t)
        else:
            for a in t.args:
                go(a)

    go(t)
    return out
