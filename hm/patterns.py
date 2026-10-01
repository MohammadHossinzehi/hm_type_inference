"""Exhaustiveness and redundancy checking for `match`.

This is Luc Maranget's usefulness algorithm ("Warnings for pattern
matching", JFP 2007) specialised to this language's constructors:

    bool    : true | false            unit : ()
    list    : [] | _ :: _             tuple: one constructor of arity n
    int, string literals: an infinite signature (never complete)

A row vector q is *useful* with respect to a pattern matrix P if some value
matches q but no row of P.  A match is exhaustive iff the wildcard row is
not useful against all arms, and arm i is redundant iff it is not useful
against arms 0..i-1.  `missing` returns a witness value pattern instead of
a boolean, which is what makes the warning messages concrete.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .syntax import Pat, PCons, PLit, PNil, PTuple, PVar, PWild


@dataclass(frozen=True)
class Con:
    name: str           # 'true', 'false', '()', '[]', '::', 'tuple/3', 'int:5', 'str:"x"'
    args: Tuple["P", ...]


class _Wild:
    def __repr__(self) -> str:
        return "_"


WILD = _Wild()
P = object               # Con | WILD


def normalise(p: Pat) -> P:
    if isinstance(p, (PVar, PWild)):
        return WILD
    if isinstance(p, PLit):
        v = p.value
        if v is None:
            return Con("()", ())
        if v is True:
            return Con("true", ())
        if v is False:
            return Con("false", ())
        if isinstance(v, int):
            return Con(f"int:{v}", ())
        return Con("str:" + repr(v), ())
    if isinstance(p, PNil):
        return Con("[]", ())
    if isinstance(p, PCons):
        return Con("::", (normalise(p.head), normalise(p.tail)))
    if isinstance(p, PTuple):
        return Con(f"tuple/{len(p.items)}", tuple(normalise(x) for x in p.items))
    raise TypeError(p)


def arity(name: str) -> int:
    if name == "::":
        return 2
    if name.startswith("tuple/"):
        return int(name.split("/")[1])
    return 0


def signature(name: str) -> Optional[List[str]]:
    """All constructors of the type `name` belongs to, or None if infinite."""
    if name in ("true", "false"):
        return ["true", "false"]
    if name == "()":
        return ["()"]
    if name in ("[]", "::"):
        return ["[]", "::"]
    if name.startswith("tuple/"):
        return [name]
    return None


Row = Tuple[P, ...]


def specialise(rows: Sequence[Row], c: str) -> List[Row]:
    a = arity(c)
    out = []
    for r in rows:
        h = r[0]
        if h is WILD:
            out.append((WILD,) * a + r[1:])
        elif h.name == c:
            out.append(tuple(h.args) + r[1:])
    return out


def default(rows: Sequence[Row]) -> List[Row]:
    return [r[1:] for r in rows if r[0] is WILD]


def head_constructors(rows: Sequence[Row]) -> List[str]:
    seen: List[str] = []
    for r in rows:
        if r[0] is not WILD and r[0].name not in seen:
            seen.append(r[0].name)
    return seen


def useful(rows: Sequence[Row], q: Row) -> bool:
    if not q:
        return not rows
    h = q[0]
    if h is not WILD:
        return useful(specialise(rows, h.name), tuple(h.args) + q[1:])
    sigma = head_constructors(rows)
    sig = signature(sigma[0]) if sigma else None
    if sig is not None and set(sig) <= set(sigma):
        return any(useful(specialise(rows, c), (WILD,) * arity(c) + q[1:]) for c in sig)
    return useful(default(rows), q[1:])


def missing(rows: Sequence[Row], n: int) -> Optional[List[P]]:
    """A vector of n patterns matched by no row, or None if rows are exhaustive."""
    if n == 0:
        return None if rows else []
    sigma = head_constructors(rows)
    sig = signature(sigma[0]) if sigma else None
    if sig is not None and set(sig) <= set(sigma):
        for c in sig:
            a = arity(c)
            w = missing(specialise(rows, c), a + n - 1)
            if w is not None:
                return [Con(c, tuple(w[:a]))] + w[a:]
        return None
    w = missing(default(rows), n - 1)
    if w is None:
        return None
    if not sigma:
        return [WILD] + w
    if sig is not None:
        absent = next(c for c in sig if c not in sigma)
        return [Con(absent, (WILD,) * arity(absent))] + w
    if sigma[0].startswith("int:"):      # pick an integer nobody mentioned
        used = {int(c[4:]) for c in sigma}
        k = 0
        while k in used:
            k += 1
        return [Con(f"int:{k}", ())] + w
    return [WILD] + w


def show_pattern(p: P, top: bool = True) -> str:
    if p is WILD:
        return "_"
    assert isinstance(p, Con)
    if p.name == "::":
        s = f"{show_pattern(p.args[0], False)} :: {show_pattern(p.args[1], True)}"
        return s if top else f"({s})"
    if p.name.startswith("tuple/"):
        return "(" + ", ".join(show_pattern(a, True) for a in p.args) + ")"
    if p.name.startswith("int:"):
        return p.name[4:]
    if p.name.startswith("str:"):
        return '"' + eval(p.name[4:]) + '"'
    return p.name


def check_match(pats: Sequence[Pat]) -> Tuple[Optional[str], List[int]]:
    """Return (example of an unmatched value or None, indices of redundant arms)."""
    rows: List[Row] = []
    redundant = []
    for i, p in enumerate(pats):
        r = (normalise(p),)
        if not useful(rows, r):
            redundant.append(i)
        rows.append(r)
    w = missing(rows, 1)
    return (show_pattern(w[0]) if w is not None else None), redundant
