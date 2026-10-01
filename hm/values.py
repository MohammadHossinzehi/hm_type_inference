"""Runtime values and the two kinds of runtime failure.

`MLRuntimeError` is a failure the language allows (division by zero,
`failwith`, a match with no applicable arm).  `StuckError` means the
evaluator reached a state that should be impossible for a well typed
program, such as applying an integer.  The soundness fuzz test asserts
that well typed programs never raise it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class MLRuntimeError(Exception):
    pass


class StuckError(Exception):
    pass


class NilT:
    __slots__ = ()

    def __repr__(self) -> str:
        return "[]"


NIL = NilT()


class ConsCell:
    __slots__ = ("head", "tail")

    def __init__(self, head: Any, tail: Any):
        self.head = head
        self.tail = tail


def from_pylist(items) -> Any:
    out: Any = NIL
    for x in reversed(list(items)):
        out = ConsCell(x, out)
    return out


def to_pylist(v: Any) -> List[Any]:
    out = []
    while isinstance(v, ConsCell):
        out.append(v.head)
        v = v.tail
    if v is not NIL:
        raise StuckError("improper list")
    return out


class Env:
    """A persistent chain of frames; extending never mutates the parent."""
    __slots__ = ("name", "value", "parent")

    def __init__(self, name: Optional[str] = None, value: Any = None, parent: "Optional[Env]" = None):
        self.name = name
        self.value = value
        self.parent = parent

    def extend(self, name: str, value: Any) -> "Env":
        return Env(name, value, self)

    def lookup(self, name: str) -> Any:
        e: Optional[Env] = self
        while e is not None:
            if e.name == name:
                return e.value
            e = e.parent
        raise StuckError(f"unbound variable {name}")


class Closure:
    __slots__ = ("param", "body", "env")

    def __init__(self, param: str, body, env: Optional[Env]):
        self.param = param
        self.body = body
        self.env = env


class Builtin:
    """A native function, curried: arguments accumulate until `arity`."""
    __slots__ = ("name", "arity", "fn", "args")

    def __init__(self, name: str, arity: int, fn: Callable[..., Any], args: tuple = ()):
        self.name = name
        self.arity = arity
        self.fn = fn
        self.args = args

    def apply(self, arg: Any) -> Any:
        args = self.args + (arg,)
        if len(args) == self.arity:
            return self.fn(*args)
        return Builtin(self.name, self.arity, self.fn, args)


def is_int(v: Any) -> bool:
    return type(v) is int


def show_value(v: Any) -> str:
    if v is None:
        return "()"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if is_int(v):
        return str(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, tuple):
        return "(" + ", ".join(show_value(x) for x in v) + ")"
    if v is NIL or isinstance(v, ConsCell):
        return "[" + "; ".join(show_value(x) for x in to_pylist(v)) + "]"
    if isinstance(v, (Closure, Builtin)):
        return "<fun>"
    return repr(v)
