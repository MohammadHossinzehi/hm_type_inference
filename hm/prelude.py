"""The initial environment: native primitives plus a prelude written in ML.

Native primitives are declared with a type signature string.  Both
inferencers parse those signatures through `parse_signature`, which returns
a neutral tree so neither algorithm depends on the other's type
representation.  The second half of the prelude is ordinary source code
whose types (map, fold_left, compose, ...) are *inferred*, not declared.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple, Union

from .values import (NIL, Builtin, ConsCell, MLRuntimeError, StuckError,
                     Closure, is_int, show_value, to_pylist, from_pylist)

SigTree = Union[Tuple[str, str], Tuple[str, str, list]]   # ('var', a) | ('con', name, args)


def parse_signature(src: str) -> SigTree:
    toks = src.replace("(", " ( ").replace(")", " ) ").replace("->", " -> ").replace("*", " * ").split()
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def arrow():
        left = tup()
        if peek() == "->":
            take()
            return ("con", "->", [left, arrow()])
        return left

    def tup():
        items = [app()]
        while peek() == "*":
            take()
            items.append(app())
        return items[0] if len(items) == 1 else ("con", "*", items)

    def app():
        t = atom()
        while peek() is not None and peek() not in ("->", "*", ")"):
            t = ("con", take(), [t])
        return t

    def atom():
        tok = take()
        if tok == "(":
            t = arrow()
            assert take() == ")"
            return t
        if tok.startswith("'"):
            return ("var", tok[1:])
        return ("con", tok, [])

    out = arrow()
    assert pos == len(toks), f"trailing tokens in signature {src!r}"
    return out


# ---------------------------------------------------------------- natives

def _int(v):
    if not is_int(v):
        raise StuckError(f"expected an int, got {show_value(v)}")
    return v


def _bool(v):
    if v is not True and v is not False:
        raise StuckError(f"expected a bool, got {show_value(v)}")
    return v


def _str(v):
    if not isinstance(v, str):
        raise StuckError(f"expected a string, got {show_value(v)}")
    return v


def _pair(v):
    if not isinstance(v, tuple) or len(v) != 2:
        raise StuckError(f"expected a pair, got {show_value(v)}")
    return v


def _div(a, b):
    a, b = _int(a), _int(b)
    if b == 0:
        raise MLRuntimeError("Division_by_zero")
    q = abs(a) // abs(b)                         # truncate toward zero like OCaml
    return q if (a >= 0) == (b >= 0) else -q


def _mod(a, b):
    a, b = _int(a), _int(b)
    if b == 0:
        raise MLRuntimeError("Division_by_zero")
    return a - b * _div(a, b)


def structural_compare(a: Any, b: Any) -> int:
    """OCaml's polymorphic compare: structural, and an error on functions."""
    if isinstance(a, (Closure, Builtin)) or isinstance(b, (Closure, Builtin)):
        raise MLRuntimeError("Invalid_argument: compare: functional value")
    if (a is NIL or isinstance(a, ConsCell)) and (b is NIL or isinstance(b, ConsCell)):
        while True:
            if a is NIL or b is NIL:
                return (a is not NIL) - (b is not NIL)
            c = structural_compare(a.head, b.head)
            if c:
                return c
            a, b = a.tail, b.tail
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        for x, y in zip(a, b):
            c = structural_compare(x, y)
            if c:
                return c
        return 0
    if type(a) is not type(b):
        raise StuckError(f"compare on values of different shapes: {show_value(a)} and {show_value(b)}")
    if a is None:
        return 0
    return (a > b) - (a < b)


def _append(a, b):
    items = to_pylist(a)
    out = b
    for x in reversed(items):
        out = ConsCell(x, out)
    return out


def _print(s):
    sys.stdout.write(_str(s))
    return None


def _failwith(s):
    raise MLRuntimeError(f"Failure: {_str(s)}")


# name -> (signature, arity, implementation)
NATIVES: Dict[str, Tuple[str, int, Any]] = {
    "+": ("int -> int -> int", 2, lambda a, b: _int(a) + _int(b)),
    "-": ("int -> int -> int", 2, lambda a, b: _int(a) - _int(b)),
    "*": ("int -> int -> int", 2, lambda a, b: _int(a) * _int(b)),
    "/": ("int -> int -> int", 2, _div),
    "mod": ("int -> int -> int", 2, _mod),
    "~-": ("int -> int", 1, lambda a: -_int(a)),
    "=": ("'a -> 'a -> bool", 2, lambda a, b: structural_compare(a, b) == 0),
    "<>": ("'a -> 'a -> bool", 2, lambda a, b: structural_compare(a, b) != 0),
    "<": ("'a -> 'a -> bool", 2, lambda a, b: structural_compare(a, b) < 0),
    ">": ("'a -> 'a -> bool", 2, lambda a, b: structural_compare(a, b) > 0),
    "<=": ("'a -> 'a -> bool", 2, lambda a, b: structural_compare(a, b) <= 0),
    ">=": ("'a -> 'a -> bool", 2, lambda a, b: structural_compare(a, b) >= 0),
    "compare": ("'a -> 'a -> int", 2, structural_compare),
    "^": ("string -> string -> string", 2, lambda a, b: _str(a) + _str(b)),
    "@": ("'a list -> 'a list -> 'a list", 2, _append),
    "not": ("bool -> bool", 1, lambda b: not _bool(b)),
    "fst": ("'a * 'b -> 'a", 1, lambda p: _pair(p)[0]),
    "snd": ("'a * 'b -> 'b", 1, lambda p: _pair(p)[1]),
    "string_of_int": ("int -> string", 1, lambda n: str(_int(n))),
    "string_of_bool": ("bool -> string", 1, lambda b: "true" if _bool(b) else "false"),
    "string_length": ("string -> int", 1, lambda s: len(_str(s))),
    "print_string": ("string -> unit", 1, _print),
    "print_int": ("int -> unit", 1, lambda n: _print(str(_int(n)))),
    "print_endline": ("string -> unit", 1, lambda s: _print(_str(s) + "\n")),
    "failwith": ("string -> 'a", 1, _failwith),
}


def native_values() -> Dict[str, Builtin]:
    return {name: Builtin(name, arity, fn) for name, (_, arity, fn) in NATIVES.items()}


PRELUDE_SOURCE = r"""
let id x = x
let const x _ = x
let compose f g x = f (g x)
let flip f x y = f y x
let rec length l = match l with [] -> 0 | _ :: t -> 1 + length t
let rec map f l = match l with [] -> [] | x :: t -> f x :: map f t
let rec filter p l =
  match l with
  | [] -> []
  | x :: t -> if p x then x :: filter p t else filter p t
let rec fold_left f acc l = match l with [] -> acc | x :: t -> fold_left f (f acc x) t
let rec fold_right f l acc = match l with [] -> acc | x :: t -> f x (fold_right f t acc)
let rev l = fold_left (fun acc x -> x :: acc) [] l
let rec iter f l = match l with [] -> () | x :: t -> f x; iter f t
let rec range a b = if a >= b then [] else a :: range (a + 1) b
let rec exists p l = match l with [] -> false | x :: t -> p x || exists p t
let rec for_all p l = match l with [] -> true | x :: t -> p x && for_all p t
let rec zip a b =
  match a, b with
  | [], _ -> []
  | _, [] -> []
  | x :: xs, y :: ys -> (x, y) :: zip xs ys
let rec assoc k l =
  match l with
  | [] -> failwith "Not_found"
  | (k2, v) :: t -> if k = k2 then v else assoc k t
let hd l = match l with x :: _ -> x | [] -> failwith "hd"
let tl l = match l with _ :: t -> t | [] -> failwith "tl"
"""
