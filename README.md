# hm_type_inference

Hindley Milner type inference for a small ML, written from scratch in pure Python with no dependencies.

You write code with no type annotations at all, and it tells you the most general type every expression can have:

```
$ python -m hm infer -e "fun f g x -> f (g x)"
('a -> 'b) -> ('c -> 'a) -> 'c -> 'b

$ python -m hm infer examples/tour.ml
val pair : 'a -> 'b -> 'a * 'b
val swap : 'a * 'b -> 'b * 'a
val twice : ('a -> 'a) -> 'a -> 'a
val compose_all : ('a -> 'a) list -> 'a -> 'a
val sort : 'a list -> 'a list
val take : int -> 'a list -> 'a list
val show_list : ('a -> string) -> 'a list -> string
...
```

and it refuses programs that could go wrong, with messages that point at the problem:

```
$ python -m hm infer -e "fun x -> x x"
Type error: line 1, col 12: This expression has type 'a -> 'b but an expression was expected of type 'a
  The type variable 'a occurs inside 'a -> 'b (this would be an infinite type)
```

Then it can actually run the program.

## Why I built this

Type inference is the piece of OCaml, Haskell, F#, Rust and TypeScript that feels most like magic, and it is surprisingly small once you see it. I wanted an implementation that is readable in one sitting, but that is still the real algorithm rather than a classroom sketch, and that proves its own correctness instead of asking you to trust it. Concretely, it has:

* **Algorithm J with level based generalisation** (`hm/infer.py`). Type variables are mutable union find cells, unification links them in place, and every variable remembers the `let` depth it was born at. Leaving a `let`, anything deeper than the current level is generalised. This is the technique OCaml's own type checker uses (Didier Rémy's levels) and it avoids scanning the environment at every `let`.
* **Textbook Algorithm W as an independent oracle** (`hm/algorithm_w.py`). Immutable types, explicit substitution dictionaries, `ftv(env)` computed at every generalisation, exactly as in Damas and Milner (1982). It shares no inference code with J.
* **Exhaustiveness and redundancy checking** (`hm/patterns.py`) using Maranget's usefulness algorithm, with a concrete counterexample in the warning:
  ```
  Warning: this pattern matching is not exhaustive.
    Here is an example of a case that is not matched: _ :: _ :: _
  ```
* **An evaluator** (`hm/interp.py`) with closures, `let rec`, curried native primitives and OCaml style structural comparison, plus a REPL.
* **A prelude written in the language itself**: `map`, `filter`, `fold_left`, `fold_right`, `zip`, `assoc` and friends are plain source code whose types are inferred at startup, not declared.

## The language

An OCaml flavoured subset: `int`, `bool`, `string`, `unit`, tuples, lists, first class functions, `let` / `let rec`, `fun`, `if`, `match` with nested patterns, sequencing with `;`, and the usual operators (`+ - * / mod`, `= <> < > <= >=`, `&& ||`, `^` for strings, `@` and `::` for lists). Operators can be used as functions: `fold_left (+) 0 xs`. Comments are `(* nested (* like this *) *)`.

```ocaml
let rec insert x l =
  match l with
  | [] -> [x]
  | y :: ys -> if x <= y then x :: l else y :: insert x ys

let sort l = fold_right insert l []

;; sort [5; 3; 9; 1; 4];;      (* [1; 3; 4; 5; 9] *)
```

## Running it

Requires Python 3.10 or newer. Nothing to install for the core:

```bash
python -m hm infer examples/tour.ml        # types of every top level binding
python -m hm infer -e "map fst"            # type of a single expression
python -m hm infer --algorithm w examples/tour.ml   # same, via Algorithm W
python -m hm run examples/tour.ml          # type check, then evaluate
python -m hm repl                          # interactive toplevel, phrases end with ;;
```

`run` type checks the whole file before evaluating anything, so an ill typed program never produces partial side effects.

Tests and benchmark:

```bash
pip install pytest
python -m pytest            # 98 tests, about 4 seconds
python bench.py             # J vs W timing
```

## How the tests earn their keep

There are four test files and they check different kinds of claim.

**`test_infer.py`** pins down principal types for a corpus of classic terms (composition, `flip`, `fold_right`, `zip`, `assoc`...), the boundaries of let polymorphism (lambda bound variables are monomorphic, `let rec` is monomorphic inside its own body), every error message, and the exhaustiveness witnesses.

**`test_cross_check.py`** runs a hand written corpus and the whole `examples/tour.ml` through both J and W and requires identical results up to renaming of type variables. Both printers name variables in order of first appearance, so alpha equivalence becomes string equality.

**`test_soundness_fuzz.py`** is the interesting one. A generator produces 4000 random, well scoped but type unaware programs (about 18% happen to type check). For each one:

1. *J and W must agree*: both reject it, or both accept it with the same principal type.
2. *Well typed programs do not go wrong*: if J accepts it, evaluating it must never reach a stuck state (applying an integer, `if` on a string, a list pattern against a tuple). The evaluator raises a dedicated `StuckError` for those, separate from the failures the language allows (division by zero, `failwith`, an unmatched case, running out of fuel).
3. *The value inhabits the inferred type*: an `int list` really is a list of ints, a `'a * bool` really is a pair whose second element is a boolean.
4. As a sanity check on the oracle itself, the *ill typed* programs, run anyway, must get stuck a reasonable number of times. Without this a broken `StuckError` check would make test 2 pass vacuously.

I also mutation tested the suite. Making `generalise` ignore levels (the classic unsound "generalise everything" bug) is caught by both fuzz properties. Removing the level adjustment during unification (the subtle bug where a variable escapes into an outer scope but still gets generalised) is caught by the cross check corpus and by a targeted unit test.

## Design notes

* **Why two algorithms?** J is the one you would ship; W is the one you can check against a paper line by line. Agreement on thousands of random programs is much stronger evidence than either implementation's unit tests alone, the same reason my Earley parser is cross checked against CYK.
* **Error messages print both types with one shared namer**, so `'a` in "has type" and `'a` in "expected type" are the same variable. When the clash is deep inside a larger type, a second line shows the innermost incompatible pair (`Type bool is not compatible with type int`).
* **Function application is checked in two steps**: first that the head is a function at all (otherwise the error is "it is not a function, it cannot be applied", positioned at the head), then that the argument matches the parameter type (positioned at the argument). This gives far better locations than unifying `f` with `arg -> result` in one go.
* **`&&` and `||` desugar to `if`**, so they short circuit at run time without any special case in the type checker. Operators are just names in the environment, which is why `(+)` works for free.
* **No value restriction.** The language has no mutable references, so generalising every `let` is sound. Adding `ref` would require OCaml's relaxed value restriction; the level machinery is already the right place to implement it.
* **Exhaustiveness runs on patterns, not types**, which is enough here because every constructor determines its type's full signature (`[]` and `::`, `true` and `false`, a tuple of arity n). Integer and string literals have an infinite signature, so for those the witness is a value nobody mentioned (`2` after `0 -> ... | 1 -> ...`).

## Benchmark

`python bench.py` on a laptop, CPython 3.10:

```
workload             n      J (ms)      W (ms)    W / J
let_chain           50         0.6        10.7     17.4
let_chain          100         1.1        31.5     28.5
let_chain          200         3.3       102.3     30.9
let_chain          400         6.8       366.7     54.0
pair_tower           4         0.4         2.8      7.7
pair_tower           5        95.4       335.5      3.5
```

`let_chain` is n nested polymorphic lets. J stays roughly linear while W's environment scan makes it quadratic. `pair_tower` is the textbook pathological program whose type has 4^n leaves. Both algorithms blow up there, because HM typability is DEXPTIME complete and no clever implementation avoids it; it is included as an honest limit rather than a win.

## Limitations

* No user defined algebraic data types, records, references or modules; the type system is core HM plus tuples and lists.
* The evaluator is a recursive tree walker, so very deep non tail recursion (lists of tens of thousands of elements) can hit Python's recursion limit. The CLI raises the limit to 20000.
* Inference stops at the first type error rather than recovering and reporting several.

## Layout

```
hm/syntax.py        lexer, AST, recursive descent parser
hm/types.py         mutable type variables, levels, pretty printer
hm/infer.py         Algorithm J with levels (the main inferencer)
hm/algorithm_w.py   textbook Algorithm W (reference oracle)
hm/patterns.py      Maranget exhaustiveness and redundancy checking
hm/prelude.py       native primitives and the ML source prelude
hm/values.py        runtime values, StuckError vs MLRuntimeError
hm/interp.py        evaluator
hm/__main__.py      CLI and REPL
tests/              unit tests, J/W cross check, soundness fuzzing
examples/tour.ml    a guided example program
bench.py            J vs W timing
```

MIT licensed.
