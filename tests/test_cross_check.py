"""Algorithm J (mutable, level based) must agree with Algorithm W (textbook)."""
import pytest

from hm import parse_expr, parse_program
from hm.algorithm_w import AlgorithmW, WTypeError, to_display
from hm.infer import HMTypeError, Inferencer
from hm.types import show

CORPUS = [
    "fun x -> x",
    "fun f g x -> f (g x)",
    "fun f x y -> f y x",
    "fun x -> let y = x in y",
    "let id = fun x -> x in (id 1, id true)",
    "fun f -> let g = fun x -> f x in g",
    "fun x -> let f = fun y -> (x, y) in (f 1, f true)",
    "let k = fun x y -> x in (k 1 true, k \"a\" ())",
    "fun l -> match l with [] -> [] | x :: t -> [x]",
    "fun p -> match p with (a, (b, c)) -> (c, b, a)",
    "map (fun p -> fst p + snd p)",
    "fold_right (fun x acc -> x :: x :: acc)",
    "let rec loop x = loop x in loop",
    "let rec f n = if n = 0 then [] else n :: f (n - 1) in f",
    "fun a b c -> if a then b else c",
    "fun f -> f f",
    "fun x -> x 1 + x true",
    "let rec f x = (f 1; f true; ()) in f",
    "fun xs -> (map id xs, map (compose not id) xs)",
    "fun x -> match x with 1 -> \"one\" | _ -> \"many\"",
    "zip (range 0 3)",
    "fun x -> (fun y -> y) (fun z -> x)",
]


def j(src):
    try:
        return show(Inferencer().infer_expr(parse_expr(src)))
    except HMTypeError:
        return "error"


def w(src):
    try:
        return to_display(AlgorithmW().infer_expr(parse_expr(src)))
    except WTypeError:
        return "error"


@pytest.mark.parametrize("src", CORPUS)
def test_agree_on_corpus(src):
    assert j(src) == w(src)


def test_agree_on_prelude_and_tour():
    import pathlib
    tour = (pathlib.Path(__file__).resolve().parent.parent / "examples" / "tour.ml").read_text()
    items = parse_program(tour)
    a = [(n, show(t)) for n, t in Inferencer().check_program(items)]
    b = [(n, to_display(t)) for n, t in AlgorithmW().check_program(items)]
    assert a == b
