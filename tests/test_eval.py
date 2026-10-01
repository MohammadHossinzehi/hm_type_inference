"""End to end: parse, type check, evaluate."""
import pytest

from hm import Inferencer, Interpreter, MLRuntimeError, parse_program, show_value
from hm.__main__ import main


def run(src):
    items = parse_program(src)
    Inferencer().check_program(items)
    interp = Interpreter()
    v = None
    for it in items:
        v = interp.run_item(it)
    return show_value(v)


@pytest.mark.parametrize("src, expected", [
    ("1 + 2 * 3;;", "7"),
    ("7 / 2, -7 / 2, -7 mod 2;;", "(3, -3, -1)"),
    ('"ab" ^ "cd";;', '"abcd"'),
    ("map (fun x -> x * x) [1; 2; 3];;", "[1; 4; 9]"),
    ("filter (fun x -> x mod 2 = 0) (range 0 10);;", "[0; 2; 4; 6; 8]"),
    ("rev [1; 2; 3];;", "[3; 2; 1]"),
    ("[1; 2] @ [3];;", "[1; 2; 3]"),
    ("zip [1; 2; 3] [true; false];;", "[(1, true); (2, false)]"),
    ("assoc 2 [(1, \"a\"); (2, \"b\")];;", '"b"'),
    ("let rec fact n = if n = 0 then 1 else n * fact (n - 1) in fact 20;;", "2432902008176640000"),
    ("[1; 2] < [1; 3], (2, \"a\") > (1, \"z\"), [] = [1];;", "(true, true, false)"),
    ("let rec fib n = if n < 2 then n else fib (n - 1) + fib (n - 2) in map fib (range 0 10);;",
     "[0; 1; 1; 2; 3; 5; 8; 13; 21; 34]"),
    ("false && (1 / 0 = 0);;", "false"),               # && short circuits
    ("true || (1 / 0 = 0);;", "true"),
    ("let f x y z = x + y + z in let g = f 1 in let h = g 2 in h 3;;", "6"),
    ("(+) 1 2;;", "3"),
    ("fold_left (-) 100 [1; 2; 3];;", "94"),
    ("exists (fun x -> x > 2) [1; 2; 3], for_all (fun x -> x > 2) [1; 2; 3];;", "(true, false)"),
])
def test_programs(src, expected):
    assert run(src) == expected


def test_closures_capture_lexically():
    assert run("let x = 1\nlet f y = x + y\nlet x = 100\n;; f 1;;") == "2"


def test_runtime_errors():
    with pytest.raises(MLRuntimeError, match="Division_by_zero"):
        run("1 / 0;;")
    with pytest.raises(MLRuntimeError, match="Failure: hd"):
        run("hd [];;")
    with pytest.raises(MLRuntimeError, match="Match_failure"):
        run("match [] with x :: _ -> x;;")
    with pytest.raises(MLRuntimeError, match="functional value"):
        run("(fun x -> x) = (fun y -> y);;")


def test_cli_run_prints_output(tmp_path, capsys):
    f = tmp_path / "p.ml"
    f.write_text('let () = print_endline "hello"\n;; 1 + 1;;\n')
    assert main(["run", str(f)]) == 0
    assert capsys.readouterr().out == "hello\n2\n"


def test_cli_refuses_ill_typed_programs(tmp_path, capsys):
    f = tmp_path / "bad.ml"
    f.write_text('let () = print_endline "side effect"\nlet x = 1 + "2"\n')
    assert main(["run", str(f)]) == 1
    captured = capsys.readouterr()
    assert "side effect" not in captured.out          # nothing ran
    assert "Type error" in captured.err


def test_cli_infer_both_algorithms(capsys):
    assert main(["infer", "-e", "fun f x -> f (f x)"]) == 0
    assert main(["infer", "--algorithm", "w", "-e", "fun f x -> f (f x)"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["('a -> 'a) -> 'a -> 'a"] * 2


def test_tour_example_runs(capsys):
    import pathlib
    tour = pathlib.Path(__file__).resolve().parent.parent / "examples" / "tour.ml"
    assert main(["run", str(tour)]) == 0
    out = capsys.readouterr().out
    assert "[1; 3; 4; 5; 9]" in out and "385" in out
