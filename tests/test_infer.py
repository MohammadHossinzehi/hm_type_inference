"""Principal types and error reporting for Algorithm J."""
import pytest

from hm import HMTypeError, Inferencer, ParseError, parse_program, show, type_of


@pytest.mark.parametrize("src, expected", [
    ("42", "int"),
    ('"hi"', "string"),
    ("()", "unit"),
    ("fun x -> x", "'a -> 'a"),
    ("fun x y -> x", "'a -> 'b -> 'a"),
    ("fun f g x -> f (g x)", "('a -> 'b) -> ('c -> 'a) -> 'c -> 'b"),
    ("fun f x y -> f y x", "('a -> 'b -> 'c) -> 'b -> 'a -> 'c"),
    ("fun f x -> f (f x)", "('a -> 'a) -> 'a -> 'a"),
    ("fun x -> (x, x)", "'a -> 'a * 'a"),
    ("fun p -> match p with (a, b) -> (b, a)", "'a * 'b -> 'b * 'a"),
    ("[]", "'a list"),
    ("[1; 2; 3]", "int list"),
    ("fun x -> [x; x]", "'a -> 'a list"),
    ("fun x l -> x :: l", "'a -> 'a list -> 'a list"),
    ("map", "('a -> 'b) -> 'a list -> 'b list"),
    ("filter", "('a -> bool) -> 'a list -> 'a list"),
    ("fold_left", "('a -> 'b -> 'a) -> 'a -> 'b list -> 'a"),
    ("fold_right", "('a -> 'b -> 'b) -> 'a list -> 'b -> 'b"),
    ("zip", "'a list -> 'b list -> ('a * 'b) list"),
    ("assoc", "'a -> ('a * 'b) list -> 'b"),
    ("map (fun x -> x + 1)", "int list -> int list"),
    ("map fst", "('a * 'b) list -> 'a list"),
    ("compose (map snd) (zip [1; 2])", "'a list -> 'a list"),
    ("fun x -> if x then 1 else 2", "bool -> int"),
    ("fun f -> (fun x -> f (x x))", None),      # occurs check fails
])
def test_principal_types(src, expected):
    if expected is None:
        with pytest.raises(HMTypeError):
            type_of(src)
    else:
        assert type_of(src) == expected


def test_let_polymorphism():
    assert type_of("let id = fun x -> x in (id 1, id true, id \"s\")") == "int * bool * string"


def test_lambda_bound_variables_are_monomorphic():
    with pytest.raises(HMTypeError):
        type_of("fun id -> (id 1, id true)")


def test_let_inside_lambda_does_not_generalise_captured_variables():
    # y's type mentions x's type, which lives in the enclosing scope, so the
    # level based generaliser must not quantify it.
    assert type_of("fun x -> let y = x in y") == "'a -> 'a"
    with pytest.raises(HMTypeError):
        type_of("fun x -> let y = x in (y + 1, y && true)")


def test_level_adjustment_on_unification():
    # Inside the let, g is unified with a lambda bound variable from the
    # outside, which must pull its level down and stop generalisation.
    with pytest.raises(HMTypeError):
        type_of("fun f -> let g = (fun x -> f x; x) in (g 1, g true)")
    assert type_of("fun f -> let g = fun x -> f x in g") == "('a -> 'b) -> 'a -> 'b"


def test_let_rec_polymorphic_after_definition():
    src = "let rec len l = match l with [] -> 0 | _ :: t -> 1 + len t in (len [1], len [true])"
    assert type_of(src) == "int * int"


def test_let_rec_is_monomorphic_inside_its_own_body():
    # Standard HM: recursive uses are monomorphic (no polymorphic recursion).
    with pytest.raises(HMTypeError):
        type_of("let rec f x = (f 1; f true; ()) in f")


def test_program_types_are_generalised_at_top_level():
    inf = Inferencer()
    out = inf.check_program(parse_program("let pair x y = (x, y)\nlet p = pair 1 \"a\""))
    assert [(n, show(t)) for n, t in out] == [("pair", "'a -> 'b -> 'a * 'b"), ("p", "int * string")]


def test_shadowing_and_sequencing():
    assert type_of("let x = 1 in let x = true in x") == "bool"
    assert type_of('print_string "a"; 3') == "int"


def test_comparison_is_polymorphic_but_homogeneous():
    assert type_of("fun a b -> a < b") == "'a -> 'a -> bool"
    with pytest.raises(HMTypeError):
        type_of('1 = "1"')


# ---------------------------------------------------------------- messages

def error(src):
    with pytest.raises(HMTypeError) as e:
        type_of(src)
    return e.value


def test_error_mismatch_message_and_position():
    err = error("let f x = x + 1 in\nf true")
    assert err.pos == (2, 3)
    assert "has type bool but an expression was expected of type int" in err.msg


def test_error_occurs_check_message():
    err = error("fun x -> x x")
    assert "occurs inside" in err.msg and "infinite type" in err.msg


def test_error_not_a_function():
    err = error("let n = 3 in n 4")
    assert "not a function" in err.msg


def test_error_unbound_value():
    assert "Unbound value nope" in error("nope 1").msg


def test_error_statement_must_be_unit():
    assert "should have type unit" in error("1; 2").msg


def test_error_shows_inner_clash():
    err = error("[(1, true); (2, 3)]")
    assert "Type bool is not compatible with type int" in err.msg or \
           "Type int is not compatible with type bool" in err.msg


def test_error_in_pattern():
    err = error("fun l -> match l with [] -> 0 | (a, b) -> 1")
    assert "This pattern matches values of type" in err.msg


def test_duplicate_pattern_variable():
    assert "bound several times" in error("fun p -> match p with (x, x) -> x").msg


def test_parse_errors_have_positions():
    with pytest.raises(ParseError) as e:
        parse_program("let f x = (x + 1\nlet y = 2")
    assert e.value.pos[0] == 2


# ---------------------------------------------------------------- warnings

def warnings_of(src):
    inf = Inferencer()
    inf.check_program(parse_program(src))
    return [m for _, m in inf.warnings]


def test_non_exhaustive_list_match_gives_witness():
    (w,) = warnings_of("let f l = match l with [] -> 0 | [x] -> x")
    assert "not exhaustive" in w and "_ :: _ :: _" in w


def test_non_exhaustive_tuple_of_bools():
    (w,) = warnings_of("let f p = match p with (true, true) -> 1 | (false, _) -> 2")
    assert "(true, false)" in w


def test_unused_case():
    (w,) = warnings_of("let f b = match b with true -> 1 | false -> 2 | _ -> 3")
    assert "unused" in w


def test_integer_witness_avoids_listed_values():
    (w,) = warnings_of("let f n = match n with 0 -> 1 | 1 -> 1")
    assert w.endswith(": 2")


def test_exhaustive_match_is_quiet():
    assert warnings_of("""
let rec merge a b =
  match a, b with
  | [], l -> l
  | l, [] -> l
  | x :: xs, y :: ys -> if x <= y then x :: merge xs b else y :: merge a ys
""") == []
