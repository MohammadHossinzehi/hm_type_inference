"""Lexer, AST and recursive descent parser for the mini ML surface language.

The grammar is a small, OCaml flavoured subset:

    program  ::= item* ;  item ::= 'let' ['rec'] IDENT param* '=' expr [';;']
                                 | expr ';;'
    expr     ::= seq
    seq      ::= prefix (';' seq)?
    prefix   ::= 'let' ['rec'] IDENT param* '=' expr 'in' expr
               | 'fun' param+ '->' expr
               | 'if' expr 'then' expr 'else' expr
               | 'match' expr 'with' ['|'] pat '->' expr ('|' pat '->' expr)*
               | tuple
    tuple    ::= or (',' or)*
    or / and / compare / concat(^ @) / cons(::) / additive / multiplicative
    unary    ::= '-' unary | app
    app      ::= atom atom*

Operators are desugared into applications of ordinary prelude functions, so
the type checker only ever sees a handful of core forms.  `&&` and `||`
become `if` expressions so they short circuit at run time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


class ParseError(Exception):
    def __init__(self, pos: Tuple[int, int], msg: str):
        self.pos = pos
        self.msg = msg
        super().__init__(f"line {pos[0]}, col {pos[1]}: {msg}")


# ---------------------------------------------------------------- tokens

KEYWORDS = {
    "let", "rec", "in", "fun", "if", "then", "else", "match", "with",
    "true", "false", "mod",
}

# Longest first so that '->' wins over '-', '::' over ':' and so on.
SYMBOLS = [
    ";;", "->", "::", "<=", ">=", "<>", "&&", "||",
    "(", ")", "[", "]", ",", ";", "|", "_", "=", "<", ">",
    "+", "-", "*", "/", "^", "@",
]


@dataclass
class Token:
    kind: str          # 'int' | 'string' | 'ident' | 'kw' | 'sym' | 'eof'
    value: object
    pos: Tuple[int, int]

    def is_(self, kind: str, value: object = None) -> bool:
        return self.kind == kind and (value is None or self.value == value)


def tokenize(src: str) -> List[Token]:
    toks: List[Token] = []
    i, line, col = 0, 1, 1
    n = len(src)

    def advance(k: int) -> None:
        nonlocal i, line, col
        for _ in range(k):
            if src[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        c = src[i]
        if c in " \t\r\n":
            advance(1)
            continue
        if src.startswith("(*", i):           # nested comments, like OCaml
            start = (line, col)
            depth = 0
            while True:
                if i >= n:
                    raise ParseError(start, "unterminated comment")
                if src.startswith("(*", i):
                    depth += 1
                    advance(2)
                elif src.startswith("*)", i):
                    depth -= 1
                    advance(2)
                    if depth == 0:
                        break
                else:
                    advance(1)
            continue
        pos = (line, col)
        if c.isdigit():
            j = i
            while j < n and src[j].isdigit():
                j += 1
            toks.append(Token("int", int(src[i:j]), pos))
            advance(j - i)
            continue
        if c.isalpha() or (c == "_" and i + 1 < n and (src[i + 1].isalnum() or src[i + 1] in "_'")):
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_'"):
                j += 1
            word = src[i:j]
            toks.append(Token("kw" if word in KEYWORDS else "ident", word, pos))
            advance(j - i)
            continue
        if c == '"':
            j = i + 1
            out = []
            escapes = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
            while True:
                if j >= n or src[j] == "\n":
                    raise ParseError(pos, "unterminated string literal")
                if src[j] == '"':
                    break
                if src[j] == "\\":
                    if j + 1 >= n or src[j + 1] not in escapes:
                        raise ParseError(pos, "bad escape in string literal")
                    out.append(escapes[src[j + 1]])
                    j += 2
                    continue
                out.append(src[j])
                j += 1
            toks.append(Token("string", "".join(out), pos))
            advance(j + 1 - i)
            continue
        for s in SYMBOLS:
            if src.startswith(s, i):
                toks.append(Token("sym", s, pos))
                advance(len(s))
                break
        else:
            raise ParseError(pos, f"unexpected character {c!r}")
    toks.append(Token("eof", None, (line, col)))
    return toks


# ---------------------------------------------------------------- AST

Pos = Tuple[int, int]


@dataclass
class Expr:
    pos: Pos = field(default=(0, 0), compare=False, repr=False, kw_only=True)


@dataclass
class Lit(Expr):
    value: Union[int, bool, str, None]     # None encodes ()


@dataclass
class Var(Expr):
    name: str


@dataclass
class Lam(Expr):
    param: str                             # '_' for an ignored argument
    body: Expr


@dataclass
class App(Expr):
    fn: Expr
    arg: Expr


@dataclass
class Let(Expr):
    name: str
    rec: bool
    value: Expr
    body: Expr


@dataclass
class If(Expr):
    cond: Expr
    then: Expr
    other: Expr


@dataclass
class Tuple_(Expr):
    items: List[Expr]


@dataclass
class Nil(Expr):
    pass


@dataclass
class Cons(Expr):
    head: Expr
    tail: Expr


@dataclass
class Seq(Expr):
    first: Expr
    second: Expr


@dataclass
class Match(Expr):
    scrutinee: Expr
    arms: List[Tuple["Pat", Expr]]


# Patterns

@dataclass
class Pat:
    pos: Pos = field(default=(0, 0), compare=False, repr=False, kw_only=True)


@dataclass
class PVar(Pat):
    name: str


@dataclass
class PWild(Pat):
    pass


@dataclass
class PLit(Pat):
    value: Union[int, bool, str, None]


@dataclass
class PTuple(Pat):
    items: List[Pat]


@dataclass
class PNil(Pat):
    pass


@dataclass
class PCons(Pat):
    head: Pat
    tail: Pat


# Top level items

@dataclass
class Decl:
    name: str
    rec: bool
    value: Expr
    pos: Pos


@dataclass
class TopExpr:
    expr: Expr
    pos: Pos


Item = Union[Decl, TopExpr]


# ---------------------------------------------------------------- parser

BINARY_LEVELS = [
    # (operators, associativity); lowest precedence first
    (("=", "<>", "<", ">", "<=", ">="), "left"),
    (("^", "@"), "right"),
    (("::",), "right"),
    (("+", "-"), "left"),
    (("*", "/", "mod"), "left"),
]

ATOM_START_SYMS = {"(", "["}


class Parser:
    def __init__(self, src: str):
        self.toks = tokenize(src)
        self.i = 0

    # -- token helpers
    @property
    def tok(self) -> Token:
        return self.toks[self.i]

    def peek(self, k: int = 1) -> Token:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def at(self, kind: str, value: object = None) -> bool:
        return self.tok.is_(kind, value)

    def at_sym(self, s: str) -> bool:
        return self.tok.is_("sym", s)

    def at_kw(self, s: str) -> bool:
        return self.tok.is_("kw", s)

    def expect_sym(self, s: str) -> Token:
        if not self.at_sym(s):
            raise ParseError(self.tok.pos, f"expected '{s}' but found {self.describe(self.tok)}")
        return self.next()

    def expect_kw(self, s: str) -> Token:
        if not self.at_kw(s):
            raise ParseError(self.tok.pos, f"expected '{s}' but found {self.describe(self.tok)}")
        return self.next()

    @staticmethod
    def describe(t: Token) -> str:
        if t.kind == "eof":
            return "end of input"
        if t.kind == "string":
            return f'string "{t.value}"'
        return f"'{t.value}'"

    # -- program
    def program(self) -> List[Item]:
        items: List[Item] = []
        while not self.at("eof"):
            if self.at_sym(";;"):
                self.next()
                continue
            pos = self.tok.pos
            if self.at_kw("let"):
                save = self.i
                name, rec, value = self.let_binding()
                if self.at_kw("in"):              # it was an expression after all
                    self.i = save
                    items.append(TopExpr(self.expr(), pos))
                    self.end_item()
                else:
                    items.append(Decl(name, rec, value, pos))
                    if self.at_sym(";;"):
                        self.next()
                continue
            items.append(TopExpr(self.expr(), pos))
            self.end_item()
        return items

    def end_item(self) -> None:
        if self.at_sym(";;"):
            self.next()
        elif not self.at("eof"):
            raise ParseError(self.tok.pos, f"expected ';;' after top level expression, found {self.describe(self.tok)}")

    def let_binding(self) -> Tuple[str, bool, Expr]:
        let_tok = self.expect_kw("let")
        rec = False
        if self.at_kw("rec"):
            self.next()
            rec = True
        name_tok = self.tok
        if self.at("ident"):
            name = self.next().value
        elif self.at_sym("_"):
            self.next()
            name = "_"
        elif self.at_sym("(") and self.peek().is_("sym", ")"):
            self.next(); self.next()
            name = "_"
        else:
            raise ParseError(name_tok.pos, f"expected a name after 'let', found {self.describe(name_tok)}")
        params = self.params()
        self.expect_sym("=")
        value = self.expr()
        for p, ppos in reversed(params):
            value = Lam(p, value, pos=ppos)
        if rec and not isinstance(value, Lam):
            raise ParseError(let_tok.pos, "'let rec' requires a function on the right hand side")
        return name, rec, value

    def params(self) -> List[Tuple[str, Pos]]:
        out = []
        while True:
            t = self.tok
            if t.kind == "ident":
                out.append((self.next().value, t.pos))
            elif t.is_("sym", "_"):
                self.next()
                out.append(("_", t.pos))
            elif t.is_("sym", "(") and self.peek().is_("sym", ")"):
                self.next(); self.next()
                out.append(("_", t.pos))
            else:
                return out

    # -- expressions
    def expr(self) -> Expr:
        e = self.prefix()
        if self.at_sym(";"):
            pos = self.next().pos
            if self.at_sym(";;") or self.at("eof") or self.at_sym(")") or self.at_sym("|"):
                return e                            # tolerate a trailing ';'
            return Seq(e, self.expr(), pos=pos)
        return e

    def prefix(self) -> Expr:
        t = self.tok
        if t.is_("kw", "let"):
            name, rec, value = self.let_binding()
            self.expect_kw("in")
            return Let(name, rec, value, self.expr(), pos=t.pos)
        if t.is_("kw", "fun"):
            self.next()
            params = self.params()
            if not params:
                raise ParseError(self.tok.pos, "expected a parameter after 'fun'")
            self.expect_sym("->")
            body = self.expr()
            for p, ppos in reversed(params):
                body = Lam(p, body, pos=ppos)
            return body
        if t.is_("kw", "if"):
            self.next()
            c = self.expr()
            self.expect_kw("then")
            a = self.prefix()
            self.expect_kw("else")
            b = self.prefix()
            return If(c, a, b, pos=t.pos)
        if t.is_("kw", "match"):
            self.next()
            scrut = self.expr()
            self.expect_kw("with")
            if self.at_sym("|"):
                self.next()
            arms = []
            while True:
                p = self.pattern()
                self.expect_sym("->")
                arms.append((p, self.expr()))
                if self.at_sym("|"):
                    self.next()
                    continue
                break
            return Match(scrut, arms, pos=t.pos)
        return self.tuple_expr()

    def tuple_expr(self) -> Expr:
        pos = self.tok.pos
        first = self.or_expr()
        if not self.at_sym(","):
            return first
        items = [first]
        while self.at_sym(","):
            self.next()
            items.append(self.or_expr())
        return Tuple_(items, pos=pos)

    def or_expr(self) -> Expr:
        left = self.and_expr()
        while self.at_sym("||"):
            pos = self.next().pos
            right = self.and_expr()
            left = If(left, Lit(True, pos=pos), right, pos=pos)
        return left

    def and_expr(self) -> Expr:
        left = self.binary(0)
        while self.at_sym("&&"):
            pos = self.next().pos
            right = self.binary(0)
            left = If(left, right, Lit(False, pos=pos), pos=pos)
        return left

    def at_op(self, ops) -> Optional[str]:
        t = self.tok
        if (t.kind == "sym" or t.is_("kw", "mod")) and t.value in ops:
            return t.value
        return None

    def binary(self, level: int) -> Expr:
        if level == len(BINARY_LEVELS):
            return self.unary()
        ops, assoc = BINARY_LEVELS[level]
        left = self.binary(level + 1)
        if assoc == "right":
            op = self.at_op(ops)
            if op is None:
                return left
            pos = self.next().pos
            right = self.binary(level)
            return self.make_binop(op, left, right, pos)
        while True:
            op = self.at_op(ops)
            if op is None:
                return left
            pos = self.next().pos
            right = self.binary(level + 1)
            left = self.make_binop(op, left, right, pos)

    @staticmethod
    def make_binop(op: str, left: Expr, right: Expr, pos: Pos) -> Expr:
        if op == "::":
            return Cons(left, right, pos=pos)
        return App(App(Var(op, pos=pos), left, pos=pos), right, pos=pos)

    def unary(self) -> Expr:
        if self.at_sym("-"):
            pos = self.next().pos
            operand = self.unary()
            if isinstance(operand, Lit) and isinstance(operand.value, int) and not isinstance(operand.value, bool):
                return Lit(-operand.value, pos=pos)
            return App(Var("~-", pos=pos), operand, pos=pos)
        return self.application()

    def starts_atom(self) -> bool:
        t = self.tok
        if t.kind in ("int", "string", "ident"):
            return True
        if t.kind == "kw" and t.value in ("true", "false"):
            return True
        return t.kind == "sym" and t.value in ATOM_START_SYMS

    def application(self) -> Expr:
        fn = self.atom()
        while self.starts_atom():
            arg = self.atom()
            fn = App(fn, arg, pos=arg.pos)
        return fn

    def atom(self) -> Expr:
        t = self.tok
        if t.kind == "int":
            self.next()
            return Lit(t.value, pos=t.pos)
        if t.kind == "string":
            self.next()
            return Lit(t.value, pos=t.pos)
        if t.is_("kw", "true") or t.is_("kw", "false"):
            self.next()
            return Lit(t.value == "true", pos=t.pos)
        if t.kind == "ident":
            self.next()
            return Var(t.value, pos=t.pos)
        if t.is_("sym", "("):
            self.next()
            if self.at_sym(")"):
                self.next()
                return Lit(None, pos=t.pos)
            op = self.operator_section()
            if op is not None:
                return Var(op, pos=t.pos)
            e = self.expr()
            self.expect_sym(")")
            return e
        if t.is_("sym", "["):
            self.next()
            items = []
            if not self.at_sym("]"):
                items.append(self.prefix())
                while self.at_sym(";"):
                    self.next()
                    if self.at_sym("]"):
                        break
                    items.append(self.prefix())
            end = self.expect_sym("]")
            out: Expr = Nil(pos=end.pos)
            for item in reversed(items):
                out = Cons(item, out, pos=item.pos)
            return out
        raise ParseError(t.pos, f"unexpected {self.describe(t)}")

    def operator_section(self) -> Optional[str]:
        """`(+)`, `(::)` is not a function in OCaml, so neither here."""
        t, nxt = self.tok, self.peek()
        ops = {"+", "-", "*", "/", "^", "@", "=", "<>", "<", ">", "<=", ">="}
        if ((t.kind == "sym" and t.value in ops) or t.is_("kw", "mod")) and nxt.is_("sym", ")"):
            self.next(); self.next()
            return t.value
        return None

    # -- patterns
    def pattern(self) -> Pat:
        pos = self.tok.pos
        first = self.cons_pattern()
        if not self.at_sym(","):
            return first
        items = [first]
        while self.at_sym(","):
            self.next()
            items.append(self.cons_pattern())
        return PTuple(items, pos=pos)

    def cons_pattern(self) -> Pat:
        head = self.atom_pattern()
        if self.at_sym("::"):
            pos = self.next().pos
            return PCons(head, self.cons_pattern(), pos=pos)
        return head

    def atom_pattern(self) -> Pat:
        t = self.tok
        if t.kind == "ident":
            self.next()
            return PVar(t.value, pos=t.pos)
        if t.is_("sym", "_"):
            self.next()
            return PWild(pos=t.pos)
        if t.kind in ("int", "string"):
            self.next()
            return PLit(t.value, pos=t.pos)
        if t.is_("sym", "-") and self.peek().kind == "int":
            self.next()
            return PLit(-self.next().value, pos=t.pos)
        if t.is_("kw", "true") or t.is_("kw", "false"):
            self.next()
            return PLit(t.value == "true", pos=t.pos)
        if t.is_("sym", "("):
            self.next()
            if self.at_sym(")"):
                self.next()
                return PLit(None, pos=t.pos)
            p = self.pattern()
            self.expect_sym(")")
            return p
        if t.is_("sym", "["):
            self.next()
            items = []
            if not self.at_sym("]"):
                items.append(self.pattern())
                while self.at_sym(";"):
                    self.next()
                    items.append(self.pattern())
            end = self.expect_sym("]")
            out: Pat = PNil(pos=end.pos)
            for item in reversed(items):
                out = PCons(item, out, pos=item.pos)
            return out
        raise ParseError(t.pos, f"unexpected {self.describe(t)} in pattern")


def parse_program(src: str) -> List[Item]:
    return Parser(src).program()


def parse_expr(src: str) -> Expr:
    p = Parser(src)
    e = p.expr()
    if p.at_sym(";;"):
        p.next()
    if not p.at("eof"):
        raise ParseError(p.tok.pos, f"unexpected {p.describe(p.tok)} after expression")
    return e
