"""Command line interface.

    python -m hm infer FILE.ml          print the type of every top level binding
    python -m hm infer -e "EXPR"        print the type of one expression
    python -m hm run FILE.ml            type check, then evaluate
    python -m hm repl                   interactive toplevel
    --algorithm w                       use the reference Algorithm W for `infer`
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import syntax as S
from .algorithm_w import AlgorithmW, WTypeError, to_display
from .infer import HMTypeError, Inferencer
from .interp import Interpreter
from .types import show
from .values import MLRuntimeError, show_value


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _report_warnings(inf: Inferencer, filename: str) -> None:
    for (line, col), msg in inf.warnings:
        print(f"{filename}:{line}:{col}: Warning: {msg}", file=sys.stderr)
    inf.warnings.clear()


def cmd_infer(src: str, filename: str, algorithm: str, single: bool) -> int:
    if algorithm == "w":
        w = AlgorithmW()
        if single:
            print(to_display(w.infer_expr(S.parse_expr(src))))
        else:
            for name, t in w.check_program(S.parse_program(src)):
                if name != "_":
                    print(f"val {name} : {to_display(t)}" if name != "-" else f"- : {to_display(t)}")
        return 0
    inf = Inferencer()
    if single:
        print(show(inf.infer_expr(S.parse_expr(src))))
    else:
        for name, t in inf.check_program(S.parse_program(src)):
            if name != "_":
                print(f"val {name} : {show(t)}" if name != "-" else f"- : {show(t)}")
    _report_warnings(inf, filename)
    return 0


def cmd_run(src: str, filename: str) -> int:
    items = S.parse_program(src)
    inf = Inferencer()
    inf.check_program(items)                 # refuse to run ill typed code
    _report_warnings(inf, filename)
    interp = Interpreter()
    for item in items:
        v = interp.run_item(item)
        if isinstance(item, S.TopExpr) and v is not None:
            sys.stdout.write(show_value(v) + "\n")
    sys.stdout.flush()
    return 0


def cmd_repl() -> int:
    inf, interp = Inferencer(), Interpreter()
    print("hm toplevel. End each phrase with ';;'. Ctrl+D to quit.")
    buf: List[str] = []
    while True:
        try:
            line = input("# " if not buf else "  ")
        except EOFError:
            print()
            return 0
        buf.append(line)
        if not line.rstrip().endswith(";;"):
            continue
        src, buf = "\n".join(buf), []
        try:
            for item in S.parse_program(src):
                name, t = inf.check_item(item)
                _report_warnings(inf, "toplevel")
                v = interp.run_item(item)
                print(f"val {name} : {show(t)} = {show_value(v)}" if name != "-"
                      else f"- : {show(t)} = {show_value(v)}")
        except (S.ParseError, HMTypeError) as err:
            print(f"Error: {err}")
        except MLRuntimeError as err:
            print(f"Exception: {err}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hm", description="Hindley Milner type inference for a small ML")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_inf = sub.add_parser("infer", help="print inferred types")
    p_inf.add_argument("file", nargs="?")
    p_inf.add_argument("-e", "--expr", help="infer a single expression instead of a file")
    p_inf.add_argument("--algorithm", choices=["j", "w"], default="j")
    p_run = sub.add_parser("run", help="type check and run a program")
    p_run.add_argument("file")
    sub.add_parser("repl", help="interactive toplevel")
    args = ap.parse_args(argv)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

    try:
        if args.cmd == "infer":
            if args.expr is not None:
                return cmd_infer(args.expr, "<expr>", args.algorithm, single=True)
            if not args.file:
                ap.error("infer needs a FILE or -e EXPR")
            return cmd_infer(_read(args.file), args.file, args.algorithm, single=False)
        if args.cmd == "run":
            return cmd_run(_read(args.file), args.file)
        return cmd_repl()
    except S.ParseError as err:
        print(f"Syntax error: {err}", file=sys.stderr)
    except HMTypeError as err:
        print(f"Type error: {err}", file=sys.stderr)
    except WTypeError as err:
        print(f"Type error (algorithm W): {err}", file=sys.stderr)
    except MLRuntimeError as err:
        print(f"Exception: {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
