"""Compare Algorithm J (levels + union find) with textbook Algorithm W.

Two workloads that stress the difference:

  * let_chain(n):  n nested lets, each binding a polymorphic function and
    using the previous one.  W rescans the free variables of the whole
    environment at every let, so its cost grows quadratically.
  * pair_tower(n): the classic pathological program `let x0 = fun y -> (y, y)
    in let x1 = fun y -> x0 (x0 y) ...`.  The principal type of x_n has 4^n
    leaves, so *both* algorithms blow up; HM typability is DEXPTIME complete
    and no implementation trick avoids that.  It is here as an honest limit.

Usage:  python bench.py
"""
import sys
import time

from hm import syntax as S
from hm.algorithm_w import AlgorithmW
from hm.infer import Inferencer


def let_chain(n: int) -> str:
    parts = ["let f0 = fun x -> x in"]
    for i in range(1, n):
        parts.append(f"let f{i} = fun x -> f{i - 1} (f{i - 1} x) in")
    parts.append(f"f{n - 1}")
    return "\n".join(parts)


def pair_tower(n: int) -> str:
    parts = ["let x0 = fun y -> (y, y) in"]
    for i in range(1, n):
        parts.append(f"let x{i} = fun y -> x{i - 1} (x{i - 1} y) in")
    parts.append(f"x{n - 1}")
    return "\n".join(parts)


def timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> None:
    sys.setrecursionlimit(100000)
    print(f"{'workload':<16}{'n':>6}{'J (ms)':>12}{'W (ms)':>12}{'W / J':>9}")
    for name, make, sizes in [("let_chain", let_chain, [50, 100, 200, 400]),
                              ("pair_tower", pair_tower, [2, 3, 4, 5])]:
        for n in sizes:
            e = S.parse_expr(make(n))
            j, w = Inferencer(), AlgorithmW()
            tj = timed(lambda: j.infer_expr(e)) * 1000
            tw = timed(lambda: w.infer_expr(e)) * 1000
            print(f"{name:<16}{n:>6}{tj:>12.1f}{tw:>12.1f}{tw / tj:>9.1f}")


if __name__ == "__main__":
    main()
