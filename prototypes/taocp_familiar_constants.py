"""TAOCP Vol.2 §4.5.3 Table 1 — 'familiar real numbers' as continued fractions.

Knuth's table demonstrates that well-known constants fall into three families:
  - Quadratic irrationals (√2, φ, ...): eventually periodic CFs
  - e: aperiodic but with a visible pattern (1, 2, 1, 1, 4, 1, 1, 6, ...)
  - Transcendentals like π, γ: no discernible pattern

Run:  python prototypes/taocp_familiar_constants.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mpmath import e, euler, floor, log, mp, mpf, phi, pi, sqrt

mp.dps = 120  # enough headroom for 20 CF terms


def cf_of(x, n: int = 20) -> list[int]:
    """Extract n CF partial quotients from a mpmath number."""
    terms = []
    x = mpf(x)
    for _ in range(n):
        a = int(floor(x))
        terms.append(a)
        frac = x - a
        if frac < mpf("1e-100"):
            break
        x = mpf(1) / frac
    return terms


CONSTANTS: list[tuple[str, object]] = [
    ("φ  = [1; 1, 1, 1, ...]", phi),
    ("√2 = [1; 2, 2, 2, ...]", sqrt(2)),
    ("√3 = [1; 1, 2, 1, 2, ...]", sqrt(3)),
    ("√5 = [2; 4, 4, 4, ...]", sqrt(5)),
    ("e  = [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]", e),
    ("π  = [3; 7, 15, 1, 292, ...]", pi),
    ("ln 2", log(2)),
    ("ln 10", log(10)),
    ("γ  (Euler-Mascheroni)", euler),
    ("log₁₀ e  = 1/ln 10", log(e, 10)),
]

print(f"{'Constant':<38}  CF (first 20 partial quotients)")
print("-" * 80)
for name, val in CONSTANTS:
    terms = cf_of(val)
    body = f"[{terms[0]}; {', '.join(str(t) for t in terms[1:])}]"
    print(f"{name:<38}  {body}")
