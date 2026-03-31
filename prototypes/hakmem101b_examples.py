"""HAKMEM 101B examples — continued fraction arithmetic showcase.

These computations follow the spirit of Gosper's 1972 HAKMEM memo item 101B,
which demonstrated that exact CF arithmetic can produce results not reducible
to simpler expressions.  Every computation here uses only integer arithmetic
and the algorithms in cfmath.gosper; no floating-point conversion is needed.

Run:  python prototypes/hakmem101b_examples.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fractions import Fraction
from cfmath import CF, E, Pi, Sqrt, Exp, Ln, cf_homographic
from cfmath.convergents import convergent


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _cf_terms(cf: CF, n: int = 14) -> list[int]:
    return list(cf.take(n))


def _show(label: str, cf: CF, n_terms: int = 14) -> None:
    terms = _cf_terms(cf, n_terms)
    # Build a human-friendly [a0; a1, a2, ...] string
    if len(terms) == 0:
        cf_str = "[]"
    elif len(terms) == 1:
        cf_str = f"[{terms[0]}]"
    else:
        cf_str = f"[{terms[0]}; {', '.join(str(t) for t in terms[1:])}  ...]"

    # Decimal approximation via the last convergent shown
    approx = float(convergent(cf, min(len(terms) - 1, 10)))
    print(f"{label}")
    print(f"  CF    = {cf_str}")
    print(f"  ≈ {approx:.10f}")
    print()


# ---------------------------------------------------------------------------
# Constants (each call produces an independent lazy CF object)
# ---------------------------------------------------------------------------

print("=" * 60)
print("HAKMEM 101B: continued-fraction arithmetic examples")
print("=" * 60)
print()

# --- sqrt(105) ---
# Exact periodic CF: the standard algorithm gives the period exactly.
# 10² = 100 < 105 < 121 = 11², so a₀ = 10.  Period is [4, 20].
_show("sqrt(105)  =  [10; (4, 20), (4, 20), ...]  (period 2)", Sqrt(105))

# --- coth(1/69) ---
# coth(x) = (e^2x + 1) / (e^2x − 1)
# Written as the Möbius transform (y+1)/(y−1) with y = e^(2/69).
# Result: coth(1/n) = [n; 3n, 5n, 7n, ...] — an arithmetic progression!
# This is one of Gosper's celebrated examples: an infinite non-periodic CF
# whose terms follow a simple linear pattern, uncoverable only by exact arithmetic.
y = Exp(Fraction(2, 69))
coth_69 = cf_homographic(y, 1, 1, 1, -1)
_show("coth(1/69)  =  [69; 3·69, 5·69, 7·69, ...]  — arithmetic progression", coth_69)

# --- e + π ---
_show("e + π", E() + Pi())

# --- e × π ---
_show("e × π", E() * Pi())

# --- e / π ---
_show("e / π", E() / Pi())

# --- e^π  (Gelfond's constant) ---
# Uses Exp(CF) path: approximates π via its convergents, feeds to exp.
_show("e^π  (Gelfond's constant)", Exp(Pi()))

# ---------------------------------------------------------------------------
# Bonus: showing that Exp(r * Ln(x)) = x^r
# ---------------------------------------------------------------------------
print("-" * 60)
print("Bonus: arbitrary rational powers via Exp(r·Ln(x))")
print()

# 2^(3/2) = √8 — should match Sqrt(8)'s CF exactly
r = Fraction(3, 2)
pow_cf   = Exp(r * Ln(2))
sqrt8_cf = Sqrt(8)
terms_pow   = _cf_terms(pow_cf,   10)
terms_sqrt8 = _cf_terms(sqrt8_cf, 10)
print(f"  2^(3/2) via Exp(3/2·Ln(2)):  {terms_pow}")
print(f"  Sqrt(8):                      {terms_sqrt8}")
print(f"  First 10 terms match:         {terms_pow == terms_sqrt8}")
print()

# ---------------------------------------------------------------------------
# Convergents of e*π as an illustration
# ---------------------------------------------------------------------------
print("-" * 60)
print("Convergents of e·π:")
epi = E() * Pi()
for i in range(8):
    c = convergent(epi, i)
    print(f"  p_{i}/q_{i} = {c}  ≈ {float(c):.8f}")
