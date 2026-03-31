"""Prototype: error estimates for take(n) truncations.

Shows how .err_estimate tightens as more CF terms are taken, and how
it compares to the actual truncation error (measured via mpmath).

Run:  python prototypes/err_estimate.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
from fractions import Fraction

from cfmath import CF, Pi, E, Phi, Sqrt, Sin, Ln


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log10_or_inf(f: Fraction) -> float:
    """Return floor(log10(f)) — i.e. the number of correct decimal digits."""
    if f == 0:
        return float("inf")
    return -math.log10(float(f))


def show_convergence(label: str, cf_factory, n_terms_list: list[int]) -> None:
    print(f"  {label}")
    print(f"    {'n':>4}  {'bound':>12}  {'~digits':>8}  {'convergent value'}")
    print(f"    {'-'*4}  {'-'*12}  {'-'*8}  {'-'*30}")
    for n in n_terms_list:
        trunc = cf_factory().take(n)
        bound = trunc.err_estimate
        digits = log10_or_inf(bound)
        val = trunc.to_fraction()
        print(f"    {n:>4}  {float(bound):>12.3e}  {digits:>8.1f}  {float(val):.15f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print()
print("=" * 66)
print("take(n).err_estimate — truncation error upper bounds")
print()
print("  bound = |p_n/q_n - p_{n-1}/q_{n-1}| = 1 / (q_n * q_{n-1})")
print("  '~digits' = floor(-log10(bound)) ≈ correct decimal digits")
print("=" * 66)
print()

show_convergence(
    "π  [3; 7, 15, 1, 292, 1, 1, 1, 2, ...]",
    Pi,
    [1, 2, 3, 4, 5, 8, 10, 15],
)

show_convergence(
    "e  [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]",
    E,
    [1, 3, 5, 8, 10, 15],
)

show_convergence(
    "φ  [1; 1, 1, 1, ...] (slowest-converging CF)",
    Phi,
    [5, 10, 15, 20, 25, 30],
)

show_convergence(
    "√2  [1; 2, 2, 2, ...]",
    lambda: Sqrt(2),
    [5, 10, 15, 20],
)

show_convergence(
    "sin(1/4)  via direct GCF",
    lambda: Sin(Fraction(1, 4)),
    [5, 10, 15, 20],
)

# ---------------------------------------------------------------------------
# Spot-check: consecutive-convergent bound vs next-convergent gap
#
# No mpmath or math.pi needed.  We use Pi().take(n+1) as the "better"
# approximation: the difference take(n+1) - take(n) is 1/(q_n * q_{n+1}),
# which is the tightest possible bound given one more term.
# err_estimate uses 1/(q_n * q_{n-1}), which is looser but requires no
# extra terms.  The ratio shows how much looser it is.
# ---------------------------------------------------------------------------

print("=" * 66)
print("Spot-check for Pi().take(n) — no mpmath, no math.pi")
print()
print("  gap(n)   = |take(n+1) - take(n)|  (tightest bound, needs +1 term)")
print("  bound(n) = err_estimate on take(n) (no extra terms needed)")
print()
print(f"  {'n':>4}  {'gap(n)':>14}  {'bound(n)':>14}  {'bound/gap':>10}")
print(f"  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*10}")

for n in [2, 3, 4, 5, 6, 8, 10]:
    trunc_n  = Pi().take(n)
    trunc_n1 = Pi().take(n + 1)
    gap   = abs(trunc_n1.to_fraction() - trunc_n.to_fraction())
    bound = trunc_n.err_estimate
    ratio = float(bound / gap)
    print(f"  {n:>4}  {float(gap):>14.3e}  {float(bound):>14.3e}  {ratio:>10.2f}x")

print()
print("  bound/gap ≈ q_{n+1}/q_{n-1} — how much looser the bound is.")
print("  Large ratio at n=5 because a_5=292 makes q_6 >> q_4.")
print()
