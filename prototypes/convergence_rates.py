"""CF convergence rate analysis.

For each function, measures how many CF terms must be consumed before
the digits() algorithm pins each decimal digit.  Answers: "how many CF
terms does it take to reach 10 / 100 correct decimal digits?"

Run:  python prototypes/convergence_rates.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fractions import Fraction
from itertools import islice

from cfmath import (
    CF,
    digits_with_debug,
    Pi, E, Phi, Tau, EulerGamma,
    Sqrt,
    Sin, Cos, Tan,
    Ln, Log2,
)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def cost_profile(cf_factory, n_digits: int = 105) -> list[int]:
    """Return cumulative CF terms consumed after each decimal digit.

    Index 0 = integer part; index k = k-th fractional digit.
    Returns a list of length min(n_digits+1, actual_digits_emitted).
    """
    cf = cf_factory()
    profile: list[int] = []
    total = 0
    for _digit, cost in islice(digits_with_debug(cf), n_digits + 1):
        total += cost
        profile.append(total)
    return profile


def terms_at(profile: list[int], n: int) -> int | None:
    """Cumulative terms to produce n fractional digits (index n in profile)."""
    return profile[n] if n < len(profile) else None


# ---------------------------------------------------------------------------
# Benchmark table
# ---------------------------------------------------------------------------

TARGETS = [10, 100]
N_PROFILE = max(TARGETS) + 5   # compute a few extra so we always hit 100

BENCHMARKS: list[tuple[str, object]] = [
    # --- constants ---
    ("π",           Pi),
    ("e",           E),
    ("φ (golden)",  Phi),
    ("τ = 2π",      Tau),
    ("√2",          lambda: Sqrt(2)),
    ("√7",          lambda: Sqrt(7)),
    ("γ (E-M)",     EulerGamma),
    # --- logarithms ---
    ("ln 2",        lambda: Ln(2)),
    ("ln 3",        lambda: Ln(3)),
    ("log₂ 123",    lambda: Log2(123)),
    # --- trig at x = 1/4 ---
    ("sin(1/4)",    lambda: Sin(Fraction(1, 4))),
    ("cos(1/4)",    lambda: Cos(Fraction(1, 4))),
    ("tan(1/4)",    lambda: Tan(Fraction(1, 4))),
    # --- trig at x = 1/2 ---
    ("sin(1/2)",    lambda: Sin(Fraction(1, 2))),
    ("cos(1/2)",    lambda: Cos(Fraction(1, 2))),
    ("tan(1/2)",    lambda: Tan(Fraction(1, 2))),
]


def print_table(benchmarks, targets, n_profile):
    profiles = {}
    for label, factory in benchmarks:
        profiles[label] = cost_profile(factory, n_profile)

    # Header
    name_w = max(len(lbl) for lbl, _ in benchmarks) + 1
    target_cols = [f"{t}d" for t in targets]
    header = f"  {'function':<{name_w}}" + "".join(f"  {h:>7}" for h in target_cols) + "  avg/digit"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, _ in benchmarks:
        p = profiles[label]
        row = f"  {label:<{name_w}}"
        for t in targets:
            v = terms_at(p, t)
            row += f"  {v:>7}" if v is not None else f"  {'—':>7}"
        v100 = terms_at(p, 100)
        row += f"  {v100 / 100:.2f}" if v100 is not None else ""
        print(row)


# ---------------------------------------------------------------------------
# Per-digit cost breakdown for selected functions
# ---------------------------------------------------------------------------

DETAIL_FUNCTIONS = [
    ("π",        Pi),
    ("φ (golden)", Phi),
    ("sin(1/4)", lambda: Sin(Fraction(1, 4))),
]
DETAIL_DIGITS = 20


def print_detail(benchmarks, n_digits):
    for label, factory in benchmarks:
        profile = cost_profile(factory, n_digits)
        # cost per digit = difference between consecutive cumulative totals
        costs = [profile[0]] + [profile[i] - profile[i - 1] for i in range(1, len(profile))]
        print(f"  {label}:")
        print(f"    {'digit':>5}  {'cost':>5}  {'cumul':>6}  bar")
        for i, (cum, cost) in enumerate(zip(profile, costs)):
            bar = "█" * cost
            kind = "int" if i == 0 else f".{i:02d}"
            print(f"    {kind:>5}  {cost:>5}  {cum:>6}  {bar}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print()
print("=" * 68)
print("CF convergence: CF terms consumed to produce N decimal digits")
print("=" * 68)
print()
print_table(BENCHMARKS, TARGETS, N_PROFILE)

print()
print("=" * 68)
print(f"Per-digit cost breakdown (first {DETAIL_DIGITS} digits)")
print("=" * 68)
print()
print_detail(DETAIL_FUNCTIONS, DETAIL_DIGITS)
