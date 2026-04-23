"""Prototype: π via arctan continued fractions + Gosper arithmetic.

Strategy:
    π = 16·arctan(1/5) − 4·arctan(1/239)   (Machin's formula)

Each arctan is computed as a generalized CF using Euler's formula:

    arctan(x) = x / (1 + 1²x² / (3 + 2²x² / (5 + 3²x² / (7 + ...))))

Written as (b_n, a_{n+1}) pairs for the denominator part:
    b_n     = 2n+1
    a_{n+1} = (n+1)² · x²     for n = 0, 1, 2, ...

Note: the signs are POSITIVE here (unlike atanh which uses negative signs).
The atanh identity has the same structural shape but with alternating subtraction,
producing negative a_{n+1} values.  arctan has all-positive nested fractions.

Then:  arctan(x) = CF.from_fraction(x_num, x_den) / CF.from_generalized_cf(pairs)
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction
from typing import Iterator

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cfmath.convergents import convergent
from cfmath.core import CF

# ---------------------------------------------------------------------------
# Generalized CF pairs for arctan(x_num/x_den)
# ---------------------------------------------------------------------------


def _arctan_pairs(x_num: int, x_den: int) -> Iterator[tuple[int, Fraction]]:
    """Yield (b_n, a_{n+1}) pairs for the denominator CF of arctan(x_num/x_den).

    Using Euler's generalized CF:
        arctan(x) = x / (1 + 1²x² / (3 + 2²x² / (5 + ...)))

    Pairs:
        b_n     = 2n+1
        a_{n+1} = (n+1)² · Fraction(x_num, x_den)²

    The a_{n+1} values are POSITIVE, matching the all-positive nested fractions
    in the arctan series.  (atanh would use negative signs here.)
    """
    x2 = Fraction(x_num * x_num, x_den * x_den)
    n = 0
    while True:
        b_n = 2 * n + 1
        a_next = (n + 1) ** 2 * x2
        yield (b_n, a_next)
        n += 1


# ---------------------------------------------------------------------------
# Counting wrapper for pairs (to measure how many pairs are consumed)
# ---------------------------------------------------------------------------


class _CountingPairs:
    """Wraps a pairs iterator and counts how many pairs have been consumed."""

    def __init__(self, pairs_iter: Iterator) -> None:
        self._iter = pairs_iter
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        val = next(self._iter)
        self.count += 1
        return val


# ---------------------------------------------------------------------------
# arctan as a CF
# ---------------------------------------------------------------------------


def arctan_cf(x_num: int, x_den: int, counter: _CountingPairs | None = None) -> CF:
    """Return a CF for arctan(x_num / x_den).

    Uses Euler's generalized CF for the denominator part, then divides x into it:
        arctan(x) = CF.from_fraction(x_num, x_den) / CF.from_generalized_cf(pairs)

    If *counter* is provided it should be a _CountingPairs wrapping the pairs
    iterator — the caller can inspect counter.count after consuming the CF.
    """
    raw_pairs = _arctan_pairs(x_num, x_den)
    if counter is not None:
        pairs = counter
        # Re-attach the raw pairs as the source
        counter._iter = raw_pairs
    else:
        pairs = raw_pairs

    denom_cf = CF.from_generalized_cf(pairs)
    return CF.from_fraction(x_num, x_den) / denom_cf


# ---------------------------------------------------------------------------
# π via Machin's formula
# ---------------------------------------------------------------------------


def Pi_gosper() -> CF:
    """Return a CF for π = 16·arctan(1/5) − 4·arctan(1/239).

    All arithmetic is done via Gosper's algorithm (implemented in cfmath.gosper),
    so no floating-point or mpmath is needed.
    """
    at5 = arctan_cf(1, 5)
    at239 = arctan_cf(1, 239)
    return CF.from_int(16) * at5 - CF.from_int(4) * at239


# ---------------------------------------------------------------------------
# Main: demonstration and verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    KNOWN_PI_TERMS = [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1]

    # --- Measure pair consumption while producing 15 terms ---
    counter5 = _CountingPairs(iter([]))  # placeholder; arctan_cf will fill it
    counter239 = _CountingPairs(iter([]))

    at5 = arctan_cf(1, 5, counter=counter5)
    at239 = arctan_cf(1, 239, counter=counter239)
    pi_cf = CF.from_int(16) * at5 - CF.from_int(4) * at239

    print("Computing first 15 CF terms of Pi_gosper()...")
    computed = list(pi_cf.take(15))
    print(f"  Computed : {computed}")
    print(f"  Expected : {KNOWN_PI_TERMS}")
    match = computed == KNOWN_PI_TERMS
    print(f"  Match    : {match}")

    print()
    print(f"  arctan(1/5)   pairs consumed: {counter5.count}")
    print(f"  arctan(1/239) pairs consumed: {counter239.count}")

    # --- Convergent check: convergent(Pi_gosper(), 3) == 355/113 ---
    print()
    print("Checking convergent(Pi_gosper(), 3) == 355/113 ...")
    pi2 = Pi_gosper()
    c3 = convergent(pi2, 3)
    expected_c3 = Fraction(355, 113)
    print(f"  convergent(Pi_gosper(), 3) = {c3}")
    print(f"  Expected                   = {expected_c3}")
    print(f"  Match: {c3 == expected_c3}")

    # --- Summary ---
    print()
    if match and c3 == expected_c3:
        print("All checks PASSED.")
    else:
        print("SOME CHECKS FAILED.")
        sys.exit(1)
