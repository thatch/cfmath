"""Prototype: lazy π using incremental Chudnovsky + rational interval arithmetic.

Generates CF terms one at a time without needing to know upfront how many
Chudnovsky terms are required.

Strategy
--------
The Chudnovsky formula gives:

    π = C · √C · Q / (12 · T)    where  C = 640320

and T/Q is an alternating series that we extend one term at a time.
Because the series alternates in sign, consecutive partial sums bracket
the true value.  This gives us a rational interval for the factor C·Q/(12T)
(the "rational part" r = π/√C).

To bound √C we use the CF convergents of Sqrt(640320): even-indexed
convergents underestimate √C, odd-indexed ones overestimate.  Multiplying
the two intervals gives a fully rational interval [π_lo, π_hi] for π.

We track which CF digits we have emitted via a Möbius (linear fractional)
state (a, b, c, d) meaning "remaining output = (a·π + b) / (c·π + d)".
A new digit n can be output when both endpoints of the transformed interval
share the same integer floor.  Emitting n updates the Möbius state:
  (a, b, c, d)  →  (c, d, a − n·c, b − n·d)

When the interval is too wide to emit a digit, we add the next Chudnovsky
term and deepen the √C bounds.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from typing import Iterator

sys.path.insert(0, "/home/claude/cfmath/src")

from cfmath.constants import Sqrt
from cfmath.convergents import convergent
from cfmath.core import CF

# ---------------------------------------------------------------------------
# Chudnovsky constants
# ---------------------------------------------------------------------------

_C = 640320
_C3_OVER_24 = _C**3 // 24  # = 10939058860032000
_A = 13591409
_B = 545140134


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


def _lazy_pi_terms() -> Iterator[int]:
    """Yield CF terms of π one at a time, consuming Chudnovsky terms on demand."""

    # --- Incremental Chudnovsky state ---
    # We keep (P_acc, Q_acc, T_acc) so that T_acc/Q_acc is the partial sum
    # Σ_{j=0}^{k} term_j, and P_acc is the product of all P_j (needed to
    # attach the next term without recomputing from scratch).
    P_acc = 1
    Q_acc = 1
    T_acc = _A  # k=0: P=1, Q=1, T=A (the first Chudnovsky term)

    def _add_chudnovsky_term(k: int) -> None:
        """Attach the k-th Chudnovsky term (k ≥ 1) to the running totals."""
        nonlocal P_acc, Q_acc, T_acc
        Pk = (6 * k - 5) * (2 * k - 1) * (6 * k - 1)
        Qk = k**3 * _C3_OVER_24
        sign = 1 if k % 2 == 0 else -1
        Tk = sign * Pk * (_A + _B * k)
        # Binary-split merge: attach [k,k+1) to accumulated [0,k)
        T_acc = T_acc * Qk + P_acc * Tk
        P_acc *= Pk
        Q_acc *= Qk

    def _rational_part() -> Fraction:
        """Current approximation: C·Q/(12·T)  ≈  π/√C."""
        return Fraction(_C * Q_acc, 12 * T_acc)

    # --- √C bounds via Sqrt(640320) CF convergents ---
    # √640320 = [800; 5, 1600, 5, 1600, ...] — the large period term 1600
    # makes convergents extremely accurate after just a few steps.
    _sqrt_cf = Sqrt(_C)
    _sqrt_cache: list[Fraction] = []

    def _sqrt_conv(n: int) -> Fraction:
        while len(_sqrt_cache) <= n:
            _sqrt_cache.append(convergent(_sqrt_cf, len(_sqrt_cache)))
        return _sqrt_cache[n]

    def _sqrt_bounds(depth: int) -> tuple[Fraction, Fraction]:
        """Return a tight [lo, hi] bracket for √C using convergents up to depth.

        Even-indexed convergents underestimate √C; odd-indexed overestimate.
        We use one even/odd pair for the bracket.
        """
        d_even = depth if depth % 2 == 0 else depth - 1
        d_odd = d_even + 1
        lo = _sqrt_conv(d_even)
        hi = _sqrt_conv(d_odd)
        # Sanity: lo should be less than hi; if somehow not, swap.
        return (lo, hi) if lo <= hi else (hi, lo)

    # --- Möbius (linear fractional) state ---
    # Emitted digits so far mean: next output = (ma·π + mb) / (mc·π + md)
    # where π is the true value of π, currently bounded by [pi_lo, pi_hi].
    # Initially the identity transform: output = π.
    ma, mb, mc, md = 1, 0, 0, 1

    # --- Seed: need two consecutive partial sums to bracket π/√C ---
    r_prev = _rational_part()  # after k=0 term
    _add_chudnovsky_term(1)
    r_curr = _rational_part()  # after k=0 and k=1
    k_next = 2

    sqrt_depth = 3  # start with convergent pair (2, 3); already very tight

    # ---------------------------------------------------------------------------
    # Main loop: emit CF terms until interval too wide, then add precision
    # ---------------------------------------------------------------------------
    while True:
        # Build a rational interval [pi_lo, pi_hi] for the true π.
        # r_prev and r_curr bracket the true rational part r∞ = π/√C,
        # and sq_lo, sq_hi bracket √C, so their products bracket π.
        r_lo = min(r_prev, r_curr)
        r_hi = max(r_prev, r_curr)
        sq_lo, sq_hi = _sqrt_bounds(sqrt_depth)
        pi_lo = r_lo * sq_lo
        pi_hi = r_hi * sq_hi

        # Evaluate the Möbius transform at both endpoints.
        # next_output = (ma·π + mb) / (mc·π + md)
        den_lo = mc * pi_lo + md
        den_hi = mc * pi_hi + md

        # Both denominators must be positive and the same sign (no pole in range).
        if den_lo > 0 and den_hi > 0:
            val_lo = (ma * pi_lo + mb) / den_lo
            val_hi = (ma * pi_hi + mb) / den_hi
            out_lo = min(val_lo, val_hi)
            out_hi = max(val_lo, val_hi)
            n = math.floor(out_lo)
            if n == math.floor(out_hi):
                yield n
                # Subtract n and take reciprocal:
                # new_output = 1/(output - n)
                #            = (mc·π + md) / ((ma − n·mc)·π + (mb − n·md))
                ma, mb, mc, md = mc, md, ma - n * mc, mb - n * md
                continue  # try to emit another digit before adding more terms

        # Interval still too wide — consume one more Chudnovsky term.
        _add_chudnovsky_term(k_next)
        r_prev = r_curr
        r_curr = _rational_part()
        k_next += 1

        # Deepen √C bounds in step with the additional Chudnovsky precision.
        # Each term adds ~14 decimal digits; √640320 converges very fast
        # (period term 1600 means each cycle multiplies accuracy by ~1600²),
        # so a small increment here is more than sufficient.
        sqrt_depth += 2


# ---------------------------------------------------------------------------
# Public CF constructor
# ---------------------------------------------------------------------------


def Pi_lazy() -> CF:
    """Return π as a CF, computed lazily via incremental Chudnovsky.

    Terms are generated on demand: no upfront count needed.
    """
    src = _lazy_pi_terms()
    static: list[int] = []
    for _ in range(10):
        try:
            static.append(next(src))
        except StopIteration:
            break
    return CF(static, _source=src)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EXPECTED_FIRST_15 = [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1]

    print("Computing lazy Chudnovsky π CF terms...\n")

    pi = Pi_lazy()
    first_15 = list(pi.take(15))
    print(f"First 15 CF terms: {first_15}")
    print(f"Expected:          {EXPECTED_FIRST_15}")
    match = first_15 == EXPECTED_FIRST_15
    print(f"Match: {match}")
    assert match, f"Terms mismatch!\n  got:      {first_15}\n  expected: {EXPECTED_FIRST_15}"

    c3 = convergent(Pi_lazy(), 3)
    expected_frac = Fraction(355, 113)
    print(f"\nconvergent(Pi_lazy(), 3) = {c3}  (expected {expected_frac})")
    assert c3 == expected_frac, f"Convergent mismatch: {c3}"
    print("Convergent check passed.")

    print("\nAll checks passed.")
