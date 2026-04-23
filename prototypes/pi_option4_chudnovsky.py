"""Prototype: π as a continued fraction using the Chudnovsky algorithm.

Uses binary splitting for efficient exact-integer partial sum computation,
then converts to decimal only at the final step.

The Chudnovsky formula:
    1/π = (12 / 640320^(3/2)) * Σ_{k=0}^∞  (-1)^k (6k)! (13591409 + 545140134·k)
                                              ------------------------------------------
                                              (3k)! (k!)³ 640320^(3k)

Each term contributes roughly 14.18 decimal digits of precision.
"""

from __future__ import annotations

import decimal
import math
import sys
from fractions import Fraction
from typing import Iterator

# Add the src directory to path so we can import cfmath
sys.path.insert(0, "/home/claude/cfmath/src")

from cfmath.convergents import convergent
from cfmath.core import CF

# ---------------------------------------------------------------------------
# Chudnovsky constants
# ---------------------------------------------------------------------------

_C = 640320
_C3_OVER_24 = _C**3 // 24  # = 10939058860032000
_A = 13591409
_B = 545140134
# Each term gives log10(C^3 / 24) / 3 ≈ 14.18 digits
_DIGITS_PER_TERM = math.log10(_C3_OVER_24) / 1  # ~ 16.38 digits per binary-split term


def _binary_split(a: int, b: int) -> tuple[int, int, int]:
    """Compute (P, Q, T) for the binary splitting of Chudnovsky sum over [a, b).

    Returns integers P, Q, T such that the partial sum from index a to b-1 is
        S(a,b) = T / (P*Q)   ... but more precisely the recurrence builds:
        sum = T/Q  (P is only used internally for the recursion)

    The Chudnovsky sum Σ_{k=a}^{b-1} term_k equals T / Q where:
        term_0 = A / (C^(3/2))   and subsequent terms use the recurrence.

    The exact relation is:
        T/Q = Σ_{k=a}^{b-1} (-1)^k * (6k)! * (A + B*k) / ((3k)! * (k!)^3 * C3^k)

    where C3 = C^3.  We factor out C^(3/2) separately at the end.
    """
    if b - a == 1:
        if a == 0:
            P = 1
            Q = 1
            T = _A
        else:
            # p(a) = (6a-5)(2a-1)(6a-1)
            P = (6 * a - 5) * (2 * a - 1) * (6 * a - 1)
            # q(a) = a^3 * C^3 / 24
            Q = a**3 * _C3_OVER_24
            # T = (-1)^a * P * (A + B*a)
            T = (-1) ** a * P * (_A + _B * a)
        return P, Q, T

    mid = (a + b) // 2
    P_left, Q_left, T_left = _binary_split(a, mid)
    P_right, Q_right, T_right = _binary_split(mid, b)

    P = P_left * P_right
    Q = Q_left * Q_right
    T = T_left * Q_right + P_left * T_right

    return P, Q, T


def _chudnovsky_sum(n_terms: int) -> tuple[int, int]:
    """Compute the Chudnovsky partial sum using exact integer binary splitting.

    Returns (numerator, denominator) as Python ints such that:
        sum = numerator / denominator
        ≈ Σ_{k=0}^{n_terms-1} (-1)^k (6k)! (A + B*k) / ((3k)! (k!)^3 C3^k)

    where C3 = 640320^3.
    """
    _, Q, T = _binary_split(0, n_terms)
    return T, Q


def _pi_chudnovsky(n_cf_terms: int) -> tuple[list[int], int]:
    """Compute n_cf_terms CF terms of π using the Chudnovsky algorithm.

    Strategy:
    - Each Chudnovsky term provides ~14 decimal digits.
    - We need enough decimal digits to extract n_cf_terms CF terms safely.
    - CF terms for π are typically small (mostly 1s), but some are large
      (e.g., term index 4 is 292), so we add a generous safety margin.
    """
    # Estimate digits needed: each CF term eats at most ~log10(a_n) digits,
    # but for π the average is about 3.5 bits ≈ 1.05 digits per term.
    # Use 5 digits per CF term + a large safety buffer.
    digits_needed = n_cf_terms * 5 + 100
    n_terms = max(2, math.ceil(digits_needed / 14) + 5)

    # Set decimal precision to cover the computation
    prec = digits_needed + 50
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    decimal.setcontext(ctx)

    # Compute exact integer partial sum via binary splitting
    T, Q = _chudnovsky_sum(n_terms)

    # Now compute π using the Chudnovsky formula:
    #   1/π = (12 / C^(3/2)) * (T / Q)
    #   π = C^(3/2) * Q / (12 * T)
    #   C^(3/2) = C * sqrt(C)
    C = decimal.Decimal(_C)
    sqrt_C = C.sqrt()
    C_3_2 = C * sqrt_C  # 640320^(3/2)

    numerator = C_3_2 * decimal.Decimal(Q)
    denominator = decimal.Decimal(12) * decimal.Decimal(T)

    pi_decimal = numerator / denominator

    # Extract CF terms via floor-and-reciprocal
    cf_terms: list[int] = []
    x = pi_decimal
    for _ in range(n_cf_terms):
        a = int(x)  # floor (ROUND_FLOOR context makes this safe)
        cf_terms.append(a)
        frac = x - decimal.Decimal(a)
        # Stop if we've lost precision
        if frac <= decimal.Decimal(10) ** (-(prec - 20)):
            break
        x = decimal.Decimal(1) / frac

    decimal.setcontext(decimal.DefaultContext)
    return cf_terms, n_terms


def Pi_chudnovsky() -> CF:
    """Return π as a CF, computed via the Chudnovsky algorithm.

    Uses the same static-prefix + lazy _source pattern as Cuberoot/Pi.
    Static prefix: first 10 terms (pre-computed).
    Lazy source: yields more terms on demand.
    """
    # Pre-compute enough for the static portion + initial lazy batch
    terms, _ = _pi_chudnovsky(80)
    static = terms[:10]
    rest = iter(terms[10:])

    def _more() -> Iterator[int]:
        yield from rest
        # Extend further on demand in batches
        offset = 80
        while True:
            more, _ = _pi_chudnovsky(offset + 50)
            yield from more[offset:]
            offset += 50

    return CF(static, _source=_more())


# ---------------------------------------------------------------------------
# Main: demo and self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EXPECTED_FIRST_15 = [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1]

    print("Computing π via Chudnovsky binary splitting...\n")

    # Get first 15 CF terms and count how many Chudnovsky terms were needed
    cf_terms, n_chudnovsky = _pi_chudnovsky(15)
    first_15 = cf_terms[:15]

    print(f"First 15 CF terms: {first_15}")
    print(f"Expected:          {EXPECTED_FIRST_15}")
    match = first_15 == EXPECTED_FIRST_15
    print(f"Match: {match}")
    assert match, f"CF terms mismatch!\n  got:      {first_15}\n  expected: {EXPECTED_FIRST_15}"

    print(f"\nChudnovsky terms used for 15 CF terms: {n_chudnovsky}")

    # Test convergent(Pi_chudnovsky(), 3) == 355/113
    print("\nBuilding Pi_chudnovsky() CF object...")
    pi_cf = Pi_chudnovsky()

    c3 = convergent(pi_cf, 3)
    print(f"convergent(Pi_chudnovsky(), 3) = {c3}")
    expected_frac = Fraction(355, 113)
    print(f"Expected: {expected_frac}")
    assert c3 == expected_frac, f"Convergent mismatch: got {c3}, expected {expected_frac}"
    print("Convergent check passed.")

    print("\nAll checks passed.")
