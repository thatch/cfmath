"""Prototype: π as a continued fraction using Machin's formula with decimal.

Machin's formula:  π = 16·arctan(1/5) − 4·arctan(1/239)

Uses only Python's standard-library `decimal` module for high-precision
arithmetic — no mpmath required.
"""

from __future__ import annotations

import decimal
from fractions import Fraction
from typing import Iterator

from cfmath.convergents import convergent
from cfmath.core import CF

# ---------------------------------------------------------------------------
# Low-level: arctan via Taylor series
# ---------------------------------------------------------------------------


def _arctan_decimal(x_num: int, x_den: int, prec: int) -> tuple[decimal.Decimal, int]:
    """Compute arctan(x_num/x_den) to `prec` decimal digits.

    Uses the alternating Taylor series:
        arctan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...

    The current decimal context must already be set to sufficient precision
    (at least prec + 50 guard digits) by the caller.

    Returns (result, n_terms) where n_terms is the number of Taylor series
    terms summed.
    """
    D = decimal.Decimal
    x = D(x_num) / D(x_den)
    x2 = x * x  # x² — we multiply each term by −x² to step forward

    threshold = D(10) ** (-(prec + 10))

    result = x  # first term: x / 1
    term = x  # running term value
    n_terms = 1

    k = 1  # denominator of the current term is 2k-1, next is 2k+1
    while True:
        term = -term * x2  # multiply by −x² to get next-order term (unsigned)
        k += 1
        denom = D(2 * k - 1)  # 3, 5, 7, …
        contribution = term / denom
        result += contribution
        n_terms += 1
        if abs(contribution) < threshold:
            break

    return result, n_terms


# ---------------------------------------------------------------------------
# CF-term extraction: π via Machin's formula
# ---------------------------------------------------------------------------


def _pi_machin_decimal(n_terms: int) -> tuple[list[int], int, int]:
    """Compute n_terms CF terms of π using Machin's formula.

    π = 16·arctan(1/5) − 4·arctan(1/239)

    Returns (terms, n_arctan5, n_arctan239) where the last two values are
    the number of Taylor series terms used for each arctan computation.
    """
    prec = n_terms * 5 + 60

    ctx = decimal.Context(prec=prec + 50, rounding=decimal.ROUND_FLOOR)
    decimal.setcontext(ctx)

    D = decimal.Decimal
    arctan5, n5 = _arctan_decimal(1, 5, prec)
    arctan239, n239 = _arctan_decimal(1, 239, prec)

    pi = D(16) * arctan5 - D(4) * arctan239

    terms: list[int] = []
    x = pi
    for _ in range(n_terms):
        a = int(x)  # floor (context rounds down)
        terms.append(a)
        frac = x - D(a)
        if frac <= D(10) ** (-(prec - 20)):
            break  # precision exhausted
        x = D(1) / frac

    decimal.setcontext(decimal.DefaultContext)
    return terms, n5, n239


# ---------------------------------------------------------------------------
# CF object
# ---------------------------------------------------------------------------


def Pi_machin() -> CF:
    """Return a CF object for π, computed via Machin's formula.

    Computes 80 terms initially; first 10 are stored as the static prefix,
    the rest are exposed via a lazy _source iterator that recomputes at
    higher precision on demand.
    """
    terms, _, _ = _pi_machin_decimal(80)
    static = terms[:10]
    rest = iter(terms[10:])

    def _more() -> Iterator[int]:
        yield from rest
        # Extend further on demand with higher precision
        offset = 80
        while True:
            more, _, _ = _pi_machin_decimal(offset + 50)
            yield from more[offset:]
            offset += 50

    return CF(static, _source=_more())


# ---------------------------------------------------------------------------
# Main: smoke-test the prototype
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    KNOWN = [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1]

    # ---- Report Taylor-series iteration counts ----
    terms15, n5, n239 = _pi_machin_decimal(15)
    print(f"arctan(1/5)   needed {n5:4d} Taylor series terms")
    print(f"arctan(1/239) needed {n239:4d} Taylor series terms")
    print()

    # ---- First 15 CF terms ----
    pi_cf = Pi_machin()
    computed = list(pi_cf.take(15))

    print(f"Computed CF terms:  {computed}")
    print(f"Known CF terms:     {KNOWN}")
    match = computed == KNOWN
    print(f"Terms match:        {match}")
    if not match:
        mismatches = [(i, c, k) for i, (c, k) in enumerate(zip(computed, KNOWN)) if c != k]
        for i, c, k in mismatches:
            print(f"  index {i}: got {c}, expected {k}")
    print()

    # ---- Convergent check: convergent(π, 3) = 355/113 ----
    pi_cf2 = Pi_machin()
    conv3 = convergent(pi_cf2, 3)
    expected = Fraction(355, 113)
    print(f"convergent(Pi_machin(), 3) = {conv3}")
    print(f"Expected:                    {expected}")
    print(f"Equals 355/113:              {conv3 == expected}")
