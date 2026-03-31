"""Convergents: the rational fractions that best approximate a continued fraction.

Each convergent p_n/q_n is built from the first n+1 terms of the CF and gives
the closest rational approximation with denominator up to q_n.  Successive
convergents alternate between slightly below and slightly above the true value,
converging toward it.  The recurrence is:

    p_n = a_n * p_{n-1} + p_{n-2},   p_{-1}=1, p_0=a_0
    q_n = a_n * q_{n-1} + q_{n-2},   q_{-1}=0, q_0=1

Key identity: p_n * q_{n-1} - p_{n-1} * q_n = (-1)^n
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .core import CF


def convergent_pairs(cf: CF) -> Iterator[tuple[int, int]]:
    """Yield (numerator, denominator) pairs for every convergent of *cf*, lazily."""
    p_prev, p_curr = 1, None
    q_prev, q_curr = 0, None

    for a in cf:
        if p_curr is None:
            # n = 0
            p_curr = a
            q_curr = 1
        else:
            p_curr, p_prev = a * p_curr + p_prev, p_curr
            q_curr, q_prev = a * q_curr + q_prev, q_curr
        yield (p_curr, q_curr)


def convergents(cf: CF) -> Iterator[Fraction]:
    """Yield every convergent of *cf* as a Fraction, lazily."""
    for p, q in convergent_pairs(cf):
        yield Fraction(p, q)


def convergent_pair(cf: CF, n: int) -> tuple[int, int]:
    """Return (p_n, q_n) for the n-th convergent (0-indexed)."""
    if n < 0:
        raise IndexError(f"convergent index {n} must be >= 0")
    for i, pq in enumerate(convergent_pairs(cf)):
        if i == n:
            return pq
    raise IndexError(f"CF has fewer than {n + 1} terms")


def convergent(cf: CF, n: int) -> Fraction:
    """Return the n-th convergent (0-indexed) as a Fraction."""
    p, q = convergent_pair(cf, n)
    return Fraction(p, q)
