"""Trigonometric functions as continued fractions."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterator

from ._backend import _annotate_cf, _coerce_trig_arg
from .core import CF

# ---------------------------------------------------------------------------
# Generalized CF generators (exact, no floating point)
# ---------------------------------------------------------------------------


def _tan_pairs(x: Fraction) -> Iterator[tuple[int, Fraction]]:
    """Yield (b_n, a_{n+1}) pairs for the Lambert generalized CF of tan(x).

    tan(x) = x / (1 - x²/(3 - x²/(5 - x²/...)))
    """
    neg_x2 = -(x * x)
    n = 0
    while True:
        yield (2 * n + 1, neg_x2)
        n += 1


def _sin_pairs(x: Fraction) -> Iterator[tuple[Fraction, Fraction]]:
    """Yield (b_n, a_{n+1}) pairs for the generalized CF of sin(x)/x."""
    x2 = x * x
    yield (Fraction(1), x2)
    k = 1
    while True:
        c = Fraction(2 * k * (2 * k + 1))
        yield (c - x2, c * x2)
        k += 1


def _cos_pairs(x: Fraction) -> Iterator[tuple[Fraction, Fraction]]:
    """Yield (b_n, a_{n+1}) pairs for the generalized CF of 1/cos(x)."""
    x2 = x * x
    yield (Fraction(1), x2)
    k = 1
    while True:
        c = Fraction((2 * k - 1) * (2 * k))
        yield (c - x2, c * x2)
        k += 1


# ---------------------------------------------------------------------------
# mpmath backends (for cross-checking; not used in dispatch)
# ---------------------------------------------------------------------------


def _sin_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of sin(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.sin(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _cos_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of cos(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.cos(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _tan_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of tan(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.tan(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


# ---------------------------------------------------------------------------
# Public functions (use exact GCF — no external library required)
# ---------------------------------------------------------------------------


def Tan(x: int | Fraction) -> CF:
    """Tangent of x (in radians), as a continued fraction.

    x may be an int or Fraction.  Returns CF([0]) for x=0.
    Uses Lambert's generalized CF (no external library required).

    Examples::

        Tan(0)                  # [0]
        Tan(Fraction(1, 4))     # [0; 3, 1, 10, 1, 18, 1, 26]
        Tan(Fraction(1, 2))     # [0; 1, 1, 4, 1, 1, 8, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(0), ("Tan", x))
    denom_cf = CF.from_generalized_cf(_tan_pairs(x))
    return _annotate_cf(CF.from_rational(x) / denom_cf, ("Tan", x))


def Sin(x: int | Fraction) -> CF:
    """Sine of x (in radians), as a continued fraction.

    x may be an int or Fraction.  Returns CF([0]) for x=0.
    Uses a direct generalized CF (no external library required).

    Examples::

        Sin(0)                  # [0]
        Sin(Fraction(1, 2))     # [0; 2, 11, 1, 1, 1, 6, ...]
        Sin(Fraction(123, 1000))# [0; 8, 6, 1, 1, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(0), ("Sin", x))
    return _annotate_cf(CF.from_rational(x) / CF.from_generalized_cf(_sin_pairs(x)), ("Sin", x))


def Cos(x: int | Fraction) -> CF:
    """Cosine of x (in radians), as a continued fraction.

    x may be an int or Fraction.  Returns CF([1]) for x=0.
    Uses a direct generalized CF (no external library required).

    Examples::

        Cos(0)                  # [1]
        Cos(Fraction(1, 2))     # [0; 1, 7, 5, 1, 12, 2, 1]
        Cos(Fraction(123, 1000))# [0; 1, 131, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(1), ("Cos", x))
    return _annotate_cf(CF.from_int(1) / CF.from_generalized_cf(_cos_pairs(x)), ("Cos", x))
