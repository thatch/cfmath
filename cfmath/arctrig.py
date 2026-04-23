"""Inverse trigonometric functions as continued fractions."""

from __future__ import annotations

from fractions import Fraction

from ._backend import _coerce_trig_arg
from .core import CF

# ---------------------------------------------------------------------------
# Generalized CF generators (exact, no floating point)
# ---------------------------------------------------------------------------


def _arctan_pairs(x: Fraction):
    """Yield (b_n, a_{n+1}) pairs for the Gauss generalized CF of arctan(x).

    arctan(x) = x / (1 + x²/(3 + (2x)²/(5 + (3x)²/(7 + ...))))
    """
    x2 = x * x
    k = 0
    while True:
        yield (2 * k + 1, (k + 1) ** 2 * x2)
        k += 1


def _arcsin_pairs(x: Fraction):
    """Yield (b_n, a_{n+1}) pairs for the Euler generalized CF of arcsin(x)/x."""
    x2 = x * x

    def _r(k):
        return Fraction((2 * k - 1) ** 2) * x2 / Fraction(2 * k * (2 * k + 1))

    yield (Fraction(1), -_r(1))
    k = 1
    while True:
        yield (Fraction(1) + _r(k), -_r(k + 1))
        k += 1


# ---------------------------------------------------------------------------
# mpmath backends (for cross-checking; not used in dispatch)
# ---------------------------------------------------------------------------


def _arctan_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of arctan(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.atan(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _arcsin_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of arcsin(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.asin(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _arccos_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of arccos(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.acos(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


# ---------------------------------------------------------------------------
# Public functions (use exact GCF — no external library required)
# ---------------------------------------------------------------------------


def Arctan(x) -> CF:
    """Arctangent of x (in radians), as a continued fraction.

    x may be an int or Fraction.  Returns CF([0]) for x=0.
    Uses the Gauss generalized CF (no external library required).

    Examples::

        Arctan(0)               # [0]
        Arctan(1)               # π/4 ≈ [0; 1, 3, 1, 5, ...]
        Arctan(Fraction(1, 2))  # [0; 2, 7, 1, 1, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(0)
    return CF.from_rational(x) / CF.from_generalized_cf(_arctan_pairs(x))


def Arcsin(x) -> CF:
    """Arcsine of x (in radians), as a continued fraction.

    x may be an int or Fraction; must satisfy |x| ≤ 1.
    Returns CF([0]) for x=0.
    Uses a generalized CF (no external library required).

    Examples::

        Arcsin(0)               # [0]
        Arcsin(1)               # π/2 ≈ [1; 1, 3, 1, 5, ...]
        Arcsin(Fraction(1, 2))  # π/6 ≈ [0; 1, 11, 1, 2, ...]
    """
    x = _coerce_trig_arg(x)
    if abs(x) > 1:
        raise ValueError(f"Arcsin argument must satisfy |x| ≤ 1, got {x}")
    if x == 0:
        return CF.from_int(0)
    return CF.from_rational(x) / CF.from_generalized_cf(_arcsin_pairs(x))


def Arccos(x) -> CF:
    """Arccosine of x (in radians), as a continued fraction.

    x may be an int or Fraction; must satisfy |x| ≤ 1.
    Computed as π/2 - arcsin(x).

    Examples::

        Arccos(0)               # π/2 ≈ [1; 1, 3, 1, 5, ...]
        Arccos(1)               # [0]
        Arccos(Fraction(1, 2))  # π/3 ≈ [1; 20, 1, 2, ...]
    """
    x = _coerce_trig_arg(x)
    if abs(x) > 1:
        raise ValueError(f"Arccos argument must satisfy |x| ≤ 1, got {x}")
    if x == 1:
        return CF.from_int(0)
    from .constants import Pi

    half_pi = Pi() / CF.from_int(2)
    if x == 0:
        return half_pi
    return half_pi - Arcsin(x)
