"""Hyperbolic functions as continued fractions."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterator

from ._backend import (
    _HAS_MPMATH,
    _annotate_cf,
    _cf_terms_from_interval_approximator,
    _coerce_trig_arg,
    _lazy_cf,
    _mpmath_cf_for_cf_arg,
)
from .core import CF

# ---------------------------------------------------------------------------
# Decimal backends for Sinh and Cosh
# ---------------------------------------------------------------------------


def _sinh_terms_from_decimal(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of sinh(x_num/x_den) using rational intervals."""
    x = Fraction(x_num, x_den)

    def _exp_positive_interval(y: Fraction, precision: int) -> tuple[Fraction, Fraction]:
        k_limit = max(precision, 2 * y.numerator // y.denominator + 4)
        term = Fraction(1)
        val = Fraction(1)
        for k in range(1, k_limit + 1):
            term *= y / k
            val += term

        next_term = term * y / (k_limit + 1)
        ratio = y / (k_limit + 2)
        if ratio >= 1:
            return _exp_positive_interval(y, precision * 2)
        tail = next_term / (1 - ratio)
        return val, val + tail

    def _interval(precision: int) -> tuple[Fraction, Fraction]:
        sign = -1 if x < 0 else 1
        y = abs(x)
        e_lo, e_hi = _exp_positive_interval(y, precision)
        inv_lo, inv_hi = Fraction(1, e_hi), Fraction(1, e_lo)
        lo = (e_lo - inv_hi) / 2
        hi = (e_hi - inv_lo) / 2
        if sign < 0:
            return -hi, -lo
        return lo, hi

    return _cf_terms_from_interval_approximator(_interval, n_terms)


def _cosh_terms_from_decimal(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of cosh(x_num/x_den) using rational intervals."""
    x = abs(Fraction(x_num, x_den))

    def _exp_positive_interval(y: Fraction, precision: int) -> tuple[Fraction, Fraction]:
        k_limit = max(precision, 2 * y.numerator // y.denominator + 4)
        term = Fraction(1)
        val = Fraction(1)
        for k in range(1, k_limit + 1):
            term *= y / k
            val += term

        next_term = term * y / (k_limit + 1)
        ratio = y / (k_limit + 2)
        if ratio >= 1:
            return _exp_positive_interval(y, precision * 2)
        tail = next_term / (1 - ratio)
        return val, val + tail

    def _interval(precision: int) -> tuple[Fraction, Fraction]:
        e_lo, e_hi = _exp_positive_interval(x, precision)
        inv_lo, inv_hi = Fraction(1, e_hi), Fraction(1, e_lo)
        return (e_lo + inv_lo) / 2, (e_hi + inv_hi) / 2

    return _cf_terms_from_interval_approximator(_interval, n_terms)


# ---------------------------------------------------------------------------
# mpmath backends for Sinh and Cosh
# ---------------------------------------------------------------------------


def _sinh_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of sinh(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.sinh(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _cosh_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of cosh(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.cosh(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


# ---------------------------------------------------------------------------
# Lambert CF generator for Tanh (exact, no floating point)
# ---------------------------------------------------------------------------


def _tanh_pairs(x: Fraction) -> Iterator[tuple[int, Fraction]]:
    """Yield (b_n, a_{n+1}) pairs for the Lambert generalized CF of tanh(x).

    tanh(x) = x / (1 + x²/(3 + x²/(5 + x²/(7 + ...))))
    """
    x2 = x * x
    k = 0
    while True:
        yield (2 * k + 1, x2)
        k += 1


def _tanh_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of tanh(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.tanh(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def Sinh(x: int | Fraction | CF) -> CF:
    """Hyperbolic sine of x, as a continued fraction.

    x may be an int or Fraction.  Returns CF([0]) for x=0.
    Uses mpmath for high-precision CF term extraction.

    Examples::

        Sinh(0)                 # [0]
        Sinh(1)                 # [1; 6, 2, 20, 1, ...]
        Sinh(Fraction(1, 2))    # [0; 2, 5, 1, 1, ...]
    """
    if isinstance(x, CF):
        from .exponential import ExpCF

        e = ExpCF(x)
        return (e - 1 / e) / 2
    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(0), ("Sinh", x))
    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _sinh_terms_mpmath(num, den, n), debug_source=("Sinh", x))
    return _lazy_cf(lambda n: _sinh_terms_from_decimal(num, den, n), debug_source=("Sinh", x))


def Cosh(x: int | Fraction | CF) -> CF:
    """Hyperbolic cosine of x, as a continued fraction.

    x may be an int or Fraction.  Returns CF([1]) for x=0.
    Uses mpmath for high-precision CF term extraction.

    Examples::

        Cosh(0)                 # [1]
        Cosh(1)                 # [1; 1, 1, 3, 1, ...]
        Cosh(Fraction(1, 2))    # [1; 12, 1, 2, ...]
    """
    if isinstance(x, CF):
        from .exponential import ExpCF

        e = ExpCF(x)
        return (e + 1 / e) / 2
    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(1), ("Cosh", x))
    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _cosh_terms_mpmath(num, den, n), debug_source=("Cosh", x))
    return _lazy_cf(lambda n: _cosh_terms_from_decimal(num, den, n), debug_source=("Cosh", x))


def Tanh(x: int | Fraction | CF) -> CF:
    """Hyperbolic tangent of x, as a continued fraction.

    x may be an int or Fraction.  Returns CF([0]) for x=0.
    Uses Lambert's generalized CF (no external library required):
        tanh(x) = x / (1 + x²/(3 + x²/(5 + x²/(7 + ...))))

    Examples::

        Tanh(0)                 # [0]
        Tanh(1)                 # [0; 1, 3, 5, 7, ...]
        Tanh(Fraction(1, 2))    # [0; 2, 6, 10, 14, ...]
    """
    if isinstance(x, CF):
        import mpmath

        return _mpmath_cf_for_cf_arg(x, mpmath.tanh)
    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(0), ("Tanh", x))
    return _annotate_cf(CF.from_rational(x) / CF.from_generalized_cf(_tanh_pairs(x)), ("Tanh", x))
