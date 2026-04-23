"""Hyperbolic functions as continued fractions."""

from __future__ import annotations

from fractions import Fraction

from ._backend import _HAS_MPMATH, _coerce_trig_arg, _lazy_cf
from .core import CF

# ---------------------------------------------------------------------------
# Decimal backends for Sinh and Cosh
# ---------------------------------------------------------------------------


def _sinh_terms_from_decimal(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of sinh(x_num/x_den) using high-precision Decimal.

    Uses the Taylor series sinh(x) = Σ x²ⁿ⁺¹/(2n+1)!  No external library required.
    """
    import decimal

    prec = n_terms * 5 + 80
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    with decimal.localcontext(ctx):
        x = decimal.Decimal(x_num) / decimal.Decimal(x_den)
        x2 = x * x
        term = x
        val = x
        k = 1
        eps = decimal.Decimal(10) ** (-(prec - 10))
        while True:
            term *= x2 / decimal.Decimal((2 * k) * (2 * k + 1))
            val += term
            if abs(term) < eps:
                break
            k += 1

        terms: list[int] = []
        for _ in range(n_terms):
            a = int(val.to_integral_value(rounding=decimal.ROUND_FLOOR))
            terms.append(a)
            frac = val - a
            if frac <= eps:
                break
            val = decimal.Decimal(1) / frac

    return terms


def _cosh_terms_from_decimal(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of cosh(x_num/x_den) using high-precision Decimal.

    Uses the Taylor series cosh(x) = Σ x²ⁿ/(2n)!  No external library required.
    """
    import decimal

    prec = n_terms * 5 + 80
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    with decimal.localcontext(ctx):
        x = decimal.Decimal(x_num) / decimal.Decimal(x_den)
        x2 = x * x
        term = decimal.Decimal(1)
        val = decimal.Decimal(1)
        k = 1
        eps = decimal.Decimal(10) ** (-(prec - 10))
        while True:
            term *= x2 / decimal.Decimal((2 * k - 1) * (2 * k))
            val += term
            if abs(term) < eps:
                break
            k += 1

        terms: list[int] = []
        for _ in range(n_terms):
            a = int(val.to_integral_value(rounding=decimal.ROUND_FLOOR))
            terms.append(a)
            frac = val - a
            if frac <= eps:
                break
            val = decimal.Decimal(1) / frac

    return terms


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


def _tanh_pairs(x: Fraction):
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


def Sinh(x) -> CF:
    """Hyperbolic sine of x, as a continued fraction.

    x may be an int or Fraction.  Returns CF([0]) for x=0.
    Uses mpmath for high-precision CF term extraction.

    Examples::

        Sinh(0)                 # [0]
        Sinh(1)                 # [1; 6, 2, 20, 1, ...]
        Sinh(Fraction(1, 2))    # [0; 2, 5, 1, 1, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(0)
    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _sinh_terms_mpmath(num, den, n))
    return _lazy_cf(lambda n: _sinh_terms_from_decimal(num, den, n))


def Cosh(x) -> CF:
    """Hyperbolic cosine of x, as a continued fraction.

    x may be an int or Fraction.  Returns CF([1]) for x=0.
    Uses mpmath for high-precision CF term extraction.

    Examples::

        Cosh(0)                 # [1]
        Cosh(1)                 # [1; 1, 1, 3, 1, ...]
        Cosh(Fraction(1, 2))    # [1; 12, 1, 2, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(1)
    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _cosh_terms_mpmath(num, den, n))
    return _lazy_cf(lambda n: _cosh_terms_from_decimal(num, den, n))


def Tanh(x) -> CF:
    """Hyperbolic tangent of x, as a continued fraction.

    x may be an int or Fraction.  Returns CF([0]) for x=0.
    Uses Lambert's generalized CF (no external library required):
        tanh(x) = x / (1 + x²/(3 + x²/(5 + x²/(7 + ...))))

    Examples::

        Tanh(0)                 # [0]
        Tanh(1)                 # [0; 1, 3, 5, 7, ...]
        Tanh(Fraction(1, 2))    # [0; 2, 6, 10, 14, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(0)
    return CF.from_rational(x) / CF.from_generalized_cf(_tanh_pairs(x))
