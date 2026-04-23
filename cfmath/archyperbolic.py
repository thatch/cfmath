"""Inverse hyperbolic functions as continued fractions."""

from __future__ import annotations

from fractions import Fraction

from ._backend import _HAS_MPMATH, _coerce_trig_arg, _lazy_cf
from .core import CF

# ---------------------------------------------------------------------------
# Decimal backends
# ---------------------------------------------------------------------------


def _arcsinh_terms_from_decimal(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of arcsinh(x_num/x_den).

    Uses arcsinh(x) = ln(x + sqrt(x²+1)) computed entirely in Decimal.
    Works for all real x.
    """
    import decimal

    prec = n_terms * 5 + 80
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    with decimal.localcontext(ctx):
        eps = decimal.Decimal(10) ** (-(prec - 10))

        def _two_atanh(t: decimal.Decimal) -> decimal.Decimal:
            t2 = t * t
            term = t
            val = t
            k = 1
            while True:
                term *= t2
                delta = term / decimal.Decimal(2 * k + 1)
                val += delta
                if abs(delta) < eps:
                    break
                k += 1
            return 2 * val

        x = decimal.Decimal(x_num) / decimal.Decimal(x_den)
        inner = x + (x * x + 1).sqrt()

        ln2 = _two_atanh(decimal.Decimal(1) / decimal.Decimal(3))
        n = 0
        reduced = inner
        while reduced >= 2:
            reduced /= 2
            n += 1
        while reduced < decimal.Decimal("0.5"):
            reduced *= 2
            n -= 1

        ln_val = decimal.Decimal(n) * ln2 + _two_atanh((reduced - 1) / (reduced + 1))

        terms: list[int] = []
        for _ in range(n_terms):
            a = int(ln_val.to_integral_value(rounding=decimal.ROUND_FLOOR))
            terms.append(a)
            frac = ln_val - a
            if frac <= eps:
                break
            ln_val = decimal.Decimal(1) / frac

    return terms


def _arccosh_terms_from_decimal(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of arccosh(x_num/x_den).

    Uses arccosh(x) = ln(x + sqrt(x²-1)) computed entirely in Decimal.
    Requires x ≥ 1.
    """
    import decimal

    prec = n_terms * 5 + 80
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    with decimal.localcontext(ctx):
        eps = decimal.Decimal(10) ** (-(prec - 10))

        def _two_atanh(t: decimal.Decimal) -> decimal.Decimal:
            t2 = t * t
            term = t
            val = t
            k = 1
            while True:
                term *= t2
                delta = term / decimal.Decimal(2 * k + 1)
                val += delta
                if abs(delta) < eps:
                    break
                k += 1
            return 2 * val

        x = decimal.Decimal(x_num) / decimal.Decimal(x_den)
        inner = x + (x * x - 1).sqrt()

        if inner <= eps:
            return [0]

        ln2 = _two_atanh(decimal.Decimal(1) / decimal.Decimal(3))
        n = 0
        reduced = inner
        while reduced >= 2:
            reduced /= 2
            n += 1
        while reduced < decimal.Decimal("0.5"):
            reduced *= 2
            n -= 1

        ln_val = decimal.Decimal(n) * ln2 + _two_atanh((reduced - 1) / (reduced + 1))

        terms: list[int] = []
        for _ in range(n_terms):
            a = int(ln_val.to_integral_value(rounding=decimal.ROUND_FLOOR))
            terms.append(a)
            frac = ln_val - a
            if frac <= eps:
                break
            ln_val = decimal.Decimal(1) / frac

    return terms


# ---------------------------------------------------------------------------
# mpmath backends
# ---------------------------------------------------------------------------


def _arcsinh_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of arcsinh(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 5 + 80
    val = mpmath.asinh(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _arccosh_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of arccosh(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 5 + 80
    val = mpmath.acosh(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _arctanh_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of arctanh(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 5 + 80
    val = mpmath.atanh(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def Arctanh(x) -> CF:
    """Inverse hyperbolic tangent of x, as a continued fraction.

    x may be an int or Fraction; must satisfy |x| < 1.
    Returns CF([0]) for x=0.
    Uses the identity arctanh(x) = ln((1+x)/(1-x)) / 2 via existing Ln
    (no external library required).

    Examples::

        Arctanh(0)               # [0]
        Arctanh(Fraction(1, 2))  # [0; 1, 1, 4, 1, 1, 3, ...]
        Arctanh(Fraction(1, 4))  # [0; 3, 1, 10, 1, ...]
    """
    x = _coerce_trig_arg(x)
    if abs(x) >= 1:
        raise ValueError(f"Arctanh argument must satisfy |x| < 1, got {x}")
    if x == 0:
        return CF.from_int(0)
    ratio = (Fraction(1) + x) / (Fraction(1) - x)
    from .logarithm import Ln

    return Ln(ratio) / CF.from_int(2)


def Arcsinh(x) -> CF:
    """Inverse hyperbolic sine of x, as a continued fraction.

    x may be an int or Fraction.  Returns CF([0]) for x=0.
    Uses arcsinh(x) = ln(x + sqrt(x²+1)).
    Dispatches to mpmath when available, else Decimal.

    Examples::

        Arcsinh(0)               # [0]
        Arcsinh(1)               # [0; 1, 5, 1, 1, ...]
        Arcsinh(Fraction(1, 2))  # [0; 2, 12, 1, 5, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(0)
    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _arcsinh_terms_mpmath(num, den, n))
    return _lazy_cf(lambda n: _arcsinh_terms_from_decimal(num, den, n))


def Arccosh(x) -> CF:
    """Inverse hyperbolic cosine of x, as a continued fraction.

    x may be an int or Fraction; must satisfy x ≥ 1.
    Returns CF([0]) for x=1.
    Uses arccosh(x) = ln(x + sqrt(x²-1)).
    Dispatches to mpmath when available, else Decimal.

    Examples::

        Arccosh(1)               # [0]
        Arccosh(2)               # [1; 3, 1, 3, 1, ...]
        Arccosh(Fraction(3, 2))  # [0; 1, 3, 3, 1, ...]
    """
    x = _coerce_trig_arg(x)
    if x < 1:
        raise ValueError(f"Arccosh argument must satisfy x ≥ 1, got {x}")
    if x == 1:
        return CF.from_int(0)
    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _arccosh_terms_mpmath(num, den, n))
    return _lazy_cf(lambda n: _arccosh_terms_from_decimal(num, den, n))
