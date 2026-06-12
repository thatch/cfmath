"""Inverse hyperbolic functions as continued fractions."""

from __future__ import annotations

from fractions import Fraction

from ._backend import _HAS_MPMATH, _annotate_cf, _coerce_trig_arg, _lazy_cf, _mpmath_cf_for_cf_arg
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
    from .logarithm import _decimal_ln2

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

        ln2 = _decimal_ln2(prec)
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
    from .logarithm import _decimal_ln2

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

        ln2 = _decimal_ln2(prec)
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


def Arctanh(x: int | Fraction | CF) -> CF:
    """Inverse hyperbolic tangent of x, as a continued fraction.

    For int/Fraction, uses arctanh(x) = ln((1+x)/(1-x)) / 2 (|x| < 1 required).
    For CF input, uses the mpmath dual-precision convergent approach.
    """
    if isinstance(x, CF):
        import mpmath
        return _mpmath_cf_for_cf_arg(x, mpmath.atanh)
    x = _coerce_trig_arg(x)
    if abs(x) >= 1:
        raise ValueError(f"Arctanh argument must satisfy |x| < 1, got {x}")
    if x == 0:
        return _annotate_cf(CF.from_int(0), ("Arctanh", x))
    ratio = (Fraction(1) + x) / (Fraction(1) - x)
    from .logarithm import Ln

    return _annotate_cf(Ln(ratio) / CF.from_int(2), ("Arctanh", x))


def Arcsinh(x: int | Fraction | CF) -> CF:
    """Inverse hyperbolic sine of x, as a continued fraction.

    For int/Fraction, uses arcsinh(x) = ln(x + sqrt(x²+1)).
    For CF input, uses the mpmath dual-precision convergent approach.
    """
    if isinstance(x, CF):
        import mpmath
        return _mpmath_cf_for_cf_arg(x, mpmath.asinh)
    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(0), ("Arcsinh", x))
    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _arcsinh_terms_mpmath(num, den, n), debug_source=("Arcsinh", x))
    return _lazy_cf(lambda n: _arcsinh_terms_from_decimal(num, den, n), debug_source=("Arcsinh", x))


def Arccosh(x: int | Fraction | CF) -> CF:
    """Inverse hyperbolic cosine of x, as a continued fraction.

    For int/Fraction, uses arccosh(x) = ln(x + sqrt(x²-1)) (x ≥ 1 required).
    For CF input, uses the mpmath dual-precision convergent approach.
    """
    if isinstance(x, CF):
        import mpmath
        return _mpmath_cf_for_cf_arg(x, mpmath.acosh)
    x = _coerce_trig_arg(x)
    if x < 1:
        raise ValueError(f"Arccosh argument must satisfy x ≥ 1, got {x}")
    if x == 1:
        return _annotate_cf(CF.from_int(0), ("Arccosh", x))
    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _arccosh_terms_mpmath(num, den, n), debug_source=("Arccosh", x))
    return _lazy_cf(lambda n: _arccosh_terms_from_decimal(num, den, n), debug_source=("Arccosh", x))
