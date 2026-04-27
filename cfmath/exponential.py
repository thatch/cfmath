"""Exponential function as a continued fraction."""

from __future__ import annotations

from fractions import Fraction

from ._backend import _HAS_MPMATH, _coerce_trig_arg, _lazy_cf
from .core import CF


def _exp_terms_from_decimal(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of exp(x_num/x_den) using high-precision Decimal.

    Uses the Taylor series exp(x) = Σ xⁿ/n!  No external library required.
    """
    import decimal

    prec = n_terms * 5 + 80
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    with decimal.localcontext(ctx):
        x = decimal.Decimal(x_num) / decimal.Decimal(x_den)
        term = decimal.Decimal(1)
        val = decimal.Decimal(1)
        k = 1
        eps = decimal.Decimal(10) ** (-(prec - 10))
        while True:
            term *= x / decimal.Decimal(k)
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


def _exp_terms_from_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of exp(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.exp(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def Exp(x: int | Fraction | CF) -> CF:
    """e raised to the power x, as a continued fraction.

    x may be an int, Fraction, or CF.  Returns CF([1]) for x=0.

    Accepting a CF argument enables arbitrary real powers:
        x ** r  ==  Exp(r * Ln(x))   for any rational r and positive x.
    (r * Ln(x) returns a CF, which is passed directly here.)

    Examples::

        Exp(1)                       # e ≈ [2; 1, 2, 1, 1, 4, ...]
        Exp(Fraction(1, 2))          # sqrt(e) ≈ [1; 1, 1, 1, 5, ...]
        Exp(Fraction(3, 2) * Ln(2))  # 2^(3/2) ≈ [2; 1, 3, 1, 5, ...]
    """
    if isinstance(x, CF):
        x_cf = x

        def _compute(n_terms: int) -> list[int]:
            import mpmath

            dps = n_terms * 4 + 50
            mpmath.mp.dps = dps
            from .convergents import convergent as _convergent

            depth = n_terms * 2 + 20
            approx: Fraction = _convergent(x_cf, depth)
            val = mpmath.exp(mpmath.mpf(approx.numerator) / mpmath.mpf(approx.denominator))
            terms: list[int] = []
            for _ in range(n_terms):
                a = int(mpmath.floor(val))
                terms.append(a)
                val = 1 / (val - a)
            return terms

        return _lazy_cf(_compute)

    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(1)
    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _exp_terms_from_mpmath(num, den, n))
    return _lazy_cf(lambda n: _exp_terms_from_decimal(num, den, n))
