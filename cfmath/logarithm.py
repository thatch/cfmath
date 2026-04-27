"""Logarithmic functions as continued fractions."""

from __future__ import annotations

from fractions import Fraction

from ._backend import _HAS_MPMATH, _lazy_cf
from .core import CF


def _ln_terms_from_decimal(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of ln(x_num/x_den) using high-precision Decimal.

    Uses ln(x) = 2·atanh((x-1)/(x+1)) with argument reduction to [½, 2).
    The atanh series 2·(y + y³/3 + y⁵/5 + ...) converges geometrically for |y| ≤ 1/3.
    No external library required.
    """
    import decimal

    prec = n_terms * 5 + 80
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    with decimal.localcontext(ctx):
        eps = decimal.Decimal(10) ** (-(prec - 10))

        def _two_atanh_dec(t: Fraction) -> decimal.Decimal:
            dt = decimal.Decimal(t.numerator) / decimal.Decimal(t.denominator)
            dt2 = dt * dt
            term = dt
            val = dt
            k = 1
            while True:
                term *= dt2
                delta = term / decimal.Decimal(2 * k + 1)
                val += delta
                if abs(delta) < eps:
                    break
                k += 1
            return 2 * val

        x = Fraction(x_num, x_den)
        n = 0
        reduced = x
        while reduced >= Fraction(2):
            reduced /= 2
            n += 1
        while reduced < Fraction(1, 2):
            reduced *= 2
            n -= 1

        t = (reduced - 1) / (reduced + 1)
        ln_val = decimal.Decimal(n) * _two_atanh_dec(Fraction(1, 3)) + _two_atanh_dec(t)

        terms: list[int] = []
        for _ in range(n_terms):
            a = int(ln_val.to_integral_value(rounding=decimal.ROUND_FLOOR))
            terms.append(a)
            frac = ln_val - a
            if frac <= eps:
                break
            ln_val = decimal.Decimal(1) / frac

    return terms


def _ln_terms_from_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of ln(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 5 + 80
    val = mpmath.log(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def Ln(x: int | Fraction | CF) -> CF:
    """Natural logarithm of x.

    x may be a positive int or Fraction.
    Uses ln(x) = 2·atanh((x-1)/(x+1)) with argument reduction when mpmath
    is unavailable, otherwise delegates to mpmath for speed.

    Examples::

        Ln(2)   # ≈ [0; 1, 2, 3, 1, 6, 3, 1, 1, 2, ...]
        Ln(3)   # ≈ [1; 10, 7, 9, 2, 2, 1, 3, 1, ...]
    """
    if isinstance(x, int):
        x = Fraction(x)
    elif not isinstance(x, Fraction):
        raise TypeError(f"Ln() expects int or Fraction, got {type(x).__name__}")
    if x <= 0:
        raise ValueError("Ln of non-positive number")
    if x == 1:
        return CF.from_int(0)

    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _ln_terms_from_mpmath(num, den, n))
    return _lazy_cf(lambda n: _ln_terms_from_decimal(num, den, n))


def Log10(x: int | Fraction | CF) -> CF:
    """Common logarithm (base 10) of x, as a continued fraction.

    x may be a positive int or Fraction.
    Computed as Ln(x) / Ln(10).

    Examples::

        Log10(10)               # [1]
        Log10(100)              # [2]
        Log10(2)                # ≈ [0; 3, 3, 9, 2, 1, 1, 2, ...]
    """
    return Ln(x) / Ln(10)


def Log(x: int | Fraction | CF, base: int | Fraction | None = None) -> CF:
    """Logarithm of x to the given base, as a continued fraction.

    x may be a positive int or Fraction.  base may be a positive int or
    Fraction ≠ 1, or None for the natural logarithm (same as Ln(x)).
    Computed as Ln(x) / Ln(base).

    Examples::

        Log(8, 2)               # [3]
        Log(Fraction(1, 4), 2)  # [-2]
        Log(10, 10)             # [1]
        Log(2)                  # same as Ln(2)
    """
    if base is None:
        return Ln(x)
    if isinstance(base, int):
        base = Fraction(base)
    elif not isinstance(base, Fraction):
        raise TypeError(f"Log() base expects int or Fraction, got {type(base).__name__}")
    if base <= 0 or base == 1:
        raise ValueError(f"Log() base must be positive and ≠ 1, got {base}")
    return Ln(x) / Ln(base)


def Log2(n: int | Fraction | CF) -> CF:
    """Logarithm base 2 of n, as a continued fraction.

    n may be a positive int or Fraction.
    Implemented as Ln(n) / Ln(2).

    Examples::

        Log2(2)                 # [1]
        Log2(4)                 # [2]
        Log2(123)               # [6; 1, 16, 2, 1, 1, 8, ...]
    """
    return Ln(n) / Ln(2)
