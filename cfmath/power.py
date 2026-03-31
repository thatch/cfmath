"""Power and root functions as continued fractions."""

from __future__ import annotations

from fractions import Fraction

from .core import CF
from ._backend import _HAS_MPMATH, _lazy_cf


def _integer_cbrt(n: int) -> int:
    """Return floor(n^(1/3)) exactly using integer Newton's method."""
    if n <= 0:
        return 0
    x = int(round(n ** (1 / 3)))
    while x ** 3 > n:
        x -= 1
    while (x + 1) ** 3 <= n:
        x += 1
    return x


def _cbrt_terms_from_decimal(n: int, n_terms: int) -> list[int]:
    """Compute CF terms for cbrt(n) using high-precision decimal arithmetic.

    Uses Python's standard-library `decimal` module (no mpmath needed).
    Newton's method: x_{k+1} = (2*x_k + n / x_k²) / 3 converges cubically.
    """
    import decimal

    prec = n_terms * 5 + 60
    with decimal.localcontext(decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)):
        dn = decimal.Decimal(n)
        x = decimal.Decimal(n ** (1 / 3))
        for _ in range(20):
            x = (2 * x + dn / (x * x)) / 3

        terms: list[int] = []
        eps = decimal.Decimal(10) ** (-(prec - 20))
        for _ in range(n_terms):
            a = int(x)
            terms.append(a)
            frac = x - a
            if frac <= eps:
                break
            x = decimal.Decimal(1) / frac

    return terms


def _cbrt_terms_from_mpmath(n: int, n_terms: int) -> list[int]:
    """Compute CF terms for cbrt(n) using mpmath."""
    import mpmath
    mpmath.mp.dps = n_terms * 5 + 60
    val = mpmath.cbrt(mpmath.mpf(n))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def Cuberoot(n: int) -> CF:
    """Return the CF for n^(1/3).

    For perfect cubes, returns the exact integer CF.
    Otherwise uses mpmath when available, else high-precision decimal arithmetic.
    """
    root = _integer_cbrt(n)
    if root ** 3 == n:
        return CF.from_int(root)
    if _HAS_MPMATH:
        return _lazy_cf(lambda n_terms: _cbrt_terms_from_mpmath(n, n_terms), initial=80)
    return _lazy_cf(lambda n_terms: _cbrt_terms_from_decimal(n, n_terms), initial=80)


def Pow(x, r) -> CF:
    """x raised to the rational power r, as a continued fraction.

    x may be a positive int or Fraction; r may be an int or Fraction.

    Special cases (exact or faster paths):
    - r is an integer: exact rational arithmetic
    - x is a positive integer and r = 1/2: uses Sqrt
    - x is a positive integer and r = 1/3: uses Cuberoot
    - general: Exp(r * Ln(x))

    Examples::

        Pow(4, Fraction(1, 2))  # [2] — exact via Sqrt
        Pow(2, Fraction(3, 2))  # 2√2 ≈ [2; 1, 3, 1, 5, ...]
        Pow(2, -3)              # [0; 8] = 1/8
        Pow(Fraction(2, 3), 2)  # [0; 2, 3, 1, ...] = 4/9
    """
    if isinstance(x, int):
        x = Fraction(x)
    elif not isinstance(x, Fraction):
        raise TypeError(f"Pow() base expects int or Fraction, got {type(x).__name__}")
    if isinstance(r, int):
        r = Fraction(r)
    elif not isinstance(r, Fraction):
        raise TypeError(f"Pow() exponent expects int or Fraction, got {type(r).__name__}")
    if x <= 0:
        raise ValueError(f"Pow() base must be positive, got {x}")
    if x == 1 or r == 0:
        return CF.from_int(1)
    if r == 1:
        return CF.from_rational(x)

    # Integer exponent: compute exactly as a rational
    if r.denominator == 1:
        return CF.from_rational(x ** r.numerator)

    # Square-root exponent: x^(p/2) = sqrt(x^p) — always a quadratic irrational
    if r.denominator == 2:
        xp = x ** r.numerator
        m, n = xp.numerator, xp.denominator
        from .quadratic import _cf_from_poly
        result = _cf_from_poly(n, 0, -m)
        if result is not None:
            return result

    # Cube-root exponent for integer base
    if x.denominator == 1 and r == Fraction(1, 3):
        return Cuberoot(x.numerator)

    # General case: exp(r * ln(x))
    from .exponential import Exp
    from .logarithm import Ln
    return Exp(r * Ln(x))
