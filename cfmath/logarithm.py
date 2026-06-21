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


def _ln_terms_from_cf(x: CF, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of ln(x) where x is a CF, using mpmath."""
    import mpmath

    from .convergents import convergent as _convergent

    mpmath.mp.dps = n_terms * 5 + 80
    approx: Fraction = _convergent(x, n_terms * 2 + 20)
    if approx <= 0:
        raise ValueError("Ln of non-positive number")
    val = mpmath.log(mpmath.mpf(approx.numerator) / mpmath.mpf(approx.denominator))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def Ln(x: int | Fraction | CF) -> CF:
    """Natural logarithm of x.

    x may be a positive int, Fraction, or CF.
    Uses ln(x) = 2·atanh((x-1)/(x+1)) with argument reduction when mpmath
    is unavailable, otherwise delegates to mpmath for speed.
    CF inputs always require mpmath (used to evaluate a convergent).

    Examples::

        Ln(2)        # ≈ [0; 1, 2, 3, 1, 6, 3, 1, 1, 2, ...]
        Ln(3)        # ≈ [1; 10, 7, 9, 2, 2, 1, 3, 1, ...]
        Ln(Sqrt(2))  # ≈ [0; 2, 1, 2, 1, 4, 1, ...]  (= ln(2)/2)
    """
    if isinstance(x, CF) and x.is_finite():
        # A finite CF is an exact rational; use the (exact) rational path rather
        # than the convergent-of-an-infinite-CF path, which needs many terms.
        x = x.to_fraction()

    if isinstance(x, CF):
        # Quick non-positive check from the first term alone.
        # a0 < 0 → definitely negative; a0 == 0 with no further terms → zero.
        a0 = next(x._iter_from(0))
        if a0 < 0 or (a0 == 0 and not x.repeating and x._source is None and len(x.terms) == 1):
            raise ValueError("Ln of non-positive number")

        x_cf: CF = x
        return _lazy_cf(lambda n: _ln_terms_from_cf(x_cf, n))

    if isinstance(x, int):
        x = Fraction(x)
    elif not isinstance(x, Fraction):
        raise TypeError(f"Ln() expects int, Fraction, or CF, got {type(x).__name__}")
    if x <= 0:
        raise ValueError("Ln of non-positive number")
    if x == 1:
        return CF.from_int(0)

    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _ln_terms_from_mpmath(num, den, n))
    return _lazy_cf(lambda n: _ln_terms_from_decimal(num, den, n))


def _coerce_log_arg(x: int | Fraction | CF) -> Fraction | None:
    """Return x as a Fraction if it is rational, else None."""
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, Fraction):
        return x
    if isinstance(x, CF) and x.is_finite():
        return x.to_fraction()
    return None


def _exact_rational_log(x: Fraction, base: Fraction) -> int | None:
    """Return k if x == base**k for integer k, else None."""
    if x <= 0 or base <= 0 or base == 1:
        return None
    if x == 1:
        return 0
    if base < 1:
        found = _exact_rational_log(x, 1 / base)
        return None if found is None else -found
    if x < 1:
        found = _exact_rational_log(1 / x, base)
        return None if found is None else -found
    power = Fraction(1)
    k = 0
    while power < x:
        power *= base
        k += 1
    return k if power == x else None


def Log10(x: int | Fraction | CF) -> CF:
    """Common logarithm (base 10) of x, as a continued fraction.

    x may be a positive int or Fraction.
    Computed as Ln(x) / Ln(10).

    Examples::

        Log10(10)               # [1]
        Log10(100)              # [2]
        Log10(2)                # ≈ [0; 3, 3, 9, 2, 1, 1, 2, ...]
    """
    x_rat = _coerce_log_arg(x)
    if x_rat is not None:
        exact = _exact_rational_log(x_rat, Fraction(10))
        if exact is not None:
            return CF.from_int(exact)
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
    x_rat = _coerce_log_arg(x)
    if x_rat is not None:
        exact = _exact_rational_log(x_rat, base)
        if exact is not None:
            return CF.from_int(exact)
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
    n_rat = _coerce_log_arg(n)
    if n_rat is not None:
        exact = _exact_rational_log(n_rat, Fraction(2))
        if exact is not None:
            return CF.from_int(exact)
    return Ln(n) / Ln(2)
