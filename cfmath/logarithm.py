"""Logarithmic functions as continued fractions."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterator

from ._backend import _HAS_MPMATH, _annotate_cf, _lazy_cf
from .core import CF


def _ln1p_meta_gcf_terms() -> Iterator[tuple[list[int], list[int]]]:
    """Yield polynomial terms for ln(1 + 1/z)'s generalized meta-CF.

    The denominator H satisfies

        ln(1 + 1/z) = 1 / H(z)
        H_n(z) = (n + 1)z - n + (n + 1)^2 z / H_{n+1}(z)

    for n = 0, 1, 2, ...
    """
    n = 0
    while True:
        yield ([-n, n + 1], [0, (n + 1) ** 2])
        n += 1


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
        if x.is_finite():
            return Ln(x.to_fraction())
        # Quick non-positive check from the first term alone.
        # a0 < 0 → definitely negative; a0 == 0 with no further terms → zero.
        a0 = next(x._iter_from(0))
        if a0 < 0 or (a0 == 0 and not x.repeating and x._source is None and len(x.terms) == 1):
            raise ValueError("Ln of non-positive number")
        x_cf: CF = x
        return _lazy_cf(lambda n: _ln_terms_from_cf(x_cf, n), debug_source=("Ln", x_cf))

    if isinstance(x, int):
        x = Fraction(x)
    elif not isinstance(x, Fraction):
        raise TypeError(f"Ln() expects int, Fraction, or CF, got {type(x).__name__}")
    if x <= 0:
        raise ValueError("Ln of non-positive number")
    if x == 1:
        return _annotate_cf(CF.from_int(0), ("Ln", x))

    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _ln_terms_from_mpmath(num, den, n), debug_source=("Ln", x))
    return _lazy_cf(lambda n: _ln_terms_from_decimal(num, den, n), debug_source=("Ln", x))


def _cf_interval(x: CF, n_terms: int) -> tuple[Fraction, Fraction]:
    """Return rational bounds known to contain x after n_terms CF terms."""
    return x.interval(n_terms)


def _prove_positive(x: CF, max_terms: int = 120) -> None:
    """Raise unless interval refinement proves x is positive."""
    for n_terms in range(1, max_terms + 1):
        lo, hi = _cf_interval(x, n_terms)
        if lo > 0:
            return
        if hi <= 0:
            raise ValueError("LnCF of non-positive number")
    raise ValueError("LnCF could not prove argument is positive")


def _pow2(k: int) -> Fraction:
    """Return 2**k as a Fraction."""
    if k >= 0:
        return Fraction(1 << k)
    return Fraction(1, 1 << -k)


def _floor_log2(q: Fraction) -> int:
    """Return floor(log2(q)) for a positive Fraction."""
    if q <= 0:
        raise ValueError("log2 floor needs a positive value")
    k = q.numerator.bit_length() - q.denominator.bit_length()
    while _pow2(k + 1) <= q:
        k += 1
    while _pow2(k) > q:
        k -= 1
    return k


def _reduce_pow2_cf(x: CF, max_terms: int = 120) -> tuple[int, CF]:
    """Return k and r such that x = 2**k * r, with 1 <= r < 2.

    The tension is boundary detection.  If x is exactly a power of two but the
    CF expression does not expose that structurally, intervals may keep
    straddling the boundary.  In that case this helper reports that reduction is
    undecided instead of choosing the wrong k.
    """
    _prove_positive(x, max_terms=max_terms)
    for n_terms in range(1, max_terms + 1):
        lo, hi = _cf_interval(x, n_terms)
        if lo <= 0:
            continue
        k_lo = _floor_log2(lo)
        k_hi = _floor_log2(hi)
        if k_lo == k_hi:
            k = k_lo
            scale = CF.from_rational(_pow2(k))
            return k, x / scale
    raise ValueError("LnCF could not reduce argument by powers of two")


def _ln1p_cf(u: Fraction | CF) -> CF:
    """Return ln(1 + u) for u > 0 using the experimental meta-CF path."""
    from .gosper import cf_metaGCF

    if isinstance(u, Fraction):
        if u == 0:
            return CF.from_int(0)
        z = 1 / CF.from_rational(u)
    else:
        z = 1 / u
    return 1 / cf_metaGCF(z, _ln1p_meta_gcf_terms())


def _ln2_cf() -> CF:
    """Return ln(2) without using the slow endpoint u = 1 expansion."""
    return _ln1p_cf(Fraction(1, 2)) + _ln1p_cf(Fraction(1, 3))


def LnCF(x: int | Fraction | CF) -> CF:
    """Natural logarithm of x, using the experimental meta-CF path.

    The hard case is keeping the log expansion away from ``ln(1 + 1)``.  This
    function reduces by powers of two, computes ``ln(2)`` as
    ``ln(3/2) + ln(4/3)``, and evaluates the remaining ``ln(1 + u)`` with
    positive generalized meta-CF numerators.  The cost is extra Gosper
    arithmetic for the reduction terms.
    """
    if isinstance(x, (int, Fraction)):
        x = Fraction(x)
        if x <= 0:
            raise ValueError("LnCF of non-positive number")
        if x == 1:
            return CF.from_int(0)

        reduced = x
        shifts = 0
        while reduced >= 2:
            reduced /= 2
            shifts += 1
        while reduced < 1:
            reduced *= 2
            shifts -= 1

        out = CF.from_int(0)
        if shifts:
            out = shifts * _ln2_cf()
        if reduced == 1:
            return out
        return out + _ln1p_cf(reduced - 1)

    if not isinstance(x, CF):
        raise TypeError(f"LnCF() expects int, Fraction, or CF, got {type(x).__name__}")
    if x.is_finite():
        return LnCF(x.to_fraction())

    shifts, reduced_cf = _reduce_pow2_cf(x)
    out = CF.from_int(0)
    if shifts:
        out = shifts * _ln2_cf()
    return out + _ln1p_cf(reduced_cf - 1)


def Log10CF(x: int | Fraction | CF) -> CF:
    """Common logarithm of x, using ``LnCF``."""
    return LnCF(x) / LnCF(10)


def LogCF(x: int | Fraction | CF, base: int | Fraction | CF | None = None) -> CF:
    """Logarithm of x to the given base, using ``LnCF``."""
    if base is None:
        return LnCF(x)
    if isinstance(base, int):
        base = Fraction(base)
    elif not isinstance(base, (Fraction, CF)):
        raise TypeError(f"LogCF() base expects int, Fraction, or CF, got {type(base).__name__}")
    if isinstance(base, Fraction) and (base <= 0 or base == 1):
        raise ValueError(f"LogCF() base must be positive and ≠ 1, got {base}")
    if isinstance(base, CF) and base.is_finite():
        base_value = base.to_fraction()
        if base_value <= 0 or base_value == 1:
            raise ValueError(f"LogCF() base must be positive and ≠ 1, got {base_value}")
        base = base_value
    return LnCF(x) / LnCF(base)


def Log2CF(n: int | Fraction | CF) -> CF:
    """Logarithm base 2 of n, using ``LnCF``."""
    return LnCF(n) / LnCF(2)


def _coerce_log_arg(name: str, x: int | Fraction | CF) -> Fraction | None:
    """Return rational log inputs as Fraction, or None for non-rational CF."""
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, Fraction):
        return x
    if isinstance(x, CF) and x.is_finite():
        return x.to_fraction()
    if isinstance(x, CF):
        return None
    raise TypeError(f"{name} expects int or Fraction, got {type(x).__name__}")


def _exact_rational_log(x: Fraction, base: Fraction) -> int | None:
    """Return k if x == base**k for an integer k, else None."""
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
    if power == x:
        return k
    return None


def Log10(x: int | Fraction | CF) -> CF:
    """Common logarithm (base 10) of x, as a continued fraction.

    x may be a positive int or Fraction.
    Computed as Ln(x) / Ln(10).

    Examples::

        Log10(10)               # [1]
        Log10(100)              # [2]
        Log10(2)                # ≈ [0; 3, 3, 9, 2, 1, 1, 2, ...]
    """
    x_rat = _coerce_log_arg("Log10()", x)
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
    x_rat = _coerce_log_arg("Log()", x)
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
    n_rat = _coerce_log_arg("Log2()", n)
    if n_rat is not None:
        exact = _exact_rational_log(n_rat, Fraction(2))
        if exact is not None:
            return CF.from_int(exact)
    return Ln(n) / Ln(2)
