"""Special functions (Gamma, Zeta) as continued fractions."""

from __future__ import annotations

from fractions import Fraction

from .core import CF
from ._backend import _HAS_MPMATH, _lazy_cf


# ---------------------------------------------------------------------------
# Bernoulli numbers
# ---------------------------------------------------------------------------

def _bernoulli(n: int) -> Fraction:
    """Compute the n-th Bernoulli number as an exact Fraction.

    Uses the recurrence (m+1) B_m = -Σ_{k=0}^{m-1} C(m+1,k) B_k.
    """
    from math import comb
    B: list[Fraction] = [Fraction(0)] * (n + 1)
    B[0] = Fraction(1)
    for m in range(1, n + 1):
        B[m] = -sum(Fraction(comb(m + 1, k)) * B[k] for k in range(m)) / (m + 1)
    return B[n]


# ---------------------------------------------------------------------------
# Riemann zeta function
# ---------------------------------------------------------------------------

def _zeta_odd_terms_from_decimal(s: int, n_terms: int) -> list[int]:
    """Compute CF terms of ζ(s) for odd integer s ≥ 5 via Euler-accelerated η.

    Uses η(s) = (1 − 2^{1−s}) ζ(s) = Σ_{k=0}^∞ (−1)^k/(k+1)^s, accelerated
    by the backward-difference Euler transform (~3.32 steps per decimal digit,
    no external library).
    """
    import decimal

    prec = n_terms * 4 + 50
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    with decimal.localcontext(ctx):
        eps = decimal.Decimal(10) ** (-(prec - 10))
        half = decimal.Decimal(1) / decimal.Decimal(2)
        N = int(3.5 * prec) + 20

        d = [decimal.Decimal(1) / decimal.Decimal(k + 1) ** s for k in range(N)]

        val = decimal.Decimal(0)
        for n in range(N):
            t = d[0] * half
            if t < eps:
                break
            val += t
            end = N - n - 1
            for k in range(end):
                d[k] = (d[k] - d[k + 1]) * half

        val /= decimal.Decimal(1) - decimal.Decimal(2) ** (1 - s)

        terms: list[int] = []
        for _ in range(n_terms):
            a = int(val.to_integral_value(rounding=decimal.ROUND_FLOOR))
            terms.append(a)
            frac = val - a
            if frac <= eps:
                break
            val = decimal.Decimal(1) / frac
    return terms


def _zeta_odd_terms_mpmath(s: int, n_terms: int) -> list[int]:
    """Compute CF terms of ζ(s) for odd integer s ≥ 5 using mpmath."""
    import mpmath
    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.zeta(s)
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def Zeta(s: int) -> CF:
    """Riemann zeta function ζ(s) for integer s ≥ 2.

    ζ(s) = Σ_{n=1}^∞ 1/n^s

    - s = 3: delegates to Apery() (same constant)
    - Even s: exact rational multiple of π^s via Bernoulli numbers
    - Odd s ≥ 5: Euler-accelerated Dirichlet η, dispatches to mpmath when available

    Examples::

        Zeta(2)   # π²/6  ≈ [1; 1, 1, 1, 4, 2, 4, 1, 1, 1, ...]
        Zeta(3)   # Apéry ≈ [1; 4, 1, 18, 1, 1, 1, 4, ...]
        Zeta(4)   # π⁴/90 ≈ [1; 12, 6, 1, 3, 1, ...]
        Zeta(5)   #        ≈ [1; 27, 1, 1, 1, 2, 1, 1, ...]
    """
    if not isinstance(s, int) or s < 2:
        raise ValueError("Zeta requires integer s >= 2")
    from .constants import Apery
    if s == 3:
        return Apery()
    if s % 2 == 0:
        n = s // 2
        from math import factorial
        B2n = _bernoulli(s)
        coeff = Fraction((-1) ** (n + 1)) * B2n * Fraction(2 ** (2 * n - 1), factorial(2 * n))
        from .constants import Pi
        pi = Pi()
        result: CF = CF.from_rational(coeff)
        for _ in range(2 * n):
            result = result * pi
        return result
    else:
        if _HAS_MPMATH:
            return _lazy_cf(lambda n_terms: _zeta_odd_terms_mpmath(s, n_terms))
        return _lazy_cf(lambda n_terms: _zeta_odd_terms_from_decimal(s, n_terms))


# ---------------------------------------------------------------------------
# Gamma function
# ---------------------------------------------------------------------------

def _gamma_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of Γ(x_num/x_den) using mpmath."""
    import mpmath
    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.gamma(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        frac = val - a
        if frac == 0:
            break
        val = 1 / frac
    return terms


def Gamma(x) -> CF:
    """Gamma function Γ(x) for positive rational x.

    x may be a positive int or Fraction.

    Special cases:
    - Positive integers: Γ(n) = (n-1)! — returned as an exact CF.
    - All other positive rationals: computed via mpmath.

    Raises TypeError for non-int/Fraction input, ValueError for x ≤ 0.

    Examples::

        Gamma(1)                 # [1]
        Gamma(5)                 # [24]
        Gamma(Fraction(1, 2))    # ≈ [1; 1, 3, 2, 1, 1, 6, 1, ...]  (√π)
        Gamma(Fraction(3, 2))    # ≈ [0; 1, 4, 1, 1, 1, ...]  (√π/2)
    """
    import math as _math
    if isinstance(x, int):
        x = Fraction(x)
    elif not isinstance(x, Fraction):
        raise TypeError(f"Gamma() expects int or Fraction, got {type(x).__name__}")
    if x <= 0:
        raise ValueError(f"Gamma() argument must be positive, got {x}")

    if x.denominator == 1:
        n = x.numerator
        return CF.from_int(_math.factorial(n - 1))

    num, den = x.numerator, x.denominator
    return _lazy_cf(lambda n: _gamma_terms_mpmath(num, den, n))
