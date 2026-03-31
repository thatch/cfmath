"""Named mathematical constants as continued fractions."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Iterator

from .core import CF
from ._backend import _HAS_MPMATH, _lazy_cf


# ---------------------------------------------------------------------------
# Exact constants
# ---------------------------------------------------------------------------

def Phi() -> CF:
    """Golden ratio φ = (1 + sqrt(5))/2 = [1; 1, 1, 1, ...]"""
    return CF([1], repeating=[1])


def _e_terms() -> Iterator[int]:
    """Yield CF terms of e beyond the first: [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]

    The pattern after the leading 2 is: 1, 2k, 1 for k = 1, 2, 3, ...
    """
    k = 1
    while True:
        yield 1
        yield 2 * k
        yield 1
        k += 1


def E() -> CF:
    """Euler's number e = [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]"""
    return CF([2], _source=_e_terms())


# ---------------------------------------------------------------------------
# π and τ
# ---------------------------------------------------------------------------

def _four_over_pi_gen():
    b = 1
    a = 1
    while True:
        yield (b, a**2)
        b += 2
        a += 1


def Pi() -> CF:
    """π = [3; 7, 15, 1, 292, ...]"""
    return 4 / CF.from_generalized_cf(_four_over_pi_gen())


def Tau() -> CF:
    """τ = 2π ≈ [6; 3, 1, 1, 7, 2, 146, 3, 6, ...]"""
    return CF.from_int(2) * Pi()


# ---------------------------------------------------------------------------
# Euler-Mascheroni constant γ
# ---------------------------------------------------------------------------

def _euler_gamma_terms(n_terms: int) -> list[int]:
    """Compute n_terms CF terms of the Euler-Mascheroni constant γ using mpmath."""
    import mpmath
    mpmath.mp.dps = n_terms * 4 + 50
    x = mpmath.euler
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(x))
        terms.append(a)
        x = 1 / (x - a)
    return terms


def EulerGamma() -> CF:
    """Euler-Mascheroni constant γ ≈ [0; 1, 1, 2, 1, 2, 1, 4, 3, 13, ...]

    γ = lim_{n→∞} (1 + 1/2 + ... + 1/n − ln n) ≈ 0.5772156649...
    """
    return _lazy_cf(_euler_gamma_terms)


# ---------------------------------------------------------------------------
# Catalan's constant G
# ---------------------------------------------------------------------------

def _catalan_terms_mpmath(n_terms: int) -> list[int]:
    """Compute n_terms CF terms of Catalan's constant using mpmath."""
    import mpmath
    mpmath.mp.dps = n_terms * 4 + 50
    x = mpmath.catalan
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(x))
        terms.append(a)
        x = 1 / (x - a)
    return terms


def _catalan_terms_from_decimal(n_terms: int) -> list[int]:
    """Compute n_terms CF terms of Catalan's constant without external libraries.

    Uses the Euler acceleration of G = Σ_{k=0}^∞ (-1)^k / (2k+1)²:
        G = Σ_{n=0}^∞ (1/2^{n+1}) · ∇^n a_0,   a_k = 1/(2k+1)²

    where ∇ is the backward-difference operator ∇a_k = a_k − a_{k+1}.
    The outer 1/2^{n+1} makes the series converge geometrically at ratio 1/2
    (~3.32 steps per decimal digit).
    """
    import decimal

    prec = n_terms * 5 + 80
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    with decimal.localcontext(ctx):
        eps = decimal.Decimal(10) ** (-(prec - 10))
        half = decimal.Decimal(1) / decimal.Decimal(2)

        N = int(3.5 * prec) + 20
        d = [decimal.Decimal(1) / decimal.Decimal((2 * k + 1) ** 2) for k in range(N)]

        val = decimal.Decimal(0)
        for n in range(N):
            t = d[0] * half
            if t < eps:
                break
            val += t
            end = N - n - 1
            for k in range(end):
                d[k] = (d[k] - d[k + 1]) * half

        terms: list[int] = []
        for _ in range(n_terms):
            a = int(val.to_integral_value(rounding=decimal.ROUND_FLOOR))
            terms.append(a)
            frac = val - a
            if frac <= eps:
                break
            val = decimal.Decimal(1) / frac

    return terms


def Catalan() -> CF:
    """Catalan's constant G ≈ [0; 1, 10, 1, 8, 1, 88, 4, 1, 1, 14, ...]

    G = β(2) = Σ_{k=0}^∞ (-1)^k / (2k+1)² ≈ 0.9159655941772190...
    """
    fn = _catalan_terms_mpmath if _HAS_MPMATH else _catalan_terms_from_decimal
    return _lazy_cf(fn)


# ---------------------------------------------------------------------------
# Apéry's constant ζ(3)
# ---------------------------------------------------------------------------

def _apery_terms_mpmath(n_terms: int) -> list[int]:
    """Compute n_terms CF terms of Apéry's constant ζ(3) using mpmath."""
    import mpmath
    mpmath.mp.dps = n_terms * 4 + 50
    x = mpmath.apery
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(x))
        terms.append(a)
        x = 1 / (x - a)
    return terms


def _apery_terms_from_decimal(n_terms: int) -> list[int]:
    """Compute n_terms CF terms of Apéry's constant ζ(3) without external libraries.

    Uses Apéry's own alternating series:
        ζ(3) = (5/2) · Σ_{k=1}^∞ (-1)^{k-1} / (k³ · C(2k, k))

    C(2k, k) grows like 4^k/√(πk), so consecutive terms have ratio → 1/4, giving
    ~1.66 terms per decimal digit.
    """
    import decimal

    prec = n_terms * 4 + 50
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    with decimal.localcontext(ctx):
        eps = decimal.Decimal(10) ** (-(prec - 10))

        N = 2 * prec + 20

        val = decimal.Decimal(0)
        binom = decimal.Decimal(2)  # C(2, 1)
        sign = 1
        for k in range(1, N + 1):
            dk = decimal.Decimal(k)
            term = decimal.Decimal(sign) / (dk ** 3 * binom)
            val += term
            if abs(term) < eps:
                break
            sign = -sign
            binom = binom * decimal.Decimal(2 * (2 * k + 1)) / decimal.Decimal(k + 1)

        val = val * decimal.Decimal(5) / decimal.Decimal(2)

        terms: list[int] = []
        for _ in range(n_terms):
            a = int(val.to_integral_value(rounding=decimal.ROUND_FLOOR))
            terms.append(a)
            frac = val - a
            if frac <= eps:
                break
            val = decimal.Decimal(1) / frac

    return terms


def Apery() -> CF:
    """Apéry's constant ζ(3) ≈ [1; 4, 1, 18, 1, 1, 1, 4, 1, 9, 9, 2, ...]

    ζ(3) = Σ_{n=1}^∞ 1/n³ ≈ 1.2020569031595942...
    """
    fn = _apery_terms_mpmath if _HAS_MPMATH else _apery_terms_from_decimal
    return _lazy_cf(fn)


# ---------------------------------------------------------------------------
# Plastic constant
# ---------------------------------------------------------------------------

def _plastic_terms_mpmath(n_terms: int) -> list[int]:
    """Compute n_terms CF terms of the plastic constant using mpmath."""
    import mpmath
    mpmath.mp.dps = n_terms * 4 + 50
    x = mpmath.findroot(lambda z: z**3 - z - 1, mpmath.mpf("1.3"))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(x))
        terms.append(a)
        x = 1 / (x - a)
    return terms


def _plastic_terms(n_terms: int) -> list[int]:
    """Compute n_terms CF terms of the plastic constant via Newton's method.

    Solves x³ − x − 1 = 0; Newton iteration converges quadratically (~log₂(prec)
    steps), so precision costs are negligible compared to the CF extraction.
    """
    import decimal

    prec = n_terms * 4 + 50
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    with decimal.localcontext(ctx):
        eps = decimal.Decimal(10) ** (-(prec - 5))
        x = decimal.Decimal(13) / decimal.Decimal(10)   # near ρ ≈ 1.3247
        while True:
            x2 = x * x
            dx = (x2 * x - x - 1) / (3 * x2 - 1)
            x -= dx
            if abs(dx) < eps:
                break
        terms: list[int] = []
        for _ in range(n_terms):
            a = int(x.to_integral_value(rounding=decimal.ROUND_FLOOR))
            terms.append(a)
            frac = x - a
            if frac <= eps:
                break
            x = decimal.Decimal(1) / frac
    return terms


def Plastic() -> CF:
    """Plastic constant ρ ≈ [1; 3, 12, 1, 1, 3, 2, 3, 2, 4, ...]

    Real root of x³ − x − 1 = 0 ≈ 1.32471795724474602596...
    A cubic irrational — aperiodic CF unlike quadratic Phi = [1; 1, 1, 1, ...].
    """
    fn = _plastic_terms_mpmath if _HAS_MPMATH else _plastic_terms
    return _lazy_cf(fn)


# ---------------------------------------------------------------------------
# Khinchin's constant
# ---------------------------------------------------------------------------

def _khinchin_terms(n_terms: int) -> list[int]:
    """Compute n_terms CF terms of Khinchin's constant K using mpmath."""
    import mpmath
    mpmath.mp.dps = n_terms * 4 + 50
    x = mpmath.khinchin
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(x))
        terms.append(a)
        x = 1 / (x - a)
    return terms


def Khinchin() -> CF:
    """Khinchin's constant K ≈ [2; 1, 2, 5, 1, 1, 2, 1, 1, 3, ...]

    For almost all real x, the geometric mean (a₁·a₂···aₙ)^{1/n} of the
    CF partial quotients converges to K ≈ 2.6854520010653064...

        K = Π_{k=1}^∞ (1 + 1/(k(k+2)))^{log₂ k}
    """
    return _lazy_cf(_khinchin_terms)
