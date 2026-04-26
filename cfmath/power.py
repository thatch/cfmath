"""Power and root functions as continued fractions."""

from __future__ import annotations

from fractions import Fraction
from math import comb, gcd
from typing import Iterator

from .core import CF


def _integer_kth_root(n: int, k: int) -> int:
    """Return floor(n^(1/k)) exactly."""
    if n <= 0:
        return 0
    x = int(round(n ** (1 / k)))
    while x**k > n:
        x -= 1
    while (x + 1) ** k <= n:
        x += 1
    return x


def _floor_kth_root_rational(a: int, b: int, k: int) -> int:
    """Return floor((a/b)^(1/k)) using binary search with exact integer arithmetic.

    Finds the largest m >= 0 with m^k * b <= a.
    """
    lo, hi = 0, _integer_kth_root(a, k) + 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid**k * b <= a:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _horner_update(coeffs: list[int], a: int) -> list[int]:
    """Coefficients of the transformed polynomial after substituting y = a + 1/z.

    Produces new_coeffs[j] = f^(j)(a) / j!, the j-th Taylor coefficient of f at a.
    These are always integers for integer-coefficient polynomials evaluated at integers.

    Computed by repeated synthetic division (Ruffini's rule): each pass reduces the
    degree by one and its final value is the next Taylor coefficient.
    """
    row = list(coeffs)
    result: list[int] = []
    while len(row) > 1:
        val = row[0]
        new_row = [val]
        for c in row[1:]:
            val = val * a + c
            new_row.append(val)
        result.append(new_row[-1])
        row = new_row[:-1]
    result.append(row[0])
    return result


def _reduce_coeffs(coeffs: list[int]) -> list[int]:
    """Divide all coefficients by their GCD to keep magnitudes small."""
    g = 0
    for c in coeffs:
        g = gcd(g, abs(c))
    return [c // g for c in coeffs] if g > 1 else coeffs


def _floor_by_sign(coeffs: list[int]) -> int:
    """Floor of the unique root > 1 via exponential + binary search.

    The root is always > 1 here (guaranteed by the Möbius structure: each tail
    is 1/(prev_tail - floor(prev_tail)), which is always > 1).

    Sign invariant: if leading > 0, f < 0 below the root and f > 0 above it;
    if leading < 0, the opposite. This holds because there is exactly one root
    in (1, +inf) and f(1) always has the "below" sign.
    """
    leading = coeffs[0]

    def _eval(m: int) -> int:
        v = 0
        for c in coeffs:
            v = v * m + c
        return v

    def _above(m: int) -> bool:
        fm = _eval(m)
        return (fm > 0) == (leading > 0)

    # Exponential search: double hi until we land above the root
    hi = 2
    while not _above(hi):
        hi *= 2

    # Binary search: find the exact crossover point
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _above(mid):
            hi = mid
        else:
            lo = mid

    return lo


def _kthroot_cf_gen(a: int, b: int, k: int) -> Iterator[int]:
    """Yield exact CF terms of (a/b)^(1/k) using integer arithmetic only.

    Maintains the integer polynomial satisfied by the current tail value and
    updates it at each step via the Möbius substitution y = floor + 1/z.
    No floating-point arithmetic is used at any step.

    The polynomial starts as b*y^k - a = 0 and its coefficients grow slowly
    (roughly O(N) bits after N terms), so the algorithm is practical for
    hundreds or thousands of terms.
    """
    # Initial polynomial: b*y^k - a = 0, coefficients in descending degree
    coeffs: list[int] = [b] + [0] * (k - 1) + [-a]

    # The integer part may be 0 (for values < 1, e.g. (1/8)^(1/3) = 1/2)
    a0 = _floor_kth_root_rational(a, b, k)
    yield a0

    # Advance polynomial past a0 to get the polynomial for the tail value 1/(alpha - a0)
    coeffs = _reduce_coeffs(_horner_update(coeffs, a0))

    # All subsequent tail values are > 1
    while True:
        af = _floor_by_sign(coeffs)
        yield af
        coeffs = _reduce_coeffs(_horner_update(coeffs, af))


def Nthroot(x: int | Fraction, k: int) -> CF:
    """n-th root of x as an exact continued fraction.

    x may be a positive int or Fraction; k must be an integer >= 2.

    Uses exact integer arithmetic throughout — no floating-point or mpmath needed.
    k=2 returns a periodic CF (quadratic irrational) via the Sqrt algorithm.
    k>=3 returns a lazy CF whose terms are produced exactly one by one by
    tracking the integer polynomial satisfied by the current tail value.

    Examples::

        Nthroot(2, 2)               # Sqrt(2) = [1; (2)] — exact periodic
        Nthroot(2, 3)               # 2^(1/3) ≈ [1; 3, 1, 5, 1, 1, 4, ...]
        Nthroot(16, 4)              # [2] — exact
        Nthroot(Fraction(1, 8), 3)  # 1/2 = [0; 2] — exact rational
        Nthroot(Fraction(2, 3), 4)  # (2/3)^(1/4) ≈ [0; 1, 3, ...]
    """
    if isinstance(x, int):
        x = Fraction(x)
    elif not isinstance(x, Fraction):
        raise TypeError(f"Nthroot base expects int or Fraction, got {type(x).__name__}")
    if x <= 0:
        raise ValueError(f"Nthroot base must be positive, got {x}")
    if not isinstance(k, int) or k < 2:
        raise ValueError(f"Nthroot degree must be an integer >= 2, got {k!r}")

    a, b = x.numerator, x.denominator  # x = a/b, both positive

    # Perfect k-th root: both numerator and denominator are perfect k-th powers
    a_root = _integer_kth_root(a, k)
    b_root = _integer_kth_root(b, k)
    if a_root**k == a and b_root**k == b:
        return CF.from_fraction(a_root, b_root)

    # k=2: exact periodic CF via quadratic algorithm
    if k == 2:
        from .quadratic import _cf_from_poly

        result = _cf_from_poly(b, 0, -a)
        if result is not None:
            return result

    # General: exact lazy CF via polynomial tracking
    return CF([], _source=_kthroot_cf_gen(a, b, k))


def Cuberoot(n: int) -> CF:
    """Return the CF for n^(1/3).

    For perfect cubes returns an exact integer CF; otherwise returns a lazy CF
    whose terms are computed exactly using integer polynomial tracking.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Cuberoot expects a positive integer, got {n!r}")
    return Nthroot(n, 3)


def Pow(x: int | Fraction, r: int | Fraction | CF) -> CF:
    """x raised to the power r, as a continued fraction.

    x may be a positive int or Fraction; r may be an int, Fraction, or CF.

    - r integer:          exact rational arithmetic
    - r Fraction p/q:     compute x^p exactly (Fraction), then Nthroot(x^p, q)
    - r CF:               Exp(r * Ln(x))

    Examples::

        Pow(4, Fraction(1, 2))              # [2] — exact square root
        Pow(2, Fraction(3, 2))              # 2√2 ≈ [2; 1, 3, 1, 5, ...]
        Pow(Fraction(1, 8), Fraction(1, 3)) # 1/2 = [0; 2] — exact cube root
        Pow(2, -3)                          # [0; 8] = 1/8
        Pow(2, Pi())                        # 2^π ≈ [8; 1, 4, 1, ...]
    """
    if isinstance(x, int):
        x = Fraction(x)
    elif not isinstance(x, Fraction):
        raise TypeError(f"Pow() base expects int or Fraction, got {type(x).__name__}")
    if x <= 0:
        raise ValueError(f"Pow() base must be positive, got {x}")

    # CF exponent: x^r = exp(r * ln(x))
    if isinstance(r, CF):
        from .exponential import Exp
        from .logarithm import Ln

        return Exp(r * Ln(x))

    if isinstance(r, int):
        r = Fraction(r)
    elif not isinstance(r, Fraction):
        raise TypeError(f"Pow() exponent expects int, Fraction, or CF, got {type(r).__name__}")

    if x == 1 or r == 0:
        return CF.from_int(1)
    if r == 1:
        return CF.from_rational(x)

    # Integer exponent: exact rational
    if r.denominator == 1:
        return CF.from_rational(x ** r.numerator)

    # Rational exponent p/q: compute x^p exactly, then take q-th root
    return Nthroot(x ** r.numerator, r.denominator)
