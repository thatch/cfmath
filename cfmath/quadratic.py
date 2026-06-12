"""Exact arithmetic for quadratic irrationals as continued fractions.

Quadratic irrationals are numbers of the form (P + √D) / Q with integer
P, Q, D.  They are precisely the numbers whose CF is eventually periodic
(Lagrange's theorem).  This module provides:

  * Sqrt(n)           — exact periodic CF for √n
  * _minimal_poly     — recover A·x²+B·x+C=0 from a periodic CF
  * _cf_from_poly     — convert that polynomial back to an explicit periodic CF
  * _periodic_square  — compute x² for a periodic CF without bihomographic stall
  * _periodic_mul     — multiply two periodic CFs in the same field Q(√d)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ._backend import _annotate_cf

if TYPE_CHECKING:
    from .core import CF


# ---------------------------------------------------------------------------
# Sqrt
# ---------------------------------------------------------------------------


def Sqrt(n: int) -> "CF":
    """Return the exact CF for sqrt(n).

    For perfect squares, returns the integer CF [floor(sqrt(n))].
    For non-square positive integers, computes the exact periodic CF
    using the standard algorithm.
    """
    from .core import CF as _CF

    if n < 0:
        raise ValueError("sqrt of negative number")
    a0 = math.isqrt(n)
    if a0 * a0 == n:
        return _annotate_cf(_CF.from_int(a0), ("Sqrt", n))

    # Standard algorithm for periodic CF of sqrt(n)
    # Invariant: sqrt(n) = (sqrt(n) + m) / d
    period: list[int] = []
    m, d, a = 0, 1, a0
    while True:
        m = d * a - m
        d = (n - m * m) // d
        a = (a0 + m) // d
        period.append(a)
        if a == 2 * a0:
            break

    return _annotate_cf(_CF([a0], repeating=period), ("Sqrt", n))


# ---------------------------------------------------------------------------
# Minimal polynomial helpers
# ---------------------------------------------------------------------------


def _minimal_poly(x: "CF") -> "tuple[int, int, int] | None":
    """Return (A, B, C) such that A·x² + B·x + C = 0 for periodic CF x.

    Every eventually-periodic CF is a quadratic irrational — it satisfies
    exactly one irreducible quadratic with integer coefficients.  We compute
    it by:

    1. Building the period matrix M = M_r1 · M_r2 · … · M_rk  (right-multiply,
       where M_a = [[a,1],[1,0]]).  This gives the self-referential equation
       for the periodic tail τ:  r·τ² + (s−p)·τ − q = 0.

    2. Unwinding the non-repeating prefix terms (innermost first) to convert
       the quadratic for τ into a quadratic for x.

    Returns None if x is not periodic or the leading coefficient is zero.
    """
    if not x.is_periodic():
        return None

    p, q, r, s = 1, 0, 0, 1  # identity
    for a in x.repeating:
        p, q, r, s = p * a + q, p, r * a + s, r

    A, B, C = r, s - p, -q  # quadratic for periodic tail τ
    for a in reversed(x.terms):
        A, B, C = C, B - 2 * C * a, A - B * a + C * a * a

    if A == 0:
        return None
    return A, B, C


def _square_free_decomp(n: int) -> "tuple[int, int]":
    """Return (d, k) where n = k²·d and d is square-free.  Requires n > 0."""
    d, k = n, 1
    i = 2
    while i * i <= d:
        while d % (i * i) == 0:
            d //= i * i
            k *= i
        i += 1
    return d, k


def _cf_from_poly(A: int, B: int, C: int) -> "CF | None":
    """Convert a minimal polynomial A·x²+B·x+C=0 to an explicit periodic CF.

    Returns a CF with correct ``terms`` and ``repeating`` (so ``is_periodic()``
    is True), rather than a lazy homographic wrapper.

    If the discriminant is a perfect square the result is a finite (rational) CF.
    If both roots are positive we take the larger one — this is correct for all
    Sqrt/Phi-type sources.  (TODO: may be wrong for other quadratics.)

    Algorithm: write x=(P+√D)/Q, run the standard period-finding loop, detect
    the cycle by recording seen states (P,Q).
    """
    from .core import CF as _CF

    D = B * B - 4 * A * C
    if D < 0:
        return None

    s = math.isqrt(D)
    if s * s == D:  # rational result
        for num in (-B + s, -B - s):
            if (num > 0) == (2 * A > 0):
                return _CF.from_fraction(num, 2 * A)
            if num == 0:
                return _CF.from_int(0)
        return _CF.from_fraction(-B + s, 2 * A)

    sqrt_D_f = math.sqrt(D)
    r1 = (-B + sqrt_D_f) / (2 * A)
    r2 = (-B - sqrt_D_f) / (2 * A)

    # Pick the positive root; prefer larger when both positive (see TODO above).
    P0, Q0 = (-B, 2 * A) if r1 >= r2 else (B, -2 * A)
    if Q0 < 0:
        P0, Q0 = -P0, -Q0
    if Q0 == 0:
        return None

    def floor_q(P: int, Q: int) -> int:
        """floor((P + √D) / Q), Q > 0, using float + integer correction."""
        a = int((P + sqrt_D_f) / Q)
        while ((a + 1) * Q - P) > 0 and ((a + 1) * Q - P) ** 2 <= D:
            a += 1
        r = a * Q - P
        while r > 0 and r * r > D:
            a -= 1
            r = a * Q - P
        return a

    a0 = floor_q(P0, Q0)
    all_terms: list[int] = [a0]
    seen: dict[tuple[int, int], int] = {(P0, Q0): 0}
    P, Q, a = P0, Q0, a0

    for _ in range(10 * (abs(A) + abs(B) + abs(C)) + 200):
        Pn = a * Q - P
        Qn = (D - Pn * Pn) // Q
        if Qn == 0:
            break
        an = floor_q(Pn, Qn)
        state = (Pn, Qn)
        if state in seen:
            k = seen[state]
            return _CF(all_terms[:k], repeating=all_terms[k:])
        seen[state] = len(all_terms)
        all_terms.append(an)
        P, Q, a = Pn, Qn, an

    return _CF(all_terms)  # fallback: finite CF (shouldn't normally reach)


# ---------------------------------------------------------------------------
# Periodic CF squaring and multiplication
# ---------------------------------------------------------------------------


def _periodic_square(x: "CF") -> "CF | None":
    """Return x² for a periodic CF x, using the equation that x satisfies.

    The normal bihomographic algorithm for x*x breaks down when the result
    is a whole number or simple fraction.  For example, Sqrt(2)**2 should
    give [2], but the algorithm loops forever: after consuming many input
    terms, some corners of the range say floor=2 and others say floor=1,
    and they never all agree.

    Why the loop happens
    --------------------
    The bihomographic treats its two inputs as *independent* values, each
    free to range over [1, ∞).  But x*x has a much tighter constraint: both
    inputs are *the same number*.  The corners of the range approach the true
    answer from both sides — some from above, some from below — and for an
    exact integer answer they oscillate forever without converging.

    The fix: skip the iteration entirely
    ------------------------------------
    Every periodic CF satisfies a quadratic equation.  If we know that
    equation, we can compute x*x by substituting directly.

    Examples:
      * Sqrt(2) = [1; 2, 2, 2, ...] satisfies  x² = 2
        So x*x is just the constant 2.
      * Phi()   = [1; 1, 1, 1, ...] satisfies  x² = x + 1
        So x*x equals x+1, computed by the (always-reliable) homographic
        algorithm: cf_homographic(x, 1, 1, 0, 1).

    Finding the equation from the repeating block
    ----------------------------------------------
    The repeating block [r₁, r₂, ..., rₖ] defines a self-referential
    formula: if we call the periodic tail τ, then after going through the
    whole period once we're back to τ.  That means:

        τ = r₁ + 1/(r₂ + 1/(... + 1/(rₖ + 1/τ)...))

    This is a fixed-point equation — τ appears on both sides.  Rearranging
    it gives a quadratic that τ satisfies.

    We find that quadratic using matrix multiplication.  Each step "a + 1/…"
    has a tidy 2×2 matrix form [[a,1],[1,0]].  Multiplying the matrices for
    the whole period gives [[p,q],[r,s]], and τ satisfies:

        r·τ² + (s−p)·τ − q = 0

    (This works because the matrix product encodes exactly the Möbius
    transformation that maps τ to itself when you go around the cycle.)

    Finally, since x = a₀ + 1/τ, we substitute τ = 1/(x−a₀) into the
    quadratic for τ and simplify to get a quadratic A·x² + B·x + C = 0
    for x itself.  From that, x² = −(B·x + C) / A, which is a homographic
    transform of x — something the normal algorithm handles without any
    issues.

    Concrete derivation for Sqrt(2) = [1; (2)]
    -------------------------------------------
      Period is [2].  Matrix M = [[2,1],[1,0]] (just one step).

      Tail equation:  1·τ² + (0−2)·τ − 1 = 0  →  τ² − 2τ − 1 = 0
      Solving: τ = 1 + √2  ✓  (the tail of [1; 2, 2, 2, ...])

      Substitute τ = 1/(x−1):
        1/(x−1)² − 2/(x−1) − 1 = 0   ×(x−1)²
        1 − 2(x−1) − (x−1)² = 0
        2 − x² = 0   →   x² = 2   ✓

      So x² = (0·x + 2) / (0·x + (−1)) / (−1) = 2.
      As a homographic: cf_homographic(x, 0, −2, 0, −1) = (−2)/(−1) = 2.

    Limitation
    ----------
    Returns None if x is not periodic, or if the math degenerates (e.g.
    the quadratic has zero leading coefficient — shouldn't happen for any
    standard Sqrt/Phi value).  In those cases the caller falls back to the
    bihomographic algorithm.
    """
    poly = _minimal_poly(x)
    if poly is None:
        return None
    A, B, C = poly

    # If Ax²+Bx+C=0 and z=x², substitute x=(-Az-C)/B (or use the B=0 shortcut).
    # Result: A²z² + (2AC−B²)z + C² = 0.
    from math import gcd

    A2, B2, C2 = A * A, 2 * A * C - B * B, C * C
    g = gcd(gcd(abs(A2), abs(B2)), abs(C2))
    return _cf_from_poly(A2 // g, B2 // g, C2 // g)


def _periodic_mul(x: "CF", y: "CF") -> "CF | None":
    """Multiply two periodic CFs that live in the same quadratic field Q(√d).

    What "same field" means
    -----------------------
    Every periodic CF satisfies a quadratic equation A·z² + B·z + C = 0
    whose discriminant D = B² − 4AC is a positive integer.  The square-free
    part of D — call it d — determines which quadratic field the number lives
    in.  For example:

      Sqrt(2)  →  D = 8  = 2²·2   →  d = 2   (lives in Q(√2))
      Sqrt(8)  →  D = 32 = 4²·2   →  d = 2   (also lives in Q(√2))
      Sqrt(3)  →  D = 12 = 2²·3   →  d = 3   (lives in Q(√3))

    Numbers in the same field can be written as P + Q·√d for rational P, Q.
    Their product is then: (P₁ + Q₁·√d)(P₂ + Q₂·√d) = (P₁P₂ + Q₁Q₂·d) + (P₁Q₂ + P₂Q₁)·√d
    which is also in Q(√d), expressible as a homographic transform of √d.

    Numbers in different fields (d₁ ≠ d₂) can't be combined this way —
    we return None and the caller falls back to the bihomographic algorithm.

    The calculation
    ---------------
    From the minimal polynomial A·x² + B·x + C = 0 we read off:
      x = (−B + ε·k·√d) / (2A)
    where D = k²·d (k is the square root of the "perfect square part" of D)
    and ε = ±1 is chosen so that x > 0 (the physically correct root).

    For x with coefficients (Ax, Bx, kx, εx) and y with (Ay, By, ky, εy):

      Rational part:     P = Bx·By + εx·εy·kx·ky·d
      Irrational part:   Q = −(εx·By·kx + εy·Bx·ky)      (coefficient of √d)
      Denominator:       denom = 4·Ax·Ay

    So x·y = (Q·√d + P) / denom = cf_homographic(Sqrt(d), Q, P, 0, denom).

    If Q = 0, the product is rational and we return a constant CF directly.

    Returns None if either CF is not periodic, their fields differ, or the
    discriminant arithmetic fails.  The caller should fall back to bihomographic.

    Worked examples
    ---------------
    Sqrt(2) × Sqrt(2)  (two separate objects, same minimal poly A=−1, B=0, C=2):
      D=8, k=2, d=2, ε=−1 for both.
      P = 0·0 + (−1)(−1)·2·2·2 = 8,  Q = 0,  denom = 4·(−1)·(−1) = 4
      Result = 8/4 = 2  ✓

    Sqrt(8) × Sqrt(2)  (both in Q(√2), d=2):
      Sqrt(8): A=−1, B=0, k=4, ε=−1;  Sqrt(2): A=−1, B=0, k=2, ε=−1
      P = 0 + 1·4·2·2 = 16,  Q = 0,  denom = 4
      Result = 16/4 = 4  ✓

    Sqrt(7) × Sqrt(7)  (d=7, k=6 after full period computation):
      P = 0 + 1·6·6·7 = 252,  Q = 0,  denom = 36
      Result = 252/36 = 7  ✓
    """
    from math import gcd

    from .core import CF as _CF

    poly_x = _minimal_poly(x)
    poly_y = _minimal_poly(y)
    if poly_x is None or poly_y is None:
        return None

    Ax, Bx, Cx = poly_x
    Ay, By, Cy = poly_y

    Dx = Bx * Bx - 4 * Ax * Cx
    Dy = By * By - 4 * Ay * Cy
    if Dx <= 0 or Dy <= 0:
        return None  # not a genuine quadratic irrational

    dx, kx = _square_free_decomp(Dx)
    dy, ky = _square_free_decomp(Dy)
    if dx != dy:
        return None  # different quadratic fields — bihomographic handles this

    d = dx
    sqrt_d_f = math.sqrt(d)

    # Determine which root of each minimal polynomial is the actual value.
    # x = (−Bx + ε·kx·√d) / (2Ax); we need x > 0.
    # When both roots are positive, prefer the larger one (consistent with _cf_from_poly).
    def _pick_eps(B: int, k: int, A: int) -> int:
        vp = (-B + k * sqrt_d_f) / (2 * A)
        vn = (-B - k * sqrt_d_f) / (2 * A)
        if vp > 0 and vn > 0:
            return 1 if vp >= vn else -1
        return 1 if vp > 0 else -1

    εx = _pick_eps(Bx, kx, Ax)
    εy = _pick_eps(By, ky, Ay)

    # x·y = P/denom + Q/denom · √d  where:
    P_num = Bx * By + εx * εy * kx * ky * d
    Q_num = -(εx * By * kx + εy * Bx * ky)
    denom = 4 * Ax * Ay

    # Normalise sign so denom > 0
    if denom < 0:
        P_num, Q_num, denom = -P_num, -Q_num, -denom

    g = gcd(gcd(abs(P_num), abs(Q_num)), denom)
    if g > 1:
        P_num //= g
        Q_num //= g
        denom //= g

    if Q_num == 0:
        return _CF.from_fraction(P_num, denom)

    # z = (P_num + Q_num·√d)/denom satisfies:
    #   denom²·z² − 2·P_num·denom·z + (P_num² − Q_num²·d) = 0
    A2 = denom * denom
    B2 = -2 * P_num * denom
    C2 = P_num * P_num - Q_num * Q_num * d
    g2 = gcd(gcd(abs(A2), abs(B2)), abs(C2))
    return _cf_from_poly(A2 // g2, B2 // g2, C2 // g2)
