"""Power and root functions as continued fractions."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Iterator

from ._backend import _annotate_cf
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
        return _annotate_cf(CF.from_fraction(a_root, b_root), ("Nthroot", x, k))

    # k=2: exact periodic CF via quadratic algorithm
    if k == 2:
        from .quadratic import _cf_from_poly

        result = _cf_from_poly(b, 0, -a)
        if result is not None:
            return _annotate_cf(result, ("Nthroot", x, k))

    # General: exact lazy CF via polynomial tracking
    return _annotate_cf(CF([], _source=_kthroot_cf_gen(a, b, k)), ("Nthroot", x, k))


def Cuberoot(n: int) -> CF:
    """Return the CF for n^(1/3).

    For perfect cubes returns an exact integer CF; otherwise returns a lazy CF
    whose terms are computed exactly using integer polynomial tracking.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Cuberoot expects a positive integer, got {n!r}")
    return _annotate_cf(Nthroot(n, 3), ("Cuberoot", n))


def _init_bracket_poly(a: int, b: int, p: int, q: int) -> list[int]:
    """Initial polynomial  b^p * y^q - a^p = 0  whose root is (a/b)^(p/q)."""
    return [b**p] + [0] * (q - 1) + [-(a**p)]


def _bracket_floor(coeffs: list[int]) -> int | None:
    """Floor of the unique root > 1, or None if the polynomial has degenerated.

    Degeneracy occurs when a bracket is an exact rational power whose CF has
    been exhausted — e.g. 2^(3/1) = 8 exactly, so after emitting the term 8
    the tail polynomial becomes [0, 1] (representing 1/0 = ∞).
    """
    return None if coeffs[0] == 0 else _floor_by_sign(coeffs)


# Bracket polynomials for α = π require degree ~33000 at the 5th convergent
# (103993/33102), with coefficients like 2^103993 (~31000 digits).  Beyond this
# threshold the per-step cost becomes prohibitive; fall through to exp/ln.
_MAX_BRACKET_DEGREE = 500


def _pow_rational_cf_gen(a: int, b: int, cf_exp: CF) -> Iterator[int]:
    """Yield exact CF terms of (a/b)^α where α is a CF.

    Maintains two polynomial bracket states evolving in parallel:
      poly_even  for  x^(p_{2k}/q_{2k})    even convergent of α, approaches from below
      poly_odd   for  x^(p_{2k+1}/q_{2k+1}) odd convergent of α, approaches from above

    Each bracket polynomial starts as  b^p * y^q - a^p = 0  (root = x^(p/q)).
    For x = a/b > 1, even convergents give x^(p/q) < x^α and odd give x^(p/q) > x^α,
    so the bracket brackets x^α from both sides.

    Emit: when both brackets agree on the floor, that integer is the next CF term.
    Ingest: when they disagree (or one bracket degenerates to a degenerate polynomial
    after an exact rational power is exhausted), pull the next pair of α-convergents
    and re-initialize both bracket polynomials.

    Re-initializing advances the new polynomial past all already-emitted output
    terms.  Two strategies are provided below; replay is the one currently implemented.

    TODO (replay — current): Apply _horner_update(poly, t) for each stored emitted
    term t.  Cost O(n * q) per re-initialization where n = number of emitted terms
    and q = degree of the new polynomial.  Requires storing the full emitted list.

    TODO (compose — future): Maintain the accumulated Möbius transform [p,q,r,s]
    encoding the substitution y = (p*z + q)/(r*z + s) built from the emitted terms.
    Substitute it into the new degree-q polynomial in one O(q^2) operation instead
    of n separate O(q) steps.  Avoids storing the emitted list and has better
    asymptotic cost per re-initialization.

    PRACTICAL LIMIT: α-convergents with large denominators create polynomials of
    very high degree with astronomically large coefficients (for α = π the 5th
    convergent has degree ~33000 and coefficients ~2^103993).  This generator stops
    and returns once convergents exceed _MAX_BRACKET_DEGREE; the caller should fall
    back to the numerical path for any remaining terms.

    Assumes  x = a/b > 1  (a > b > 0)  and  α > 0.
    """
    from .convergents import convergent_pairs

    conv_iter = convergent_pairs(cf_exp)

    # Seed with first two convergents of α (even = below, odd = above)
    try:
        p_even, q_even = next(conv_iter)
        p_odd, q_odd = next(conv_iter)
    except StopIteration:
        return

    if q_even > _MAX_BRACKET_DEGREE or q_odd > _MAX_BRACKET_DEGREE:
        return

    poly_even = _init_bracket_poly(a, b, p_even, q_even)
    poly_odd = _init_bracket_poly(a, b, p_odd, q_odd)

    # Stored emitted terms used for replay when re-initializing bracket polynomials.
    # TODO (compose): replace with accumulated Möbius transform [p, q, r, s].
    emitted: list[int] = []

    while True:
        f_even = _bracket_floor(poly_even)
        f_odd = _bracket_floor(poly_odd)

        if f_even is not None and f_even == f_odd:
            # Brackets agree: emit the shared floor as the next CF term
            c = f_even
            yield c
            emitted.append(c)
            poly_even = _reduce_coeffs(_horner_update(poly_even, c))
            poly_odd = _reduce_coeffs(_horner_update(poly_odd, c))
        else:
            # Brackets degenerate or disagree: tighten by pulling the next pair
            # of α-convergents and replaying all emitted terms onto each new poly.
            try:
                p_even, q_even = next(conv_iter)
                p_odd, q_odd = next(conv_iter)
            except StopIteration:
                return  # α is finite; no more convergents

            if q_even > _MAX_BRACKET_DEGREE or q_odd > _MAX_BRACKET_DEGREE:
                return  # coefficient blowup territory; stop exact computation

            # TODO (replay): O(n * q) replay — replace with compose for O(q^2)
            poly_even = _init_bracket_poly(a, b, p_even, q_even)
            for t in emitted:
                poly_even = _reduce_coeffs(_horner_update(poly_even, t))

            poly_odd = _init_bracket_poly(a, b, p_odd, q_odd)
            for t in emitted:
                poly_odd = _reduce_coeffs(_horner_update(poly_odd, t))


def Pow(x: int | Fraction, r: int | Fraction | CF) -> CF:
    """x raised to the power r, as a continued fraction.

    x may be a positive int or Fraction; r may be an int, Fraction, or CF.

    - r integer:          exact rational arithmetic
    - r Fraction p/q:     compute x^p exactly (Fraction), then Nthroot(x^p, q)
    - r CF:               bracket polynomial approach up to _MAX_BRACKET_DEGREE,
                          then Exp(r * Ln(x)) for remaining terms

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

    if isinstance(r, CF):
        from .exponential import Exp
        from .logarithm import Ln

        # _pow_rational_cf_gen yields exact terms via the bracket polynomial approach
        # until α's convergents exceed _MAX_BRACKET_DEGREE (coefficient blowup).
        # Exp(r * Ln(x)) covers the full range numerically.
        # TODO: once the compose strategy is implemented (eliminating blowup), replace
        # the Exp path entirely with _pow_rational_cf_gen.
        return _annotate_cf(Exp(r * Ln(x)), ("Pow", x, r))

    if isinstance(r, int):
        r = Fraction(r)
    elif not isinstance(r, Fraction):
        raise TypeError(f"Pow() exponent expects int, Fraction, or CF, got {type(r).__name__}")

    if x == 1 or r == 0:
        return _annotate_cf(CF.from_int(1), ("Pow", x, r))
    if r == 1:
        return _annotate_cf(CF.from_rational(x), ("Pow", x, r))

    # Integer exponent: exact rational
    if r.denominator == 1:
        return _annotate_cf(CF.from_rational(x**r.numerator), ("Pow", x, r))

    # Rational exponent p/q: compute x^p exactly, then take q-th root
    return _annotate_cf(Nthroot(x**r.numerator, r.denominator), ("Pow", x, r))
