"""Gosper's algorithms for exact arithmetic on continued fractions.

Bill Gosper (in a 1972 MIT memo called HAKMEM) discovered how to add, subtract,
multiply, and divide continued fractions directly, without ever converting them
to decimals.  The key insight is that a CF represents a number via a sequence
of integer terms, and arithmetic can be done term-by-term, alternating between
reading input terms and writing output terms.

Two building blocks:

  Single-input (homographic) formula — one CF input x:
      y = (ax + b) / (cx + d)
  This is a linear fractional (Möbius) transformation — a ratio of two linear
  expressions in x.  It covers negation, reciprocal, and rational scaling.

  Two-input (bihomographic) formula — two CF inputs x and y:
      z = (axy + bx + cy + d) / (exy + fx + gy + h)
  This covers addition, subtraction, multiplication, and division of two CFs.

The algorithm maintains a state matrix of 8 (or 4) integer coefficients.  At
each step it either:
  1. Emits the next output term, when all possible tail values round to the same integer.
  2. Reads the next term from whichever input reduces uncertainty the most.

"Output is determined" means evaluating the formula at the extreme corners of
the remaining input range (x', y' ∈ [1, ∞) for CF tails) and checking that all
four (or two) corner values share the same integer floor:

  Single-input corners (x' ∈ [1, ∞)):
    x'→∞:   a/c
    x'=1:   (a+b)/(c+d)

  Two-input corners (x', y' ∈ [1, ∞)):
    (∞,∞):   a/e
    (∞, 1):  (a+b)/(e+f)
    (1, ∞):  (a+c)/(e+g)
    (1,  1): (a+b+c+d)/(e+f+g+h)

All corner denominators must be non-zero and the same sign (no division-by-zero
in the range), and all floor values must agree, before a term can be emitted.

References
----------
Bill Gosper, HAKMEM, MIT AI Memo 239, 1972, items 101-101B.
"""

from __future__ import annotations

import os
from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Iterator

from ._backend import _annotate_cf
from .quadratic import _periodic_mul, _periodic_square

if TYPE_CHECKING:
    from .core import CF


# Max non-emitting iterations before _metaCF_simple_terms gives up and raises.
# The simple reference path cannot detect an exact-rational result; given more
# budget it emits a spurious near-rational term (like the mpmath backend) rather
# than raising, so this stays small enough to raise first (garbage appears past
# ~125).  Override with CFRAC_METACF_STALL_LIMIT.
_METACF_STALL_LIMIT = int(os.environ.get("CFRAC_METACF_STALL_LIMIT", "100"))

# Max extra terms of x read in the main path when gimme is OFF
# (gimme_min_term_digits=None) before raising on a stall.  Unlike the simple
# path, the main path never emits a spurious term, so this can be generous: it
# bounds how large a legitimate partial quotient an irrational result may have
# before it is wrongly rejected (~one digit per ~2.4 terms), so 200 admits
# ~80-digit terms.  Override with CFRAC_METACF_NONE_STALL_LIMIT.
_METACF_NONE_STALL_LIMIT = int(os.environ.get("CFRAC_METACF_NONE_STALL_LIMIT", "200"))

# "Gimme" mode: when a stall persists, the value sits within 10^-N of an integer
# boundary, where N (the digit count of the partial quotient we would otherwise
# emit) grows as we refine.  Once N reaches this threshold, gimme declares the
# value exactly that boundary — the best rational the large suppressed term
# reveals — and stops, with error < 10^-N.  See the discussion in
# _metaCF_terms.  The threshold must exceed the largest partial quotient (in
# digits) of any value you consider legitimate rather than a coincidental
# near-rational.  Calibration points: Ramanujan's constant e^(pi*sqrt(163)) is
# an integer to ~12 digits; fib(360)/fib(216) is a near-integer (~Lucas(144))
# to ~30 digits, with a 31-digit partial quotient.  The default clears both.
# Set to None to disable gimme (raise on stall instead).  Override with
# CFRAC_GIMME_MIN_TERM_DIGITS.
_GIMME_MIN_TERM_DIGITS = int(os.environ.get("CFRAC_GIMME_MIN_TERM_DIGITS", "50"))

# Safety net: even with gimme on, cap total refinements so a pathological input
# (bracket failing to shrink) cannot loop forever.  Refinement reaches N digits
# in ~2.4*N terms in the worst (all-ones) case, so this leaves wide margin.
_GIMME_REFINE_CAP = 50 * _GIMME_MIN_TERM_DIGITS + 1000

# Consecutive non-emitting bihomographic terms before the gimme boundary check
# kicks in.  Normal arithmetic emits within a few terms (stall stays below this),
# so it never pays the exact-Fraction corner cost; only true stalls do.  Well
# below the term count any digit threshold needs, so it never delays a gimme.
_BI_GIMME_WARMUP = 16


# ---------------------------------------------------------------------------
# Helper: correct floor-based output check
# ---------------------------------------------------------------------------


def _homo_output(a: int, b: int, c: int, d: int) -> int | None:
    """Return the next output CF term for (ax'+b)/(cx'+d) with x' ∈ [1,∞), or None.

    Evaluates the formula at both endpoints of the tail range (x'=1 and x'→∞).
    If both denominators are non-zero, same-signed (no pole in range), and both
    values share the same integer floor, that floor is the next output term.
    """
    den_inf = c
    den_one = c + d
    if den_inf == 0 or den_one == 0:
        return None
    # Pole check: denominators must have the same sign
    if (den_inf > 0) != (den_one > 0):
        return None
    q_inf = a // c  # floor(a/c) — Python // is floor division
    q_one = (a + b) // (c + d)
    return q_inf if q_inf == q_one else None


def _homo_output_simple(a: int, b: int, c: int, d: int) -> int | None:
    """Return the next output CF term for (ax'+b)/(cx'+d) with x' ∈ [1,∞), or None.

    Evaluates the formula at both endpoints of the tail range (x'=1 and x'→∞).
    Assumes both denominators are non-zero, same-signed (no pole in range).
    If both values share the same integer floor, that floor is the next output term.
    """
    q_inf = a // c  # floor(a/c) — Python // is floor division
    q_one = (a + b) // (c + d)
    return q_inf if q_inf == q_one else None


def _bi_output(
    a: int,
    b: int,
    c: int,
    d: int,
    e: int,
    f: int,
    g: int,
    h: int,
) -> int | None:
    """Return the next output CF term for the bihomographic (two-input) formula, or None.

    Evaluates (axy+bx+cy+d)/(exy+fx+gy+h) at all four corners of the tail
    range (x',y' ∈ [1,∞)).  If all four denominators are non-zero, same-signed
    (no pole in range), and all four floors agree, that floor is the next term.

    Corners: (∞,∞) → a/e; (∞,1) → (a+b)/(e+f);
             (1,∞) → (a+c)/(e+g); (1,1) → (a+b+c+d)/(e+f+g+h).
    """
    corners = [
        (a, e),
        (a + b, e + f),
        (a + c, e + g),
        (a + b + c + d, e + f + g + h),
    ]
    signs = set()
    floors = []
    for num, den in corners:
        if den == 0:
            return None
        signs.add(den > 0)
        if len(signs) > 1:
            return None  # mixed signs → pole in range
        floors.append(num // den)
    if len(set(floors)) == 1:
        return floors[0]
    return None


# ---------------------------------------------------------------------------
# Unary: y = (ax + b) / (cx + d)
# ---------------------------------------------------------------------------


def _homographic_terms(
    x_iter: Iterator[int],
    a: int,
    b: int,
    c: int,
    d: int,
) -> Iterator[int]:
    """Yield CF terms of (ax+b)/(cx+d) given the CF terms of x, one at a time."""

    x_done = False
    stall = 0

    while True:
        n = _homo_output(a, b, c, d)
        if n is not None:
            yield n
            stall = 0
            # Subtract n, take reciprocal: (a,b,c,d) → (c, d, a-nc, b-nd)
            a, b, c, d = c, d, a - n * c, b - n * d
            continue

        if c == 0 and d == 0:
            # Denominator is identically zero: the value is ∞, i.e. the CF has
            # terminated (a finite/rational result fully emitted, e.g. a constant
            # map like (0x+5)/(0x+1) = 5).  Nothing remains.
            return

        if x_done:
            # x exhausted: tail x' → ∞, value is a/c
            if c == 0:
                return
            from .core import CF as _CF

            yield from _CF.from_fraction(a, c)
            return

        try:
            t = next(x_iter)
        except StopIteration:
            x_done = True
            stall = 0
            continue

        # Ingest: substitute x = t + 1/x'
        a, b, c, d = a * t + b, a, c * t + d, c
        stall += 1
        # A one-input transform of a canonical CF always emits or terminates:
        # an exact-rational output is the degenerate map handled above, and a
        # near-rational output needs a near-rational *input*, whose large partial
        # quotient pins the floor in a single read before the bracket can shrink
        # to a boundary gimme.  (The two-input path differs: correlated inputs,
        # e.g. x - x, narrow gradually with no large term, so it has a gimme.)
        # A long stall therefore means a malformed/non-canonical input; fail loud.
        if stall >= _GIMME_REFINE_CAP:
            raise ArithmeticError(
                f"homographic stalled: read {stall} terms without pinning an "
                f"output term (likely a malformed input)"
            )


def cf_homographic(x: CF, a: int, b: int, c: int, d: int) -> CF:
    """Return the CF for (ax+b)/(cx+d) — a homographic (Möbius) transformation of x.

    A Möbius transformation is any formula of the form (ax+b)/(cx+d).  This
    covers negation (a=-1,b=0,c=0,d=1), reciprocal (a=0,b=1,c=1,d=0), and
    any rational scaling, all computed exactly without converting to decimals.
    """
    from .core import CF as _CF

    return _annotate_cf(
        _CF([], _source=_homographic_terms(x._iter_from(0), a, b, c, d)),
        ("cf_homographic", x, a, b, c, d),
    )


# ---------------------------------------------------------------------------
# Binary bihomographic: z = (axy + bx + cy + d) / (exy + fx + gy + h)
# ---------------------------------------------------------------------------
#
# Ingest term t from x (substitute x = t + 1/x'):
#   new_a = a*t+c,  new_b = b*t+d,  new_c = a,  new_d = b
#   new_e = e*t+g,  new_f = f*t+h,  new_g = e,  new_h = f
#
# Ingest term t from y (substitute y = t + 1/y'):
#   new_a = a*t+b,  new_b = a,  new_c = c*t+d,  new_d = c
#   new_e = e*t+f,  new_f = e,  new_g = g*t+h,  new_h = g


def _bi_boundary(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int) -> int | None:
    """Gimme for the two-input transform, by the same rule as the metaCF gimme.

    The four corners (x', y' ∈ {1, ∞}) bound the output value.  If they are all
    finite and same-signed (no pole in the [1,∞)×[1,∞) rectangle) and span less
    than 10^-_GIMME_MIN_TERM_DIGITS while straddling an integer, the partial
    quotient we would otherwise emit has at least that many digits — strong
    evidence the value is that near-rational.  Return the straddled integer (the
    best rational the large suppressed term reveals); otherwise None.

    This replaces the old fixed "emit after N input terms" rule, so an
    exact-rational result like Ln(2)/Ln(2) = 1 resolves at the configured digit
    threshold rather than a hard-coded term count.  See _metaCF_terms.
    """
    dens = (e, e + f, e + g, e + f + g + h)
    if any(den == 0 for den in dens):
        return None  # a corner at infinity: bracket unbounded, cannot pin
    if len({den > 0 for den in dens}) != 1:
        return None  # mixed signs: a pole lies in the rectangle
    nums = (a, a + b, a + c, a + b + c + d)
    vals = [Fraction(num, den) for num, den in zip(nums, dens)]
    lo, hi = min(vals), max(vals)
    if hi - lo >= Fraction(1, 10**_GIMME_MIN_TERM_DIGITS):
        return None  # not pinned tightly enough yet
    k = hi.numerator // hi.denominator  # floor(hi): the straddled integer
    if lo >= k:
        return None  # all corners share floor k — a normal emit, not a straddle
    return k


def _bihomographic_terms(
    x_iter: Iterator[int],
    y_iter: Iterator[int],
    a: int,
    b: int,
    c: int,
    d: int,
    e: int,
    f: int,
    g: int,
    h: int,
) -> Iterator[int]:

    x_done = False
    y_done = False
    stall = 0
    # Track whether at least one term has been consumed from each input.
    # The corner check (x', y' ∈ [1, ∞)) is only valid once both inputs have
    # had their first term consumed; before that, the "variable" in the state
    # represents the full CF (which may be < 1), not a tail ≥ 1.
    x_started = False
    y_started = False

    while True:
        # Only evaluate output when both tails are in [1, ∞) domain.
        if x_started and y_started:
            n = _bi_output(a, b, c, d, e, f, g, h)
        else:
            n = None

        if n is not None:
            yield n
            stall = 0
            a, b, c, d, e, f, g, h = (
                e,
                f,
                g,
                h,
                a - n * e,
                b - n * f,
                c - n * g,
                d - n * h,
            )
            continue

        if x_done and y_done:
            # Both tails → ∞; value is a/e
            if e != 0:
                from .core import CF as _CF

                yield from _CF.from_fraction(a, e)
            return

        if x_done:
            # x' → ∞: reduce to homographic in y': (ay+b)/(ey+f)
            yield from _homographic_terms(y_iter, a, b, e, f)
            return

        if y_done:
            # y' → ∞: reduce to homographic in x': (ax+c)/(ex+g)
            yield from _homographic_terms(x_iter, a, c, e, g)
            return

        # Prefer consuming from whichever input hasn't been started yet,
        # to ensure both tails are in the [1, ∞) domain before emitting.
        #
        # The float ranking can't separate corners closer than ~1e-16, so once a
        # stall is underway (the value is near a boundary and we need to narrow
        # past float precision to reach the gimme threshold) switch to the exact
        # ranking.  Only stalling sequences pay it.
        decide = _should_ingest_x_exact if stall >= _BI_GIMME_WARMUP else _should_ingest_x
        if not x_started or (y_started and decide(a, b, c, d, e, f, g, h)):
            try:
                t = next(x_iter)
                x_started = True
                a, b, c, d, e, f, g, h = (
                    a * t + c,
                    b * t + d,
                    a,
                    b,
                    e * t + g,
                    f * t + h,
                    e,
                    f,
                )
            except StopIteration:
                x_done = True
                x_started = True
                stall = 0
                continue
        else:
            try:
                t = next(y_iter)
                y_started = True
                a, b, c, d, e, f, g, h = (
                    a * t + b,
                    a,
                    c * t + d,
                    c,
                    e * t + f,
                    e,
                    g * t + h,
                    g,
                )
            except StopIteration:
                y_done = True
                y_started = True
                stall = 0
                continue

        stall += 1
        # Integer-boundary stall: the corners straddle an integer because the
        # value is exactly (or extremely near) a rational, e.g. Ln(2)/Ln(2) = 1.
        # Accept it once the digit-based gimme is satisfied.  The warmup skips
        # the check for normal fast-emitting arithmetic (stall stays small), so
        # only genuinely stalling sequences pay the exact-Fraction corner cost.
        if stall >= _BI_GIMME_WARMUP:
            boundary = _bi_boundary(a, b, c, d, e, f, g, h)
            if boundary is not None:
                yield boundary
                return
        if stall >= _GIMME_REFINE_CAP:
            raise ArithmeticError(
                f"bihomographic stalled: read {stall} terms without pinning an "
                f"output term or a near-rational boundary within "
                f"10^-{_GIMME_MIN_TERM_DIGITS} (raise CFRAC_GIMME_MIN_TERM_DIGITS)"
            )


def _corner_val(num: int, den: int) -> float:
    """Float value of num/den for spread calculations.

    Returns `float("inf")` for all infinite results, never `float("-inf")`
    (there is only one infinity in projective space P^1).  Raises OverflowError
    when num/den exceeds the float range; callers fall back to exact arithmetic.
    """
    if den == 0:
        return float("inf")
    return num / den


def _should_ingest_x(
    a: int,
    b: int,
    c: int,
    d: int,
    e: int,
    f: int,
    g: int,
    h: int,
) -> bool:
    """Return True if reading the next term from x narrows the output range more than from y.

    Compares how much the four corner values spread apart when x varies (1→∞)
    versus when y varies (1→∞).  Whichever input causes more spread in the output
    is the one that most needs to be pinned down by reading its next term.

    The decision runs on floats (this is the hot per-iteration path).  A corner
    past the float range (~1e308, only with hundreds of digits in a coefficient)
    overflows; that call falls back to exact rational arithmetic, which is slow
    but reached only by pathological inputs.
    """
    # Corners: (∞,∞), (∞,1), (1,∞), (1,1)
    try:
        c00 = _corner_val(a, e)  # (∞, ∞)
        c01 = _corner_val(a + b, e + f)  # (∞, 1)
        c10 = _corner_val(a + c, e + g)  # (1, ∞)
        c11 = _corner_val(a + b + c + d, e + f + g + h)  # (1, 1)
    except OverflowError:
        return _should_ingest_x_exact(a, b, c, d, e, f, g, h)

    # Compare the x-direction range vs the y-direction range.
    sx_inf = abs(c00 - c10)  # x: ∞ vs 1, with y=∞
    sx_one = abs(c01 - c11)  # x: ∞ vs 1, with y=1
    sy_inf = abs(c00 - c01)  # y: ∞ vs 1, with x=∞
    sy_one = abs(c10 - c11)  # y: ∞ vs 1, with x=1

    spread_x = max(sx_inf, sx_one)
    spread_y = max(sy_inf, sy_one)

    if spread_x == float("inf") and spread_y == float("inf"):
        return True  # default: x first
    if spread_x == float("inf"):
        return True
    if spread_y == float("inf"):
        return False
    return spread_x >= spread_y


def _should_ingest_x_exact(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int) -> bool:
    """Exact-rational fallback for _should_ingest_x when a corner overflows float.

    ``None`` is projective infinity (den == 0).  Two infinities are the *same*
    point, so a direction with both endpoints at infinity has zero spread; a
    direction mixing finite and infinite has infinite spread.
    """

    def corner(num: int, den: int) -> Fraction | None:
        return None if den == 0 else Fraction(num, den)

    c00, c01 = corner(a, e), corner(a + b, e + f)
    c10, c11 = corner(a + c, e + g), corner(a + b + c + d, e + f + g + h)

    def dist(p: Fraction | None, q: Fraction | None) -> Fraction | None:
        if p is None and q is None:
            return Fraction(0)
        if p is None or q is None:
            return None  # infinite spread
        return abs(p - q)

    def spread(d1: Fraction | None, d2: Fraction | None) -> Fraction | None:
        return None if (d1 is None or d2 is None) else max(d1, d2)

    spread_x = spread(dist(c00, c10), dist(c01, c11))
    spread_y = spread(dist(c00, c01), dist(c10, c11))

    if spread_x is None:
        return True
    if spread_y is None:
        return False
    return spread_x >= spread_y


def _bihomographic(x: CF, y: CF, a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int) -> CF:
    from .core import CF as _CF

    xi = x._iter_from(0)
    yi = y._iter_from(0)
    return _CF([], _source=_bihomographic_terms(xi, yi, a, b, c, d, e, f, g, h))


# ---------------------------------------------------------------------------
# META-GOSPER
# "Unary": y = (a(x)·F + b(x)) / (c(x)·F + d(x))
# a,b,c,d are monotonic polynomials, stored as lists of coefficients
# F has a continued fraction form s.t. F = t(x) + 1/F' at each step
#   with t(x) >= 1 when x >= 1
# x is a CF object with x >= 1
# ---------------------------------------------------------------------------
#
# Ingest term t(x) from F (substitute F = t(x) + 1/F'):
#   new_a = a*t+c,  new_b = a,
#   new_c = b*t+d,  new_d = b.
#
# Terms are never "ingested" from x in this function.
# That is handled internally within the CF class.
# Instead, CF.interval() is used to make the necessary estimations.


# This first implementation runs very slowly,
# but is a good sketch of the control flow or intention of metaCF.


def _metaCF_simple_terms(x: CF, F_iter: Iterator[Callable[[CF], CF]]) -> Iterator[int]:
    from .core import CF as _CF

    one = _CF.from_int(1)
    zero = _CF.from_int(0)
    stall = 0
    F_done = False
    t = next(F_iter)(x)
    a, b, c, d = t, one, one, zero
    while True:
        n0 = (a / c).take(1).terms[0]
        n1 = ((a + b) / (c + d)).take(1).terms[0]
        if n0 == n1:
            yield n0
            # Subtract n0, take reciprocal:
            a, b, c, d = c, d, a - n0 * c, b - n0 * d
            continue

        if F_done:
            # F exhausted: tail F' → ∞, value is a/c
            yield from a / c
            return

        try:
            t = next(F_iter)(x)
        except StopIteration:
            F_done = True
            stall = 0
            continue

        # Ingest: substitute F = t + 1/F'
        a, b, c, d = b + a * t, a, d + c * t, c

        # As in _metaCF_terms, an exact-boundary value (e.g. Exp(Ln(2)) = 2)
        # leaves the two corners floored one apart no matter how many F terms
        # we ingest — no interval check can confirm it.  Fail loudly instead of
        # silently truncating (which would return a wrong CF for values like
        # Exp(Ln(7/4)) = 8/3 = [2; 1, 2]).
        stall += 1
        if stall >= _METACF_STALL_LIMIT:
            raise ArithmeticError(
                f"metaCF stalled: ingested {stall} terms of F without pinning an "
                f"output term (corners floor to {n0} and {n1}).  The value is "
                f"likely exactly the integer boundary {max(n0, n1)}, which an "
                f"interval corner check cannot confirm.  Raise "
                f"CFRAC_METACF_STALL_LIMIT to allow more iterations."
            )


def cf_metaCF_simple(x: CF, F_iter: Iterator[Callable[[CF], CF]]) -> CF:
    """Return the CF for F(x), where F is a CF of functions of x.

    This allows for generation of CF for Tanh, Exp, and others.

    In order to ensure efficient convergence, F should have
    (all but finitely many) terms guaranteed to be in [1, ∞]
    for the domain of x-values F operates on.
    """
    from .core import CF as _CF

    return _CF([], _source=_metaCF_simple_terms(x, F_iter))


def _poly_eval(coeffs: list[int], inp):
    """Evaluate a low-to-high polynomial at inp."""
    out = 0
    for i, coeff in enumerate(coeffs):
        if coeff:
            out += coeff * inp**i
    return out


def _poly_eval_scaled(coeffs: list[int], p: int, q: int, degree: int) -> int:
    """Evaluate poly(p/q) * q**degree without constructing a Fraction."""
    if len(coeffs) == 1:
        return coeffs[0] * q**degree
    if degree == 1:
        return coeffs[0] * q + coeffs[1] * p

    out = 0
    for i, coeff in enumerate(coeffs):
        if coeff:
            out += coeff * p**i * q**(degree - i)
    return out


def _poly_add(coeffs0: list[int], coeffs1: list[int]) -> list[int]:
    """Add low-to-high polynomials."""
    coeffs0, coeffs1 = sorted((coeffs0, coeffs1), key=len)
    out = list(coeffs1)
    for i, coeff in enumerate(coeffs0):
        out[i] += coeff
    return out


def _poly_mul(coeffs0: list[int], coeffs1: list[int]) -> list[int]:
    """Multiply low-to-high polynomials."""
    if len(coeffs0) == 1:
        k = coeffs0[0]
        if k == 0:
            return [0]
        if k == 1:
            return list(coeffs1)
        return [k * c for c in coeffs1]
    if len(coeffs1) == 1:
        k = coeffs1[0]
        if k == 0:
            return [0]
        if k == 1:
            return list(coeffs0)
        return [k * c for c in coeffs0]

    nonzero0 = [i for i, c in enumerate(coeffs0) if c != 0]
    if len(nonzero0) == 1:
        shift = nonzero0[0]
        k = coeffs0[shift]
        return [0] * shift + [k * c for c in coeffs1]

    nonzero1 = [i for i, c in enumerate(coeffs1) if c != 0]
    if len(nonzero1) == 1:
        shift = nonzero1[0]
        k = coeffs1[shift]
        return [0] * shift + [k * c for c in coeffs0]

    out = [0] * (len(coeffs0) + len(coeffs1) - 1)
    for i in nonzero0:
        c0 = coeffs0[i]
        for j in nonzero1:
            out[i + j] += c0 * coeffs1[j]
    return out


def _meta_emit_value(
    a: int, b: int, c: int, d: int, term: int
) -> tuple[int, int, int, int]:
    """Subtract term from a homographic state and invert the tail."""
    return c, d, a - term * c, b - term * d


def _meta_ingest_values(
    a: int,
    b: int,
    c: int,
    d: int,
    b_term: int,
    a_term: int,
    scale: int,
) -> tuple[int, int, int, int]:
    """Ingest F = b(x) + a(x)/F' into scalar endpoint state."""
    return (
        b_term * a + scale * b,
        a_term * a,
        b_term * c + scale * d,
        a_term * c,
    )


def _metaCF_replay_state(
    terms: list[tuple[list[int], int]],
    emits: list[tuple[int, int]],
    p: int,
    q: int,
) -> tuple[int, int, int, int]:
    """Replay simple meta-CF terms at p/q using only integer arithmetic."""
    if not terms:
        return 0, 0, 1, 0

    term, degree = terms[0]
    scale = q**degree
    a = _poly_eval_scaled(term, p, q, degree)
    b = scale
    c = scale
    d = 0

    emit_idx = 0
    while emit_idx < len(emits) and emits[emit_idx][0] == 1:
        a, b, c, d = _meta_emit_value(a, b, c, d, emits[emit_idx][1])
        emit_idx += 1

    for term_idx in range(1, len(terms)):
        term, degree = terms[term_idx]
        scale = q**degree
        term_value = _poly_eval_scaled(term, p, q, degree)
        a, b, c, d = _meta_ingest_values(a, b, c, d, term_value, scale, scale)

        while emit_idx < len(emits) and emits[emit_idx][0] == term_idx + 1:
            a, b, c, d = _meta_emit_value(a, b, c, d, emits[emit_idx][1])
            emit_idx += 1

    return a, b, c, d


def _meta_apply_poly_emits(
    emits: list[tuple[int, int]],
    emit_idx: int,
    level_count: int,
    pa: list[int],
    pb: list[int],
    pc: list[int],
    pd: list[int],
) -> tuple[int, list[int], list[int], list[int], list[int]]:
    """Apply output terms emitted after level_count symbolic ingests."""
    while emit_idx < len(emits) and emits[emit_idx][0] == level_count:
        term = emits[emit_idx][1]
        pa, pb, pc, pd = (
            pc,
            pd,
            _poly_add(pa, _poly_mul([-term], pc)),
            _poly_add(pb, _poly_mul([-term], pd)),
        )
        emit_idx += 1
    return emit_idx, pa, pb, pc, pd


def _metaCF_rebuild_polys(
    terms: list[tuple[list[int], int]], emits: list[tuple[int, int]]
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Rebuild symbolic polynomials for a terminating simple meta-CF."""
    if not terms:
        return [1], [0], [1], [0]

    term, _ = terms[0]
    pa, pb, pc, pd = list(term), [1], [1], [0]
    emit_idx = 0

    emit_idx, pa, pb, pc, pd = _meta_apply_poly_emits(
        emits, emit_idx, 1, pa, pb, pc, pd
    )
    for term_idx in range(1, len(terms)):
        term, _ = terms[term_idx]
        old_pa, old_pb, old_pc, old_pd = pa, pb, pc, pd
        pa = _poly_add(old_pb, _poly_mul(term, old_pa))
        pb = old_pa
        pc = _poly_add(old_pd, _poly_mul(term, old_pc))
        pd = old_pc
        emit_idx, pa, pb, pc, pd = _meta_apply_poly_emits(
            emits, emit_idx, term_idx + 1, pa, pb, pc, pd
        )

    return pa, pb, pc, pd


def _metaCF_terms(
    x: CF,
    F_iter: Iterator[list[int]],
    gimme_min_term_digits: int | None = _GIMME_MIN_TERM_DIGITS,
) -> Iterator[int]:
    """Yield terms for a simple meta-CF evaluated at x.

    The hot path stores consumed polynomial terms and emitted output terms,
    then replays scalar integer state at each rational interval endpoint.  This
    avoids growing the symbolic homographic polynomials during ordinary input
    ingestion.  If the input stream ends before the interval collapses, the
    symbolic state is rebuilt once for the fallback CF evaluation.

    ``gimme_min_term_digits`` controls gimme mode: when the value hugs an
    integer boundary so closely that the next partial quotient would have at
    least this many digits, accept it as that exact rational rather than
    refining the x-bracket forever.  Set to None to raise on stall instead.
    """
    from .core import CF as _CF

    terms: list[tuple[list[int], int]] = []
    emits: list[tuple[int, int]] = []

    try:
        term = next(F_iter)
    except StopIteration:
        return
    degree = max(len(term), 1) - 1
    terms.append((term, degree))

    x_i = 1
    stall = 0
    refine_stall = 0
    F_done = False
    x_CHANGED = True

    p0 = q0 = p1 = q1 = 0
    same = True
    have_interval = False

    a0 = b0 = c0 = d0 = 0
    a1 = b1 = c1 = d1 = 0
    n0 = n1 = None

    while True:
        if x_CHANGED:
            p0, q0, p1, q1, same = x.interval_ints(x_i)
            have_interval = True

            a0, b0, c0, d0 = _metaCF_replay_state(terms, emits, p0, q0)
            n0 = _homo_output_simple(a0, b0, c0, d0)

            if same:
                a1, b1, c1, d1, n1 = a0, b0, c0, d0, n0
            else:
                a1, b1, c1, d1 = _metaCF_replay_state(terms, emits, p1, q1)
                n1 = _homo_output_simple(a1, b1, c1, d1)

        x_CHANGED = False

        if n0 is not None and n0 == n1:
            term_out = n0
            yield term_out
            stall = 0
            refine_stall = 0
            emits.append((len(terms), term_out))

            a0, b0, c0, d0 = _meta_emit_value(a0, b0, c0, d0, term_out)
            n0 = _homo_output_simple(a0, b0, c0, d0)
            if same:
                a1, b1, c1, d1, n1 = a0, b0, c0, d0, n0
            else:
                a1, b1, c1, d1 = _meta_emit_value(a1, b1, c1, d1, term_out)
                n1 = _homo_output_simple(a1, b1, c1, d1)
            continue

        if n0 is not None and n1 is not None:
            # coefficient polynomials were precise enough to determine n0, n1
            # at each q0, q1 with q0 <= x <= q1, but x was too vague for n0 to
            # equal n1; read another term of x to tighten the [q0, q1] bracket.
            #
            # This makes progress only if the true value lies strictly inside
            # the bracket.  When the value sits *exactly* on the integer
            # boundary between n0 and n1 (e.g. Exp(Ln(2)) = 2, an exact integer
            # produced from an infinite input), every bracket around x straddles
            # it, so n0 and n1 never agree and this branch refines x forever.
            # No interval-based corner check can resolve an exact-boundary value.
            #
            # The width of the value bracket bounds how close the true value is
            # to the straddled integer K = max(n0, n1): the four corners are the
            # homographic evaluated at each x-endpoint (q0, q1) and each F'-tail
            # endpoint (1, inf).  If that width is below 10^-gimme_min_term_digits
            # the partial quotient we would otherwise emit has at least that many
            # digits — strong evidence the value is the near-rational K.  In
            # gimme mode, declare it exactly K and stop (error < 10^-digits);
            # otherwise refine until the stall cap, then raise.
            if gimme_min_term_digits is not None:
                corners = [
                    Fraction(a0, c0),
                    Fraction(a0 + b0, c0 + d0),
                    Fraction(a1, c1),
                    Fraction(a1 + b1, c1 + d1),
                ]
                width = max(corners) - min(corners)
                if width != 0 and width < Fraction(1, 10**gimme_min_term_digits):
                    yield max(n0, n1)
                    return

            refine_stall += 1
            cap = _METACF_NONE_STALL_LIMIT if gimme_min_term_digits is None else _GIMME_REFINE_CAP
            if refine_stall >= cap:
                raise ArithmeticError(
                    f"metaCF stalled: read {refine_stall} extra terms of x without "
                    f"pinning an output term (corners floor to {n0} and {n1}). "
                    f"The value is likely exactly the integer boundary {max(n0, n1)}, "
                    f"which an interval corner check cannot confirm.  Enable gimme "
                    f"mode (gimme_min_term_digits=) to accept the near-rational, or "
                    f"raise CFRAC_METACF_STALL_LIMIT."
                )
            x_i += 1
            x_CHANGED = True
            continue

        if F_done and same:
            if c0 == 0:
                return
            yield from _CF.from_fraction(a0, c0)
            return

        if F_done:
            pa, pb, pc, pd = _metaCF_rebuild_polys(terms, emits)
            yield from _poly_eval(pa, x) / _poly_eval(pc, x)
            return

        try:
            term = next(F_iter)
        except StopIteration:
            F_done = True
            stall = 0
            continue

        degree = len(term) - 1
        terms.append((term, degree))

        if have_interval:
            scale0 = q0**degree
            term0 = _poly_eval_scaled(term, p0, q0, degree)
            a0, b0, c0, d0 = _meta_ingest_values(
                a0, b0, c0, d0, term0, scale0, scale0
            )
            n0 = _homo_output_simple(a0, b0, c0, d0)
            if same:
                a1, b1, c1, d1, n1 = a0, b0, c0, d0, n0
            else:
                scale1 = q1**degree
                term1 = _poly_eval_scaled(term, p1, q1, degree)
                a1, b1, c1, d1 = _meta_ingest_values(
                    a1, b1, c1, d1, term1, scale1, scale1
                )
                n1 = _homo_output_simple(a1, b1, c1, d1)
        else:
            n0 = n1 = None

        stall += 1
        if stall >= _METACF_STALL_LIMIT:
            # F failed to determine an output term after this many ingestions
            # (F not converging at the current x-bracket).  A near-rational
            # *result* is caught earlier by the x-refinement gimme; reaching here
            # is a genuine failure, so raise rather than silently truncate.
            raise ArithmeticError(
                f"metaCF stalled: ingested {stall} terms of F without determining "
                f"an output term (F not converging at the current x-bracket)"
            )


def cf_metaCF(
    x: CF,
    F: Iterator[list[int]],
    gimme_min_term_digits: int | None = _GIMME_MIN_TERM_DIGITS,
) -> CF:
    """Return the CF for F(x), where F is a CF of polynomials of x.

    Each polynomial is represented as a list of int coefficients, where
      [a0, a1, a2, ..., an]
    represents
      a0 + a1*x + a2*x**2, ..., an*x**n

    This allows for generation of CF for Tanh, Exp, and others.

    In order to ensure efficient convergence, F should have
    (all but finitely many) terms guaranteed to be in [1, ∞]
    for the domain of x-values F operates on.

    ``gimme_min_term_digits`` controls gimme mode: when the result hugs an
    integer boundary so closely that the next partial quotient would have at
    least this many digits, accept it as that exact rational rather than
    refining forever (see ``_metaCF_terms``).  Set to None to raise on stall
    instead.
    """
    from .core import CF as _CF

    return _CF([], _source=_metaCF_terms(x, F, gimme_min_term_digits))


def _metaGCF_replay_state(
    levels: list[tuple[list[int], list[int], int]],
    emits: list[tuple[int, int]],
    p: int,
    q: int,
) -> tuple[int, int, int, int]:
    """Replay generalized meta-CF levels at p/q using integer arithmetic."""
    if not levels:
        return 0, 0, 1, 0

    b_poly, a_poly, degree = levels[0]
    a = _poly_eval_scaled(b_poly, p, q, degree)
    b = _poly_eval_scaled(a_poly, p, q, degree)
    c = q**degree
    d = 0

    emit_idx = 0
    while emit_idx < len(emits) and emits[emit_idx][0] == 1:
        a, b, c, d = _meta_emit_value(a, b, c, d, emits[emit_idx][1])
        emit_idx += 1

    for level_idx in range(1, len(levels)):
        b_poly, a_poly, degree = levels[level_idx]
        scale = q**degree
        b_term = _poly_eval_scaled(b_poly, p, q, degree)
        a_term = _poly_eval_scaled(a_poly, p, q, degree)
        a, b, c, d = _meta_ingest_values(a, b, c, d, b_term, a_term, scale)

        while emit_idx < len(emits) and emits[emit_idx][0] == level_idx + 1:
            a, b, c, d = _meta_emit_value(a, b, c, d, emits[emit_idx][1])
            emit_idx += 1

    return a, b, c, d


def _metaGCF_rebuild_polys(
    levels: list[tuple[list[int], list[int], int]], emits: list[tuple[int, int]]
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Rebuild symbolic polynomials for a terminating generalized meta-CF."""
    if not levels:
        return [1], [0], [1], [0]

    b_poly, a_poly, _ = levels[0]
    pa, pb, pc, pd = list(b_poly), list(a_poly), [1], [0]
    emit_idx = 0
    emit_idx, pa, pb, pc, pd = _meta_apply_poly_emits(
        emits, emit_idx, 1, pa, pb, pc, pd
    )

    for level_idx in range(1, len(levels)):
        b_poly, a_poly, _ = levels[level_idx]
        old_pa, old_pb, old_pc, old_pd = pa, pb, pc, pd
        pa = _poly_add(_poly_mul(old_pa, b_poly), old_pb)
        pb = _poly_mul(old_pa, a_poly)
        pc = _poly_add(_poly_mul(old_pc, b_poly), old_pd)
        pd = _poly_mul(old_pc, a_poly)
        emit_idx, pa, pb, pc, pd = _meta_apply_poly_emits(
            emits, emit_idx, level_idx + 1, pa, pb, pc, pd
        )

    return pa, pb, pc, pd


def _metaGCF_terms(x: CF, F_iter: Iterator[tuple[list[int], list[int]]]) -> Iterator[int]:
    """Yield terms for a generalized meta-CF evaluated at x.

    The input stream gives levels of ``F = b(x) + a(x)/F'``.  The driver keeps
    the scalar endpoint state hot and records consumed levels plus emitted
    output terms.  When the x interval changes, it replays that history at the
    new integer endpoint pair instead of maintaining growing symbolic
    polynomials on every ingest.
    """
    from .core import CF as _CF

    levels: list[tuple[list[int], list[int], int]] = []
    emits: list[tuple[int, int]] = []

    try:
        b_poly, a_poly = next(F_iter)
    except StopIteration:
        return
    degree = max(len(b_poly), len(a_poly), 1) - 1
    levels.append((b_poly, a_poly, degree))

    x_i = 1
    stall = 0
    F_done = False
    x_CHANGED = True

    p0 = q0 = p1 = q1 = 0
    same = True
    have_interval = False

    a0 = b0 = c0 = d0 = 0
    a1 = b1 = c1 = d1 = 0
    n0 = n1 = None

    while True:
        if x_CHANGED:
            p0, q0, p1, q1, same = x.interval_ints(x_i)
            have_interval = True

            a0, b0, c0, d0 = _metaGCF_replay_state(levels, emits, p0, q0)
            n0 = _homo_output(a0, b0, c0, d0)

            if same:
                a1, b1, c1, d1, n1 = a0, b0, c0, d0, n0
            else:
                a1, b1, c1, d1 = _metaGCF_replay_state(levels, emits, p1, q1)
                n1 = _homo_output(a1, b1, c1, d1)

        x_CHANGED = False

        if n0 is not None and n0 == n1:
            term_out = n0
            yield term_out
            stall = 0
            emits.append((len(levels), term_out))

            a0, b0, c0, d0 = _meta_emit_value(a0, b0, c0, d0, term_out)
            n0 = _homo_output(a0, b0, c0, d0)
            if same:
                a1, b1, c1, d1, n1 = a0, b0, c0, d0, n0
            else:
                a1, b1, c1, d1 = _meta_emit_value(a1, b1, c1, d1, term_out)
                n1 = _homo_output(a1, b1, c1, d1)
            continue

        if n0 is not None and n1 is not None:
            x_i += 1
            x_CHANGED = True
            continue

        # If the lower endpoint is determined but the upper is not (e.g. a pole
        # produced by negative GCF numerators), narrow the z interval rather
        # than consuming more GCF levels.  The pole is an artifact of using the
        # wrong (too large) upper bound; tightening x_i removes it.
        if not same and n0 is not None and n1 is None:
            x_i += 1
            x_CHANGED = True
            continue

        if F_done and same:
            if c0 == 0:
                return
            yield from _CF.from_fraction(a0, c0)
            return

        if F_done:
            pa, pb, pc, pd = _metaGCF_rebuild_polys(levels, emits)
            yield from _poly_eval(pa, x) / _poly_eval(pc, x)
            return

        try:
            b_poly, a_poly = next(F_iter)
        except StopIteration:
            F_done = True
            stall = 0
            continue

        degree = max(len(b_poly), len(a_poly), 1) - 1
        levels.append((b_poly, a_poly, degree))

        if have_interval:
            scale0 = q0**degree
            b_term0 = _poly_eval_scaled(b_poly, p0, q0, degree)
            a_term0 = _poly_eval_scaled(a_poly, p0, q0, degree)
            a0, b0, c0, d0 = _meta_ingest_values(
                a0, b0, c0, d0, b_term0, a_term0, scale0
            )
            n0 = _homo_output(a0, b0, c0, d0)
            if same:
                a1, b1, c1, d1, n1 = a0, b0, c0, d0, n0
            else:
                scale1 = q1**degree
                b_term1 = _poly_eval_scaled(b_poly, p1, q1, degree)
                a_term1 = _poly_eval_scaled(a_poly, p1, q1, degree)
                a1, b1, c1, d1 = _meta_ingest_values(
                    a1, b1, c1, d1, b_term1, a_term1, scale1
                )
                n1 = _homo_output(a1, b1, c1, d1)
        else:
            n0 = n1 = None

        stall += 1
        if stall >= _METACF_STALL_LIMIT:
            raise ValueError(
                f"cf_metaGCF stalled: no term emitted after consuming {_METACF_STALL_LIMIT} consecutive pairs"
            )


def cf_metaGCF(x: CF, F: Iterator[tuple[list[int], list[int]]]) -> CF:
    """Return the CF for a generalized meta-CF evaluated at x.

    The input stream gives ``(b(x), a(x))`` polynomial pairs for:

        b0(x) + a1(x)/(b1(x) + a2(x)/(b2(x) + ...))

    Coefficient lists are low-to-high order.  ``cf_metaCF`` is the special case
    where every numerator polynomial is ``[1]``.
    """
    from .core import CF as _CF

    return _CF([], _source=_metaGCF_terms(x, F))


# ---------------------------------------------------------------------------
# Public arithmetic
# ---------------------------------------------------------------------------


def cf_add(x: CF, y: CF) -> CF:
    """Return x + y as a continued fraction, computed exactly using Gosper's algorithm."""
    return _annotate_cf(_bihomographic(x, y, 0, 1, 1, 0, 0, 0, 0, 1), ("cf_add", x, y))


def cf_sub(x: CF, y: CF) -> CF:
    """Return x - y as a continued fraction, computed exactly using Gosper's algorithm.

    The ``x is y`` check is a fast path: the bihomographic would reach the
    same answer via the integer-boundary stall fix, but this avoids 1000 steps.
    """
    if x is y:
        from .core import CF as _CF

        return _annotate_cf(_CF.from_int(0), ("cf_sub", x, y))
    return _annotate_cf(_bihomographic(x, y, 0, 1, -1, 0, 0, 0, 0, 1), ("cf_sub", x, y))


def cf_mul(x: CF, y: CF) -> CF:
    """Return x * y as a continued fraction, computed exactly using Gosper's algorithm.

    For periodic CFs (square roots, golden ratio, etc.) we use the minimal
    polynomial path, which gives an exact periodic result rather than a finite
    approximation from the bihomographic algorithm.

    The ``x is y`` check routes squaring to ``_periodic_square`` as a fast
    path; if skipped, the ``elif`` branch below calls ``_periodic_mul(x, y)``
    which gives the same result.
    """
    if x is y:
        result = _periodic_square(x)
        if result is not None:
            return _annotate_cf(result, ("cf_mul", x, y))
    elif x.is_periodic() and y.is_periodic():
        result = _periodic_mul(x, y)
        if result is not None:
            return _annotate_cf(result, ("cf_mul", x, y))
    return _annotate_cf(_bihomographic(x, y, 1, 0, 0, 0, 0, 0, 0, 1), ("cf_mul", x, y))


def cf_div(x: CF, y: CF) -> CF:
    """Return x / y as a continued fraction, computed exactly using Gosper's algorithm.

    The ``x is y`` check is a fast path: the integer-boundary stall fix in
    the bihomographic would reach the same answer, but this avoids 1000 steps.
    """
    if x is y:
        from .core import CF as _CF

        return _annotate_cf(_CF.from_int(1), ("cf_div", x, y))
    return _annotate_cf(_bihomographic(x, y, 0, 1, 0, 0, 0, 0, 1, 0), ("cf_div", x, y))


def cf_min(x: CF, y: CF) -> CF:
    """Return the smaller of x and y."""
    return x if x < y else y


def cf_max(x: CF, y: CF) -> CF:
    """Return the larger of x and y."""
    return x if x > y else y
