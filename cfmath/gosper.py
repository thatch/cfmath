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

from typing import TYPE_CHECKING, Iterator

from .quadratic import _periodic_mul, _periodic_square

if TYPE_CHECKING:
    from .core import CF


_MAX_STALL = 1000  # max input terms consumed per output term before giving up


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
        if stall >= _MAX_STALL:
            return


def cf_homographic(x: CF, a: int, b: int, c: int, d: int) -> CF:
    """Return the CF for (ax+b)/(cx+d) — a homographic (Möbius) transformation of x.

    A Möbius transformation is any formula of the form (ax+b)/(cx+d).  This
    covers negation (a=-1,b=0,c=0,d=1), reciprocal (a=0,b=1,c=1,d=0), and
    any rational scaling, all computed exactly without converting to decimals.
    """
    from .core import CF as _CF

    return _CF([], _source=_homographic_terms(x._iter_from(0), a, b, c, d))


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
        if not x_started or (y_started and _should_ingest_x(a, b, c, d, e, f, g, h)):
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
        if stall >= _MAX_STALL:
            # Integer-boundary stall: the corners permanently straddle an
            # integer because both inputs represent the same value
            # (e.g. Ln(2)/Ln(2) = 1).  The max corner floor is the correct
            # answer — a wrong answer would require the true value to be
            # within ~10^-209 of the boundary (sound for all practical use).
            # Emit it and terminate; there is nothing left after an exact integer.
            corners_nd = [
                (a, e),
                (a + b, e + f),
                (a + c, e + g),
                (a + b + c + d, e + f + g + h),
            ]
            valid = [(num, den) for num, den in corners_nd if den != 0]
            if valid and len({den > 0 for _, den in valid}) == 1:
                yield max(num // den for num, den in valid)
            return


def _corner_val(num: int, den: int) -> float:
    """Safe float value of num/den for spread calculations."""
    if den == 0:
        return float("inf") if num >= 0 else float("-inf")
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
    """
    # Corners: (∞,∞), (∞,1), (1,∞), (1,1)
    c00 = _corner_val(a, e)  # (∞, ∞)
    c01 = _corner_val(a + b, e + f)  # (∞, 1)
    c10 = _corner_val(a + c, e + g)  # (1, ∞)
    c11 = _corner_val(a + b + c + d, e + f + g + h)  # (1, 1)

    def _spread(v1: float, v2: float, v3: float, v4: float) -> float:
        finite = [v for v in (v1, v2, v3, v4) if abs(v) != float("inf")]
        if len(finite) < 2:
            return float("inf")
        return max(finite) - min(finite)

    # x-spread: corners varying x' (1→∞) with y' fixed
    _spread(c00, c10, c01, c11)  # all four corners
    # y-spread: corners varying y' (1→∞) with x' fixed
    # As a proxy, compare the x-direction range vs y-direction range:
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


def _bihomographic(x: CF, y: CF, a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int) -> CF:
    from .core import CF as _CF

    xi = x._iter_from(0)
    yi = y._iter_from(0)
    return _CF([], _source=_bihomographic_terms(xi, yi, a, b, c, d, e, f, g, h))


# ---------------------------------------------------------------------------
# Public arithmetic
# ---------------------------------------------------------------------------


def cf_add(x: CF, y: CF) -> CF:
    """Return x + y as a continued fraction, computed exactly using Gosper's algorithm."""
    return _bihomographic(x, y, 0, 1, 1, 0, 0, 0, 0, 1)


def cf_sub(x: CF, y: CF) -> CF:
    """Return x - y as a continued fraction, computed exactly using Gosper's algorithm.

    The ``x is y`` check is a fast path: the bihomographic would reach the
    same answer via the integer-boundary stall fix, but this avoids 1000 steps.
    """
    if x is y:
        from .core import CF as _CF

        return _CF.from_int(0)
    return _bihomographic(x, y, 0, 1, -1, 0, 0, 0, 0, 1)


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
            return result
    elif x.is_periodic() and y.is_periodic():
        result = _periodic_mul(x, y)
        if result is not None:
            return result
    return _bihomographic(x, y, 1, 0, 0, 0, 0, 0, 0, 1)


def cf_div(x: CF, y: CF) -> CF:
    """Return x / y as a continued fraction, computed exactly using Gosper's algorithm.

    The ``x is y`` check is a fast path: the integer-boundary stall fix in
    the bihomographic would reach the same answer, but this avoids 1000 steps.
    """
    if x is y:
        from .core import CF as _CF

        return _CF.from_int(1)
    return _bihomographic(x, y, 0, 1, 0, 0, 0, 0, 1, 0)


def cf_min(x: CF, y: CF) -> CF:
    """Return the smaller of x and y."""
    return x if x < y else y


def cf_max(x: CF, y: CF) -> CF:
    """Return the larger of x and y."""
    return x if x > y else y
