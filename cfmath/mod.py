"""Integer rounding and modular arithmetic for continued fractions."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .core import CF

_MAX_FLOOR_ITERS = 10_000


def _bounds_iter(cf: CF) -> Iterator[tuple[Fraction, Fraction]]:
    """Yield tightening (lo, hi) rational bounds on *cf*, forever.

    Even-indexed convergents lie below the true value; odd-indexed lie above.
    Waits for the second convergent before yielding the first pair so the
    upper bound is a real convergent, not a guess.  For a one-term (exact
    rational) CF the exact value is yielded immediately without guessing.
    For a finite CF the last pair is (exact, exact) and repeats indefinitely.
    """
    from .convergents import convergent_pairs

    lo: Fraction | None = None
    hi: Fraction | None = None
    last: Fraction | None = None

    for k, (p, q) in enumerate(convergent_pairs(cf)):
        c = Fraction(p, q)
        last = c
        if k % 2 == 0:
            lo = c
        else:
            hi = c
        if lo is not None and hi is not None:
            yield lo, hi

    # Finite CF: the final convergent is the exact value; repeat forever.
    assert last is not None
    while True:
        yield last, last


def _floor_quotient(x: CF, y: CF) -> int:
    """Return floor(x/y) as a plain int using rational convergent bounds.

    Maintains rational intervals [lo_x, hi_x] and [lo_y, hi_y] derived from
    the convergents of *x* and *y*, and resolves floor(x/y) as soon as the
    interval for x/y stops straddling an integer.

    This avoids building a lazy CF for x/y and bypasses the _MAX_STALL limit
    in the bihomographic algorithm.  The fundamental hard case — x/y exactly
    equal to an integer, with both inputs independently irrational — cannot
    terminate with any rational-approximation approach.
    """
    x_bounds = _bounds_iter(x)
    y_bounds = _bounds_iter(y)

    for _ in range(_MAX_FLOOR_ITERS):
        lo_x, hi_x = next(x_bounds)
        lo_y, hi_y = next(y_bounds)

        if lo_y > 0:
            # x/y is maximised at hi_x/lo_y and minimised at lo_x/hi_y
            f_lo = math.floor(lo_x / hi_y)
            f_hi = math.floor(hi_x / lo_y)
            if f_lo == f_hi:
                return f_lo
        elif hi_y < 0:
            # Negative y flips the monotonicity
            f_lo = math.floor(hi_x / hi_y)
            f_hi = math.floor(lo_x / lo_y)
            if f_lo == f_hi:
                return f_lo
        # else: y bounds straddle zero (e.g. first convergent of a fraction in (0,1)
        # is 0); keep narrowing — true y != 0 will resolve on the next iteration.

    raise ArithmeticError(f"floor(x/y) did not converge after {_MAX_FLOOR_ITERS} iterations (x/y may be arbitrarily close to an integer)")


def cf_floor(x: CF) -> CF:
    """Return floor(x) as a finite CF.

    For a simple continued fraction the first term *is* the floor by definition,
    so this is O(1) for any CF — finite, periodic, or lazy.
    """
    from .core import CF as _CF

    return _CF.from_int(next(x._iter_from(0)))


def cf_ceil(x: CF) -> CF:
    """Return ceil(x) as a finite CF.

    Uses the identity ceil(x) = -floor(-x).  This correctly handles the
    x-is-an-integer case without peeking at the second CF term: negation via
    the homographic transform naturally maps [n] → [-n] and [n; ...] → [-n-1; ...]
    so the floor of -x is always -(ceil x).
    """
    from .core import CF as _CF

    neg_x = -x
    return _CF.from_int(-next(neg_x._iter_from(0)))


def cf_floordiv(x: CF, y: CF) -> CF:
    """Return floor(x/y) as a finite CF."""
    from .core import CF as _CF

    return _CF.from_int(_floor_quotient(x, y))


def cf_mod(x: CF, y: CF) -> CF:
    """Return x mod y as a CF.  Defined as x - y*floor(x/y).

    The result always has the same sign as y (Python/floor semantics).
    Uses _floor_quotient to obtain the integer n = floor(x/y), then computes
    x - n*y via a homographic scale (n*y) and a bihomographic subtraction,
    avoiding a redundant intermediate lazy CF for x/y.
    """
    from .gosper import cf_homographic, cf_sub

    n = _floor_quotient(x, y)
    return cf_sub(x, cf_homographic(y, n, 0, 0, 1))
