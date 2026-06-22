"""Integer rounding and modular arithmetic for continued fractions."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING, Iterator

# Read gosper's gimme constants at runtime so one config governs every path.
from . import gosper as _gosper

if TYPE_CHECKING:
    from .core import CF

_MAX_FLOOR_ITERS = 10_000

# Sentinel for gimme_min_term_digits: use gosper's current global threshold.
# An explicit int overrides it; None disables gimme (raise on an exact boundary).
_GIMME_DEFAULT = -1


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


def _floor_quotient(x: CF, y: CF, gimme_min_term_digits: int | None = _GIMME_DEFAULT) -> int:
    """Return floor(x/y) as a plain int using rational convergent bounds.

    Maintains rational intervals [lo_x, hi_x] and [lo_y, hi_y] derived from
    the convergents of *x* and *y* (exact Fractions — no float, at any
    magnitude), and resolves floor(x/y) as soon as the interval for x/y stops
    straddling an integer.

    The hard case is x/y exactly an integer with both inputs irrational (e.g.
    Pi/Pi = 1): the interval straddles that integer forever.  By the same
    gimme rule as the Gosper paths, once the interval is narrower than
    10^-gimme_min_term_digits while straddling integer K, accept x/y = K (so
    floor = K).  ``None`` disables gimme and raises on such a boundary instead
    (the old behaviour); the default uses gosper's shared threshold.
    """
    if gimme_min_term_digits == _GIMME_DEFAULT:
        gimme_min_term_digits = _gosper._GIMME_MIN_TERM_DIGITS

    x_bounds = _bounds_iter(x)
    y_bounds = _bounds_iter(y)

    for _ in range(_MAX_FLOOR_ITERS):
        lo_x, hi_x = next(x_bounds)
        lo_y, hi_y = next(y_bounds)

        if lo_y == hi_y == 0:
            raise ZeroDivisionError("floor division by zero")

        if not (lo_y > 0 or hi_y < 0):
            # y bounds straddle zero (e.g. first convergent of a fraction in
            # (0,1) is 0); keep narrowing — true y != 0 resolves next iteration.
            continue

        # With y bounded away from 0, x/y is monotonic in each of x and y, so
        # its extremes over the box lie at the corners.  Which corners depends on
        # the signs of both x and y (the monotonicity in y flips with the sign of
        # x), so take the min and max over all four rather than assume x > 0.
        corners = (lo_x / lo_y, lo_x / hi_y, hi_x / lo_y, hi_x / hi_y)
        val_lo, val_hi = min(corners), max(corners)

        f_lo, f_hi = math.floor(val_lo), math.floor(val_hi)
        if f_lo == f_hi:
            return f_lo
        # Straddle: floors disagree.  If the value is pinned within the gimme
        # threshold of the straddled integer f_hi, accept it.
        if gimme_min_term_digits is not None and val_hi - val_lo < Fraction(1, 10**gimme_min_term_digits):
            return f_hi

    raise ArithmeticError(f"floor(x/y) did not converge after {_MAX_FLOOR_ITERS} iterations (x/y may be arbitrarily close to an integer)")


def cf_floordiv(x: CF, y: CF, gimme_min_term_digits: int | None = _GIMME_DEFAULT) -> CF:
    """Return floor(x/y) as a finite CF.  See _floor_quotient for gimme."""
    from .core import CF as _CF

    return _CF.from_int(_floor_quotient(x, y, gimme_min_term_digits))


def cf_mod(x: CF, y: CF, gimme_min_term_digits: int | None = _GIMME_DEFAULT) -> CF:
    """Return x mod y as a CF.  Defined as x - y*floor(x/y).

    The result always has the same sign as y (Python/floor semantics).
    Uses _floor_quotient to obtain the integer n = floor(x/y), then computes
    x - n*y via a homographic scale (n*y) and a bihomographic subtraction,
    avoiding a redundant intermediate lazy CF for x/y.  See _floor_quotient
    for the gimme parameter.
    """
    from .gosper import cf_homographic, cf_sub

    n = _floor_quotient(x, y, gimme_min_term_digits)
    return cf_sub(x, cf_homographic(y, n, 0, 0, 1))
