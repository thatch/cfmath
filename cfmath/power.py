"""Power and root functions as continued fractions."""

from __future__ import annotations

import math as _math
from enum import Enum
from fractions import Fraction
from typing import Any, Iterator

from ._backend import _HAS_MPMATH, _annotate_cf, _lazy_cf
from ._poly import content as _poly_content
from .core import CF

PowArg = int | Fraction | CF

# Max input terms consumed per output term before the interval pow engine gives
# up and stops emitting.  The meta-CF engine in gosper.py has its own, smaller
# stall caps for a different algorithm; this path needs a generous bound because
# a legitimate large partial quotient can take many input refinements to pin.
_MAX_STALL = 1000


class PowMode(Enum):
    """Select the implementation used by ``Pow``."""

    AUTO = "auto"
    INT = "int"
    CF = "cf"
    MP = "mp"
    INTERVAL = "interval"


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
    """Divide all coefficients by their GCD to keep magnitudes small.

    The Nthroot engine stores coefficients in descending degree, but the content
    is just the GCD of the integers, so the order does not matter.  Skip the
    rebuild when the GCD is already 1 — the common case once content is stripped.
    """
    g = _poly_content(coeffs)
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


def _cf_interval(cf: CF, k: int) -> tuple[Fraction, Fraction]:
    """Rational bounds [lo, hi] known to bracket cf after k convergent terms."""
    return cf.interval(k)


def _v_cmp(px: int, qx: int, py: int, qy: int, t: Fraction) -> int:
    """Sign of v - t where v = (px/qx)^(py/qy).

    Returns negative, zero, or positive. Requires px >= 0, qx > 0, py >= 0, qy > 0.
    Python's Fraction invariant ensures t.denominator > 0.
    """
    tn, td = t.numerator, t.denominator
    if tn < 0:
        return 1  # v >= 0 > t
    if tn == 0:
        # v = 0 iff px == 0 and py > 0; otherwise v > 0
        return 0 if (px == 0 and py > 0) else 1
    # tn > 0, td > 0: compare (px/qx)^py with (tn/td)^qy
    lhs = px**py * td**qy
    rhs = tn**qy * qx**py
    return -1 if lhs < rhs else (0 if lhs == rhs else 1)


def _floor_at_corner(
    a: int,
    b: int,
    c: int,
    d: int,
    px: int,
    qx: int,
    py: int,
    qy: int,
    n: int,
) -> bool:
    """Check floor((a·v+b)/(c·v+d)) == n at v = (px/qx)^(py/qy).

    Uses pure integer arithmetic. Assumes c·v+d > 0 (valid CF Möbius state).
    """

    def sat(rhs: int, coeff: int, strict: bool) -> bool:
        """Check coeff·v >= rhs, or < rhs when strict=True."""
        if coeff == 0:
            return (rhs > 0) if strict else (rhs <= 0)
        t = Fraction(rhs, coeff)  # Python normalizes sign so denominator > 0
        cv = _v_cmp(px, qx, py, qy, t)
        if coeff > 0:
            return (cv < 0) if strict else (cv >= 0)
        # coeff < 0: dividing flips the inequality direction
        return (cv > 0) if strict else (cv <= 0)

    # n ≤ (a·v+b)/(c·v+d):  (a − n·c)·v ≥ n·d − b
    if not sat(n * d - b, a - n * c, strict=False):
        return False
    # (a·v+b)/(c·v+d) < n+1:  (a − (n+1)·c)·v < (n+1)·d − b
    return sat((n + 1) * d - b, a - (n + 1) * c, strict=True)


def _pow_cf_terms(x: CF, y: CF) -> Iterator[int]:
    """Yield CF terms for x^y via rational convergent interval arithmetic.

    Maintains a Möbius state (a,b,c,d) tracking emitted terms, and tightens
    rational bounds on x and y until all four interval corners agree on the
    next CF term.

    Convergents are tracked incrementally as integer pairs (p_k, q_k) to
    avoid creating Fraction objects in the refinement loop.  For the k-th
    convergent of a CF [a_0; a_1, ...]:

        p_{-1}=1, q_{-1}=0
        p_k = a_k * p_{k-1} + p_{k-2}
        q_k = a_k * q_{k-1} + q_{k-2}

    The perturbed convergent (CF with last term incremented) is
    (p_k + p_{k-1}, q_k + q_{k-1}).  For even k, swap lo/hi to keep lo < hi.

    When the float corners all share the same floor by a comfortable margin,
    that floor is emitted directly.  Near an integer boundary (and only when
    the convergent numerators are small), integer arithmetic verifies the
    floor exactly.
    """
    a, b, c, d = 1, 0, 0, 1

    # Convergent state: xp[k+1] = p_k, xq[k+1] = q_k with p_{-1}=1, q_{-1}=0
    x_iter = iter(x)
    xp: list[int] = [1]  # xp[0] = p_{-1} = 1
    xq: list[int] = [0]  # xq[0] = q_{-1} = 0
    xi = 0  # current depth; xp[-1]/xq[-1] = p_{xi-1}/q_{xi-1}

    y_iter = iter(y)
    yp: list[int] = [1]
    yq: list[int] = [0]
    yi = 0
    y_exact: tuple[int, int] | None = None  # set when y is finite and fully known

    def grow_x() -> bool:
        nonlocal xi
        try:
            t = next(x_iter)
        except StopIteration:
            return False
        xp.append(t * xp[-1] + (xp[-2] if len(xp) >= 2 else 0))
        xq.append(t * xq[-1] + (xq[-2] if len(xq) >= 2 else 1))
        xi += 1
        return True

    def grow_y() -> bool:
        nonlocal yi, y_exact
        if y_exact is not None:
            return True  # y already pinned; only x needs refining
        try:
            t = next(y_iter)
        except StopIteration:
            # y is a finite CF; the last convergent is the exact rational value
            y_exact = (yp[-1], yq[-1])
            return True
        yp.append(t * yp[-1] + (yp[-2] if len(yp) >= 2 else 0))
        yq.append(t * yq[-1] + (yq[-2] if len(yq) >= 2 else 1))
        yi += 1
        return True

    def xy_corners() -> list[tuple[int, int, int, int]]:
        """Four (px, qx, py, qy) integer corners for the current intervals."""
        # x interval
        pxc, qxc = xp[-1], xq[-1]
        pxp, qxp = pxc + xp[-2], qxc + xq[-2]
        if not (xi & 1):
            pxc, qxc, pxp, qxp = pxp, qxp, pxc, qxc
        # y interval — point when y is a fully-consumed finite CF
        if y_exact is not None:
            pyc, qyc = y_exact
            pyp, qyp = pyc, qyc
        else:
            pyc, qyc = yp[-1], yq[-1]
            pyp, qyp = pyc + yp[-2], qyc + yq[-2]
            if not (yi & 1):
                pyc, qyc, pyp, qyp = pyp, qyp, pyc, qyc
        return [
            (pxc, qxc, pyc, qyc),
            (pxc, qxc, pyp, qyp),
            (pxp, qxp, pyc, qyc),
            (pxp, qxp, pyp, qyp),
        ]

    # Initialise both to depth 1
    if not grow_x() or not grow_y():
        return

    stall = 0

    while True:
        corners = xy_corners()

        # Float estimate of the homographic transform at each corner.
        # No Fraction objects are created here — just plain int / int → float.
        # When state magnitudes are large and both numerator/denominator
        # undergo catastrophic cancellation, fall back to mpmath for that corner.
        fvals: list[float] = []
        bad = False
        for px, qx, py, qy in corners:
            v = (px / qx) ** (py / qy)
            if _math.isnan(v):
                bad = True
                break
            num = a * v + b
            denom = c * v + d
            if _math.isnan(denom):
                bad = True
                break
            # Detect catastrophic cancellation: result is tiny vs. individual terms.
            # Check this BEFORE the denom==0 bail-out: float gives denom==0.0 when
            # state magnitudes are large and the true denominator is sub-ULP.
            max_term = max(abs(a) * abs(v), abs(b), abs(c) * abs(v), abs(d))
            min_result = min(abs(num), abs(denom))
            if max_term > 1e6 and (denom == 0.0 or min_result < 1e-5 * max_term):
                # Use mpmath to recover precision lost to cancellation.
                # When denom==0.0 the float has no bits left; estimate cancellation
                # from state magnitude alone.
                try:
                    import mpmath as _mpmath

                    if min_result > 0:
                        cancel = int(-_math.log10(min_result / max_term)) + 20
                    else:
                        cancel = int(_math.log10(max_term)) + 30
                    dps = max(30, min(cancel, 150))
                    with _mpmath.workdps(dps):
                        vmp = (_mpmath.mpf(px) / _mpmath.mpf(qx)) ** (_mpmath.mpf(py) / _mpmath.mpf(qy))
                        d_mp = c * vmp + d
                        if not _mpmath.isfinite(d_mp):
                            bad = True
                            break
                        fvals.append(float((a * vmp + b) / d_mp))
                except ImportError:
                    if denom == 0.0:
                        bad = True
                        break
                    fvals.append(num / denom)
            elif denom == 0.0:
                bad = True
                break
            else:
                fvals.append(num / denom)

        if not bad and all(_math.isfinite(f) for f in fvals):
            n_lo = int(_math.floor(min(fvals)))
            n_hi = int(_math.floor(max(fvals)))

            if n_lo == n_hi:
                n = n_lo
                low_margin = min(f - n for f in fvals)
                hi_margin = min(n + 1 - f for f in fvals)
                boundary_margin = min(low_margin, hi_margin)

                if boundary_margin > 1e-7:
                    # All corners solidly inside [n, n+1); float is reliable.
                    yield n
                    a, b, c, d = c, d, a - n * c, b - n * d
                    stall = 0
                    continue

                # Close to an integer boundary.  Use integer verification
                # when the convergent exponents are small enough.
                py_max = max(py for _, _, py, _ in corners)
                qy_max = max(qy for _, _, _, qy in corners)
                if py_max < 2000 and qy_max < 2000:
                    if all(_floor_at_corner(a, b, c, d, px, qx, py, qy, n) for px, qx, py, qy in corners):
                        yield n
                        a, b, c, d = c, d, a - n * c, b - n * d
                        stall = 0
                        continue
                else:
                    # Deep convergents imply very high precision; trust float.
                    yield n
                    a, b, c, d = c, d, a - n * c, b - n * d
                    stall = 0
                    continue

        stall += 1
        if stall >= _MAX_STALL:
            return

        # Spread heuristic: refine whichever input contributes more spread.
        if bad or not all(_math.isfinite(f) for f in fvals):
            if not grow_x():
                return
        else:
            # corners order: (xl,yl) (xl,yh) (xh,yl) (xh,yh)
            x_spread = max(abs(fvals[2] - fvals[0]), abs(fvals[3] - fvals[1]))
            y_spread = max(abs(fvals[1] - fvals[0]), abs(fvals[3] - fvals[2]))
            if x_spread >= y_spread:
                if not grow_x():
                    return
            else:
                if not grow_y():
                    return


def cf_pow(x: CF, y: CF) -> CF:
    """Return x^y as a continued fraction, computed via convergent interval arithmetic.

    Both x and y must be positive infinite CFs.  For finite CFs or rational
    inputs, use ``Pow``, which dispatches to exact paths before falling through
    to this function.
    """
    return CF([], _source=_pow_cf_terms(x, y))


def _coerce_pow_args(x: PowArg, r: PowArg) -> tuple[Fraction | CF, Fraction | CF]:
    """Coerce finite inputs once so every power implementation sees the same domain."""
    if isinstance(x, CF) and x.is_finite():
        x = x.to_fraction()
    if isinstance(r, CF) and r.is_finite():
        r = r.to_fraction()

    if isinstance(x, int):
        x = Fraction(x)
    elif not isinstance(x, (Fraction, CF)):
        raise TypeError(f"Pow() base expects int, Fraction, or CF, got {type(x).__name__}")

    if isinstance(r, int):
        r = Fraction(r)
    elif not isinstance(r, (Fraction, CF)):
        raise TypeError(f"Pow() exponent expects int, Fraction, or CF, got {type(r).__name__}")

    if isinstance(x, Fraction) and x <= 0:
        raise ValueError(f"Pow() base must be positive, got {x}")

    return x, r


def _as_cf(x: Fraction | CF) -> CF:
    """Return x as a CF without changing infinite CF inputs."""
    return x if isinstance(x, CF) else CF.from_rational(x)


def _pow_trivial(x: Fraction | CF, r: Fraction | CF) -> CF | None:
    """Return exact identity cases that every implementation shares."""
    if isinstance(r, Fraction) and r == 0:
        return CF.from_int(1)
    if isinstance(x, Fraction) and x == 1:
        return CF.from_int(1)
    if isinstance(r, Fraction) and r == 1:
        return _as_cf(x)
    return None


def PowIntExponent(x: PowArg, r: PowArg) -> CF:
    """Raise x to an integer exponent.

    This path is exact.  It accepts a CF base, but the exponent must be an
    integer after finite-CF reduction.  A CF base uses CF repeated squaring.
    """
    x, r = _coerce_pow_args(x, r)
    if not isinstance(r, Fraction) or r.denominator != 1:
        raise ValueError("PowIntExponent requires an integer exponent")
    if isinstance(x, Fraction):
        return CF.from_rational(x**r.numerator)
    return x**r.numerator


def _pow_rational_special(x: Fraction | CF, r: Fraction | CF) -> CF | None:
    """Return exact rational-base shortcuts, or None when no shortcut applies."""
    if not isinstance(x, Fraction) or not isinstance(r, Fraction):
        return None

    if r.denominator == 1:
        return PowIntExponent(x, r)

    # Any rational exponent p/q on a rational base can be rewritten as
    # (x^p)^(1/q), which stays exact and usually beats the generic CF path.
    return Nthroot(x**r.numerator, r.denominator)


def PowCF(x: PowArg, r: PowArg) -> CF:
    """Raise x to r using CF logarithm and exponential implementations.

    The calculation is ``ExpCF(r * LnCF(x))`` after shared coercion and trivial
    identity handling.  It is slower than exact integer/radical paths, but it
    works for finite rationals and non-finite CF inputs with one comparable
    mechanism.
    """
    x, r = _coerce_pow_args(x, r)
    trivial = _pow_trivial(x, r)
    if trivial is not None:
        return trivial
    special = _pow_rational_special(x, r)
    if special is not None:
        return special

    from .exponential import ExpCF
    from .logarithm import LnCF

    return ExpCF(_as_cf(r) * LnCF(_as_cf(x)))


def PowMP(x: PowArg, r: PowArg) -> CF:
    """Raise x to r using mpmath numeric evaluation.

    This path exists for comparison.  It approximates CF inputs by convergents
    at the requested output depth, then extracts CF terms from the mpmath value.
    """
    if not _HAS_MPMATH:
        raise RuntimeError("PowMP requires mpmath")

    x, r = _coerce_pow_args(x, r)
    trivial = _pow_trivial(x, r)
    if trivial is not None:
        return trivial
    special = _pow_rational_special(x, r)
    if special is not None:
        return special

    def _compute(n_terms: int) -> list[int]:
        import mpmath

        from .convergents import convergent as _convergent

        mpmath.mp.dps = n_terms * 5 + 80
        depth = n_terms * 2 + 20

        def mp_arg(v: Fraction | CF) -> Any:
            if isinstance(v, Fraction):
                return mpmath.mpf(v.numerator) / mpmath.mpf(v.denominator)
            q = _convergent(v, depth)
            return mpmath.mpf(q.numerator) / mpmath.mpf(q.denominator)

        val = mpmath.power(mp_arg(x), mp_arg(r))
        terms: list[int] = []
        for _ in range(n_terms):
            a = int(mpmath.floor(val))
            terms.append(a)
            frac = val - a
            if frac == 0:
                break
            val = 1 / frac
        return terms

    return _lazy_cf(_compute)


def PowInterval(x: PowArg, r: PowArg) -> CF:
    """Raise x to r using convergent interval arithmetic.

    This exposes the interval engine for comparison.  It handles exact identity
    and integer-exponent cases first.  For general finite-rational powers use
    ``PowCF`` or ``PowMP``; the raw interval engine needs live CF uncertainty to
    refine.
    """
    x, r = _coerce_pow_args(x, r)
    trivial = _pow_trivial(x, r)
    if trivial is not None:
        return trivial
    if isinstance(r, Fraction) and r.denominator == 1:
        return PowIntExponent(x, r)
    if isinstance(x, Fraction) and isinstance(r, Fraction):
        raise ValueError("PowInterval needs a non-finite CF input for non-integer powers")

    return cf_pow(_as_cf(x), _as_cf(r))


def _coerce_pow_mode(mode: PowMode | str | None) -> PowMode:
    """Return a PowMode, accepting strings as a compatibility convenience."""
    if mode is None:
        return PowMode.AUTO
    if isinstance(mode, PowMode):
        return mode
    if isinstance(mode, str):
        try:
            return PowMode(mode)
        except ValueError as exc:
            raise ValueError(f"unknown Pow mode {mode!r}") from exc
    raise TypeError(f"Pow mode expects PowMode, str, or None, got {type(mode).__name__}")


def Pow(x: PowArg, r: PowArg, mode: PowMode | str | None = None) -> CF:
    """x raised to the power r, as a continued fraction.

    x may be a positive int, Fraction, or CF; r may be an int, Fraction, or CF.

    ``mode`` selects the implementation:
    - ``None`` or ``PowMode.AUTO``: exact shortcuts, then interval for non-finite CFs,
      else ``PowCF``
    - ``PowMode.INT``: exact integer exponent path
    - ``PowMode.CF``: ``ExpCF(r * LnCF(x))``
    - ``PowMode.MP``: mpmath evaluation
    - ``PowMode.INTERVAL``: convergent interval arithmetic

    Special cases (exact or faster paths):
    - r is an integer: exact rational arithmetic, or CF repeated squaring
    - x is a positive integer and r = 1/2: uses Sqrt
    - x is a positive integer and r = 1/3: uses Cuberoot
    - general: ExpCF(r * LnCF(x))

    Examples::

        Pow(4, Fraction(1, 2))  # [2] — exact via Sqrt
        Pow(2, Fraction(3, 2))  # 2√2 ≈ [2; 1, 3, 1, 5, ...]
        Pow(2, -3)              # [0; 8] = 1/8
        Pow(Fraction(2, 3), 2)  # [0; 2, 3, 1, ...] = 4/9
    """
    mode = _coerce_pow_mode(mode)
    if mode is PowMode.AUTO:
        x_coerced, r_coerced = _coerce_pow_args(x, r)
        trivial = _pow_trivial(x_coerced, r_coerced)
        if trivial is not None:
            result = trivial
        else:
            special = _pow_rational_special(x_coerced, r_coerced)
            if special is not None:
                result = special
            elif isinstance(x_coerced, CF) or isinstance(r_coerced, CF):
                result = PowInterval(x_coerced, r_coerced)
            else:
                result = PowCF(x_coerced, r_coerced)
    elif mode is PowMode.INT:
        result = PowIntExponent(x, r)
    elif mode is PowMode.CF:
        result = PowCF(x, r)
    elif mode is PowMode.MP:
        result = PowMP(x, r)
    elif mode is PowMode.INTERVAL:
        result = PowInterval(x, r)
    else:
        raise AssertionError(f"unhandled Pow mode {mode!r}")
    return _annotate_cf(result, ("Pow", x, r))
