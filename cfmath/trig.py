"""Trigonometric functions as continued fractions."""

from __future__ import annotations

from enum import Enum
from fractions import Fraction
from typing import Iterator

from ._backend import _annotate_cf, _coerce_trig_arg, _lazy_cf, _mpmath_cf_for_cf_arg
from .core import CF


class TrigMode(Enum):
    """Select the implementation used by trigonometric functions."""

    AUTO = "auto"
    GCF = "gcf"
    CF = "cf"
    MP = "mp"


def _coerce_trig_mode(mode: TrigMode | str | None) -> TrigMode:
    """Return a TrigMode, accepting strings as a compatibility convenience."""
    if mode is None:
        return TrigMode.AUTO
    if isinstance(mode, TrigMode):
        return mode
    if isinstance(mode, str):
        try:
            return TrigMode(mode)
        except ValueError as exc:
            raise ValueError(f"unknown trig mode {mode!r}") from exc
    raise TypeError(f"trig mode expects TrigMode, str, or None, got {type(mode).__name__}")


# ---------------------------------------------------------------------------
# Generalized CF generators (exact, no floating point)
# ---------------------------------------------------------------------------


def _tan_pairs(x: Fraction) -> Iterator[tuple[int, Fraction]]:
    """Yield (b_n, a_{n+1}) pairs for the Lambert generalized CF of tan(x).

    tan(x) = x / (1 - x²/(3 - x²/(5 - x²/...)))
    """
    neg_x2 = -(x * x)
    n = 0
    while True:
        yield (2 * n + 1, neg_x2)
        n += 1


def _tan_meta_gcf_terms() -> Iterator[tuple[list[int], list[int]]]:
    """Yield polynomial terms for tan(1/z)'s Lambert generalized CF.

    tan(1/z) = 1 / (z - 1/(3z - 1/(5z - ...))).
    """
    n = 0
    while True:
        yield ([0, 2 * n + 1], [-1])
        n += 1


def _coerce_meta_trig_arg(x: int | Fraction | CF) -> CF:
    """Validate and coerce an experimental meta-trig argument to CF."""
    coerced = CF._coerce(x)
    if coerced is None:
        raise TypeError(f"expected int, Fraction, or CF, got {type(x).__name__}")
    return coerced


def _cf_floor(x: CF) -> int:
    """Return floor(x) from the first simple-CF term."""
    return x.take(1).terms[0]


def _tan_cf_small_positive(x: CF) -> CF:
    """Return tan(x) for 0 < x <= 1 using the direct meta-CF."""
    from .gosper import cf_metaGCF

    z = 1 / x
    return 1 / cf_metaGCF(z, _tan_meta_gcf_terms())


def _tan_cf_reduced(r: CF, half_pi: CF) -> CF:
    """Return tan(r) for r in [-π/2, π/2)."""
    if r == CF.from_int(0):
        return CF.from_int(0)
    if r < CF.from_int(0):
        return -_tan_cf_reduced(-r, half_pi)
    if r > CF.from_int(1):
        return 1 / _tan_cf_reduced(half_pi - r, half_pi)
    return _tan_cf_small_positive(r)


def _sin_meta_gcf_terms() -> Iterator[tuple[list[int], list[int]]]:
    """Yield (b_poly, a_poly) for the Lambert GCF of sin(x)/x with z = x².

    sin(x)/x = 1 / (1 + z/(6−z + 6z/(20−z + 20z/(42−z + ···))))

    Level 0: b=1, a=z          →  b_poly=[1],     a_poly=[0, 1]
    Level k: b=c−z, a=c·z      →  b_poly=[c, −1], a_poly=[0, c]   c = 2k(2k+1)

    a_poly is always positive (c·z > 0 for z > 0), so the Gosper homographic
    state has no pole in the tail range [1, ∞).  b_poly[1] = −1 means
    b = c − z can be negative when z > c, which occurs for z > 6 (k=1 gives
    c=6).  The reduction in SinCF keeps z < (π−1)² < 5 so this never fires.
    """
    yield ([1], [0, 1])
    k = 1
    while True:
        c = 2 * k * (2 * k + 1)
        yield ([c, -1], [0, c])
        k += 1


def _cos_meta_gcf_terms() -> Iterator[tuple[list[int], list[int]]]:
    """Yield (b_poly, a_poly) for the Lambert GCF of 1/cos(x) with z = x².

    1/cos(x) = 1 + z/(2−z + 2z/(12−z + 12z/(30−z + ···)))

    Level 0: b=1, a=z          →  b_poly=[1],     a_poly=[0, 1]
    Level k: b=c−z, a=c·z      →  b_poly=[c, −1], a_poly=[0, c]   c = (2k−1)(2k)

    a_poly is always positive.  b_poly[1] = −1 means b = c − z turns negative
    when z > c; for k=1 c=2, so the GCF is unsafe when z > 2 (x > √2 ≈ 1.414).
    CosCF restricts r ≤ 1 before calling this, keeping z ≤ 1 < 2.
    """
    yield ([1], [0, 1])
    k = 1
    while True:
        c = (2 * k - 1) * (2 * k)
        yield ([c, -1], [0, c])
        k += 1


def _cos_cf_positive(r: CF, half_pi: CF) -> CF:
    """cos(r) for r in (0, π) via the 1/cos meta-GCF or the sin identity.

    For r ≤ 1: apply 1/cos-GCF with z = r² < 1 < 2, keeping all b terms > 0.
    For r > 1: cos(r) = sin(π/2 − r), delegating to SinCF which handles the
               resulting argument in (−π/2, π/2 − 1).
    """
    from .gosper import cf_metaGCF

    if r > CF.from_int(1):
        return _SinCF(half_pi - r)
    return 1 / cf_metaGCF(r * r, _cos_meta_gcf_terms())


def _sin_cf_positive(r: CF, pi: CF) -> CF:
    """Return sin(r) for r in (0, π) using the sin/x meta-GCF with z = r².

    Reduces r > 1 via sin(r) = sin(π − r), keeping z < (π−1)² < 5 so that
    all b terms in the GCF stay positive and the Gosper state has no pole.
    """
    from .gosper import cf_metaGCF

    if r > CF.from_int(1):
        r = pi - r
    return r / cf_metaGCF(r * r, _sin_meta_gcf_terms())


def _sin_pairs(x: Fraction) -> Iterator[tuple[Fraction, Fraction]]:
    """Yield (b_n, a_{n+1}) pairs for the generalized CF of sin(x)/x."""
    x2 = x * x
    yield (Fraction(1), x2)
    k = 1
    while True:
        c = Fraction(2 * k * (2 * k + 1))
        yield (c - x2, c * x2)
        k += 1


def _cos_pairs(x: Fraction) -> Iterator[tuple[Fraction, Fraction]]:
    """Yield (b_n, a_{n+1}) pairs for the generalized CF of 1/cos(x)."""
    x2 = x * x
    yield (Fraction(1), x2)
    k = 1
    while True:
        c = Fraction((2 * k - 1) * (2 * k))
        yield (c - x2, c * x2)
        k += 1


# ---------------------------------------------------------------------------
# mpmath backends (for cross-checking; not used in dispatch)
# ---------------------------------------------------------------------------


def _sin_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of sin(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.sin(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _cos_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of cos(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.cos(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _tan_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of tan(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.tan(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


# ---------------------------------------------------------------------------
# Public functions (use exact GCF — no external library required)
# ---------------------------------------------------------------------------


def _TanGCF(x: int | Fraction) -> CF:
    """Tangent of x (in radians), as a continued fraction.

    x may be an int or Fraction.  Returns CF([0]) for x=0.
    Uses Lambert's generalized CF (no external library required).

    Examples::

        Tan(0)                  # [0]
        Tan(Fraction(1, 4))     # [0; 3, 1, 10, 1, 18, 1, 26]
        Tan(Fraction(1, 2))     # [0; 1, 1, 4, 1, 1, 8, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(0), ("Tan", x))
    denom_cf = CF.from_generalized_cf(_tan_pairs(x))
    return _annotate_cf(CF.from_rational(x) / denom_cf, ("Tan", x))


_TanGCF.__name__ = _TanGCF.__qualname__ = "TanGCF"


def _TanCF(x: int | Fraction | CF, mode: TrigMode | str | None = None) -> CF:
    """Tangent of x, using the experimental meta-CF path.

    The direct Lambert meta-CF converges reliably for ``|x| <= 1``.  Larger
    inputs are reduced modulo π, then values in ``(1, π/2)`` use
    ``tan(x) = 1/tan(π/2 - x)``.  The cost is that reduction now depends on
    comparing against the CF for π.
    """
    x = _coerce_meta_trig_arg(x)
    if x == CF.from_int(0):
        return CF.from_int(0)

    from .constants import Pi

    pi = Pi()
    half_pi = pi / 2
    k = _cf_floor((x + half_pi) / pi)
    r = x - k * pi
    return _tan_cf_reduced(r, half_pi)


_TanCF.__name__ = _TanCF.__qualname__ = "TanCF"


def _TanMP(x: int | Fraction | CF) -> CF:
    """Tangent of x using mpmath term extraction."""
    if isinstance(x, CF):
        import mpmath

        return _mpmath_cf_for_cf_arg(x, mpmath.tan)
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(0)
    return _lazy_cf(lambda n: _tan_terms_mpmath(x.numerator, x.denominator, n))


_TanMP.__name__ = _TanMP.__qualname__ = "TanMP"


def Tan(x: int | Fraction | CF, mode: TrigMode | str | None = None) -> CF:
    """Tangent of x, dispatching among GCF, CF, and mpmath implementations."""
    mode = _coerce_trig_mode(mode)
    if mode is TrigMode.AUTO:
        if isinstance(x, CF):
            return _TanCF(x)
        return _TanGCF(x)
    if mode is TrigMode.GCF:
        return _TanGCF(x)  # type: ignore[arg-type]
    if mode is TrigMode.CF:
        return _TanCF(x)
    if mode is TrigMode.MP:
        return _TanMP(x)
    raise AssertionError(f"unhandled trig mode {mode!r}")


def _SinGCF(x: int | Fraction) -> CF:
    """Sine of x (in radians), as a continued fraction.

    x may be an int or Fraction.  Returns CF([0]) for x=0.
    Uses a direct generalized CF (no external library required).

    Examples::

        Sin(0)                  # [0]
        Sin(Fraction(1, 2))     # [0; 2, 11, 1, 1, 1, 6, ...]
        Sin(Fraction(123, 1000))# [0; 8, 6, 1, 1, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(0), ("Sin", x))
    return _annotate_cf(CF.from_rational(x) / CF.from_generalized_cf(_sin_pairs(x)), ("Sin", x))


_SinGCF.__name__ = _SinGCF.__qualname__ = "SinGCF"


def _SinCF(x: int | Fraction | CF) -> CF:
    """Sine of x using the sin/x meta-GCF with z = x².

    Reduces x modulo 2π, then maps to (0, 1] or (0, π−1] via
    sin(r) = sin(π − r), keeping z = r² < 5 so the GCF has no pole.
    Accepts int, Fraction, or CF input.
    """
    x = _coerce_meta_trig_arg(x)
    if x == CF.from_int(0):
        return CF.from_int(0)

    from .constants import Pi

    pi = Pi()
    tau = 2 * pi
    k = _cf_floor((x + pi) / tau)
    r = x - k * tau  # r in [−π, π)
    if r == CF.from_int(0):
        return CF.from_int(0)
    if r < CF.from_int(0):
        return -_sin_cf_positive(-r, pi)
    return _sin_cf_positive(r, pi)


_SinCF.__name__ = _SinCF.__qualname__ = "SinCF"


def _SinMP(x: int | Fraction | CF) -> CF:
    """Sine of x using mpmath term extraction.

    For int/Fraction input, extracts CF terms directly from mpmath.sin.
    For CF input, first approximates x via a convergent deep enough to
    make the rational-approximation error negligible at working precision,
    then uses dual-precision verification to emit only confirmed terms.
    """
    if isinstance(x, CF):
        import mpmath

        from ._backend import _mpmath_cf
        from .convergents import convergent
        x_cf = x

        def _value_fn() -> object:
            dps = mpmath.mp.dps
            depth = max(5 * dps, 60)
            try:
                approx = convergent(x_cf, depth)
            except IndexError:
                approx = x_cf.to_fraction()
            return mpmath.sin(mpmath.mpf(approx.numerator) / mpmath.mpf(approx.denominator))

        return _mpmath_cf(_value_fn)

    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(0)
    return _lazy_cf(lambda n: _sin_terms_mpmath(x.numerator, x.denominator, n))


_SinMP.__name__ = _SinMP.__qualname__ = "SinMP"


def Sin(x: int | Fraction | CF, mode: TrigMode | str | None = None) -> CF:
    """Sine of x, dispatching among GCF, CF, and mpmath implementations."""
    mode = _coerce_trig_mode(mode)
    if mode is TrigMode.AUTO:
        if isinstance(x, CF):
            return _SinCF(x)
        return _SinGCF(x)
    if mode is TrigMode.GCF:
        return _SinGCF(x)  # type: ignore[arg-type]
    if mode is TrigMode.CF:
        return _SinCF(x)
    if mode is TrigMode.MP:
        return _SinMP(x)
    raise AssertionError(f"unhandled trig mode {mode!r}")


def _CosGCF(x: int | Fraction) -> CF:
    """Cosine of x (in radians), as a continued fraction.

    x may be an int or Fraction.  Returns CF([1]) for x=0.
    Uses a direct generalized CF (no external library required).

    Examples::

        Cos(0)                  # [1]
        Cos(Fraction(1, 2))     # [0; 1, 7, 5, 1, 12, 2, 1]
        Cos(Fraction(123, 1000))# [0; 1, 131, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(1), ("Cos", x))
    return _annotate_cf(CF.from_int(1) / CF.from_generalized_cf(_cos_pairs(x)), ("Cos", x))


_CosGCF.__name__ = _CosGCF.__qualname__ = "CosGCF"


def _CosCF(x: int | Fraction | CF) -> CF:
    """Cosine of x using the 1/cos meta-GCF with z = x².

    Reduces x modulo 2π, uses even symmetry, then maps r > 1 to
    sin(π/2 − r) via _SinCF, keeping z = r² < 1 for the cos GCF itself.
    Accepts int, Fraction, or CF input.
    """
    x = _coerce_meta_trig_arg(x)
    if x == CF.from_int(0):
        return CF.from_int(1)

    from .constants import Pi

    pi = Pi()
    tau = 2 * pi
    k = _cf_floor((x + pi) / tau)
    r = x - k * tau  # r in [−π, π)
    if r == CF.from_int(0):
        return CF.from_int(1)
    if r < CF.from_int(0):
        r = -r  # cos is even
    return _cos_cf_positive(r, pi / 2)


_CosCF.__name__ = _CosCF.__qualname__ = "CosCF"


def _CosMP(x: int | Fraction | CF) -> CF:
    """Cosine of x using mpmath term extraction."""
    if isinstance(x, CF):
        import mpmath

        return _mpmath_cf_for_cf_arg(x, mpmath.cos)
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(1)
    return _lazy_cf(lambda n: _cos_terms_mpmath(x.numerator, x.denominator, n))


_CosMP.__name__ = _CosMP.__qualname__ = "CosMP"


def Cos(x: int | Fraction | CF, mode: TrigMode | str | None = None) -> CF:
    """Cosine of x, dispatching among GCF, CF, and mpmath implementations."""
    mode = _coerce_trig_mode(mode)
    if mode is TrigMode.AUTO:
        if isinstance(x, CF):
            return _CosMP(x)
        return _CosGCF(x)
    if mode is TrigMode.GCF:
        return _CosGCF(x)  # type: ignore[arg-type]
    if mode is TrigMode.CF:
        return _CosCF(x)
    if mode is TrigMode.MP:
        return _CosMP(x)
    raise AssertionError(f"unhandled trig mode {mode!r}")
