"""Trigonometric functions as continued fractions."""

from __future__ import annotations

from enum import Enum
from fractions import Fraction
from typing import Iterator

from ._backend import _annotate_cf, _coerce_trig_arg, _lazy_cf
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


def TanGCF(x: int | Fraction) -> CF:
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


def TanCF(x: int | Fraction | CF, mode: TrigMode | str | None = None) -> CF:
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


def TanMP(x: int | Fraction) -> CF:
    """Tangent of x using mpmath term extraction."""
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(0)
    return _lazy_cf(lambda n: _tan_terms_mpmath(x.numerator, x.denominator, n))


def Tan(x: int | Fraction | CF, mode: TrigMode | str | None = None) -> CF:
    """Tangent of x, dispatching among GCF, CF, and mpmath implementations."""
    mode = _coerce_trig_mode(mode)
    if mode is TrigMode.AUTO:
        if isinstance(x, CF):
            return TanCF(x)
        return TanGCF(x)
    if mode is TrigMode.GCF:
        return TanGCF(x)  # type: ignore[arg-type]
    if mode is TrigMode.CF:
        return TanCF(x)
    if mode is TrigMode.MP:
        return TanMP(x)  # type: ignore[arg-type]
    raise AssertionError(f"unhandled trig mode {mode!r}")


def SinGCF(x: int | Fraction) -> CF:
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


def SinCF(x: int | Fraction | CF, mode: TrigMode | str | None = None) -> CF:
    """Sine of x, using ``TanCF(x/2)`` when the meta-CF path applies."""
    x = _coerce_meta_trig_arg(x)
    if x == CF.from_int(0):
        return CF.from_int(0)

    from .constants import Pi

    pi = Pi()
    tau = 2 * pi
    k = _cf_floor((x + pi) / tau)
    r = x - k * tau
    if r == CF.from_int(0):
        return CF.from_int(0)

    t = TanCF(r / 2, mode=mode)
    return 2 * t / (1 + t * t)


def SinMP(x: int | Fraction) -> CF:
    """Sine of x using mpmath term extraction."""
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(0)
    return _lazy_cf(lambda n: _sin_terms_mpmath(x.numerator, x.denominator, n))


def Sin(x: int | Fraction | CF, mode: TrigMode | str | None = None) -> CF:
    """Sine of x, dispatching among GCF, CF, and mpmath implementations."""
    mode = _coerce_trig_mode(mode)
    if mode is TrigMode.AUTO:
        if isinstance(x, CF):
            return SinCF(x)
        return SinGCF(x)
    if mode is TrigMode.GCF:
        return SinGCF(x)  # type: ignore[arg-type]
    if mode is TrigMode.CF:
        return SinCF(x)
    if mode is TrigMode.MP:
        return SinMP(x)  # type: ignore[arg-type]
    raise AssertionError(f"unhandled trig mode {mode!r}")


def CosGCF(x: int | Fraction) -> CF:
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


def CosCF(x: int | Fraction | CF, mode: TrigMode | str | None = None) -> CF:
    """Cosine of x, using ``TanCF(x/2)`` when the meta-CF path applies."""
    x = _coerce_meta_trig_arg(x)
    if x == CF.from_int(0):
        return CF.from_int(1)

    from .constants import Pi

    pi = Pi()
    tau = 2 * pi
    k = _cf_floor((x + pi) / tau)
    r = x - k * tau
    if r == CF.from_int(0):
        return CF.from_int(1)

    t = TanCF(r / 2, mode=mode)
    t2 = t * t
    return (1 - t2) / (1 + t2)


def CosMP(x: int | Fraction) -> CF:
    """Cosine of x using mpmath term extraction."""
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(1)
    return _lazy_cf(lambda n: _cos_terms_mpmath(x.numerator, x.denominator, n))


def Cos(x: int | Fraction | CF, mode: TrigMode | str | None = None) -> CF:
    """Cosine of x, dispatching among GCF, CF, and mpmath implementations."""
    mode = _coerce_trig_mode(mode)
    if mode is TrigMode.AUTO:
        if isinstance(x, CF):
            return CosCF(x)
        return CosGCF(x)
    if mode is TrigMode.GCF:
        return CosGCF(x)  # type: ignore[arg-type]
    if mode is TrigMode.CF:
        return CosCF(x)
    if mode is TrigMode.MP:
        return CosMP(x)  # type: ignore[arg-type]
    raise AssertionError(f"unhandled trig mode {mode!r}")
