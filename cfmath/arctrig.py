"""Inverse trigonometric functions as continued fractions."""

from __future__ import annotations

from enum import Enum
from fractions import Fraction
from typing import Iterator

from ._backend import _annotate_cf, _coerce_trig_arg, _lazy_cf, _mpmath_cf_for_cf_arg
from .core import CF


class ArctrigMode(Enum):
    """Select the implementation used by inverse trigonometric functions."""

    AUTO = "auto"
    GCF = "gcf"
    CF = "cf"
    MP = "mp"


def _coerce_arctrig_mode(mode: ArctrigMode | str | None) -> ArctrigMode:
    """Return an ArctrigMode, accepting strings as a compatibility convenience."""
    if mode is None:
        return ArctrigMode.AUTO
    if isinstance(mode, ArctrigMode):
        return mode
    if isinstance(mode, str):
        try:
            return ArctrigMode(mode)
        except ValueError as exc:
            raise ValueError(f"unknown inverse trig mode {mode!r}") from exc
    raise TypeError(f"inverse trig mode expects ArctrigMode, str, or None, got {type(mode).__name__}")


# ---------------------------------------------------------------------------
# Generalized CF generators (exact, no floating point)
# ---------------------------------------------------------------------------


def _arctan_pairs(x: Fraction) -> Iterator[tuple[int, Fraction]]:
    """Yield (b_n, a_{n+1}) pairs for the Gauss generalized CF of arctan(x).

    arctan(x) = x / (1 + x²/(3 + (2x)²/(5 + (3x)²/(7 + ...))))
    """
    x2 = x * x
    k = 0
    while True:
        yield (2 * k + 1, (k + 1) ** 2 * x2)
        k += 1


def _arctan_meta_gcf_terms() -> Iterator[tuple[list[int], list[int]]]:
    """Yield polynomial terms for arctan(1/z)'s Gauss generalized CF.

    arctan(1/z) = 1 / (z + 1²/(3z + 2²/(5z + ...))).
    """
    k = 0
    while True:
        yield ([0, 2 * k + 1], [(k + 1) ** 2])
        k += 1


def _arcsin_pairs(x: Fraction) -> Iterator[tuple[Fraction, Fraction]]:
    """Yield (b_n, a_{n+1}) pairs for the Euler generalized CF of arcsin(x)/x."""
    x2 = x * x

    def _r(k: int) -> Fraction:
        return Fraction((2 * k - 1) ** 2) * x2 / Fraction(2 * k * (2 * k + 1))

    yield (Fraction(1), -_r(1))
    k = 1
    while True:
        yield (Fraction(1) + _r(k), -_r(k + 1))
        k += 1


# ---------------------------------------------------------------------------
# mpmath backends (for cross-checking; not used in dispatch)
# ---------------------------------------------------------------------------


def _arctan_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of arctan(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.atan(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _arcsin_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of arcsin(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.asin(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _arccos_terms_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of arccos(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.acos(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


# ---------------------------------------------------------------------------
# Public functions (use exact GCF — no external library required)
# ---------------------------------------------------------------------------


def ArctanGCF(x: int | Fraction) -> CF:
    """Arctangent of x (in radians), as a continued fraction.

    x may be an int or Fraction.  Returns CF([0]) for x=0.
    Uses the Gauss generalized CF (no external library required).

    Examples::

        Arctan(0)               # [0]
        Arctan(1)               # π/4 ≈ [0; 1, 3, 1, 5, ...]
        Arctan(Fraction(1, 2))  # [0; 2, 7, 1, 1, ...]
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(0), ("Arctan", x))
    return _annotate_cf(CF.from_rational(x) / CF.from_generalized_cf(_arctan_pairs(x)), ("Arctan", x))


def ArctanCF(x: int | Fraction | CF, mode: ArctrigMode | str | None = None) -> CF:
    """Arctangent of x, using the meta-GCF path.

    Rewrites arctan(x) as arctan(1/z) where z = 1/x ≥ 1, evaluating the
    denominator via the Gauss meta-GCF.  a_poly = [(k+1)²] is always positive,
    so the Gosper homographic state has no pole in the tail range [1, ∞).

    Accepts int, Fraction, or CF input.  Large |x| (x > 1) reduces via
    arctan(x) = π/2 − arctan(1/x).  Negative x uses arctan(−x) = −arctan(x).
    """
    from .gosper import cf_metaGCF

    x_cf = CF._coerce(x)
    if x_cf is None:
        raise TypeError(f"expected int, Fraction, or CF, got {type(x).__name__}")

    zero = CF.from_int(0)
    one = CF.from_int(1)

    if x_cf == zero:
        return zero
    if x_cf < zero:
        return -ArctanCF(-x_cf, mode=mode)
    if x_cf > one:
        from .constants import Pi

        return Pi() / 2 - ArctanCF(1 / x_cf, mode=mode)

    z = 1 / x_cf  # z ≥ 1
    return 1 / cf_metaGCF(z, _arctan_meta_gcf_terms())


def ArctanMP(x: int | Fraction | CF) -> CF:
    """Arctangent of x using mpmath term extraction."""
    if isinstance(x, CF):
        import mpmath

        return _mpmath_cf_for_cf_arg(x, mpmath.atan)
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(0)
    return _lazy_cf(lambda n: _arctan_terms_mpmath(x.numerator, x.denominator, n))


def Arctan(x: int | Fraction | CF, mode: ArctrigMode | str | None = None) -> CF:
    """Arctangent of x, dispatching among GCF, CF, and mpmath implementations."""
    mode = _coerce_arctrig_mode(mode)
    if mode is ArctrigMode.AUTO:
        if isinstance(x, CF):
            return ArctanCF(x)
        return ArctanGCF(x)
    if mode is ArctrigMode.GCF:
        if isinstance(x, CF):
            raise TypeError("ArctanGCF requires int or Fraction, not CF")
        return ArctanGCF(x)
    if mode is ArctrigMode.CF:
        return ArctanCF(x)
    if mode is ArctrigMode.MP:
        return ArctanMP(x)
    raise AssertionError(f"unhandled inverse trig mode {mode!r}")


def Arcsin(x: int | Fraction | CF) -> CF:
    """Arcsine of x (in radians), as a continued fraction.

    For int/Fraction, uses a generalized CF (|x| ≤ 1 required).
    For CF input, uses the mpmath dual-precision convergent approach.
    """
    if isinstance(x, CF):
        import mpmath

        return _mpmath_cf_for_cf_arg(x, mpmath.asin)
    x = _coerce_trig_arg(x)
    if abs(x) > 1:
        raise ValueError(f"Arcsin argument must satisfy |x| ≤ 1, got {x}")
    if x == 0:
        return _annotate_cf(CF.from_int(0), ("Arcsin", x))
    return _annotate_cf(CF.from_rational(x) / CF.from_generalized_cf(_arcsin_pairs(x)), ("Arcsin", x))


def Arccos(x: int | Fraction | CF) -> CF:
    """Arccosine of x (in radians), as a continued fraction.

    For int/Fraction, uses π/2 − arcsin(x) (|x| ≤ 1 required).
    For CF input, uses the mpmath dual-precision convergent approach.
    """
    if isinstance(x, CF):
        import mpmath

        return _mpmath_cf_for_cf_arg(x, mpmath.acos)
    x = _coerce_trig_arg(x)
    if abs(x) > 1:
        raise ValueError(f"Arccos argument must satisfy |x| ≤ 1, got {x}")
    if x == 1:
        return _annotate_cf(CF.from_int(0), ("Arccos", x))
    from .constants import Pi

    half_pi = Pi() / CF.from_int(2)
    if x == 0:
        return _annotate_cf(half_pi, ("Arccos", x))
    return _annotate_cf(half_pi - Arcsin(x), ("Arccos", x))
