"""Inverse trigonometric functions as continued fractions."""

from __future__ import annotations

from enum import Enum
from fractions import Fraction
from typing import Iterator

from ._backend import _annotate_cf, _coerce_trig_arg, _lazy_cf
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


def ArctanCF(x: int | Fraction, mode: ArctrigMode | str | None = None) -> CF:
    """Arctangent of x, using the experimental meta-CF path.

    The public ``Arctan`` function keeps using the direct Gauss generalized CF.
    This variant first rewrites small positive inputs as ``arctan(1/z)`` and
    evaluates the denominator as a generalized meta-CF in ``z``.  That keeps the
    numerator terms positive, but adds the cost of Gosper evaluation around the
    polynomial term stream.
    """
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(0)
    if x < 0:
        return -ArctanCF(-x, mode=mode)
    if x > 1:
        from .constants import Pi

        return Pi() / CF.from_int(2) - ArctanCF(Fraction(1, 1) / x, mode=mode)

    from .gosper import cf_metaGCF

    z = 1 / CF.from_rational(x)
    return 1 / cf_metaGCF(z, _arctan_meta_gcf_terms())


def ArctanMP(x: int | Fraction) -> CF:
    """Arctangent of x using mpmath term extraction."""
    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(0)
    return _lazy_cf(lambda n: _arctan_terms_mpmath(x.numerator, x.denominator, n))


def Arctan(x: int | Fraction, mode: ArctrigMode | str | None = None) -> CF:
    """Arctangent of x, dispatching among GCF, CF, and mpmath implementations."""
    mode = _coerce_arctrig_mode(mode)
    if mode is ArctrigMode.AUTO:
        return ArctanGCF(x)
    if mode is ArctrigMode.GCF:
        return ArctanGCF(x)
    if mode is ArctrigMode.CF:
        return ArctanCF(x)
    if mode is ArctrigMode.MP:
        return ArctanMP(x)
    raise AssertionError(f"unhandled inverse trig mode {mode!r}")


def Arcsin(x: int | Fraction) -> CF:
    """Arcsine of x (in radians), as a continued fraction.

    x may be an int or Fraction; must satisfy |x| ≤ 1.
    Returns CF([0]) for x=0.
    Uses a generalized CF (no external library required).

    Examples::

        Arcsin(0)               # [0]
        Arcsin(1)               # π/2 ≈ [1; 1, 3, 1, 5, ...]
        Arcsin(Fraction(1, 2))  # π/6 ≈ [0; 1, 11, 1, 2, ...]
    """
    x = _coerce_trig_arg(x)
    if abs(x) > 1:
        raise ValueError(f"Arcsin argument must satisfy |x| ≤ 1, got {x}")
    if x == 0:
        return _annotate_cf(CF.from_int(0), ("Arcsin", x))
    return _annotate_cf(CF.from_rational(x) / CF.from_generalized_cf(_arcsin_pairs(x)), ("Arcsin", x))


def Arccos(x: int | Fraction) -> CF:
    """Arccosine of x (in radians), as a continued fraction.

    x may be an int or Fraction; must satisfy |x| ≤ 1.
    Computed as π/2 - arcsin(x).

    Examples::

        Arccos(0)               # π/2 ≈ [1; 1, 3, 1, 5, ...]
        Arccos(1)               # [0]
        Arccos(Fraction(1, 2))  # π/3 ≈ [1; 20, 1, 2, ...]
    """
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
