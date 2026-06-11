"""Exponential function as a continued fraction."""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Iterator

from ._backend import (
    _HAS_MPMATH,
    _annotate_cf,
    _cf_terms_from_interval_approximator,
    _coerce_trig_arg,
    _lazy_cf,
)
from .constants import E
from .core import CF
from .gosper import _GIMME_MIN_TERM_DIGITS


def _exp_terms_from_decimal(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of exp(x_num/x_den) using rational intervals."""
    x = Fraction(x_num, x_den)

    def _positive_interval(y: Fraction, precision: int) -> tuple[Fraction, Fraction]:
        k_limit = max(precision, 2 * y.numerator // y.denominator + 4)
        term = Fraction(1)
        val = Fraction(1)
        for k in range(1, k_limit + 1):
            term *= y / k
            val += term

        next_term = term * y / (k_limit + 1)
        ratio = y / (k_limit + 2)
        if ratio >= 1:
            return _positive_interval(y, precision * 2)
        tail = next_term / (1 - ratio)
        return val, val + tail

    def _interval(precision: int) -> tuple[Fraction, Fraction]:
        if x >= 0:
            return _positive_interval(x, precision)
        lo, hi = _positive_interval(-x, precision)
        return Fraction(1, hi), Fraction(1, lo)

    return _cf_terms_from_interval_approximator(_interval, n_terms)


def _exp_terms_from_mpmath(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of exp(x_num/x_den) using mpmath."""
    import mpmath

    mpmath.mp.dps = n_terms * 4 + 50
    val = mpmath.exp(mpmath.mpf(x_num) / mpmath.mpf(x_den))
    terms: list[int] = []
    for _ in range(n_terms):
        a = int(mpmath.floor(val))
        terms.append(a)
        val = 1 / (val - a)
    return terms


def _halfexpm1_metaCF_simple_terms() -> Iterator[Callable[[CF], CF]]:
    """Return metaCF for (Exp(1/z) - 1)/2 = [2z-1; 6z, 10z, 14z, ...]

    Efficient convergence guaranteed for z in [1, ∞].
    """
    k = 2
    yield lambda x: k * x - 1
    while True:
        k += 4
        yield lambda x: k * x


def _halfexpm1_metaCF_terms() -> Iterator[list[int]]:
    """Return metaCF for (Exp(1/z) - 1)/2 = [2z-1; 6z, 10z, 14z, ...]

    Efficient convergence guaranteed for z in [1, ∞].
    """
    k = 2
    yield [-1, k]
    while True:
        k += 4
        yield [0, k]


def ExpCF(
    x: CF,
    mode: str | None = None,
    gimme_min_term_digits: int | None = _GIMME_MIN_TERM_DIGITS,
) -> CF:
    """e raised to the power x, as a continued fraction.

    Only continued fraction computation is used in the backend.

    When e**x is exactly rational (e.g. x = Ln(2) gives 2) the metaCF would
    otherwise refine forever trying to confirm the integer boundary.
    ``gimme_min_term_digits`` lets it accept the near-rational once the evidence
    (digits of the suppressed partial quotient) is strong enough; see
    ``cf_metaCF``.  Set to None to raise on stall instead.  Only the default
    mode supports gimme; mode="simple" (the slow reference path) always raises.

    TODO: Test heavily, then merge in features from ExpMP.
    """
    # TODO: Test heavily, then merge in features from ExpMP.

    modes = (None, "simple")
    if mode not in modes:
        raise ValueError("mode must be one of: {modes}")

    x_coerced = CF._coerce(x)
    if x_coerced is None:
        raise TypeError("Unable to coerce x to CF")
    x = x_coerced

    #if x0 == 0 and len(head.terms) == 1:
    if x == CF.from_int(0):
        # Exact zero is the only exponent we can finish before the meta-CF path.
        return _annotate_cf(CF.from_int(1), ("Exp", x))

    #x0 = head.terms[0]
    x0 = x.take(1).terms[0]
    if x0 != 0:
        # Handle integer part of x separately.
        # That is, Exp([x0; ...])
        #        = Exp(x0 + [0; ...])
        #        = Exp(x0) * Exp([0; ...])
        #        = e**x0   * Exp([0; ...])
        return E() ** x0 * ExpCF(x - x0, mode=mode, gimme_min_term_digits=gimme_min_term_digits)

    from .gosper import cf_metaCF, cf_metaCF_simple

    # Use metaCF for Exp(1/x) for the remainder.
    # That is, since x is in (0, 1), we can compute
    #   z = 1/x = [x1; x2, x3, ...]
    # and then plug that into the metaCF within:
    #   Exp(1/z) = 1 + 2/[2z-1; 6z, 10z, 14z, ...]
    # and it will converge efficiently with z and each term in (1, ∞)
    if mode is None:
        return _annotate_cf(
            1 + 2 / cf_metaCF(1 / x, _halfexpm1_metaCF_terms(), gimme_min_term_digits=gimme_min_term_digits),
            ("ExpCF", x),
        )
    if mode == "simple":
        return _annotate_cf(1 + 2 / cf_metaCF_simple(1 / x, _halfexpm1_metaCF_simple_terms()), ("ExpCF", x))
    assert False


# TODO: Move to tests
def test_ExpCF(mode: None = None) -> None:
    from .constants import EulerGamma, Pi
    from .quadratic import Sqrt

    assert E() == ExpCF(CF.from_int(1))
    for val in (1 / Sqrt(2), Sqrt(2), Pi(), EulerGamma() + Pi() + Sqrt(2)):
        y0 = ExpMP(val).take(80).terms
        y1 = ExpCF(val, mode).take(80).terms
        assert y0 == y1
    from timeit import timeit

    n = 1000
    setup = "from cfmath import Pi, exponential; pi = Pi(); pi.take(20)"
    if mode is None:
        t0 = timeit("exponential.ExpMP(pi).take(20)", setup, number=n)
        t1 = timeit("exponential.ExpCF(pi).take(20)", setup, number=n)
        print(f"{(t1 - t0) / t0:6.0%} ({t1:.0f}s) time taken for ExpCF(Pi) for {n=} runs")
    if mode == "simple":
        t0 = timeit("exponential.ExpMP(pi).take(20)", setup, number=n)
        t1 = timeit('exponential.ExpCF(pi, "simple").take(20)', setup, number=n)
        print(f"{(t1 - t0) / t0:6.0%} ({t1:.0f}s) time taken for ExpCF(Pi) for {n=} runs")


def ExpMP(x: int | Fraction | CF) -> CF:
    """e raised to the power x, as a continued fraction.

    x may be an int, Fraction, or CF.  Returns CF([1]) for x=0.

    Accepting a CF argument enables arbitrary real powers:
        x ** r  ==  Exp(r * Ln(x))   for any rational r and positive x.
    (r * Ln(x) returns a CF, which is passed directly here.)

    Examples::

        Exp(1)                       # e ≈ [2; 1, 2, 1, 1, 4, ...]
        Exp(Fraction(1, 2))          # sqrt(e) ≈ [1; 1, 1, 1, 5, ...]
        Exp(Fraction(3, 2) * Ln(2))  # 2^(3/2) ≈ [2; 1, 3, 1, 5, ...]
    """
    if isinstance(x, CF) and x.is_finite():
        # A finite CF is an exact rational; use the rational path below rather
        # than the convergent-of-an-infinite-CF path, which needs many terms.
        x = x.to_fraction()

    if isinstance(x, CF):
        if x.is_finite():
            return ExpMP(x.to_fraction())
        x_cf = x

        def _compute(n_terms: int) -> list[int]:
            import mpmath

            dps = n_terms * 4 + 50
            mpmath.mp.dps = dps
            from .convergents import convergent as _convergent

            depth = n_terms * 2 + 20
            approx: Fraction = _convergent(x_cf, depth)
            val = mpmath.exp(mpmath.mpf(approx.numerator) / mpmath.mpf(approx.denominator))
            terms: list[int] = []
            for _ in range(n_terms):
                a = int(mpmath.floor(val))
                terms.append(a)
                val = 1 / (val - a)
            return terms

        return _lazy_cf(_compute, debug_source=("Exp", x_cf))

    x = _coerce_trig_arg(x)
    if x == 0:
        return _annotate_cf(CF.from_int(1), ("Exp", x))
    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _exp_terms_from_mpmath(num, den, n), debug_source=("Exp", x))
    return _lazy_cf(lambda n: _exp_terms_from_decimal(num, den, n), debug_source=("Exp", x))


Exp = ExpMP  # TODO: Move into .core or wherever default behavior is selected
# The line is here for now for selecting which to run `make test` on.
