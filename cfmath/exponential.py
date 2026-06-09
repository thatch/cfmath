"""Exponential function as a continued fraction."""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Iterator

from ._backend import _HAS_MPMATH, _coerce_trig_arg, _lazy_cf
from .constants import E
from .core import CF


def _exp_terms_from_decimal(x_num: int, x_den: int, n_terms: int) -> list[int]:
    """Compute n_terms CF terms of exp(x_num/x_den) using high-precision Decimal.

    Uses the Taylor series exp(x) = Σ xⁿ/n!  No external library required.
    """
    import decimal

    prec = n_terms * 5 + 80
    ctx = decimal.Context(prec=prec, rounding=decimal.ROUND_FLOOR)
    with decimal.localcontext(ctx):
        x = decimal.Decimal(x_num) / decimal.Decimal(x_den)
        term = decimal.Decimal(1)
        val = decimal.Decimal(1)
        k = 1
        eps = decimal.Decimal(10) ** (-(prec - 10))
        while True:
            term *= x / decimal.Decimal(k)
            val += term
            if abs(term) < eps:
                break
            k += 1

        terms: list[int] = []
        for _ in range(n_terms):
            a = int(val.to_integral_value(rounding=decimal.ROUND_FLOOR))
            terms.append(a)
            frac = val - a
            if frac <= eps:
                break
            val = decimal.Decimal(1) / frac

    return terms


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


def ExpCF(x: CF, mode: str | None = None) -> CF:
    """e raised to the power x, as a continued fraction.

    Only continued fraction computation is used in the backend.

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

    if x == CF.from_int(0):
        # I can do this one in my head.
        return CF.from_int(1)

    x0 = x.take(1).terms[0]
    if x0 != 0:
        # Handle integer part of x separately.
        # That is, Exp([x0; ...])
        #        = Exp(x0 + [0; ...])
        #        = Exp(x0) * Exp([0; ...])
        #        = e**x0   * Exp([0; ...])
        return E() ** x0 * ExpCF(x - x0, mode=mode)

    from .gosper import cf_metaCF, cf_metaCF_simple

    # Use metaCF for Exp(1/x) for the remainder.
    # That is, since x is in (0, 1), we can compute
    #   z = 1/x = [x1; x2, x3, ...]
    # and then plug that into the metaCF within:
    #   Exp(1/z) = 1 + 2/[2z-1; 6z, 10z, 14z, ...]
    # and it will converge efficiently with z and each term in (1, ∞)
    if mode is None:
        return 1 + 2 / cf_metaCF(1 / x, _halfexpm1_metaCF_terms())
    if mode == "simple":
        return 1 + 2 / cf_metaCF_simple(1 / x, _halfexpm1_metaCF_simple_terms())
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
    if isinstance(x, CF):
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

        return _lazy_cf(_compute)

    x = _coerce_trig_arg(x)
    if x == 0:
        return CF.from_int(1)
    num, den = x.numerator, x.denominator
    if _HAS_MPMATH:
        return _lazy_cf(lambda n: _exp_terms_from_mpmath(num, den, n))
    return _lazy_cf(lambda n: _exp_terms_from_decimal(num, den, n))


Exp = ExpMP  # TODO: Move into .core or wherever default behavior is selected
# The line is here for now for selecting which to run `make test` on.
