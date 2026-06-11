"""Benchmarks for power implementations.

Run these with pytest-benchmark installed, for example:

    pytest tests/test_bench_power.py --benchmark-only
"""

from fractions import Fraction

import pytest

pytest.importorskip("pytest_benchmark")

from cfmath import Pi
from cfmath.power import Pow, PowCF, PowIntExponent, PowInterval, PowMP

RATIONAL_BASE = 2
RATIONAL_EXPONENT = Fraction(2, 5)
CF_BASE_EXPONENT = Fraction(1, 2)


def _terms(cf, n: int) -> tuple[int, ...]:
    """Force lazy CF evaluation and return a stable benchmark result."""
    return tuple(cf.take(n).terms)


@pytest.mark.benchmark(group="pow-rational", max_time=0.2, min_rounds=3)
@pytest.mark.parametrize(
    ("factory", "terms"),
    [
        pytest.param(lambda n: _terms(Pow(RATIONAL_BASE, RATIONAL_EXPONENT), n), 1, id="auto-n1"),
        pytest.param(lambda n: _terms(Pow(RATIONAL_BASE, RATIONAL_EXPONENT), n), 2, id="auto-n2"),
        pytest.param(lambda n: _terms(Pow(RATIONAL_BASE, RATIONAL_EXPONENT), n), 4, id="auto-n4"),
        pytest.param(lambda n: _terms(Pow(RATIONAL_BASE, RATIONAL_EXPONENT), n), 8, id="auto-n8"),
        pytest.param(lambda n: _terms(Pow(RATIONAL_BASE, RATIONAL_EXPONENT), n), 12, id="auto-n12"),
        pytest.param(lambda n: _terms(Pow(RATIONAL_BASE, RATIONAL_EXPONENT), n), 20, id="auto-n20"),
        pytest.param(lambda n: _terms(PowCF(RATIONAL_BASE, RATIONAL_EXPONENT), n), 1, id="cf-n1"),
        pytest.param(lambda n: _terms(PowCF(RATIONAL_BASE, RATIONAL_EXPONENT), n), 2, id="cf-n2"),
        pytest.param(lambda n: _terms(PowCF(RATIONAL_BASE, RATIONAL_EXPONENT), n), 4, id="cf-n4"),
        pytest.param(lambda n: _terms(PowCF(RATIONAL_BASE, RATIONAL_EXPONENT), n), 8, id="cf-n8"),
        pytest.param(lambda n: _terms(PowCF(RATIONAL_BASE, RATIONAL_EXPONENT), n), 12, id="cf-n12"),
        pytest.param(lambda n: _terms(PowCF(RATIONAL_BASE, RATIONAL_EXPONENT), n), 20, id="cf-n20"),
        pytest.param(lambda n: _terms(PowMP(RATIONAL_BASE, RATIONAL_EXPONENT), n), 1, id="mp-n1"),
        pytest.param(lambda n: _terms(PowMP(RATIONAL_BASE, RATIONAL_EXPONENT), n), 2, id="mp-n2"),
        pytest.param(lambda n: _terms(PowMP(RATIONAL_BASE, RATIONAL_EXPONENT), n), 4, id="mp-n4"),
        pytest.param(lambda n: _terms(PowMP(RATIONAL_BASE, RATIONAL_EXPONENT), n), 8, id="mp-n8"),
        pytest.param(lambda n: _terms(PowMP(RATIONAL_BASE, RATIONAL_EXPONENT), n), 12, id="mp-n12"),
        pytest.param(lambda n: _terms(PowMP(RATIONAL_BASE, RATIONAL_EXPONENT), n), 20, id="mp-n20"),
    ],
)
def test_bench_pow_rational_paths(benchmark, factory, terms):
    result = benchmark(lambda: factory(terms))
    assert len(result) == terms


@pytest.mark.benchmark(group="pow-cf-base", max_time=0.2, min_rounds=3)
@pytest.mark.parametrize(
    ("factory", "terms"),
    [
        pytest.param(lambda n: _terms(Pow(Pi(), CF_BASE_EXPONENT), n), 1, id="auto-n1"),
        pytest.param(lambda n: _terms(Pow(Pi(), CF_BASE_EXPONENT), n), 2, id="auto-n2"),
        pytest.param(lambda n: _terms(Pow(Pi(), CF_BASE_EXPONENT), n), 4, id="auto-n4"),
        pytest.param(lambda n: _terms(Pow(Pi(), CF_BASE_EXPONENT), n), 8, id="auto-n8"),
        pytest.param(lambda n: _terms(Pow(Pi(), CF_BASE_EXPONENT), n), 12, id="auto-n12"),
        pytest.param(lambda n: _terms(PowInterval(Pi(), CF_BASE_EXPONENT), n), 1, id="interval-n1"),
        pytest.param(lambda n: _terms(PowInterval(Pi(), CF_BASE_EXPONENT), n), 2, id="interval-n2"),
        pytest.param(lambda n: _terms(PowInterval(Pi(), CF_BASE_EXPONENT), n), 4, id="interval-n4"),
        pytest.param(lambda n: _terms(PowInterval(Pi(), CF_BASE_EXPONENT), n), 8, id="interval-n8"),
        pytest.param(lambda n: _terms(PowInterval(Pi(), CF_BASE_EXPONENT), n), 12, id="interval-n12"),
        pytest.param(lambda n: _terms(PowCF(Pi(), CF_BASE_EXPONENT), n), 1, id="cf-n1"),
        pytest.param(lambda n: _terms(PowCF(Pi(), CF_BASE_EXPONENT), n), 2, id="cf-n2"),
        pytest.param(lambda n: _terms(PowCF(Pi(), CF_BASE_EXPONENT), n), 4, id="cf-n4"),
        pytest.param(lambda n: _terms(PowCF(Pi(), CF_BASE_EXPONENT), n), 8, id="cf-n8"),
        pytest.param(lambda n: _terms(PowCF(Pi(), CF_BASE_EXPONENT), n), 12, id="cf-n12"),
        pytest.param(lambda n: _terms(PowMP(Pi(), CF_BASE_EXPONENT), n), 1, id="mp-n1"),
        pytest.param(lambda n: _terms(PowMP(Pi(), CF_BASE_EXPONENT), n), 2, id="mp-n2"),
        pytest.param(lambda n: _terms(PowMP(Pi(), CF_BASE_EXPONENT), n), 4, id="mp-n4"),
        pytest.param(lambda n: _terms(PowMP(Pi(), CF_BASE_EXPONENT), n), 8, id="mp-n8"),
        pytest.param(lambda n: _terms(PowMP(Pi(), CF_BASE_EXPONENT), n), 12, id="mp-n12"),
    ],
)
def test_bench_pow_cf_base_paths(benchmark, factory, terms):
    result = benchmark(lambda: factory(terms))
    assert len(result) == terms


@pytest.mark.benchmark(group="pow-cf-int-exponent", max_time=0.2, min_rounds=3)
@pytest.mark.parametrize(
    ("factory", "terms"),
    [
        pytest.param(lambda n: _terms(Pow(Pi(), 2), n), 1, id="auto-n1"),
        pytest.param(lambda n: _terms(Pow(Pi(), 2), n), 2, id="auto-n2"),
        pytest.param(lambda n: _terms(Pow(Pi(), 2), n), 4, id="auto-n4"),
        pytest.param(lambda n: _terms(Pow(Pi(), 2), n), 8, id="auto-n8"),
        pytest.param(lambda n: _terms(Pow(Pi(), 2), n), 12, id="auto-n12"),
        pytest.param(lambda n: _terms(Pow(Pi(), 2), n), 20, id="auto-n20"),
        pytest.param(lambda n: _terms(PowIntExponent(Pi(), 2), n), 1, id="int-exponent-n1"),
        pytest.param(lambda n: _terms(PowIntExponent(Pi(), 2), n), 2, id="int-exponent-n2"),
        pytest.param(lambda n: _terms(PowIntExponent(Pi(), 2), n), 4, id="int-exponent-n4"),
        pytest.param(lambda n: _terms(PowIntExponent(Pi(), 2), n), 8, id="int-exponent-n8"),
        pytest.param(lambda n: _terms(PowIntExponent(Pi(), 2), n), 12, id="int-exponent-n12"),
        pytest.param(lambda n: _terms(PowIntExponent(Pi(), 2), n), 20, id="int-exponent-n20"),
    ],
)
def test_bench_pow_cf_int_exponent_paths(benchmark, factory, terms):
    result = benchmark(lambda: factory(terms))
    assert len(result) == terms
