"""Benchmarks for trig and inverse-trig implementations.

Run these with pytest-benchmark installed, for example:

    pytest tests/test_bench_trig.py --benchmark-only
"""

from fractions import Fraction

import pytest

pytest.importorskip("pytest_benchmark")

from cfmath import Pi
from cfmath.arctrig import Arctan, ArctanCF, ArctanGCF, ArctanMP
from cfmath.trig import Cos, CosCF, CosGCF, CosMP, Sin, SinCF, SinGCF, SinMP, Tan, TanCF, TanGCF, TanMP

RATIONAL_X = Fraction(1, 3)
CF_X = Pi() + RATIONAL_X
TERM_COUNTS = (1, 4, 8, 12)
CF_TERM_COUNTS = (1, 4, 8)


def _terms(cf, n: int) -> tuple[int, ...]:
    """Force lazy CF evaluation and return a stable benchmark result."""
    return tuple(cf.take(n).terms)


def _params(paths, counts):
    return [
        pytest.param(lambda n, factory=factory: _terms(factory(), n), terms, id=f"{name}-n{terms}")
        for name, factory in paths
        for terms in counts
    ]


@pytest.mark.benchmark(group="tan-rational", max_time=0.1, min_rounds=3)
@pytest.mark.parametrize(
    ("factory", "terms"),
    _params(
        [
            ("auto", lambda: Tan(RATIONAL_X)),
            ("gcf", lambda: TanGCF(RATIONAL_X)),
            ("cf", lambda: TanCF(RATIONAL_X)),
            ("mp", lambda: TanMP(RATIONAL_X)),
        ],
        TERM_COUNTS,
    ),
)
def test_bench_tan_rational_paths(benchmark, factory, terms):
    result = benchmark(lambda: factory(terms))
    assert len(result) == terms


@pytest.mark.benchmark(group="sin-rational", max_time=0.1, min_rounds=3)
@pytest.mark.parametrize(
    ("factory", "terms"),
    _params(
        [
            ("auto", lambda: Sin(RATIONAL_X)),
            ("gcf", lambda: SinGCF(RATIONAL_X)),
            ("cf", lambda: SinCF(RATIONAL_X)),
            ("mp", lambda: SinMP(RATIONAL_X)),
        ],
        TERM_COUNTS,
    ),
)
def test_bench_sin_rational_paths(benchmark, factory, terms):
    result = benchmark(lambda: factory(terms))
    assert len(result) == terms


@pytest.mark.benchmark(group="cos-rational", max_time=0.1, min_rounds=3)
@pytest.mark.parametrize(
    ("factory", "terms"),
    _params(
        [
            ("auto", lambda: Cos(RATIONAL_X)),
            ("gcf", lambda: CosGCF(RATIONAL_X)),
            ("cf", lambda: CosCF(RATIONAL_X)),
            ("mp", lambda: CosMP(RATIONAL_X)),
        ],
        TERM_COUNTS,
    ),
)
def test_bench_cos_rational_paths(benchmark, factory, terms):
    result = benchmark(lambda: factory(terms))
    assert len(result) == terms


@pytest.mark.benchmark(group="arctan-rational", max_time=0.1, min_rounds=3)
@pytest.mark.parametrize(
    ("factory", "terms"),
    _params(
        [
            ("auto", lambda: Arctan(RATIONAL_X)),
            ("gcf", lambda: ArctanGCF(RATIONAL_X)),
            ("cf", lambda: ArctanCF(RATIONAL_X)),
            ("mp", lambda: ArctanMP(RATIONAL_X)),
        ],
        TERM_COUNTS,
    ),
)
def test_bench_arctan_rational_paths(benchmark, factory, terms):
    result = benchmark(lambda: factory(terms))
    assert len(result) == terms


@pytest.mark.benchmark(group="trig-cf-input", max_time=0.1, min_rounds=3)
@pytest.mark.parametrize(
    ("factory", "terms"),
    _params(
        [
            ("tan-auto", lambda: Tan(CF_X)),
            ("tan-cf", lambda: TanCF(CF_X)),
            ("sin-auto", lambda: Sin(CF_X)),
            ("sin-cf", lambda: SinCF(CF_X)),
            ("cos-auto", lambda: Cos(CF_X)),
            ("cos-cf", lambda: CosCF(CF_X)),
        ],
        CF_TERM_COUNTS,
    ),
)
def test_bench_trig_cf_input_paths(benchmark, factory, terms):
    result = benchmark(lambda: factory(terms))
    assert len(result) == terms
