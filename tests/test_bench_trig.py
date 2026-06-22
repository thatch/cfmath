"""Benchmarks for trig and inverse-trig implementations.

Run these with pytest-benchmark installed, for example:

    pytest tests/test_bench_trig.py --benchmark-only
"""

from fractions import Fraction

import pytest

pytest.importorskip("pytest_benchmark")

from cfmath import Pi
from cfmath.archyperbolic import Arccosh, Arcsinh, Arctanh
from cfmath.arctrig import Arccos, Arcsin, Arctan, ArctanCF, ArctanGCF, ArctanMP
from cfmath.hyperbolic import Cosh, Sinh, Tanh
from cfmath.quadratic import Sqrt
from cfmath.trig import (
    Cos,
    Sin,
    Tan,
)
from cfmath.trig import (
    _CosCF as CosCF,
)
from cfmath.trig import (
    _CosGCF as CosGCF,
)
from cfmath.trig import (
    _CosMP as CosMP,
)
from cfmath.trig import (
    _SinCF as SinCF,
)
from cfmath.trig import (
    _SinGCF as SinGCF,
)
from cfmath.trig import (
    _SinMP as SinMP,
)
from cfmath.trig import (
    _TanCF as TanCF,
)
from cfmath.trig import (
    _TanGCF as TanGCF,
)
from cfmath.trig import (
    _TanMP as TanMP,
)

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


# CF-input benchmarks for functions newly accepting CF (Option A: mpmath fallback)
# CF_X = Pi() + 1/3 ≈ 3.47, kept in domain for all functions below.
# For arcsin/arccos we need |x| ≤ 1 so we use a separate CF_X_SMALL = Sqrt(2)/2.
CF_X_SMALL = Sqrt(2) / 2  # ≈ 0.707, in domain for arcsin/arccos


# Option A (mpmath convergent) vs Option B (ExpCF) for hyperbolic CF input
@pytest.mark.benchmark(group="hyperbolic-cf-input", max_time=0.1, min_rounds=3)
@pytest.mark.parametrize(
    ("factory", "terms"),
    _params(
        [
            ("sinh-exp-cf", lambda: Sinh(CF_X)),  # Option B: (ExpCF ± 1/ExpCF)/2
            ("cosh-exp-cf", lambda: Cosh(CF_X)),  # Option B
            ("tanh-mp", lambda: Tanh(CF_X)),  # Option A: mpmath fallback
        ],
        CF_TERM_COUNTS,
    ),
)
def test_bench_hyperbolic_cf_input(benchmark, factory, terms):
    result = benchmark(lambda: factory(terms))
    assert len(result) == terms


@pytest.mark.benchmark(group="archyperbolic-cf-input", max_time=0.1, min_rounds=3)
@pytest.mark.parametrize(
    ("factory", "terms"),
    _params(
        [
            ("arcsinh-mp", lambda: Arcsinh(CF_X)),  # Option A
            ("arccosh-mp", lambda: Arccosh(CF_X)),  # Option A
            ("arctanh-mp", lambda: Arctanh(CF_X_SMALL)),  # Option A
        ],
        CF_TERM_COUNTS,
    ),
)
def test_bench_archyperbolic_cf_input(benchmark, factory, terms):
    result = benchmark(lambda: factory(terms))
    assert len(result) == terms


# Option A (ArctanMP) vs Option C (ArctanCF) for arctrig CF input
@pytest.mark.benchmark(group="arctrig-cf-input", max_time=0.1, min_rounds=3)
@pytest.mark.parametrize(
    ("factory", "terms"),
    _params(
        [
            ("arctan-cf", lambda: ArctanCF(CF_X)),  # Option C: meta-GCF
            ("arctan-mp", lambda: ArctanMP(CF_X)),  # Option A: mpmath fallback
            ("arcsin-mp", lambda: Arcsin(CF_X_SMALL)),  # Option A
            ("arccos-mp", lambda: Arccos(CF_X_SMALL)),  # Option A
            ("cos-mp", lambda: CosMP(CF_X)),  # Option A
            ("cos-cf", lambda: CosCF(CF_X)),  # Option B: 1/cos meta-GCF
            ("tan-mp", lambda: TanMP(CF_X)),  # Option A
        ],
        CF_TERM_COUNTS,
    ),
)
def test_bench_arctrig_cf_input(benchmark, factory, terms):
    result = benchmark(lambda: factory(terms))
    assert len(result) == terms
