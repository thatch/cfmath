"""Property-based tests using Hypothesis."""

from datetime import timedelta
from fractions import Fraction

from hypothesis import given, settings
from hypothesis import strategies as st

from cfmath import CF, convergent_pairs, convergents

# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

# Fractions with bounded numerator AND denominator to keep CFs short.
# st.fractions() with only min/max value can still generate huge denominators.
_fracs = st.builds(
    Fraction,
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=1, max_value=100),
)
_nonzero_fracs = _fracs.filter(lambda f: f != 0)


def _cf(frac: Fraction) -> CF:
    return CF.from_rational(frac)


def _last_convergent(cf: CF, max_terms: int = 60) -> Fraction:
    """Return the last convergent of cf, taking up to max_terms terms."""
    last: tuple[int, int] | None = None
    for last in convergent_pairs(cf.take(max_terms)):
        pass
    if last is None:
        raise ValueError("empty CF")
    p, q = last
    return Fraction(p, q)


def _positive_bounded_fracs():
    """Fractions in [1/10, 10] with small denominators."""
    return st.integers(min_value=1, max_value=100).flatmap(
        lambda q: st.builds(
            Fraction,
            st.integers(min_value=(q + 9) // 10, max_value=10 * q),
            st.just(q),
        )
    )


# ---------------------------------------------------------------------------
# Roundtrip: Fraction → CF → Fraction
# ---------------------------------------------------------------------------


class TestRoundtrip:
    @given(_fracs)
    @settings(deadline=timedelta(milliseconds=200))
    def test_fraction_roundtrip(self, f: Fraction):
        cf = CF.from_rational(f)
        assert cf.to_fraction() == f

    @given(st.integers(min_value=-1000, max_value=1000))
    @settings(deadline=timedelta(milliseconds=200))
    def test_int_roundtrip(self, n: int):
        cf = CF.from_int(n)
        assert cf.to_fraction() == Fraction(n)

    @given(_fracs)
    @settings(deadline=timedelta(milliseconds=200))
    def test_convergent_sequence_ends_at_value(self, f: Fraction):
        cf = CF.from_rational(f)
        convs = list(convergents(cf))
        assert convs[-1] == f


# ---------------------------------------------------------------------------
# Arithmetic matches Fraction arithmetic
# Uses to_fraction() since rational+rational=rational (CF is finite)
# ---------------------------------------------------------------------------


class TestArithmeticCorrectness:
    @given(_fracs, _fracs)
    @settings(max_examples=200, deadline=timedelta(milliseconds=500))
    def test_add_matches_fraction(self, a: Fraction, b: Fraction):
        ca = _cf(a)
        cb = _cf(b)
        # Sum of two rational CFs is a finite rational CF
        result_cf = ca + cb
        result = _last_convergent(result_cf)
        assert result == a + b

    @given(_fracs, _fracs)
    @settings(max_examples=200, deadline=timedelta(milliseconds=500))
    def test_sub_matches_fraction(self, a: Fraction, b: Fraction):
        ca = _cf(a)
        cb = _cf(b)
        result = _last_convergent(ca - cb)
        assert result == a - b

    @given(_fracs, _fracs)
    @settings(max_examples=200, deadline=timedelta(milliseconds=500))
    def test_mul_matches_fraction(self, a: Fraction, b: Fraction):
        ca = _cf(a)
        cb = _cf(b)
        result = _last_convergent(ca * cb)
        assert result == a * b

    @given(_fracs, _nonzero_fracs)
    @settings(max_examples=200, deadline=timedelta(milliseconds=500))
    def test_div_matches_fraction(self, a: Fraction, b: Fraction):
        ca = _cf(a)
        cb = _cf(b)
        result = _last_convergent(ca / cb)
        assert result == a / b


# ---------------------------------------------------------------------------
# Determinant identity: p_n * q_{n-1} - p_{n-1} * q_n = (-1)^(n-1)
# ---------------------------------------------------------------------------


class TestDeterminantIdentity:
    @given(st.lists(st.integers(min_value=1, max_value=20), min_size=2, max_size=10))
    @settings(deadline=timedelta(milliseconds=200))
    def test_determinant(self, coeffs: list[int]):
        # a0 can be 0; rest must be >= 1 for a simple CF
        cf = CF([0] + coeffs)
        pairs = list(convergent_pairs(cf))
        for n in range(1, len(pairs)):
            p_n, q_n = pairs[n]
            p_nm1, q_nm1 = pairs[n - 1]
            det = p_n * q_nm1 - p_nm1 * q_n
            assert det == (-1) ** (n - 1), f"Identity failed at n={n}, pairs={pairs}"


# ---------------------------------------------------------------------------
# Best-approximation bound: |p_n/q_n - x| ≤ 1/q_n^2
# ---------------------------------------------------------------------------


class TestBestApproximationBound:
    @given(
        _positive_bounded_fracs(),
        st.integers(min_value=0, max_value=6),
    )
    @settings(deadline=timedelta(milliseconds=200))
    def test_bound(self, f: Fraction, n: int):
        cf = CF.from_rational(f)
        convs_list = list(convergents(cf))
        if n >= len(convs_list):
            return
        c = convs_list[n]
        p, q = c.numerator, c.denominator
        err = abs(c - f)
        if q > 0:
            assert err <= Fraction(1, q * q), f"Bound 1/q² violated: |{p}/{q} - {f}| = {err} >= 1/{q}²"


# ---------------------------------------------------------------------------
# CF constructed from list equals CF from Fraction
# ---------------------------------------------------------------------------


class TestCFConstruction:
    @given(_fracs)
    @settings(deadline=timedelta(milliseconds=200))
    def test_from_fraction_vs_from_rational(self, f: Fraction):
        a = CF.from_fraction(f.numerator, f.denominator)
        b = CF.from_rational(f)
        assert a.to_fraction() == b.to_fraction()

    @given(_fracs)
    @settings(deadline=timedelta(milliseconds=200))
    def test_is_finite(self, f: Fraction):
        assert CF.from_rational(f).is_finite()

    @given(st.integers(min_value=1, max_value=1000))
    @settings(deadline=timedelta(milliseconds=200))
    def test_positive_cf_terms(self, n: int):
        """All CF terms except the first must be positive."""
        f = Fraction(n, n + 1)  # always proper fraction
        cf = CF.from_rational(f)
        terms = cf.terms
        if len(terms) > 1:
            for a in terms[1:]:
                assert a >= 1, f"Non-first term {a} must be >= 1"


# ---------------------------------------------------------------------------
# Negation: cf + (-cf) == 0
# ---------------------------------------------------------------------------


class TestNegation:
    @given(_fracs)
    @settings(deadline=timedelta(milliseconds=200))
    def test_add_neg_is_zero(self, f: Fraction):
        cf = _cf(f)
        result = _last_convergent(cf + (-cf))
        assert result == Fraction(0)


# ---------------------------------------------------------------------------
# Commutativity
# ---------------------------------------------------------------------------


class TestAlgebraicLaws:
    @given(_fracs, _fracs)
    @settings(deadline=timedelta(milliseconds=200))
    def test_add_commutative(self, a: Fraction, b: Fraction):
        ca, cb = _cf(a), _cf(b)
        lhs = _last_convergent(ca + cb)
        rhs = _last_convergent(cb + ca)
        assert lhs == rhs

    @given(_fracs, _fracs)
    @settings(deadline=timedelta(milliseconds=200))
    def test_mul_commutative(self, a: Fraction, b: Fraction):
        ca, cb = _cf(a), _cf(b)
        lhs = _last_convergent(ca * cb)
        rhs = _last_convergent(cb * ca)
        assert lhs == rhs
