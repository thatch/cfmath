"""Tests for quadratic irrationals (Sqrt) and quadratic surd arithmetic."""

import math
from fractions import Fraction

import pytest

from cfmath import CF, Sqrt, convergent
from cfmath.quadratic import (
    _cf_from_poly,
    _minimal_poly,
    _periodic_mul,
    _periodic_square,
    _square_free_decomp,
)


class TestSqrt:
    def test_sqrt_2_terms(self):
        cf = Sqrt(2)
        assert cf.terms == [1]
        assert cf.repeating == [2]

    def test_sqrt_3_terms(self):
        cf = Sqrt(3)
        assert cf.terms == [1]
        assert cf.repeating == [1, 2]

    def test_sqrt_5_terms(self):
        cf = Sqrt(5)
        assert cf.terms == [2]
        assert cf.repeating == [4]

    def test_sqrt_7_terms(self):
        cf = Sqrt(7)
        assert cf.terms == [2]
        assert cf.repeating == [1, 1, 1, 4]

    def test_perfect_square(self):
        assert Sqrt(4).terms == [2]
        assert Sqrt(4).repeating == []
        assert Sqrt(9).terms == [3]
        assert Sqrt(25).terms == [5]
        assert Sqrt(100).terms == [10]

    def test_sqrt_1_is_one(self):
        assert Sqrt(1).terms == [1]

    def test_sqrt_is_periodic(self):
        for n in [2, 3, 5, 6, 7, 8, 10, 11, 13]:
            cf = Sqrt(n)
            assert cf.is_periodic(), f"Sqrt({n}) should be periodic"

    def test_sqrt_convergent_bounds(self):
        """Convergents of sqrt(2) approach sqrt(2) from alternating sides."""
        cf = Sqrt(2)
        expected = [
            Fraction(1, 1),
            Fraction(3, 2),
            Fraction(7, 5),
            Fraction(17, 12),
            Fraction(41, 29),
            Fraction(99, 70),
        ]
        for i, exp in enumerate(expected):
            assert convergent(cf, i) == exp

    def test_sqrt_negative_raises(self):
        with pytest.raises(ValueError):
            Sqrt(-1)

    def test_sqrt_zero(self):
        assert Sqrt(0).terms == [0]

    def test_sqrt_value_close(self):
        for n in [2, 3, 5, 7, 10, 13, 17]:
            cf = Sqrt(n)
            val = float(convergent(cf.take(20), 19))
            assert abs(val - math.sqrt(n)) < 1e-8, f"Sqrt({n}) inaccurate"


class TestQuadraticSurds:
    def test_10_to_the_3_halves(self):
        """10^(3/2) = 10 * sqrt(10) ≈ 31.6228"""
        ten_sqrt10 = CF.from_int(10) * Sqrt(10)
        val = float(convergent(ten_sqrt10.take(20), 19))
        assert abs(val - 10 * math.sqrt(10)) < 1e-8

    def test_sqrt2_plus_sqrt3(self):
        """sqrt(2) + sqrt(3) ≈ 3.1462"""
        result = Sqrt(2).take(20) + Sqrt(3).take(20)
        val = float(convergent(result.take(20), 19))
        assert abs(val - (math.sqrt(2) + math.sqrt(3))) < 1e-8

    def test_sqrt2_times_sqrt3_approx_sqrt6(self):
        """sqrt(2) * sqrt(3) ≈ sqrt(6)"""
        result = Sqrt(2).take(20) * Sqrt(3).take(20)
        val = float(convergent(result.take(20), 19))
        assert abs(val - math.sqrt(6)) < 1e-6

    def test_sqrt6_direct_vs_product(self):
        """Sqrt(6) matches sqrt(2)*sqrt(3) to many decimal places."""
        v1 = float(convergent(Sqrt(6).take(20), 19))
        v2 = float(convergent(Sqrt(2).take(20) * Sqrt(3).take(20), 19))
        assert abs(v1 - v2) < 1e-8

    def test_1_plus_sqrt5_over_2_is_phi(self):
        """(1 + sqrt(5)) / 2 = φ ≈ 1.618"""
        one_plus_sqrt5 = CF.from_int(1) + Sqrt(5).take(20)
        phi_approx = CF.from_fraction(1, 2) * one_plus_sqrt5
        val = float(convergent(phi_approx.take(25), 24))
        assert abs(val - (1 + math.sqrt(5)) / 2) < 1e-8

    def test_sqrt_large_composite(self):
        """Sqrt(1000) ≈ 31.6228 (same as 10^(3/2))"""
        val = float(convergent(Sqrt(1000).take(20), 19))
        assert abs(val - math.sqrt(1000)) < 1e-8

    def test_silver_ratio(self):
        """Silver ratio = 1 + sqrt(2) ≈ 2.4142"""
        silver = CF.from_int(1) + Sqrt(2).take(20)
        val = float(convergent(silver.take(20), 19))
        assert abs(val - (1 + math.sqrt(2))) < 1e-8


class TestQuadraticHelpers:
    def test_minimal_poly_for_non_periodic_returns_none(self):
        assert _minimal_poly(CF.from_int(2)) is None

    def test_minimal_poly_for_sqrt2(self):
        A, B, C = _minimal_poly(Sqrt(2))
        assert A * 2 + B * math.sqrt(2) + C == 0
        assert (A, B, C) == (-1, 0, 2)

    def test_square_free_decomp(self):
        assert _square_free_decomp(72) == (2, 6)
        assert _square_free_decomp(13) == (13, 1)

    def test_cf_from_poly_negative_discriminant_returns_none(self):
        assert _cf_from_poly(1, 0, 1) is None

    def test_cf_from_poly_rational_roots(self):
        assert _cf_from_poly(1, -3, 2) == CF.from_int(2)
        assert _cf_from_poly(1, 0, 0) == CF.from_int(0)

    def test_cf_from_poly_sqrt2(self):
        cf = _cf_from_poly(1, 0, -2)
        assert cf.terms == Sqrt(2).terms
        assert cf.repeating == Sqrt(2).repeating

    def test_periodic_square_non_periodic_returns_none(self):
        assert _periodic_square(CF.from_int(2)) is None

    def test_periodic_square_sqrt2_is_exact(self):
        assert _periodic_square(Sqrt(2)) == CF.from_int(2)

    def test_periodic_mul_non_periodic_returns_none(self):
        assert _periodic_mul(CF.from_int(2), Sqrt(2)) is None

    def test_periodic_mul_different_fields_returns_none(self):
        assert _periodic_mul(Sqrt(2), Sqrt(3)) is None

    def test_periodic_mul_same_field(self):
        assert _periodic_mul(Sqrt(8), Sqrt(2)) == CF.from_int(4)
