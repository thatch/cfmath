"""Tests for power and cube-root functions."""

import math
from fractions import Fraction

import pytest

from cfmath import CF, Sqrt, Pow, Cuberoot, convergent, convergents
from cfmath.power import _cbrt_terms_from_decimal, _cbrt_terms_from_mpmath


class TestCuberoot:
    def test_cuberoot_8_is_2(self):
        cf = Cuberoot(8)
        convs = list(convergents(cf))
        val = float(convs[-1])
        assert abs(val - 2.0) < 1e-8

    def test_cuberoot_2_value(self):
        cf = Cuberoot(2)
        val = float(convergent(cf.take(15), 14))
        expected = 2 ** (1 / 3)
        assert abs(val - expected) < 1e-8

    def test_cuberoot_27_is_3(self):
        cf = Cuberoot(27)
        convs = list(convergents(cf))
        val = float(convs[-1])
        assert abs(val - 3.0) < 1e-8

    def test_cuberoot_decimal_matches_mpmath(self):
        """Decimal Newton and mpmath backends agree on CF terms."""
        for n in (2, 3, 5, 7):
            dec = _cbrt_terms_from_decimal(n, 30)
            mpm = _cbrt_terms_from_mpmath(n, 30)
            assert dec == mpm, f"Cuberoot({n}): decimal vs mpmath mismatch"


class TestPow:
    def test_pow_zero_exponent(self):
        assert Pow(5, 0).terms == [1]

    def test_pow_one_exponent(self):
        assert Pow(Fraction(3, 4), 1) == CF.from_rational(Fraction(3, 4))

    def test_pow_integer_exponent_exact(self):
        assert Pow(2, 3) == CF.from_int(8)
        assert Pow(Fraction(2, 3), 2) == CF.from_rational(Fraction(4, 9))

    def test_pow_negative_integer_exponent(self):
        assert Pow(2, -3) == CF.from_rational(Fraction(1, 8))

    def test_pow_half_dispatches_to_sqrt(self):
        result = Pow(2, Fraction(1, 2))
        sq = Sqrt(2)
        assert result.terms == sq.terms
        assert result.repeating == sq.repeating

    def test_pow_fractional_base_half_exponent_periodic(self):
        result = Pow(Fraction(9, 2), Fraction(3, 2))
        assert result.is_periodic()
        val = float(convergent(result.take(20), 19))
        assert abs(val - (4.5 ** 1.5)) < 1e-10

    def test_pow_third_dispatches_to_cuberoot(self):
        assert Pow(8, Fraction(1, 3)).terms == [2]
        v1 = float(convergent(Pow(2, Fraction(1, 3)).take(20), 19))
        v2 = float(convergent(Cuberoot(2).take(20), 19))
        assert abs(v1 - v2) < 1e-12

    def test_pow_general_fractional(self):
        val = float(convergent(Pow(2, Fraction(3, 2)).take(20), 19))
        assert abs(val - 2 ** 1.5) < 1e-8

    def test_pow_bad_type_base(self):
        with pytest.raises(TypeError):
            Pow(1.5, Fraction(1, 2))

    def test_pow_bad_type_exponent(self):
        with pytest.raises(TypeError):
            Pow(2, 0.5)

    def test_pow_nonpositive_base(self):
        with pytest.raises(ValueError):
            Pow(0, Fraction(1, 2))
        with pytest.raises(ValueError):
            Pow(-1, Fraction(1, 2))
