"""Tests for hyperbolic functions."""

import math
from fractions import Fraction

import pytest

from cfmath import Arccosh, Arcsinh, Arctanh, Cosh, Sinh, Tanh, convergent
from cfmath.archyperbolic import (
    _arccosh_terms_from_decimal,
    _arccosh_terms_mpmath,
    _arcsinh_terms_from_decimal,
    _arcsinh_terms_mpmath,
    _arctanh_terms_mpmath,
)
from cfmath.hyperbolic import (
    _cosh_terms_from_decimal,
    _cosh_terms_mpmath,
    _sinh_terms_from_decimal,
    _sinh_terms_mpmath,
    _tanh_terms_mpmath,
)


class TestSinh:
    def test_sinh_zero(self):
        assert Sinh(0).terms == [0]

    def test_sinh_bad_type_raises(self):
        with pytest.raises(TypeError):
            Sinh(1.5)

    def test_sinh_first_terms(self):
        s = Sinh(1)
        assert list(s.take(8)) == [1, 5, 1, 2, 2, 2, 1, 2]

    def test_sinh_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(1)):
            val = float(convergent(Sinh(x).take(15), 14))
            assert abs(val - math.sinh(float(x))) < 1e-8, f"Sinh({x})"

    def test_sinh_negative(self):
        """sinh(-x) = -sinh(x)"""
        pos = float(convergent(Sinh(Fraction(1, 2)).take(15), 14))
        neg = float(convergent(Sinh(Fraction(-1, 2)).take(15), 14))
        assert abs(pos + neg) < 1e-8


class TestCosh:
    def test_cosh_zero(self):
        assert Cosh(0).terms == [1]

    def test_cosh_bad_type_raises(self):
        with pytest.raises(TypeError):
            Cosh(1.5)

    def test_cosh_first_terms(self):
        c = Cosh(1)
        assert list(c.take(8)) == [1, 1, 1, 5, 3, 3, 2, 1]

    def test_cosh_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(1)):
            val = float(convergent(Cosh(x).take(15), 14))
            assert abs(val - math.cosh(float(x))) < 1e-8, f"Cosh({x})"

    def test_cosh_even(self):
        """cosh(-x) = cosh(x)"""
        pos = float(convergent(Cosh(Fraction(1, 2)).take(15), 14))
        neg = float(convergent(Cosh(Fraction(-1, 2)).take(15), 14))
        assert abs(pos - neg) < 1e-8

    def test_hyperbolic_identity(self):
        """cosh²(x) - sinh²(x) = 1"""
        x = Fraction(1, 2)
        c2 = Cosh(x).take(15) * Cosh(x).take(15)
        s2 = Sinh(x).take(15) * Sinh(x).take(15)
        val = float(convergent((c2 - s2).take(15), 14))
        assert abs(val - 1.0) < 1e-6


class TestTanh:
    def test_tanh_zero(self):
        assert Tanh(0).terms == [0]

    def test_tanh_bad_type_raises(self):
        with pytest.raises(TypeError):
            Tanh(1.5)

    def test_tanh_first_terms(self):
        t = Tanh(1)
        assert list(t.take(8)) == [0, 1, 3, 5, 7, 9, 11, 13]

    def test_tanh_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(2)):
            val = float(convergent(Tanh(x).take(20), 19))
            assert abs(val - math.tanh(float(x))) < 1e-8, f"Tanh({x})"

    def test_tanh_negative(self):
        """tanh(-x) = -tanh(x)"""
        pos = float(convergent(Tanh(Fraction(1, 3)).take(15), 14))
        neg = float(convergent(Tanh(Fraction(-1, 3)).take(15), 14))
        assert abs(pos + neg) < 1e-8

    def test_tanh_bounded(self):
        """tanh is always in (-1, 1)"""
        for x in (Fraction(1), Fraction(2), Fraction(5)):
            val = float(convergent(Tanh(x).take(20), 19))
            assert abs(val) < 1.0

    def test_tanh_gcf_matches_mpmath(self):
        """Lambert GCF backend agrees with mpmath on CF terms."""
        for x in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)):
            gcf = list(Tanh(x).take(20))
            mpm = _tanh_terms_mpmath(x.numerator, x.denominator, 20)
            assert gcf == mpm, f"Tanh({x}): GCF vs mpmath mismatch"

    def test_sinh_mpmath_matches_directly(self):
        """Sinh via mpmath terms match direct mpmath computation."""
        for x in (Fraction(1, 4), Fraction(1, 2)):
            a = list(Sinh(x).take(20))
            b = _sinh_terms_mpmath(x.numerator, x.denominator, 20)
            assert a == b, f"Sinh({x}) mismatch"

    def test_cosh_mpmath_matches_directly(self):
        """Cosh via mpmath terms match direct mpmath computation."""
        for x in (Fraction(1, 4), Fraction(1, 2)):
            a = list(Cosh(x).take(20))
            b = _cosh_terms_mpmath(x.numerator, x.denominator, 20)
            assert a == b, f"Cosh({x}) mismatch"


class TestDecimalBackends:
    def test_sinh_decimal_matches_mpmath(self):
        """Decimal Taylor series for sinh agrees with mpmath term-for-term."""
        for x in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(1)):
            dec = _sinh_terms_from_decimal(x.numerator, x.denominator, 30)
            mpm = _sinh_terms_mpmath(x.numerator, x.denominator, 30)
            assert dec == mpm, f"sinh({x}): decimal vs mpmath mismatch"

    def test_cosh_decimal_matches_mpmath(self):
        """Decimal Taylor series for cosh agrees with mpmath term-for-term."""
        for x in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(1)):
            dec = _cosh_terms_from_decimal(x.numerator, x.denominator, 30)
            mpm = _cosh_terms_mpmath(x.numerator, x.denominator, 30)
            assert dec == mpm, f"cosh({x}): decimal vs mpmath mismatch"


class TestArctanh:
    def test_arctanh_zero(self):
        assert Arctanh(0).terms == [0]

    def test_arctanh_bad_type_raises(self):
        with pytest.raises(TypeError):
            Arctanh(1.5)

    def test_arctanh_out_of_range_raises(self):
        for x in (Fraction(1), Fraction(-1), Fraction(2)):
            with pytest.raises(ValueError):
                Arctanh(x)

    def test_arctanh_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(-1, 3)):
            val = float(convergent(Arctanh(x).take(20), 19))
            assert abs(val - math.atanh(float(x))) < 1e-8, f"Arctanh({x})"

    def test_arctanh_matches_mpmath(self):
        for x in (Fraction(1, 4), Fraction(1, 2)):
            actual = list(Arctanh(x).take(20))
            expected = _arctanh_terms_mpmath(x.numerator, x.denominator, 20)
            assert actual == expected


class TestArcsinh:
    def test_arcsinh_zero(self):
        assert Arcsinh(0).terms == [0]

    def test_arcsinh_bad_type_raises(self):
        with pytest.raises(TypeError):
            Arcsinh(1.5)

    def test_arcsinh_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(-1, 3)):
            val = float(convergent(Arcsinh(x).take(20), 19))
            assert abs(val - math.asinh(float(x))) < 1e-8, f"Arcsinh({x})"

    def test_arcsinh_backends_match_mpmath(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(-1, 3)):
            dec = _arcsinh_terms_from_decimal(x.numerator, x.denominator, 20)
            mpm = _arcsinh_terms_mpmath(x.numerator, x.denominator, 20)
            assert dec == mpm, f"arcsinh({x}) backend mismatch"

    def test_arcsinh_decimal_reduction_branches_match_mpmath(self):
        for x in (Fraction(1, 20), Fraction(10), Fraction(-10)):
            dec = _arcsinh_terms_from_decimal(x.numerator, x.denominator, 18)
            mpm = _arcsinh_terms_mpmath(x.numerator, x.denominator, 18)
            assert dec == mpm, f"arcsinh({x}) reduction mismatch"


class TestArccosh:
    def test_arccosh_one(self):
        assert Arccosh(1).terms == [0]

    def test_arccosh_bad_type_raises(self):
        with pytest.raises(TypeError):
            Arccosh(1.5)

    def test_arccosh_out_of_range_raises(self):
        for x in (0, Fraction(1, 2), -1):
            with pytest.raises(ValueError):
                Arccosh(x)

    def test_arccosh_value(self):
        for x in (Fraction(3, 2), Fraction(2), Fraction(5)):
            val = float(convergent(Arccosh(x).take(20), 19))
            assert abs(val - math.acosh(float(x))) < 1e-8, f"Arccosh({x})"

    def test_arccosh_backends_match_mpmath(self):
        for x in (Fraction(3, 2), Fraction(2)):
            dec = _arccosh_terms_from_decimal(x.numerator, x.denominator, 20)
            mpm = _arccosh_terms_mpmath(x.numerator, x.denominator, 20)
            assert dec == mpm, f"arccosh({x}) backend mismatch"

    def test_arccosh_decimal_reduction_branches_match_mpmath(self):
        for x in (Fraction(11, 10), Fraction(10)):
            dec = _arccosh_terms_from_decimal(x.numerator, x.denominator, 18)
            mpm = _arccosh_terms_mpmath(x.numerator, x.denominator, 18)
            assert dec == mpm, f"arccosh({x}) reduction mismatch"
