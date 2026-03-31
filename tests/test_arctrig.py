"""Tests for inverse trigonometric functions."""

import math
from fractions import Fraction

import pytest

from cfmath import Arctan, Arcsin, Arccos, convergent
from cfmath.arctrig import _arctan_terms_mpmath, _arcsin_terms_mpmath, _arccos_terms_mpmath


class TestArctan:
    def test_arctan_zero(self):
        assert Arctan(0).terms == [0]

    def test_arctan_bad_type_raises(self):
        with pytest.raises(TypeError):
            Arctan(1.5)

    def test_arctan_first_terms(self):
        t = Arctan(Fraction(1, 4))
        assert list(t.take(8)) == [0, 4, 12, 5, 12, 1, 1, 1]

    def test_arctan_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(1, 3), Fraction(1)):
            val = float(convergent(Arctan(x).take(20), 19))
            assert abs(val - math.atan(float(x))) < 1e-8, f"Arctan({x})"

    def test_arctan_negative(self):
        """arctan(-x) = -arctan(x)"""
        pos = float(convergent(Arctan(Fraction(1, 3)).take(15), 14))
        neg = float(convergent(Arctan(Fraction(-1, 3)).take(15), 14))
        assert abs(pos + neg) < 1e-8

    def test_arctan_pi_over_4(self):
        """arctan(1) = π/4"""
        val = float(convergent(Arctan(1).take(20), 19))
        assert abs(val - math.pi / 4) < 1e-8

    def test_arctan_gcf_matches_mpmath(self):
        """Gauss GCF backend agrees with mpmath on CF terms."""
        for x in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)):
            gcf = list(Arctan(x).take(20))
            mpm = _arctan_terms_mpmath(x.numerator, x.denominator, 20)
            assert gcf == mpm, f"Arctan({x}): GCF vs mpmath mismatch"


class TestArcsin:
    def test_arcsin_zero(self):
        assert Arcsin(0).terms == [0]

    def test_arcsin_bad_type_raises(self):
        with pytest.raises(TypeError):
            Arcsin(1.5)

    def test_arcsin_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Arcsin(Fraction(3, 2))

    def test_arcsin_first_terms(self):
        s = Arcsin(Fraction(1, 2))
        assert list(s.take(8)) == [0, 1, 1, 10, 10, 1, 1, 1]

    def test_arcsin_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(1, 3), Fraction(3, 4)):
            val = float(convergent(Arcsin(x).take(20), 19))
            assert abs(val - math.asin(float(x))) < 1e-8, f"Arcsin({x})"

    def test_arcsin_negative(self):
        """arcsin(-x) = -arcsin(x)"""
        pos = float(convergent(Arcsin(Fraction(1, 3)).take(20), 19))
        neg = float(convergent(Arcsin(Fraction(-1, 3)).take(20), 19))
        assert abs(pos + neg) < 1e-8

    def test_arcsin_gcf_matches_mpmath(self):
        """Euler GCF backend agrees with mpmath on CF terms."""
        for x in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)):
            gcf = list(Arcsin(x).take(20))
            mpm = _arcsin_terms_mpmath(x.numerator, x.denominator, 20)
            assert gcf == mpm, f"Arcsin({x}): GCF vs mpmath mismatch"


class TestArccos:
    def test_arccos_one(self):
        assert Arccos(1).terms == [0]

    def test_arccos_bad_type_raises(self):
        with pytest.raises(TypeError):
            Arccos(1.5)

    def test_arccos_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Arccos(Fraction(3, 2))

    def test_arccos_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(1, 3), Fraction(3, 4)):
            val = float(convergent(Arccos(x).take(25), 24))
            assert abs(val - math.acos(float(x))) < 1e-6, f"Arccos({x})"

    def test_arccos_complementary(self):
        """arcsin(x) + arccos(x) = π/2"""
        x = Fraction(1, 3)
        s = float(convergent(Arcsin(x).take(20), 19))
        c = float(convergent(Arccos(x).take(25), 24))
        assert abs(s + c - math.pi / 2) < 1e-6

    def test_arccos_gcf_matches_mpmath(self):
        """π/2 - Arcsin backend agrees with mpmath on CF terms."""
        for x in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)):
            gcf = list(Arccos(x).take(20))
            mpm = _arccos_terms_mpmath(x.numerator, x.denominator, 20)
            assert gcf == mpm, f"Arccos({x}): GCF vs mpmath mismatch"
