"""Tests for trigonometric functions."""

import math
from fractions import Fraction

import pytest

from cfmath import Sin, Cos, Tan, convergent
from cfmath.trig import _sin_terms_mpmath, _cos_terms_mpmath, _tan_terms_mpmath


def _gcf_terms(fn, x: Fraction, n: int) -> list[int]:
    """Get n CF terms from a trig function (GCF backend)."""
    return list(fn(x).take(n))


class TestTan:
    def test_tan_zero(self):
        assert Tan(0).terms == [0]

    def test_tan_bad_type_raises(self):
        with pytest.raises(TypeError):
            Tan(1.5)

    def test_tan_first_terms(self):
        t = Tan(Fraction(1, 4))
        assert list(t.take(8)) == [0, 3, 1, 10, 1, 18, 1, 26]

    def test_tan_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(1, 3)):
            val = float(convergent(Tan(x).take(15), 14))
            assert abs(val - math.tan(float(x))) < 1e-8, f"Tan({x})"

    def test_tan_negative(self):
        """tan(-x) = -tan(x)"""
        pos = float(convergent(Tan(Fraction(1, 4)).take(15), 14))
        neg = float(convergent(Tan(Fraction(-1, 4)).take(15), 14))
        assert abs(pos + neg) < 1e-8

    def test_tan_gcf_matches_mpmath(self):
        """GCF backend agrees with mpmath on CF terms."""
        for x in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)):
            gcf = _gcf_terms(Tan, x, 20)
            mpm = _tan_terms_mpmath(x.numerator, x.denominator, 20)
            assert gcf == mpm, f"Tan({x}): GCF vs mpmath mismatch"


class TestSin:
    def test_sin_zero(self):
        assert Sin(0).terms == [0]

    def test_sin_bad_type_raises(self):
        with pytest.raises(TypeError):
            Sin(1.5)

    def test_sin_first_terms(self):
        s = Sin(Fraction(1, 2))
        assert list(s.take(8)) == [0, 2, 11, 1, 1, 1, 6, 2]

    def test_sin_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(1, 3), Fraction(123, 1000)):
            val = float(convergent(Sin(x).take(15), 14))
            assert abs(val - math.sin(float(x))) < 1e-8, f"Sin({x})"

    def test_sin_negative(self):
        """sin(-x) = -sin(x)"""
        pos = float(convergent(Sin(Fraction(1, 3)).take(15), 14))
        neg = float(convergent(Sin(Fraction(-1, 3)).take(15), 14))
        assert abs(pos + neg) < 1e-8

    def test_sin_gcf_matches_mpmath(self):
        """GCF backend agrees with mpmath on CF terms."""
        for x in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)):
            gcf = _gcf_terms(Sin, x, 20)
            mpm = _sin_terms_mpmath(x.numerator, x.denominator, 20)
            assert gcf == mpm, f"Sin({x}): GCF vs mpmath mismatch"


class TestCos:
    def test_cos_zero(self):
        assert Cos(0).terms == [1]

    def test_cos_bad_type_raises(self):
        with pytest.raises(TypeError):
            Cos(1.5)

    def test_cos_first_terms(self):
        c = Cos(Fraction(1, 2))
        assert list(c.take(8)) == [0, 1, 7, 5, 1, 12, 2, 1]

    def test_cos_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(1, 3), Fraction(123, 1000)):
            val = float(convergent(Cos(x).take(15), 14))
            assert abs(val - math.cos(float(x))) < 1e-8, f"Cos({x})"

    def test_cos_even(self):
        """cos(-x) = cos(x)"""
        pos = float(convergent(Cos(Fraction(1, 3)).take(15), 14))
        neg = float(convergent(Cos(Fraction(-1, 3)).take(15), 14))
        assert abs(pos - neg) < 1e-8

    def test_pythagorean_identity(self):
        """sin²(x) + cos²(x) ≈ 1 for rational x."""
        for x in (Fraction(1, 4), Fraction(1, 2)):
            s2 = Sin(x).take(20) * Sin(x).take(20)
            c2 = Cos(x).take(20) * Cos(x).take(20)
            val = float(convergent((s2 + c2).take(15), 14))
            assert abs(val - 1.0) < 1e-6, f"sin²+cos² ≠ 1 for x={x}"

    def test_cos_gcf_matches_mpmath(self):
        """GCF backend agrees with mpmath on CF terms."""
        for x in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2)):
            gcf = _gcf_terms(Cos, x, 20)
            mpm = _cos_terms_mpmath(x.numerator, x.denominator, 20)
            assert gcf == mpm, f"Cos({x}): GCF vs mpmath mismatch"
