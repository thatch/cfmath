"""Tests for the exponential function."""

import math
from fractions import Fraction

import pytest

from cfmath import CF, E, Exp, Ln, convergent
from cfmath.exponential import _exp_terms_from_decimal, _exp_terms_from_mpmath


class TestExp:
    def test_exp_zero(self):
        assert Exp(0) == CF.from_int(1)

    def test_exp_bad_type(self):
        with pytest.raises(TypeError):
            Exp(1.5)

    def test_exp_1_matches_e(self):
        """Exp(1) should give the same value as E()."""
        val_exp = float(convergent(Exp(1).take(15), 14))
        val_e = float(convergent(E().take(15), 14))
        assert abs(val_exp - val_e) < 1e-10

    def test_exp_half(self):
        """Exp(1/2) = sqrt(e) ≈ 1.6487..."""
        val = float(convergent(Exp(Fraction(1, 2)).take(15), 14))
        assert abs(val - math.exp(0.5)) < 1e-8

    def test_exp_value(self):
        for x in (Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(2, 3)):
            val = float(convergent(Exp(x).take(15), 14))
            assert abs(val - math.exp(float(x))) < 1e-8, f"Exp({x})"

    def test_exp_negative(self):
        """Exp(-1/2) = 1/sqrt(e)"""
        val = float(convergent(Exp(Fraction(-1, 2)).take(15), 14))
        assert abs(val - math.exp(-0.5)) < 1e-8

    def test_exp_ln_inverse(self):
        """Exp(Ln(x)) ≈ x for rational x > 0."""
        for x in (2, 3, 5):
            val = float(convergent(Exp(Ln(x)).take(20), 19))
            assert abs(val - x) < 1e-8, f"Exp(Ln({x})) ≠ {x}"

    def test_exp_cf_input(self):
        """Exp accepts a CF argument (for Pow(x, r) via Exp(r * Ln(x)))."""
        # 2^(3/2) = Exp(3/2 * Ln(2))
        result = Exp(Fraction(3, 2) * Ln(2))
        val = float(convergent(result.take(20), 19))
        assert abs(val - 2**1.5) < 1e-8


class TestDecimalBackend:
    def test_exp_decimal_matches_mpmath(self):
        """Decimal Taylor series for exp agrees with mpmath term-for-term."""
        for x in (
            Fraction(1, 4),
            Fraction(1, 3),
            Fraction(1, 2),
            Fraction(1),
            Fraction(2, 3),
        ):
            dec = _exp_terms_from_decimal(x.numerator, x.denominator, 30)
            mpm = _exp_terms_from_mpmath(x.numerator, x.denominator, 30)
            assert dec == mpm, f"exp({x}): decimal vs mpmath mismatch"

    def test_exp_decimal_negative(self):
        """Decimal backend handles negative exponents."""
        for x in (Fraction(-1, 4), Fraction(-1, 2)):
            dec = _exp_terms_from_decimal(x.numerator, x.denominator, 30)
            mpm = _exp_terms_from_mpmath(x.numerator, x.denominator, 30)
            assert dec == mpm, f"exp({x}): decimal vs mpmath mismatch"
