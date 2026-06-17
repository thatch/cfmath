"""Tests for the exponential function."""

import math
from fractions import Fraction

import pytest

from cfmath import CF, E, EulerGamma, Exp, Ln, Pi, convergent
from cfmath.exponential import ExpCF, ExpCFMode, ExpMP, _exp_terms_from_decimal, _exp_terms_from_mpmath
from cfmath.quadratic import Sqrt


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


class _TestExpCF:
    N_TERMS = 80
    pi = Pi()
    pi.take(N_TERMS)

    def _test_exp_cf_1_matches_e(self):
        """ExpCF(1) should give the same value as E()."""
        cf0 = E()
        cf1 = ExpCF(CF.from_int(1), self.mode)
        assert cf0 == cf1

    def _test_exp_cf_matches_mpmath(self):
        """MetaCF agrees with mpmath term-for-term."""
        for val in (1 / Sqrt(2), Sqrt(2), Pi(), EulerGamma() + Pi() + Sqrt(2)):
            y0 = ExpMP(val).take(self.N_TERMS).terms
            y1 = ExpCF(val, self.mode).take(self.N_TERMS).terms
            assert y0 == y1

    def _exp_take(self, inp, n_terms):
        """Take n_terms of ExpCF(inp)."""
        return ExpCF(inp, self.mode).take(n_terms)

    def _test_exp_cf_exp_pi(self, benchmark):
        """Benchmark ExpCF(Pi())."""
        n_terms = self.N_TERMS // 4
        cf_terms = benchmark(self._exp_take, self.pi, n_terms)
        mp_terms = ExpMP(self.pi).take(n_terms)
        assert cf_terms == mp_terms


class TestExpCFSimple(_TestExpCF):
    mode = ExpCFMode.SIMPLE
    test_exp_simple_1_matches_e = _TestExpCF._test_exp_cf_1_matches_e
    test_exp_simple_matches_mpmath = _TestExpCF._test_exp_cf_matches_mpmath
    test_exp_simple_exp_pi = _TestExpCF._test_exp_cf_exp_pi


class TestExpCFPoly(_TestExpCF):
    mode = ExpCFMode.POLY
    test_exp_poly_1_matches_e = _TestExpCF._test_exp_cf_1_matches_e
    test_exp_poly_matches_mpmath = _TestExpCF._test_exp_cf_matches_mpmath
    test_exp_poly_exp_pi = _TestExpCF._test_exp_cf_exp_pi


class TestExpMpmath(_TestExpCF):
    def _exp_take(self, inp, n_terms):
        return ExpMP(inp).take(n_terms)

    def test_exp_mpmath_exp_pi(self, benchmark):
        """Benchmark ExpMP(Pi()) as a control."""
        n_terms = self.N_TERMS // 4
        benchmark(self._exp_take, self.pi, n_terms)


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
