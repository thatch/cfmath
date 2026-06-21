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


class TestMetaCFStall:
    """ExpCF via the meta-CF backend cannot *prove* an exact-rational result.

    Exp(Ln(2)) = 2 exactly.  The meta-CF value lands on an integer boundary, and
    an interval corner check can never confirm a value sitting exactly on the
    boundary (the bracket always straddles it).  "Gimme" mode accepts the value
    as that boundary once the suppressed partial quotient would have at least
    gimme_min_term_digits digits; with gimme=None it raises instead.
    """

    def test_expcf_gimme_returns_exact(self):
        """Default (gimme on) returns the exact rational Exp(Ln(2)) = 2."""
        from cfmath.exponential import ExpCF

        assert ExpCF(Ln(2)).take(4).terms == [2]
        # integer-part extraction path too: Exp(Ln(3)) = 3
        assert ExpCF(Ln(3)).take(4).terms == [3]

    def test_cf_metacf_repro_gimme(self):
        """The minimal repro now resolves to [2] instead of spinning."""
        import cfmath.gosper as gosper
        from cfmath.exponential import _halfexpm1_metaCF_terms

        assert gosper.cf_metaCF(1 / Ln(2), _halfexpm1_metaCF_terms()).take(4).terms == [2]

    def test_gimme_disabled_raises(self, monkeypatch):
        """gimme_min_term_digits=None restores the raise-on-stall behavior."""
        import cfmath.gosper as gosper
        from cfmath.exponential import ExpCF, _halfexpm1_metaCF_terms

        monkeypatch.setattr(gosper, "_METACF_STALL_LIMIT", 8)
        with pytest.raises(ArithmeticError, match="metaCF stalled"):
            ExpCF(Ln(2), gimme_min_term_digits=None).take(4)
        with pytest.raises(ArithmeticError, match="metaCF stalled"):
            gosper.cf_metaCF(1 / Ln(2), _halfexpm1_metaCF_terms(), gimme_min_term_digits=None).take(4)

    def test_simple_mode_still_raises(self):
        """The slow reference path does not support gimme; it raises."""
        from cfmath.exponential import ExpCF

        with pytest.raises(ArithmeticError, match="metaCF stalled"):
            ExpCF(Ln(2), mode="simple").take(4)

    def test_gimme_does_not_misfire_on_irrational(self):
        """A genuine irrational result is computed exactly, gimme never firing."""
        import mpmath
        from cfmath.exponential import ExpCF

        mpmath.mp.dps = 80
        got = ExpCF(Fraction(2, 3)).take(15).terms
        v = mpmath.exp(mpmath.mpf(2) / 3)
        truth = []
        for _ in range(15):
            a = int(mpmath.floor(v))
            truth.append(a)
            v = 1 / (v - a)
        assert got == truth[: len(got)]


class TestGimmeThreshold:
    """The threshold must clear the largest partial quotient of any *legitimate*
    value, not just famous near-integers.

    Ramanujan's constant e^(pi*sqrt(163)) is an integer to ~12 digits;
    fib(360)/fib(216) is a near-integer (~Lucas(144)) to ~30 digits, with a
    31-digit partial quotient.  The default must sit above both.
    """

    def test_default_clears_fibonacci_near_integer(self):
        from cfmath.gosper import _GIMME_MIN_TERM_DIGITS

        # fib(360)/fib(216) has a 31-digit partial quotient; Ramanujan ~12.
        assert _GIMME_MIN_TERM_DIGITS > 31

    def test_threshold_is_load_bearing(self):
        """A value with a ~40-digit partial quotient (fib-scale) is preserved at
        the default but clobbered by a too-low threshold."""
        from cfmath.core import CF
        from cfmath.exponential import ExpCF

        # Infinite input ~ Ln(2); e^x is just below 2 with a ~40-digit term,
        # i.e. a genuine irrational, not an exact rational.
        x = CF(Ln(2).take(48).terms, repeating=[1])
        reference = ExpCF(x, gimme_min_term_digits=200).take(6).terms
        assert len(str(reference[2])) >= 40  # the legitimate large term
        # Default (50 > 40): the term is faithfully reproduced.
        assert ExpCF(x).take(6).terms == reference
        # Too low (20 < 40): the value is wrongly rationalized to [2].
        assert ExpCF(x, gimme_min_term_digits=20).take(6).terms == [2]


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
