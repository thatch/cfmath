"""Tests for logarithmic functions."""

import math
from fractions import Fraction

import pytest

from cfmath import CF, Ln, Log, Log2, Log10, convergent
from cfmath.logarithm import _ln_terms_from_decimal, _ln_terms_from_mpmath


class TestLn:
    def test_ln_1_is_zero(self):
        assert Ln(1) == CF.from_int(0)

    def test_ln_2_first_terms(self):
        assert list(Ln(2).take(8)) == [0, 1, 2, 3, 1, 6, 3, 1]

    def test_ln_2_value(self):
        val = float(convergent(Ln(2).take(15), 14))
        assert abs(val - math.log(2)) < 1e-8

    def test_ln_3_value(self):
        val = float(convergent(Ln(3).take(15), 14))
        assert abs(val - math.log(3)) < 1e-8

    def test_ln_fraction(self):
        val = float(convergent(Ln(Fraction(3, 2)).take(15), 14))
        assert abs(val - math.log(1.5)) < 1e-8

    def test_ln_nonpositive_raises(self):
        with pytest.raises(ValueError):
            Ln(0)
        with pytest.raises(ValueError):
            Ln(-1)

    def test_ln_bad_type_raises(self):
        with pytest.raises(TypeError):
            Ln(1.5)

    def test_ln_additivity(self):
        """ln(a*b) = ln(a) + ln(b) via CF arithmetic."""
        ln6 = Ln(6)
        ln2_plus_ln3 = Ln(2).take(20) + Ln(3).take(20)
        v1 = float(convergent(ln6.take(15), 14))
        v2 = float(convergent(ln2_plus_ln3.take(20), 19))
        assert abs(v1 - v2) < 1e-6

    def test_ln_change_of_base(self):
        """log_2(123) = Ln(123) / Ln(2) == Log2(123)."""
        log2_123 = Ln(123) / Ln(2)
        val = float(convergent(log2_123.take(20), 19))
        assert abs(val - math.log2(123)) < 1e-8
        assert log2_123.take(12) == Log2(123).take(12)

    def test_ln_decimal_matches_mpmath(self):
        """Decimal and mpmath backends produce identical CF terms."""
        for x_num, x_den in [(2, 1), (3, 1), (355, 113)]:
            dec = _ln_terms_from_decimal(x_num, x_den, 30)
            mpm = _ln_terms_from_mpmath(x_num, x_den, 30)
            assert dec == mpm, f"Ln({x_num}/{x_den}): decimal vs mpmath mismatch"


class TestLog2:
    def test_log2_value(self):
        for n in (3, 10, 123):
            val = float(convergent(Log2(n).take(15), 14))
            assert abs(val - math.log2(n)) < 1e-8, f"Log2({n})"

    def test_log2_not_periodic(self):
        assert not Log2(123).is_periodic()


class TestLog10:
    def test_log10_exact(self):
        assert Log10(10) == CF.from_int(1)
        assert Log10(100) == CF.from_int(2)

    def test_log10_value(self):
        val = float(convergent(Log10(2).take(20), 19))
        assert abs(val - math.log10(2)) < 1e-8


class TestLog:
    def test_log_no_base_is_ln(self):
        v1 = float(convergent(Log(2).take(20), 19))
        v2 = float(convergent(Ln(2).take(20), 19))
        assert abs(v1 - v2) < 1e-14

    def test_log_base_2_exact(self):
        assert Log(8, 2) == CF.from_int(3)

    def test_log_base_10_exact(self):
        assert Log(1000, 10) == CF.from_int(3)

    def test_log_bad_base_type(self):
        with pytest.raises(TypeError):
            Log(2, 1.5)

    def test_log_bad_base_value(self):
        with pytest.raises(ValueError):
            Log(2, 1)
        with pytest.raises(ValueError):
            Log(2, 0)

    def test_log_value(self):
        val = float(convergent(Log(3, 2).take(20), 19))
        assert abs(val - math.log(3, 2)) < 1e-8
