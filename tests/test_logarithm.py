"""Tests for logarithmic functions."""

import math
from fractions import Fraction

import pytest

from cfmath import CF, Ln, Log, Log2, Log10, convergent
from cfmath.logarithm import _ln_terms_from_decimal, _ln_terms_from_mpmath
from cfmath.quadratic import Sqrt
from cfmath import Pi
from cfmath.logarithm import LnCF, Log10CF, Log2CF, LogCF, _ln1p_cf


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

    def test_ln_cf_sqrt2_equals_half_ln2(self):
        """Ln(sqrt(2)) = Ln(2)/2 since ln(√2) = ½·ln(2)."""
        v1 = float(convergent(Ln(Sqrt(2)).take(15), 14))
        v2 = float(convergent((Ln(2) / 2).take(15), 14))
        assert abs(v1 - v2) < 1e-10

    def test_ln_cf_value(self):
        """Ln(Sqrt(2)) ≈ 0.3466."""
        val = float(convergent(Ln(Sqrt(2)).take(15), 14))
        assert abs(val - math.log(2) / 2) < 1e-10

    def test_ln_cf_pow(self):
        """2^sqrt(2) = Exp(Sqrt(2) * Ln(2)) — full pow(CF, CF) pipeline."""
        from cfmath.exponential import Exp

        result = float(convergent(Exp(Sqrt(2) * Ln(2)).take(15), 14))
        assert abs(result - 2 ** math.sqrt(2)) < 1e-8

    def test_ln_cf_nonpositive_raises(self):
        """Ln(CF) rejects zero and negative CFs eagerly."""
        with pytest.raises(ValueError):
            Ln(CF.from_int(0))
        with pytest.raises(ValueError):
            Ln(CF.from_int(-3))

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
        for x_num, x_den in [(2, 1), (3, 1), (355, 113), (1, 3)]:
            dec = _ln_terms_from_decimal(x_num, x_den, 30)
            mpm = _ln_terms_from_mpmath(x_num, x_den, 30)
            assert dec == mpm, f"Ln({x_num}/{x_den}): decimal vs mpmath mismatch"

    def test_ln1p_cf_zero_is_exact(self):
        assert _ln1p_cf(Fraction(0)) == CF.from_int(0)

    def test_ln_cf_matches_existing_terms(self):
        """Experimental meta-CF backend emits the same terms as Ln."""
        for x in (
            Fraction(2),
            Fraction(3),
            Fraction(3, 2),
            Fraction(355, 113),
            Fraction(1, 3),
            Fraction(10),
        ):
            meta = list(LnCF(x).take(20))
            current = list(Ln(x).take(20))
            mpm = _ln_terms_from_mpmath(x.numerator, x.denominator, 20)
            assert meta == current == mpm, f"LnCF({x}) term mismatch"

    def test_ln_cf_value(self):
        """Experimental meta-CF backend agrees with math.log."""
        for x in (
            Fraction(2),
            Fraction(3),
            Fraction(3, 2),
            Fraction(355, 113),
            Fraction(1, 3),
            Fraction(10),
        ):
            val = float(convergent(LnCF(x).take(25), 24))
            assert abs(val - math.log(float(x))) < 1e-8, f"LnCF({x})"

    def test_ln_cf_accepts_finite_cf(self):
        assert list(LnCF(CF.from_int(2)).take(8)) == list(LnCF(2).take(8))
        assert LnCF(CF.from_int(1)) == CF.from_int(0)
        with pytest.raises(ValueError):
            LnCF(CF.from_int(0))

    def test_ln_cf_bad_type_raises(self):
        with pytest.raises(TypeError):
            LnCF(1.5)  # type: ignore[arg-type]

    def test_ln_cf_accepts_irrational_cf(self):
        for x, expected in (
            (Sqrt(2), math.log(math.sqrt(2))),
            (Pi(), math.log(math.pi)),
        ):
            val = float(convergent(LnCF(x).take(12), 11))
            assert abs(val - expected) < 1e-8


class TestLog2:
    def test_log2_value(self):
        for n in (3, 10, 123):
            val = float(convergent(Log2(n).take(15), 14))
            assert abs(val - math.log2(n)) < 1e-8, f"Log2({n})"

    def test_log2_not_periodic(self):
        assert not Log2(123).is_periodic()

    def test_log2_cf_value(self):
        val = float(convergent(Log2CF(123).take(20), 19))
        assert abs(val - math.log2(123)) < 1e-8

    def test_log2_exact_reciprocal_power(self):
        assert Log2(Fraction(1, 8)) == CF.from_int(-3)


class TestLog10:
    def test_log10_exact(self):
        assert Log10(10) == CF.from_int(1)
        assert Log10(100) == CF.from_int(2)
        assert Log10(Fraction(1, 100)) == CF.from_int(-2)

    def test_log10_value(self):
        val = float(convergent(Log10(2).take(20), 19))
        assert abs(val - math.log10(2)) < 1e-8

    def test_log10_cf_value(self):
        val = float(convergent(Log10CF(2).take(20), 19))
        assert abs(val - math.log10(2)) < 1e-8


class TestLog:
    def test_log_no_base_is_ln(self):
        v1 = float(convergent(Log(2).take(20), 19))
        v2 = float(convergent(Ln(2).take(20), 19))
        assert abs(v1 - v2) < 1e-14

    def test_log_base_2_exact(self):
        assert Log(8, 2) == CF.from_int(3)
        assert Log(Fraction(1, 4), 2) == CF.from_int(-2)
        assert Log(8, Fraction(1, 2)) == CF.from_int(-3)
        assert Log(Fraction(1, 8), Fraction(1, 2)) == CF.from_int(3)

    def test_log_base_10_exact(self):
        assert Log(1000, 10) == CF.from_int(3)
        assert Log(Fraction(1, 1000), 10) == CF.from_int(-3)

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

    def test_log_cf_value(self):
        val = float(convergent(LogCF(3, 2).take(20), 19))
        assert abs(val - math.log(3, 2)) < 1e-8

    def test_log_cf_no_base_is_ln_cf(self):
        assert LogCF(2).take(12) == LnCF(2).take(12)

    def test_log_cf_bad_base_type(self):
        with pytest.raises(TypeError):
            LogCF(2, 1.5)  # type: ignore[arg-type]

    def test_log_cf_bad_base_value(self):
        for base in (1, 0, CF.from_int(1), CF.from_int(0)):
            with pytest.raises(ValueError):
                LogCF(2, base)
