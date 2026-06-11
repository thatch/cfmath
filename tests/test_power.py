"""Tests for power and root functions."""

import math
from fractions import Fraction

import pytest

from cfmath import (
    CF,
    Cuberoot,
    Nthroot,
    Pow,
    PowCF,
    PowIntExponent,
    PowInterval,
    PowMode,
    PowMP,
    Sqrt,
    convergent,
    convergents,
)
from cfmath.power import (
    _cf_interval,
    _floor_at_corner,
    _integer_kth_root,
    _v_cmp,
)

def _integer_cbrt(n: int) -> int:
    return _integer_kth_root(n, 3)


class TestCuberoot:
    def test_integer_cbrt_floor_edges(self):
        assert _integer_cbrt(-1) == 0
        assert _integer_cbrt(0) == 0
        assert _integer_cbrt(63) == 3
        assert _integer_cbrt(64) == 4
        assert _integer_cbrt(65) == 4

    def test_cuberoot_8_is_2(self):
        cf = Cuberoot(8)
        convs = list(convergents(cf))
        val = float(convs[-1])
        assert abs(val - 2.0) < 1e-8

    def test_cuberoot_2_value(self):
        val = float(convergent(Cuberoot(2).take(15), 14))
        assert abs(val - 2 ** (1 / 3)) < 1e-10

    def test_cuberoot_exact_matches_mpmath(self):
        import mpmath

        for n in (2, 3, 5, 7):
            exact = list(Cuberoot(n).take(30))
            mpmath.mp.dps = 200
            val = mpmath.cbrt(n)
            mpmath_terms = []
            for _ in range(30):
                a = int(mpmath.floor(val))
                mpmath_terms.append(a)
                val = 1 / (val - a)
            assert exact == mpmath_terms, f"Cuberoot({n}): exact vs mpmath mismatch"

    def test_cuberoot_bad_input(self):
        with pytest.raises(ValueError):
            Cuberoot(0)
        with pytest.raises(ValueError):
            Cuberoot(-1)


class TestNthroot:
    def test_perfect_powers(self):
        assert Nthroot(16, 4).terms == [2]
        assert Nthroot(81, 4).terms == [3]
        assert Nthroot(32, 5).terms == [2]

    def test_k2_gives_periodic_cf(self):
        result = Nthroot(2, 2)
        sq = Sqrt(2)
        assert result.terms == sq.terms
        assert result.repeating == sq.repeating

    def test_k3_agrees_with_cuberoot(self):
        for n in (2, 3, 5, 7):
            v1 = list(Nthroot(n, 3).take(20))
            v2 = list(Cuberoot(n).take(20))
            assert v1 == v2

    def test_k4_value(self):
        val = float(convergent(Nthroot(2, 4).take(20), 19))
        assert abs(val - 2**0.25) < 1e-10

    def test_k5_value(self):
        val = float(convergent(Nthroot(2, 5).take(20), 19))
        assert abs(val - 2**0.2) < 1e-10

    def test_k10_value(self):
        val = float(convergent(Nthroot(2, 10).take(20), 19))
        assert abs(val - 2**0.1) < 1e-10

    def test_fraction_perfect_root(self):
        # (1/8)^(1/3) = 1/2 exactly
        assert Nthroot(Fraction(1, 8), 3) == CF.from_rational(Fraction(1, 2))
        # (4/9)^(1/2) = 2/3 exactly
        assert Nthroot(Fraction(4, 9), 2) == CF.from_rational(Fraction(2, 3))
        # (8/27)^(1/3) = 2/3 exactly
        assert Nthroot(Fraction(8, 27), 3) == CF.from_rational(Fraction(2, 3))

    def test_fraction_k2_is_periodic(self):
        # sqrt(2/3) is a quadratic irrational — should have repeating part
        result = Nthroot(Fraction(2, 3), 2)
        assert result.is_periodic()
        val = float(convergent(result.take(20), 19))
        assert abs(val - math.sqrt(2 / 3)) < 1e-10

    def test_fraction_k3_value(self):
        # (2/3)^(1/3) ≈ 0.8736
        val = float(convergent(Nthroot(Fraction(2, 3), 3).take(20), 19))
        assert abs(val - (2 / 3) ** (1 / 3)) < 1e-10

    def test_fraction_k3_less_than_one(self):
        # (1/8)^(1/3) = 0.5, CF starts with [0; ...]
        cf = Nthroot(Fraction(1, 8), 3)
        assert cf.terms == [0, 2]  # exact rational 1/2

    def test_bad_base(self):
        with pytest.raises(ValueError):
            Nthroot(0, 4)
        with pytest.raises((ValueError, TypeError)):
            Nthroot(-1, 4)

    def test_bad_degree(self):
        with pytest.raises(ValueError):
            Nthroot(2, 1)
        with pytest.raises(ValueError):
            Nthroot(2, 0)


class TestPow:
    def test_zero_exponent(self):
        assert Pow(5, 0).terms == [1]

    def test_one_exponent(self):
        assert Pow(Fraction(3, 4), 1) == CF.from_rational(Fraction(3, 4))

    def test_pow_base_one_short_circuits_even_for_cf_exponent(self):
        from cfmath import Pi

        assert Pow(1, Pi()) == CF.from_int(1)

    def test_pow_integer_exponent_exact(self):
        assert Pow(2, 3) == CF.from_int(8)
        assert Pow(Fraction(2, 3), 2) == CF.from_rational(Fraction(4, 9))

    def test_negative_integer_exponent(self):
        assert Pow(2, -3) == CF.from_rational(Fraction(1, 8))

    def test_half_exponent_gives_periodic(self):
        result = Pow(2, Fraction(1, 2))
        sq = Sqrt(2)
        assert result.terms == sq.terms
        assert result.repeating == sq.repeating

    def test_half_exponent_fraction_base(self):
        result = Pow(Fraction(9, 2), Fraction(3, 2))
        assert result.is_periodic()
        val = float(convergent(result.take(20), 19))
        assert abs(val - 4.5**1.5) < 1e-10

    def test_third_exponent_exact(self):
        assert Pow(8, Fraction(1, 3)).terms == [2]

    def test_rational_base_rational_exponent_exact(self):
        # (1/8)^(1/3) = 1/2 — exact via Nthroot
        result = Pow(Fraction(1, 8), Fraction(1, 3))
        assert result == CF.from_rational(Fraction(1, 2))

    def test_rational_base_rational_exponent_value(self):
        # (2/3)^(2/3): compute (2/3)^2 = 4/9, then Nthroot(4/9, 3)
        result = Pow(Fraction(2, 3), Fraction(2, 3))
        expected = (2 / 3) ** (2 / 3)
        val = float(convergent(result.take(20), 19))
        assert abs(val - expected) < 1e-10

    def test_general_fractional(self):
        val = float(convergent(Pow(2, Fraction(3, 2)).take(20), 19))
        assert abs(val - 2**1.5) < 1e-8

    def test_pow_mode_cf(self):
        val = float(convergent(Pow(2, Fraction(3, 2), PowMode.CF).take(20), 19))
        assert abs(val - 2**1.5) < 1e-8

    def test_pow_mode_mp(self):
        val = float(convergent(Pow(2, Fraction(3, 2), PowMode.MP).take(20), 19))
        assert abs(val - 2**1.5) < 1e-8

    def test_pow_mode_int(self):
        assert Pow(2, 3, PowMode.INT) == CF.from_int(8)

    def test_pow_string_mode_still_works(self):
        assert Pow(2, 3, "int") == CF.from_int(8)

    def test_pow_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            Pow(2, 3, "bogus")

    def test_pow_bad_mode_type_raises(self):
        with pytest.raises(TypeError):
            Pow(2, 3, object())  # type: ignore[arg-type]

    def test_pow_bad_type_base(self):
        with pytest.raises(TypeError):
            Pow(1.5, Fraction(1, 2))

    def test_bad_type_exponent(self):
        with pytest.raises(TypeError):
            Pow(2, 0.5)

    def test_nonpositive_base(self):
        with pytest.raises(ValueError):
            Pow(0, Fraction(1, 2))
        with pytest.raises(ValueError):
            Pow(-1, Fraction(1, 2))


class TestPowIntervalHelpers:
    def test_cf_interval_for_finite_cf_collapses(self):
        assert _cf_interval(CF.from_rational(Fraction(3, 2)), 3) == (Fraction(3, 2), Fraction(3, 2))

    def test_cf_interval_orders_odd_and_even_depths(self):
        lo1, hi1 = _cf_interval(Sqrt(2), 1)
        lo2, hi2 = _cf_interval(Sqrt(2), 2)

        assert lo1 < hi1
        assert lo2 < hi2
        assert lo2 <= Fraction(99, 70) <= hi2

    def test_v_cmp_handles_negative_zero_and_positive_thresholds(self):
        assert _v_cmp(2, 1, 3, 1, Fraction(-1)) > 0
        assert _v_cmp(0, 1, 1, 1, Fraction(0)) == 0
        assert _v_cmp(2, 1, 3, 1, Fraction(8)) == 0
        assert _v_cmp(2, 1, 3, 1, Fraction(9)) < 0

    def test_floor_at_corner_positive_and_negative_coefficients(self):
        # v = 8.  The identity transform has floor 8, while 9 is too high.
        assert _floor_at_corner(1, 0, 0, 1, 2, 1, 3, 1, 8)
        assert not _floor_at_corner(1, 0, 0, 1, 2, 1, 3, 1, 9)

        # 1 / v = 1/8 has floor 0; this exercises the coeff < 0 branch.
        assert _floor_at_corner(0, 1, 1, 0, 2, 1, 3, 1, 0)


class TestPowImplementations:
    def test_pow_int_exponent_direct(self):
        assert PowIntExponent(Fraction(2, 3), -2) == CF.from_rational(Fraction(9, 4))

    def test_pow_int_exponent_rejects_non_integer_exponent(self):
        with pytest.raises(ValueError):
            PowIntExponent(2, Fraction(1, 2))

    def test_pow_cf_direct(self):
        val = float(convergent(PowCF(2, Fraction(3, 2)).take(20), 19))
        assert abs(val - 2**1.5) < 1e-8

    def test_pow_mp_direct(self):
        val = float(convergent(PowMP(2, Fraction(3, 2)).take(20), 19))
        assert abs(val - 2**1.5) < 1e-8

    def test_pow_interval_direct_rejects_general_rationals(self):
        with pytest.raises(ValueError):
            PowInterval(2, Fraction(3, 2))

    def test_pow_interval_direct_cf_base(self):
        from cfmath import Pi

        val = float(convergent(PowInterval(Pi(), 2).take(20), 19))
        assert abs(val - math.pi**2) < 1e-10


class TestPowCF:
    """Pow() with CF base or exponent."""

    def test_finite_cf_base_reduces_to_exact(self):
        # CF([4]) is finite → reduces to Fraction(4) → exact quadratic path
        assert Pow(CF([4]), Fraction(1, 2)).terms == [2]
        assert Pow(CF([9]), Fraction(1, 2)).terms == [3]

    def test_finite_cf_exponent_reduces_to_exact(self):
        # CF([3]) is finite → reduces to Fraction(3) → integer exponent path
        assert Pow(2, CF([3])) == CF.from_int(8)

    def test_finite_cf_half_exponent_reduces_to_sqrt(self):
        # CF([0, 2]) = 1/2 is finite → reduces to Fraction(1, 2) → exact Sqrt path
        assert Pow(2, CF([0, 2])).terms == Sqrt(2).terms

    def test_infinite_cf_base_integer_exponent_repeated_squaring(self):
        # Pi**2 via repeated squaring, no log involved
        from cfmath import Pi
        pi = Pi()
        result = Pow(pi, 2)
        val = float(convergent(result.take(20), 19))
        assert abs(val - math.pi**2) < 1e-10

    def test_infinite_cf_base_integer_exponent_negative(self):
        from cfmath import Pi
        result = Pow(Pi(), -1)
        val = float(convergent(result.take(20), 19))
        assert abs(val - 1/math.pi) < 1e-10

    def test_infinite_cf_base_fraction_exponent(self):
        # sqrt(Pi) via ExpCF/LnCF path
        from cfmath import Pi
        result = Pow(Pi(), Fraction(1, 2))
        val = float(convergent(result.take(20), 19))
        assert abs(val - math.sqrt(math.pi)) < 1e-8

    def test_infinite_cf_base_cf_exponent(self):
        # Pi ** (1/Pi): both infinite CFs
        from cfmath import Pi
        pi = Pi()
        result = Pow(pi, 1/Pi())
        val = float(convergent(result.take(20), 19))
        assert abs(val - math.pi ** (1/math.pi)) < 1e-8

    def test_pow_cf_base_matches_float(self):
        from cfmath import Pi
        from cfmath.constants import E
        # e**Pi (Gelfond's constant)
        result = Pow(E(), Pi())
        val = float(convergent(result.take(20), 19))
        assert abs(val - math.e**math.pi) < 1e-8
