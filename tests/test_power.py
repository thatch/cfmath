"""Tests for power and root functions."""

import math
import sys
from fractions import Fraction

import pytest

from cfmath import CF, Cuberoot, Nthroot, Pi, Pow, Sqrt, convergent
from cfmath.logarithm import Ln
from cfmath.power import (
    _bracket_floor,
    _floor_kth_root_rational,
    _horner_update,
    _init_bracket_poly,
    _pow_rational_cf_gen,
)


class TestHornerUpdate:
    def test_degree3_matches_manual_formula(self):
        # For f(y) = Ay^3 + By^2 + Cy + D after substituting y = a + 1/z:
        # new coefficients should be [f(a), f'(a), f''(a)/2, A]
        A, B, C, D, a = -1, 3, 3, 1, 3
        result = _horner_update([A, B, C, D], a)
        fa = A * a**3 + B * a**2 + C * a + D
        fp = 3 * A * a**2 + 2 * B * a + C
        fpp = 3 * A * a + B
        assert result == [fa, fp, fpp, A]

    def test_degree2_matches_sqrt_expansion(self):
        # f(y) = -y^2 + 2 (for sqrt(2), with leading -1 after normalization)
        # at a=1: new coeffs = [f(1), f'(1), -1] = [1, -2, -1]
        result = _horner_update([-1, 0, 2], 1)
        assert result == [1, -2, -1]

    def test_preserves_root(self):
        # After substitution, the root of the new polynomial at any a should
        # be 1/(root - a) when root > a.
        # For cbrt(2): starts with [1, 0, 0, -2], root = cbrt(2) ≈ 1.2599, a=1
        # new polynomial root should be 1/(cbrt(2) - 1) ≈ 3.849
        new_coeffs = _horner_update([1, 0, 0, -2], 1)
        assert new_coeffs == [-1, 3, 3, 1]


class TestFloorKthRootRational:
    def test_integer_inputs(self):
        assert _floor_kth_root_rational(2, 1, 3) == 1  # cbrt(2) ≈ 1.26
        assert _floor_kth_root_rational(8, 1, 3) == 2  # cbrt(8) = 2 exactly
        assert _floor_kth_root_rational(9, 1, 3) == 2  # cbrt(9) ≈ 2.08

    def test_fraction_less_than_one(self):
        assert _floor_kth_root_rational(1, 8, 3) == 0  # (1/8)^(1/3) = 0.5

    def test_fraction_between_integers(self):
        assert _floor_kth_root_rational(2, 3, 3) == 0  # (2/3)^(1/3) ≈ 0.874
        assert _floor_kth_root_rational(3, 2, 3) == 1  # (3/2)^(1/3) ≈ 1.145


class TestCuberoot:
    def test_perfect_cubes(self):
        assert Cuberoot(1).terms == [1]
        assert Cuberoot(8).terms == [2]
        assert Cuberoot(27).terms == [3]
        assert Cuberoot(125).terms == [5]

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

    def test_integer_exponent_exact(self):
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

    def test_cf_exponent(self):
        result = Pow(2, Pi())
        expected = math.exp(math.pi * math.log(2))
        assert abs(float(result) - expected) < 1e-8

    def test_bad_type_base(self):
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


class TestCFPow:
    def test_fraction_half_finite_base(self):
        result = CF.from_int(4) ** Fraction(1, 2)
        assert result.terms == [2]

    def test_fraction_half_infinite_base(self):
        result = Sqrt(2) ** Fraction(1, 2)
        val = float(convergent(result.take(20), 19))
        assert abs(val - 2**0.25) < 1e-10

    def test_cf_exponent(self):
        result = Sqrt(2) ** Pi()
        expected = math.exp(math.pi / 2 * math.log(2))
        assert abs(float(result) - expected) < 1e-8

    def test_rpow_int(self):
        result = 2 ** Pi()
        expected = math.exp(math.pi * math.log(2))
        assert abs(float(result) - expected) < 1e-8

    #@pytest.mark.skipif(
    #    sys.version_info < (3, 12),
    #    reason="Python 3.10/3.11 raises TypeError instead of returning NotImplemented from Fraction.__pow__",
    #)
    #def test_rpow_fraction(self):
    #    result = Fraction(1, 2) ** Pi()
    #    expected = math.exp(-math.pi * math.log(2))
    #    assert abs(float(result) - expected) < 1e-8

    def test_rpow_nonpositive_raises(self):
        with pytest.raises(ValueError):
            0 ** Pi()

    def test_integer_pow_unchanged(self):
        result = Sqrt(2) ** 2
        assert result.to_fraction() == 2


class TestBracketFloor:
    def test_normal_polynomial(self):
        # y^2 - 2 = 0, root = sqrt(2) ≈ 1.414
        assert _bracket_floor([-1, 0, 2]) == 1

    def test_degenerate_leading_zero(self):
        # [0, 1] is degenerate (represents 1/0 = ∞)
        assert _bracket_floor([0, 1]) is None

    def test_exact_integer_root(self):
        # y - 8 = 0, root = 8
        assert _bracket_floor([1, -8]) == 8


class TestInitBracketPoly:
    def test_integer_base_degree1(self):
        # (2/1)^(3/1): polynomial y - 8 = 0
        assert _init_bracket_poly(2, 1, 3, 1) == [1, -8]

    def test_integer_base_degree7(self):
        # (2/1)^(22/7): polynomial y^7 - 2^22 = 0
        poly = _init_bracket_poly(2, 1, 22, 7)
        assert poly[0] == 1
        assert poly[-1] == -(2**22)
        assert len(poly) == 8
        assert poly[1:-1] == [0] * 6

    def test_fraction_base(self):
        # (2/3)^(1/2): polynomial 3*y^2 - 2 = 0
        assert _init_bracket_poly(2, 3, 1, 2) == [3, 0, -2]


class TestPowRationalCF:
    def test_first_term_2_pi(self):
        # floor(2^π) = floor(8.824...) = 8
        gen = _pow_rational_cf_gen(2, 1, Pi())
        assert next(gen) == 8

    def test_first_few_terms_2_pi(self):
        # 2^π CF = [8; 1, 4, 1, 2, ...] — verify exact terms, then convergent accuracy
        result = CF([], _source=_pow_rational_cf_gen(2, 1, Pi()))
        terms = list(result.take(5))
        assert terms == [8, 1, 4, 1, 2]
        val = float(convergent(result.take(5), 4))
        assert abs(val - math.exp(math.pi * math.log(2))) < 2e-3

    def test_agrees_with_numerical_for_small_base(self):
        # 3^π: exact bracket terms should agree with numerical float value
        result = CF([], _source=_pow_rational_cf_gen(3, 1, Pi()))
        terms = list(result.take(4))
        assert terms[0] == 31
        val = float(convergent(result.take(4), 3))
        assert abs(val - math.exp(math.pi * math.log(3))) < 2e-3

    def test_degenerate_bracket_handled(self):
        # 2^(3/1) = 8 exactly; the lower bracket degenerates after emitting 8,
        # forcing a re-initialization — verify we still get the correct 2nd term
        result = CF([], _source=_pow_rational_cf_gen(2, 1, Pi()))
        terms = list(result.take(3))
        assert terms[0] == 8  # floor(2^π)
        assert terms[1] == 1  # next CF term of 2^π

    def test_fraction_base(self):
        # (3/2)^π ≈ 3.574..., CF = [3; 1, 1, 2, 1, 6, ...]
        result = CF([], _source=_pow_rational_cf_gen(3, 2, Pi()))
        terms = list(result.take(6))
        assert terms[0] == 3
        val = float(convergent(result.take(6), 5))
        assert abs(val - math.exp(math.pi * math.log(1.5))) < 1e-4


class TestLnCF:
    def test_ln_sqrt2_equals_half_ln2(self):
        ln_sqrt2 = Ln(Sqrt(2))
        ln_2_half = Ln(2) * Fraction(1, 2)
        v1 = float(convergent(ln_sqrt2.take(20), 19))
        v2 = float(convergent(ln_2_half.take(20), 19))
        assert abs(v1 - v2) < 1e-10

    def test_ln_pi_value(self):
        val = float(convergent(Ln(Pi()).take(20), 19))
        assert abs(val - math.log(math.pi)) < 1e-8
