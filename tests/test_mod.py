"""Tests for cf_floor, cf_ceil, cf_floordiv, cf_mod and the matching dunders."""

import math
from fractions import Fraction
from itertools import islice

from cfmath import CF, Pi, Sqrt, cf_ceil, cf_floor, cf_floordiv, cf_mod


def frac(p, q=1):
    return CF.from_rational(Fraction(p, q))


# ---------------------------------------------------------------------------
# cf_floor
# ---------------------------------------------------------------------------


class TestCfFloor:
    def test_positive_integer(self):
        assert cf_floor(frac(7)) == frac(7)

    def test_positive_rational(self):
        assert cf_floor(frac(7, 3)) == frac(2)

    def test_negative_rational(self):
        assert cf_floor(frac(-7, 3)) == frac(-3)

    def test_negative_integer(self):
        assert cf_floor(frac(-4)) == frac(-4)

    def test_pi(self):
        assert cf_floor(Pi()) == frac(3)

    def test_sqrt2(self):
        assert cf_floor(Sqrt(2)) == frac(1)

    def test_math_floor_dunder(self):
        assert math.floor(frac(7, 3)) == 2
        assert math.floor(frac(-7, 3)) == -3
        assert math.floor(Pi()) == 3


# ---------------------------------------------------------------------------
# cf_ceil
# ---------------------------------------------------------------------------


class TestCfCeil:
    def test_positive_integer(self):
        assert cf_ceil(frac(5)) == frac(5)

    def test_positive_rational(self):
        assert cf_ceil(frac(7, 3)) == frac(3)

    def test_negative_rational(self):
        assert cf_ceil(frac(-7, 3)) == frac(-2)

    def test_negative_integer(self):
        assert cf_ceil(frac(-4)) == frac(-4)

    def test_pi(self):
        assert cf_ceil(Pi()) == frac(4)

    def test_sqrt2(self):
        assert cf_ceil(Sqrt(2)) == frac(2)

    def test_math_ceil_dunder(self):
        assert math.ceil(frac(7, 3)) == 3
        assert math.ceil(frac(-7, 3)) == -2
        assert math.ceil(Pi()) == 4

    def test_floor_ceil_differ_by_one_for_non_integers(self):
        for cf in [Pi(), Sqrt(2), frac(7, 3), frac(-7, 3)]:
            assert math.ceil(cf) == math.floor(cf) + 1

    def test_floor_ceil_equal_on_integers(self):
        for n in [-3, 0, 1, 7]:
            cf = frac(n)
            assert math.floor(cf) == math.ceil(cf) == n


# ---------------------------------------------------------------------------
# __trunc__
# ---------------------------------------------------------------------------


class TestTrunc:
    def test_positive(self):
        assert math.trunc(frac(7, 3)) == 2

    def test_negative(self):
        assert math.trunc(frac(-7, 3)) == -2

    def test_integer(self):
        assert math.trunc(frac(-5)) == -5


# ---------------------------------------------------------------------------
# cf_floordiv / __floordiv__
# ---------------------------------------------------------------------------


class TestCfFloordiv:
    def test_rational_rational(self):
        assert cf_floordiv(frac(10), frac(3)) == frac(3)
        assert frac(10) // frac(3) == frac(3)

    def test_negative_dividend(self):
        assert cf_floordiv(frac(-7), frac(3)) == frac(-3)

    def test_negative_divisor(self):
        assert cf_floordiv(frac(7), frac(-3)) == frac(-3)

    def test_exact_division(self):
        assert cf_floordiv(frac(12), frac(4)) == frac(3)

    def test_pi_floordiv_int(self):
        assert Pi() // frac(1) == frac(3)
        assert Pi() // 1 == frac(3)

    def test_sqrt2_floordiv_rational(self):
        # Sqrt(2) ≈ 1.414; floor(1.414 / 0.5) = floor(2.828) = 2
        assert cf_floordiv(Sqrt(2), frac(1, 2)) == frac(2)


# ---------------------------------------------------------------------------
# cf_mod / __mod__
# ---------------------------------------------------------------------------


class TestCfMod:
    def test_basic_rational(self):
        assert cf_mod(frac(10), frac(3)) == frac(1)
        assert frac(10) % frac(3) == frac(1)

    def test_exact_multiple(self):
        assert cf_mod(frac(12), frac(4)) == frac(0)

    def test_negative_dividend(self):
        # -7 mod 3: floor(-7/3) = -3, result = -7 - 3*(-3) = 2
        assert cf_mod(frac(-7), frac(3)) == frac(2)

    def test_negative_divisor(self):
        # 7 mod -3: floor(7/-3) = -3, result = 7 - (-3)*(-3) = 7 - 9 = -2
        assert cf_mod(frac(7), frac(-3)) == frac(-2)

    def test_fractional(self):
        # (7/6) mod (1/3): floor((7/6)/(1/3)) = floor(7/2) = 3
        # result = 7/6 - 3*(1/3) = 7/6 - 1 = 1/6
        assert cf_mod(frac(7, 6), frac(1, 3)) == frac(1, 6)

    def test_pi_mod_1(self):
        result = Pi() % 1
        expected = Pi() - 3
        assert list(islice(result.digits(10), 15)) == list(islice(expected.digits(10), 15))

    def test_sqrt2_mod_1(self):
        result = Sqrt(2) % 1
        expected = Sqrt(2) - 1
        assert list(islice(result.digits(10), 15)) == list(islice(expected.digits(10), 15))

    def test_int_coercion(self):
        assert frac(10) % 3 == frac(1)

    def test_mod_floordiv_identity(self):
        cases = [
            (frac(17), frac(5)),
            (frac(-17), frac(5)),
            (frac(17), frac(-5)),
            (frac(7, 3), frac(1, 3)),
        ]
        for x, y in cases:
            q, r = divmod(x, y)
            assert q * y + r == x, f"identity failed for {x}, {y}"


# ---------------------------------------------------------------------------
# divmod
# ---------------------------------------------------------------------------


class TestDivmod:
    def test_basic(self):
        q, r = divmod(frac(17), frac(5))
        assert q == frac(3)
        assert r == frac(2)

    def test_negative(self):
        q, r = divmod(frac(-17), frac(5))
        assert q == frac(-4)
        assert r == frac(3)
