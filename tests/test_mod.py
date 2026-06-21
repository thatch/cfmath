"""Tests for floor/ceil/floordiv/mod and the matching dunders."""

import math
from fractions import Fraction
from itertools import islice

import pytest

from cfmath import CF, Pi, Sqrt


def frac(p, q=1):
    return CF.from_rational(Fraction(p, q))


# ---------------------------------------------------------------------------
# math.floor / __floor__
# ---------------------------------------------------------------------------


class TestFloor:
    def test_positive_integer(self):
        assert math.floor(frac(7)) == 7

    def test_positive_rational(self):
        assert math.floor(frac(7, 3)) == 2

    def test_negative_rational(self):
        assert math.floor(frac(-7, 3)) == -3

    def test_negative_integer(self):
        assert math.floor(frac(-4)) == -4

    def test_pi(self):
        assert math.floor(Pi()) == 3

    def test_sqrt2(self):
        assert math.floor(Sqrt(2)) == 1


# ---------------------------------------------------------------------------
# math.ceil / __ceil__
# ---------------------------------------------------------------------------


class TestCeil:
    def test_positive_integer(self):
        assert math.ceil(frac(5)) == 5

    def test_positive_rational(self):
        assert math.ceil(frac(7, 3)) == 3

    def test_negative_rational(self):
        assert math.ceil(frac(-7, 3)) == -2

    def test_negative_integer(self):
        assert math.ceil(frac(-4)) == -4

    def test_pi(self):
        assert math.ceil(Pi()) == 4

    def test_sqrt2(self):
        assert math.ceil(Sqrt(2)) == 2

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
# // (__floordiv__ and __rfloordiv__)
# ---------------------------------------------------------------------------


class TestFloordiv:
    def test_rational_rational(self):
        assert frac(10) // frac(3) == frac(3)

    def test_negative_dividend(self):
        assert frac(-7) // frac(3) == frac(-3)

    def test_negative_divisor(self):
        assert frac(7) // frac(-3) == frac(-3)

    def test_exact_division(self):
        assert frac(12) // frac(4) == frac(3)

    def test_pi_floordiv_int(self):
        assert Pi() // frac(1) == frac(3)
        assert Pi() // 1 == frac(3)

    def test_sqrt2_floordiv_rational(self):
        # Sqrt(2) ≈ 1.414; floor(1.414 / 0.5) = floor(2.828) = 2
        assert Sqrt(2) // frac(1, 2) == frac(2)

    def test_rfloordiv_int(self):
        assert 7 // frac(3) == frac(2)

    def test_rfloordiv_negative(self):
        assert (-7) // frac(3) == frac(-3)

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            frac(5) // frac(0)


# ---------------------------------------------------------------------------
# % (__mod__ and __rmod__)
# ---------------------------------------------------------------------------


class TestMod:
    def test_basic_rational(self):
        assert frac(10) % frac(3) == frac(1)

    def test_exact_multiple(self):
        assert frac(12) % frac(4) == frac(0)

    def test_negative_dividend(self):
        # -7 mod 3: floor(-7/3) = -3, result = -7 - 3*(-3) = 2
        assert frac(-7) % frac(3) == frac(2)

    def test_negative_divisor(self):
        # 7 mod -3: floor(7/-3) = -3, result = 7 - (-3)*(-3) = 7 - 9 = -2
        assert frac(7) % frac(-3) == frac(-2)

    def test_fractional(self):
        # (7/6) mod (1/3): floor((7/6)/(1/3)) = floor(7/2) = 3
        # result = 7/6 - 3*(1/3) = 7/6 - 1 = 1/6
        assert frac(7, 6) % frac(1, 3) == frac(1, 6)

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

    def test_rmod_int(self):
        assert 10 % frac(3) == frac(1)

    def test_rmod_negative(self):
        assert (-7) % frac(3) == frac(2)

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
# divmod (__divmod__ and __rdivmod__)
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

    def test_rdivmod_int(self):
        q, r = divmod(17, frac(5))
        assert q == frac(3)
        assert r == frac(2)

    def test_rdivmod_negative(self):
        q, r = divmod(-17, frac(5))
        assert q == frac(-4)
        assert r == frac(3)


class TestFloorQuotientGimme:
    """floor(x/y) at an exact-integer boundary (e.g. Pi/Pi = 1) straddles
    forever.  By default the shared gimme resolves it; gimme_min_term_digits=None
    restores the old raise behaviour.  The bounds are exact Fractions, so there
    is no float plateau — only the boundary itself is undecidable."""

    def test_exact_boundary_resolves_by_default(self):
        from cfmath.mod import _floor_quotient

        assert _floor_quotient(Pi(), Pi()) == 1
        assert (Pi() // Pi()).take(2).terms == [1]

    def test_gimme_none_raises(self, monkeypatch):
        import cfmath.mod as mod

        # Small cap keeps it fast — without gimme it never resolves Pi/Pi.
        monkeypatch.setattr(mod, "_MAX_FLOOR_ITERS", 50)
        with pytest.raises(ArithmeticError, match="did not converge"):
            mod._floor_quotient(Pi(), Pi(), gimme_min_term_digits=None)

    def test_normal_cases_unaffected(self):
        from cfmath.mod import _floor_quotient

        assert _floor_quotient(Pi(), Sqrt(2)) == 2  # ~2.22
        assert _floor_quotient(CF.from_int(10), CF.from_int(3)) == 3
