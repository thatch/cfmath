"""Tests for the generalized n-input Gosper arithmetic (gosper_generalized.py).

These mirror test_gosper.py but call functions from gosper_generalized directly,
exercising the bit-manipulation tensor path rather than the optimised 2-input
hand-coded path in gosper.py.
"""

from fractions import Fraction

from cfmath import CF, Phi, Sqrt, convergent, convergents
from cfmath.gosper_generalized import (
    cf_add,
    cf_div,
    cf_homographic,
    cf_max,
    cf_min,
    cf_mul,
    cf_n_ary,
    cf_sub,
)


def _eval(cf: CF, depth: int = 15) -> Fraction:
    """Evaluate a CF by taking the last convergent up to `depth` terms."""
    convs = list(convergents(cf.take(depth)))
    if not convs:
        raise ValueError("empty CF")
    return convs[-1]


def _ref(p: int, q: int = 1) -> CF:
    return CF.from_fraction(p, q)


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------


class TestAddition:
    def test_int_plus_int(self):
        a = _ref(3)
        b = _ref(4)
        assert _eval(cf_add(a, b)) == Fraction(7)

    def test_fraction_plus_fraction(self):
        a = CF.from_fraction(22, 7)
        b = CF.from_fraction(1, 3)
        assert _eval(cf_add(a, b)) == Fraction(22, 7) + Fraction(1, 3)

    def test_add_zero(self):
        a = CF.from_fraction(5, 3)
        z = _ref(0)
        assert _eval(cf_add(a, z)) == Fraction(5, 3)

    def test_add_one(self):
        a = CF.from_fraction(355, 113)
        one = _ref(1)
        assert _eval(cf_add(a, one)) == Fraction(355, 113) + 1

    def test_commutativity(self):
        a = CF.from_fraction(3, 7)
        b = CF.from_fraction(5, 11)
        assert _eval(cf_add(a, b)) == _eval(cf_add(b, a))

    def test_associativity(self):
        a = CF.from_fraction(1, 2)
        b = CF.from_fraction(1, 3)
        c = CF.from_fraction(1, 5)
        lhs = _eval(cf_add(cf_add(a, b), c))
        rhs = _eval(cf_add(a, cf_add(b, c)))
        assert lhs == rhs

    def test_many_fractions(self):
        pairs = [(3, 5), (7, 11), (13, 17), (100, 101)]
        for p1, q1 in pairs:
            for p2, q2 in pairs:
                a = CF.from_fraction(p1, q1)
                b = CF.from_fraction(p2, q2)
                expected = Fraction(p1, q1) + Fraction(p2, q2)
                assert _eval(cf_add(a, b)) == expected


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------


class TestSubtraction:
    def test_simple_sub(self):
        a = CF.from_fraction(5, 3)
        b = CF.from_fraction(1, 3)
        assert _eval(cf_sub(a, b)) == Fraction(4, 3)

    def test_sub_self(self):
        a = CF.from_fraction(7, 5)
        assert _eval(cf_sub(a, a)) == Fraction(0)

    def test_sub_gives_negative(self):
        a = CF.from_fraction(1, 4)
        b = CF.from_fraction(1, 2)
        assert _eval(cf_sub(a, b)) == Fraction(-1, 4)

    def test_sub_is_neg_add(self):
        a = CF.from_fraction(8, 3)
        b = CF.from_fraction(5, 7)
        assert _eval(cf_sub(a, b)) == Fraction(8, 3) - Fraction(5, 7)


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


class TestMultiplication:
    def test_int_times_int(self):
        a = _ref(3)
        b = _ref(5)
        assert _eval(cf_mul(a, b)) == Fraction(15)

    def test_fraction_times_fraction(self):
        a = CF.from_fraction(3, 7)
        b = CF.from_fraction(7, 3)
        assert _eval(cf_mul(a, b)) == Fraction(1)

    def test_multiply_by_zero(self):
        a = CF.from_fraction(5, 3)
        z = _ref(0)
        assert _eval(cf_mul(a, z)) == Fraction(0)

    def test_multiply_by_one(self):
        a = CF.from_fraction(22, 7)
        one = _ref(1)
        assert _eval(cf_mul(a, one)) == Fraction(22, 7)

    def test_commutativity(self):
        a = CF.from_fraction(3, 11)
        b = CF.from_fraction(7, 5)
        assert _eval(cf_mul(a, b)) == _eval(cf_mul(b, a))

    def test_many_pairs(self):
        pairs = [(2, 3), (5, 7), (11, 13)]
        for p1, q1 in pairs:
            for p2, q2 in pairs:
                a = CF.from_fraction(p1, q1)
                b = CF.from_fraction(p2, q2)
                expected = Fraction(p1, q1) * Fraction(p2, q2)
                assert _eval(cf_mul(a, b)) == expected


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------


class TestDivision:
    def test_simple_div(self):
        a = CF.from_fraction(3, 4)
        b = CF.from_fraction(3, 4)
        assert _eval(cf_div(a, b)) == Fraction(1)

    def test_div_gives_fraction(self):
        a = CF.from_fraction(22, 7)
        b = CF.from_fraction(11, 7)
        assert _eval(cf_div(a, b)) == Fraction(2)

    def test_div_reciprocal(self):
        a = CF.from_fraction(5, 3)
        one = _ref(1)
        assert _eval(cf_div(one, a)) == Fraction(3, 5)

    def test_many_divs(self):
        pairs = [(3, 5), (7, 11), (2, 9)]
        for p1, q1 in pairs:
            for p2, q2 in pairs:
                a = CF.from_fraction(p1, q1)
                b = CF.from_fraction(p2, q2)
                expected = Fraction(p1, q1) / Fraction(p2, q2)
                assert _eval(cf_div(a, b)) == expected


# ---------------------------------------------------------------------------
# Powers (via repeated cf_mul / cf_homographic)
# ---------------------------------------------------------------------------


class TestPow:
    def test_square(self):
        a = CF.from_fraction(3, 2)
        assert _eval(cf_mul(a, CF.from_fraction(3, 2))) == Fraction(9, 4)

    def test_cube(self):
        a = CF.from_fraction(2, 1)
        a2 = cf_mul(a, CF.from_fraction(2, 1))
        a3 = cf_mul(a2, CF.from_fraction(2, 1))
        assert _eval(a3) == Fraction(8)

    def test_reciprocal(self):
        a = CF.from_fraction(2, 3)
        recip = cf_homographic(a, 0, 1, 1, 0)  # 1/x
        assert _eval(recip) == Fraction(3, 2)


# ---------------------------------------------------------------------------
# Negation (via homographic)
# ---------------------------------------------------------------------------


class TestNegation:
    def test_negate(self):
        a = CF.from_fraction(3, 4)
        neg = cf_homographic(a, -1, 0, 0, 1)
        assert _eval(neg) == Fraction(-3, 4)

    def test_double_negate(self):
        a = CF.from_fraction(5, 7)
        assert _eval(cf_homographic(cf_homographic(a, -1, 0, 0, 1), -1, 0, 0, 1)) == Fraction(5, 7)


# ---------------------------------------------------------------------------
# Mixed integer/CF arithmetic
# ---------------------------------------------------------------------------


class TestMixedArithmetic:
    def test_add_integer(self):
        cf = CF.from_fraction(1, 7)
        three = _ref(3)
        assert _eval(cf_add(three, cf)) == Fraction(22, 7)

    def test_mul_integer(self):
        cf = CF.from_fraction(1, 7)
        twentytwo = _ref(22)
        assert _eval(cf_mul(twentytwo, cf)) == Fraction(22, 7)


# ---------------------------------------------------------------------------
# Phi identity: φ² = φ + 1
# ---------------------------------------------------------------------------


class TestPhiIdentity:
    def test_phi_squared_is_phi_plus_one(self):
        phi = Phi()
        phi_sq = cf_mul(phi, Phi())  # two independent Phi() objects
        phi_plus_1 = cf_add(Phi(), _ref(1))
        depth = 15
        lhs = convergent(phi_sq.take(depth), depth - 1)
        rhs = convergent(phi_plus_1.take(depth), depth - 1)
        assert abs(float(lhs) - float(rhs)) < 1e-10


# ---------------------------------------------------------------------------
# sqrt(2) * sqrt(2) ≈ 2
# ---------------------------------------------------------------------------


class TestSqrt2Squared:
    def test_sqrt2_times_sqrt2(self):
        s2 = Sqrt(2).take(15)
        result = cf_mul(s2, s2)
        convs = list(convergents(result))
        val = float(convs[-1])
        assert abs(val - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# Homographic (unary) transform
# ---------------------------------------------------------------------------


class TestHomographic:
    def test_shift(self):
        x = CF.from_fraction(3, 7)
        y = cf_homographic(x, 1, 2, 0, 1)  # x + 2
        assert _eval(y) == Fraction(3, 7) + 2

    def test_scale(self):
        x = CF.from_fraction(5, 11)
        y = cf_homographic(x, 3, 0, 0, 1)  # 3x
        assert _eval(y) == Fraction(15, 11)


# ---------------------------------------------------------------------------
# cf_n_ary — direct API tests
# ---------------------------------------------------------------------------


class TestNAry:
    def test_n_ary_add(self):
        """cf_n_ary with n=2 add coefficients matches cf_add."""
        a = CF.from_fraction(3, 5)
        b = CF.from_fraction(7, 11)
        result = cf_n_ary([a, b], num=[0, 1, 1, 0], den=[1, 0, 0, 0])
        expected = Fraction(3, 5) + Fraction(7, 11)
        assert _eval(result) == expected

    def test_n_ary_homographic(self):
        """cf_n_ary with n=1 reproduces cf_homographic."""
        x = CF.from_fraction(22, 7)
        # 2x + 1 = (2x + 1) / 1
        r1 = cf_n_ary([x], num=[1, 2], den=[1, 0])
        r2 = cf_homographic(CF.from_fraction(22, 7), 2, 1, 0, 1)
        assert _eval(r1) == _eval(r2)

    def test_n_ary_mul(self):
        """cf_n_ary with n=2 mul coefficients matches cf_mul."""
        a = CF.from_fraction(5, 3)
        b = CF.from_fraction(7, 2)
        result = cf_n_ary([a, b], num=[0, 0, 0, 1], den=[1, 0, 0, 0])
        expected = Fraction(5, 3) * Fraction(7, 2)
        assert _eval(result) == expected


# ---------------------------------------------------------------------------
# Min / Max
# ---------------------------------------------------------------------------


class TestMinMax:
    def test_min(self):
        a = CF.from_fraction(1, 3)
        b = CF.from_fraction(1, 2)
        assert cf_min(a, b).to_fraction() == Fraction(1, 3)

    def test_max(self):
        a = CF.from_fraction(1, 3)
        b = CF.from_fraction(1, 2)
        assert cf_max(a, b).to_fraction() == Fraction(1, 2)
