"""Tests for PolyTransform, a rational function P(x)/Q(x) of one CF source."""

from fractions import Fraction

import pytest

from cfmath import CF, Phi, Sqrt
from cfmath.polyratio import (
    PolyTransform,
    _add,
    _content,
    _divexact,
    _mul,
    _normalize,
    _poly_gcd,
    _poly_to_tensor,
    _primitive,
    _pseudorem,
    _reduce_by_minpoly,
    _strip,
    _sub,
)


def _eval(cf: CF, depth: int = 20) -> Fraction:
    from cfmath import convergents

    convs = list(convergents(cf.take(depth)))
    if not convs:
        raise ValueError("empty CF")
    return convs[-1]


class TestStrip:
    def test_removes_trailing_zeros(self):
        assert _strip([1, 0, 0]) == [1]

    def test_preserves_single_zero(self):
        assert _strip([0]) == [0]

    def test_noop_when_clean(self):
        assert _strip([1, 2, 3]) == [1, 2, 3]


class TestContent:
    def test_gcd_of_coefficients(self):
        assert _content([6, 4, 2]) == 2

    def test_single_coeff(self):
        assert _content([15]) == 15

    def test_zero_poly_returns_one(self):
        assert _content([0]) == 1

    def test_coprime(self):
        assert _content([3, 7]) == 1


class TestPrimitive:
    def test_divides_out_content(self):
        assert _primitive([6, 4, 2]) == [3, 2, 1]

    def test_already_primitive(self):
        assert _primitive([3, 2, 1]) == [3, 2, 1]


class TestAdd:
    def test_same_length(self):
        assert _add([1, 2], [3, 4]) == [4, 6]

    def test_different_length(self):
        assert _add([1, 2], [3]) == [4, 2]

    def test_strips_trailing_zeros(self):
        assert _add([1, 1], [-1, -1]) == [0]


class TestSub:
    def test_simple(self):
        assert _sub([3, 2], [1]) == [2, 2]

    def test_self_is_zero(self):
        assert _sub([1, 2, 3], [1, 2, 3]) == [0]


class TestMul:
    def test_constant_times_constant(self):
        assert _mul([3], [4]) == [12]

    def test_linear_times_linear(self):
        assert _mul([1, 1], [1, 1]) == [1, 2, 1]

    def test_degree_additive(self):
        assert _mul([0, 1], [0, 0, 1]) == [0, 0, 0, 1]


class TestPseudorem:
    def test_remainder(self):
        assert _pseudorem([-1, 0, 1], [-1, 1]) == [0]

    def test_lower_degree_unchanged(self):
        assert _pseudorem([1], [0, 1]) == [1]


class TestPolyGcd:
    def test_common_linear_factor(self):
        g = _poly_gcd([-1, 0, 1], [-1, 1])
        assert g == [-1, 1] or g == [1, -1]

    def test_coprime(self):
        assert _poly_gcd([1, 0, 1], [0, 1]) == [1]

    def test_identical(self):
        assert _poly_gcd([1, 2], [1, 2]) == [1, 2]


class TestDivexact:
    def test_linear_factor(self):
        assert _divexact([-1, 0, 1], [-1, 1]) == [1, 1]

    def test_constant_divisor(self):
        assert _divexact([6, 4], [2]) == [3, 2]


class TestNormalize:
    def test_integer_content(self):
        num, den = _normalize([0, 2], [2])
        assert num == [0, 1]
        assert den == [1]

    def test_polynomial_gcd(self):
        num, den = _normalize([-1, 0, 1], [-1, 1])
        assert num == [1, 1]
        assert den == [1]

    def test_negative_leading_den_flipped(self):
        num, den = _normalize([1], [-2])
        assert den[-1] > 0
        assert Fraction(num[0], den[0]) == Fraction(-1, 2)

    def test_zero_numerator(self):
        num, den = _normalize([0], [5])
        assert num == [0]
        assert Fraction(num[0], den[0]) == Fraction(0)

    def test_zero_denominator_raises(self):
        with pytest.raises(ZeroDivisionError):
            _normalize([1], [0])


class TestReduceByMinpoly:
    def test_linear_unchanged(self):
        p, d = _reduce_by_minpoly([3, 5], 1, -1, -1)
        assert p == [3, 5]
        assert d == 1

    def test_phi_minpoly(self):
        p, d = _reduce_by_minpoly([0, 0, 1], 1, -1, -1)
        assert p == [1, 1]
        assert d == 1

    def test_sqrt2_minpoly(self):
        p, d = _reduce_by_minpoly([0, 0, 1], 1, 0, -2)
        assert p == [2]
        assert d == 1


class TestPolyToTensor:
    def test_n1(self):
        assert _poly_to_tensor([3, 5], 1) == [3, 5]

    def test_n2_quadratic(self):
        assert _poly_to_tensor([1, 2, 3], 2) == [1, 2, 0, 3]

    def test_n2_constant(self):
        assert _poly_to_tensor([7], 2) == [7, 0, 0, 0]


class TestPolyTransformConstruct:
    def test_constant_degree_zero(self):
        src = CF.from_fraction(7, 3)
        pr = PolyTransform(src, [3], [4])
        assert _eval(pr) == Fraction(3, 4)

    def test_linear_identity(self):
        src = CF.from_fraction(7, 3)
        pr = PolyTransform(src, [0, 1], [1])
        assert _eval(pr) == Fraction(7, 3)

    def test_linear_shift(self):
        src = CF.from_fraction(5, 11)
        pr = PolyTransform(src, [2, 1], [1])
        assert _eval(pr) == Fraction(5, 11) + 2

    def test_linear_scale(self):
        src = CF.from_fraction(5, 11)
        pr = PolyTransform(src, [0, 3], [1])
        assert _eval(pr) == Fraction(15, 11)

    def test_linear_mobius(self):
        src = CF.from_fraction(5, 7)
        pr = PolyTransform(src, [1, 2], [3, 1])
        expected = (2 * Fraction(5, 7) + 1) / (Fraction(5, 7) + 3)
        assert abs(float(_eval(pr)) - float(expected)) < 1e-10

    def test_normalizes_on_construct(self):
        src = CF.from_fraction(1, 2)
        pr = PolyTransform(src, [0, 4], [2])
        assert pr._num == [0, 2]
        assert pr._den == [1]

    def test_repr(self):
        src = CF.from_fraction(1, 2)
        pr = PolyTransform(src, [1, 2], [3])
        r = repr(pr)
        assert "PolyTransform" in r
        assert "num=" in r
        assert "den=" in r


class TestPolyTransformAddition:
    def test_add_same_source(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 1], [1])
        b = PolyTransform(src, [0, 1], [1])
        assert _eval(a + b) == Fraction(6, 7)

    def test_add_int(self):
        src = CF.from_fraction(1, 3)
        a = PolyTransform(src, [0, 1], [1])
        assert _eval(a + 2) == Fraction(1, 3) + 2

    def test_radd_int(self):
        src = CF.from_fraction(1, 3)
        a = PolyTransform(src, [0, 1], [1])
        assert _eval(2 + a) == 2 + Fraction(1, 3)

    def test_add_different_source_falls_back(self):
        src1 = CF.from_fraction(1, 3)
        src2 = CF.from_fraction(1, 5)
        a = PolyTransform(src1, [0, 1], [1])
        b = PolyTransform(src2, [0, 1], [1])
        result = a + b
        assert abs(float(_eval(result)) - (1 / 3 + 1 / 5)) < 1e-8


class TestPolyTransformSubtraction:
    def test_sub_same_source(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 1], [1])
        b = PolyTransform(src, [1], [1])
        assert _eval(a - b) == Fraction(3, 7) - 1

    def test_sub_self_is_zero(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 1], [1])
        r = a - a
        assert _eval(r) == Fraction(0)

    def test_sub_int(self):
        src = CF.from_fraction(4, 3)
        a = PolyTransform(src, [0, 1], [1])
        assert _eval(a - 1) == Fraction(4, 3) - 1

    def test_rsub_int(self):
        src = CF.from_fraction(1, 3)
        a = PolyTransform(src, [0, 1], [1])
        assert _eval(2 - a) == 2 - Fraction(1, 3)


class TestPolyTransformMultiplication:
    def test_mul_same_source(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 1], [1])
        b = PolyTransform(src, [1], [1])
        assert _eval(a * b) == Fraction(3, 7)

    def test_mul_int(self):
        src = CF.from_fraction(2, 5)
        a = PolyTransform(src, [0, 1], [1])
        assert _eval(a * 3) == Fraction(6, 5)

    def test_rmul_int(self):
        src = CF.from_fraction(2, 5)
        a = PolyTransform(src, [0, 1], [1])
        assert _eval(3 * a) == Fraction(6, 5)

    def test_mul_int_zero(self):
        src = CF.from_fraction(2, 5)
        a = PolyTransform(src, [0, 1], [1])
        assert _eval(a * 0) == Fraction(0)


class TestPolyTransformDivision:
    def test_div_same_source_identity(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 1], [1])
        b = PolyTransform(src, [0, 1], [1])
        r = a / b
        assert _eval(r) == Fraction(1)

    def test_div_int(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 3], [1])
        assert _eval(a / 3) == Fraction(3, 7)

    def test_rdiv_int(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 1], [1])
        assert _eval(21 / a) == Fraction(49)

    def test_div_by_zero_raises(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 1], [1])
        with pytest.raises(ZeroDivisionError):
            _ = a / 0

    def test_mul_then_div_cancels(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 1], [1])
        b = PolyTransform(src, [1, 1], [1])
        r = (a * b) / b
        assert isinstance(r, PolyTransform)
        assert _eval(r) == Fraction(3, 7)


class TestPolyTransformNegation:
    def test_neg(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 1], [1])
        assert _eval(-a) == Fraction(-3, 7)

    def test_double_neg(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 1], [1])
        assert _eval(-(-a)) == Fraction(3, 7)

    def test_neg_then_add(self):
        src = CF.from_fraction(3, 7)
        a = PolyTransform(src, [0, 1], [1])
        assert _eval(a + (-a)) == Fraction(0)


class TestPolyTransformPeriodicSource:
    def test_phi_squared_reduces_degree(self):
        phi = Phi()
        pr = PolyTransform(phi, [0, 0, 1], [1])
        assert max(len(pr._num) - 1, len(pr._den) - 1) <= 1

    def test_phi_squared_value(self):
        phi = Phi()
        pr = PolyTransform(phi, [0, 0, 1], [1])
        phi_val = float(_eval(PolyTransform(phi, [0, 1], [1])))
        assert abs(float(_eval(pr)) - (phi_val + 1)) < 1e-8

    def test_sqrt2_squared_is_two(self):
        s2 = Sqrt(2)
        pr = PolyTransform(s2, [0, 0, 1], [1])
        assert pr._num == [2]
        assert pr._den == [1]
        assert _eval(pr) == Fraction(2)

    def test_phi_minus_phi_is_zero(self):
        phi = Phi()
        a = PolyTransform(phi, [0, 1], [1])
        assert _eval(a - a) == Fraction(0)

    def test_phi_identity_transform(self):
        phi = Phi()
        a = PolyTransform(phi, [0, 1], [1])
        phi_val = float(_eval(a))
        assert abs(phi_val - (1 + 5**0.5) / 2) < 1e-8
