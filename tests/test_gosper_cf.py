"""Tests for GosperMono, GosperBi, GosperGeneric CF subclasses."""

from fractions import Fraction

import pytest

from cfmath import CF
from cfmath.convergents import convergents
from cfmath.gosper import cf_homographic
from cfmath.gosper_cf import GosperBi, GosperGeneric, GosperMono


def _eval(cf: CF, depth: int = 15) -> Fraction:
    convs = list(convergents(cf.take(depth)))
    return convs[-1]


# ===========================================================================
# GosperMono
# ===========================================================================


class TestGosperMonoBasic:
    def test_is_a_cf(self):
        x = CF.from_fraction(3, 7)
        assert isinstance(GosperMono(x, 1, 0, 0, 1), CF)

    def test_identity_yields_source_value(self):
        x = CF.from_fraction(3, 7)
        assert _eval(GosperMono(x, 1, 0, 0, 1)) == Fraction(3, 7)

    def test_scale(self):
        x = CF.from_fraction(3, 7)
        assert _eval(GosperMono(x, 3, 0, 0, 1)) == Fraction(9, 7)

    def test_shift(self):
        x = CF.from_fraction(3, 7)
        assert _eval(GosperMono(x, 1, 5, 0, 1)) == Fraction(3, 7) + 5

    def test_reciprocal(self):
        x = CF.from_fraction(3, 7)
        assert _eval(GosperMono(x, 0, 1, 1, 0)) == Fraction(7, 3)

    def test_matches_cf_homographic(self):
        x = CF.from_fraction(5, 11)
        assert _eval(GosperMono(x, 3, 1, 2, 5)) == _eval(cf_homographic(x, 3, 1, 2, 5))

    def test_stores_source_and_matrix(self):
        x = CF.from_fraction(3, 7)
        g = GosperMono(x, 2, 3, 1, 4)
        assert g._source_cf is x
        assert g._mono_mat == (2, 3, 1, 4)

    def test_repr(self):
        x = CF.from_fraction(1, 1)
        assert repr(GosperMono(x, 2, 3, 1, 4)) == "GosperMono(source, 2, 3, 1, 4)"

    def test_gcd_normalization_in_init(self):
        x = CF.from_fraction(3, 7)
        g = GosperMono(x, 6, 4, 2, 8)  # GCD=2 → (3,2,1,4)
        assert g._mono_mat == (3, 2, 1, 4)
        assert _eval(g) == _eval(GosperMono(x, 3, 2, 1, 4))

    def test_gcd_normalization_after_ops(self):
        x = CF.from_fraction(3, 7)
        g = GosperMono(x, 1, 0, 0, 1)
        result = (g * 4) / 2  # (4a,4b,c,d) then /2 → (4a,4b,2c,2d) → GCD=2
        assert result._mono_mat == (2, 0, 0, 1)


class TestGosperMonoScalarOps:
    """Each op stays in-class and produces the correct numeric value."""

    def setup_method(self):
        self.x = CF.from_fraction(3, 7)
        self.val = Fraction(3, 7)

    def _gm(self):
        return GosperMono(self.x, 1, 0, 0, 1)  # identity: g(x) = x

    def test_mul_right_stays_in_class(self):
        result = self._gm() * 5
        assert isinstance(result, GosperMono)
        assert _eval(result) == 5 * self.val

    def test_mul_left_stays_in_class(self):
        result = 5 * self._gm()
        assert isinstance(result, GosperMono)
        assert _eval(result) == 5 * self.val

    def test_add_right_stays_in_class(self):
        result = self._gm() + 2
        assert isinstance(result, GosperMono)
        assert _eval(result) == self.val + 2

    def test_add_left_stays_in_class(self):
        result = 2 + self._gm()
        assert isinstance(result, GosperMono)
        assert _eval(result) == self.val + 2

    def test_sub_right_stays_in_class(self):
        result = self._gm() - 2
        assert isinstance(result, GosperMono)
        assert _eval(result) == self.val - 2

    def test_sub_left_stays_in_class(self):
        result = 5 - self._gm()
        assert isinstance(result, GosperMono)
        assert _eval(result) == 5 - self.val

    def test_neg_stays_in_class(self):
        result = -self._gm()
        assert isinstance(result, GosperMono)
        assert _eval(result) == -self.val

    def test_truediv_stays_in_class(self):
        result = self._gm() / 3
        assert isinstance(result, GosperMono)
        assert _eval(result) == self.val / 3

    def test_rtruediv_stays_in_class(self):
        result = 1 / self._gm()
        assert isinstance(result, GosperMono)
        assert _eval(result) == 1 / self.val

    def test_truediv_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            self._gm() / 0

    def test_fraction_mul_stays_in_class(self):
        result = self._gm() * Fraction(2, 3)
        assert isinstance(result, GosperMono)
        assert _eval(result) == self.val * Fraction(2, 3)

    def test_fraction_add_stays_in_class(self):
        result = self._gm() + Fraction(1, 2)
        assert isinstance(result, GosperMono)
        assert _eval(result) == self.val + Fraction(1, 2)

    def test_fraction_sub_stays_in_class(self):
        result = self._gm() - Fraction(1, 4)
        assert isinstance(result, GosperMono)
        assert _eval(result) == self.val - Fraction(1, 4)

    def test_fraction_rsub_stays_in_class(self):
        result = Fraction(1, 2) - self._gm()
        assert isinstance(result, GosperMono)
        assert _eval(result) == Fraction(1, 2) - self.val

    def test_fraction_div_stays_in_class(self):
        result = self._gm() / Fraction(2, 3)
        assert isinstance(result, GosperMono)
        assert _eval(result) == self.val / Fraction(2, 3)

    def test_fraction_rdiv_stays_in_class(self):
        # Use a non-identity source to avoid d=0 in the result matrix.
        g = GosperMono(self.x, 2, 1, 0, 1)  # 2x+1
        result = Fraction(1, 2) / g
        assert isinstance(result, GosperMono)
        assert _eval(result) == Fraction(1, 2) / (2 * self.val + 1)

    def test_chained_ops_stay_in_class(self):
        g = self._gm()
        result = 4 - g * 2
        assert isinstance(result, GosperMono)
        assert _eval(result) == 4 - 2 * self.val

    def test_matrix_is_pre_multiplied(self):
        g = self._gm()
        result = (g + 1) * 3
        assert isinstance(result, GosperMono)
        assert result._mono_mat == (3, 3, 0, 1)


class TestGosperMonoSameSourceOps:
    """Binary ops on two GosperMono with the same source return GosperBi."""

    def setup_method(self):
        self.x = CF.from_fraction(3, 7)
        self.xv = Fraction(3, 7)

    def _gm(self, a, b, c, d):
        return GosperMono(self.x, a, b, c, d)

    def test_add_same_source_returns_gosperbi(self):
        g1 = self._gm(1, 0, 0, 1)  # x
        g2 = self._gm(2, 0, 0, 1)  # 2x
        result = g1 + g2
        assert isinstance(result, GosperBi)
        assert _eval(result) == self.xv + 2 * self.xv

    def test_sub_same_source_returns_gosperbi(self):
        g1 = self._gm(1, 0, 0, 1)  # x
        g2 = self._gm(2, 0, 0, 1)  # 2x
        result = g1 - g2
        assert isinstance(result, GosperBi)
        assert _eval(result) == self.xv - 2 * self.xv

    def test_mul_same_source_returns_gosperbi(self):
        g1 = self._gm(1, 0, 0, 1)  # x
        g2 = self._gm(2, 1, 0, 1)  # 2x+1
        result = g1 * g2
        assert isinstance(result, GosperBi)
        assert _eval(result) == self.xv * (2 * self.xv + 1)

    def test_div_same_source_returns_gosperbi(self):
        g1 = self._gm(1, 0, 0, 1)  # x
        g2 = self._gm(2, 1, 0, 1)  # 2x+1
        result = g1 / g2
        assert isinstance(result, GosperBi)
        assert _eval(result) == self.xv / (2 * self.xv + 1)

    def test_different_source_add_falls_to_cf(self):
        y = CF.from_fraction(5, 11)
        g1 = GosperMono(self.x, 1, 0, 0, 1)
        g2 = GosperMono(y, 1, 0, 0, 1)
        result = g1 + g2
        assert not isinstance(result, GosperMono)
        assert not isinstance(result, GosperBi)


class TestGosperMonoComposition:
    """@ composes matrices and is equivalent to nested cf_homographic calls."""

    def test_mono_at_mono_matches_nested_homographic(self):
        x = CF.from_fraction(7, 3)
        g1 = GosperMono(x, 1, 5, 0, 1)  # x + 5
        g2 = GosperMono(x, -3, 2, 0, 1)  # 2 - 3y
        composed = g2 @ g1  # 2 - 3(x+5) = -3x - 13

        step1 = cf_homographic(x, 1, 5, 0, 1)
        longhand = cf_homographic(step1, -3, 2, 0, 1)

        assert isinstance(composed, GosperMono)
        assert _eval(composed) == _eval(longhand) == Fraction(-20)

    def test_mono_at_mono_same_source(self):
        x = CF.from_fraction(5, 11)
        g1 = GosperMono(x, 1, 0, 0, 1)  # identity on x
        g2 = GosperMono(x, 2, 0, 0, 1)  # 2y (same source x)
        composed = g2 @ g1
        assert isinstance(composed, GosperMono)
        assert composed._source_cf is x
        assert _eval(composed) == 2 * Fraction(5, 11)

    def test_mono_at_mono_wrapped_source_allowed(self):
        x = CF.from_fraction(5, 11)
        g1 = GosperMono(x, 1, 3, 0, 1)  # x+3
        g2 = GosperMono(g1, 2, 0, 0, 1)  # 2*(x+3) via g1 as source
        composed = g2 @ g1  # g2(g1(x)) = 2*(x+3)
        assert isinstance(composed, GosperMono)
        assert composed._source_cf is x
        assert _eval(composed) == 2 * (Fraction(5, 11) + 3)

    def test_mono_at_mono_mismatched_sources_raises(self):
        x = CF.from_fraction(5, 11)
        y = CF.from_fraction(7, 3)
        g1 = GosperMono(x, 1, 0, 0, 1)
        g2 = GosperMono(y, 2, 0, 0, 1)
        with pytest.raises(ValueError, match="mismatched sources"):
            g2 @ g1

    def test_mono_at_mono_associative(self):
        x = CF.from_fraction(2, 9)
        g1 = GosperMono(x, 1, 3, 0, 1)  # x+3
        g2 = GosperMono(x, 2, 0, 0, 1)  # 2y
        g3 = GosperMono(x, 0, 1, 1, 0)  # 1/z
        assert _eval((g3 @ g2) @ g1) == _eval(g3 @ (g2 @ g1))

    def test_mono_at_plain_cf_wraps_new_source(self):
        x = CF.from_fraction(5, 11)
        new_src = CF.from_fraction(7, 3)
        gm = GosperMono(x, 2, 1, 0, 1)  # 2x+1
        wrapped = gm @ new_src
        assert isinstance(wrapped, GosperMono)
        assert wrapped._source_cf is new_src
        assert _eval(wrapped) == 2 * Fraction(7, 3) + 1

    def test_three_transforms_chain(self):
        x = CF.from_fraction(5, 3)
        g1 = GosperMono(x, 1, 7, 0, 1)
        g2 = GosperMono(x, 2, 0, 0, 1)
        g3 = GosperMono(x, 0, 1, 1, 0)
        composed = g3 @ g2 @ g1

        step1 = cf_homographic(x, 1, 7, 0, 1)
        step2 = cf_homographic(step1, 2, 0, 0, 1)
        longhand = cf_homographic(step2, 0, 1, 1, 0)

        assert _eval(composed) == _eval(longhand)

    def test_chained_scalar_then_compose(self):
        x = CF.from_fraction(7, 3)
        g_id = GosperMono(x, 1, 0, 0, 1)
        g2 = 2 - g_id * 3  # 2-3x as GosperMono
        g1 = GosperMono(x, 1, 5, 0, 1)  # x+5
        composed = g2 @ g1  # 2 - 3(x+5) = -3x-13

        step1 = cf_homographic(x, 1, 5, 0, 1)
        longhand = cf_homographic(step1, -3, 2, 0, 1)

        assert isinstance(composed, GosperMono)
        assert _eval(composed) == _eval(longhand) == Fraction(-20)


# ===========================================================================
# GosperBi
# ===========================================================================


class TestGosperBiBasic:
    def test_is_a_cf(self):
        x = CF.from_fraction(3, 7)
        y = CF.from_fraction(5, 11)
        assert isinstance(GosperBi(x, y, 0, 1, 1, 0, 0, 0, 0, 1), CF)

    def test_addition_formula(self):
        x = CF.from_fraction(3, 7)
        y = CF.from_fraction(5, 11)
        g = GosperBi(x, y, 0, 1, 1, 0, 0, 0, 0, 1)  # x+y
        assert _eval(g) == Fraction(3, 7) + Fraction(5, 11)

    def test_multiplication_formula(self):
        x = CF.from_fraction(3, 7)
        y = CF.from_fraction(5, 11)
        g = GosperBi(x, y, 1, 0, 0, 0, 0, 0, 0, 1)  # x*y
        assert _eval(g) == Fraction(3, 7) * Fraction(5, 11)

    def test_stores_sources_and_matrix(self):
        x = CF.from_fraction(1, 2)
        y = CF.from_fraction(1, 3)
        g = GosperBi(x, y, 1, 2, 3, 4, 5, 6, 7, 8)
        assert g._source_x is x
        assert g._source_y is y
        assert g._bi_mat == (1, 2, 3, 4, 5, 6, 7, 8)

    def test_gcd_normalization_in_init(self):
        x = CF.from_fraction(3, 7)
        y = CF.from_fraction(5, 11)
        # x+y with all coefficients doubled: GCD=2 → normalised back
        g = GosperBi(x, y, 0, 2, 2, 0, 0, 0, 0, 2)
        assert g._bi_mat == (0, 1, 1, 0, 0, 0, 0, 1)
        assert _eval(g) == Fraction(3, 7) + Fraction(5, 11)


class TestGosperBiScalarOps:
    def setup_method(self):
        self.x = CF.from_fraction(3, 7)
        self.y = CF.from_fraction(5, 11)
        self.xy_sum = Fraction(3, 7) + Fraction(5, 11)

    def _g(self):
        return GosperBi(self.x, self.y, 0, 1, 1, 0, 0, 0, 0, 1)  # x+y

    def test_mul_stays_in_class(self):
        result = self._g() * 3
        assert isinstance(result, GosperBi)
        assert _eval(result) == 3 * self.xy_sum

    def test_rmul_stays_in_class(self):
        result = 3 * self._g()
        assert isinstance(result, GosperBi)
        assert _eval(result) == 3 * self.xy_sum

    def test_add_stays_in_class(self):
        result = self._g() + 7
        assert isinstance(result, GosperBi)
        assert _eval(result) == self.xy_sum + 7

    def test_radd_stays_in_class(self):
        result = 7 + self._g()
        assert isinstance(result, GosperBi)
        assert _eval(result) == self.xy_sum + 7

    def test_sub_stays_in_class(self):
        result = self._g() - 1
        assert isinstance(result, GosperBi)
        assert _eval(result) == self.xy_sum - 1

    def test_rsub_stays_in_class(self):
        result = 10 - self._g()
        assert isinstance(result, GosperBi)
        assert _eval(result) == 10 - self.xy_sum

    def test_neg_stays_in_class(self):
        result = -self._g()
        assert isinstance(result, GosperBi)
        assert _eval(result) == -self.xy_sum

    def test_truediv_stays_in_class(self):
        result = self._g() / 2
        assert isinstance(result, GosperBi)
        assert _eval(result) == self.xy_sum / 2

    def test_rtruediv_stays_in_class(self):
        result = 1 / self._g()
        assert isinstance(result, GosperBi)
        assert _eval(result) == 1 / self.xy_sum

    def test_truediv_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            self._g() / 0

    def test_fraction_mul_stays_in_class(self):
        result = self._g() * Fraction(1, 3)
        assert isinstance(result, GosperBi)
        assert _eval(result) == self.xy_sum * Fraction(1, 3)

    def test_fraction_add_stays_in_class(self):
        result = self._g() + Fraction(1, 4)
        assert isinstance(result, GosperBi)
        assert _eval(result) == self.xy_sum + Fraction(1, 4)

    def test_fraction_div_stays_in_class(self):
        result = self._g() / Fraction(3, 2)
        assert isinstance(result, GosperBi)
        assert _eval(result) == self.xy_sum / Fraction(3, 2)


class TestGosperBiRmatmul:
    def test_gosper_scale_at_gosperbi(self):
        x = CF.from_fraction(3, 7)
        y = CF.from_fraction(5, 11)
        g = GosperBi(x, y, 1, 0, 0, 0, 0, 0, 0, 1)  # x*y
        assert _eval(g) == Fraction(3, 7) * Fraction(5, 11)


# ===========================================================================
# GosperGeneric
# ===========================================================================


class TestGosperGenericBasic:
    def test_is_a_cf(self):
        x = CF.from_fraction(3, 7)
        y = CF.from_fraction(5, 11)
        assert isinstance(GosperGeneric([x, y], [0, 1, 1, 0], [1, 0, 0, 0]), CF)

    def test_binary_add(self):
        x = CF.from_fraction(3, 7)
        y = CF.from_fraction(5, 11)
        g = GosperGeneric([x, y], [0, 1, 1, 0], [1, 0, 0, 0])
        assert _eval(g) == Fraction(3, 7) + Fraction(5, 11)

    def test_binary_mul(self):
        x = CF.from_fraction(3, 7)
        y = CF.from_fraction(5, 11)
        g = GosperGeneric([x, y], [0, 0, 0, 1], [1, 0, 0, 0])  # xy
        assert _eval(g) == Fraction(3, 7) * Fraction(5, 11)

    def test_wrong_array_length_raises(self):
        x = CF.from_fraction(1, 2)
        with pytest.raises(ValueError):
            GosperGeneric([x], [1, 0, 0], [0, 0, 1])  # n=1 needs length 2

    def test_stores_sources_and_arrays(self):
        x = CF.from_fraction(1, 2)
        y = CF.from_fraction(1, 3)
        num = [0, 1, 1, 0]
        den = [1, 0, 0, 0]
        g = GosperGeneric([x, y], num, den)
        assert g._source_cfs[0] is x
        assert g._source_cfs[1] is y
        assert g._gen_num == num
        assert g._gen_den == den

    def test_gcd_normalization_in_init(self):
        x = CF.from_fraction(3, 7)
        y = CF.from_fraction(5, 11)
        g = GosperGeneric([x, y], [0, 2, 2, 0], [2, 0, 0, 0])
        assert g._gen_num == [0, 1, 1, 0]
        assert g._gen_den == [1, 0, 0, 0]
        assert _eval(g) == Fraction(3, 7) + Fraction(5, 11)


class TestGosperGenericScalarOps:
    def setup_method(self):
        self.x = CF.from_fraction(3, 7)
        self.y = CF.from_fraction(5, 11)
        self.val = Fraction(3, 7) + Fraction(5, 11)

    def _g(self):
        return GosperGeneric([self.x, self.y], [0, 1, 1, 0], [1, 0, 0, 0])

    def test_mul_stays_in_class(self):
        result = self._g() * 3
        assert isinstance(result, GosperGeneric)
        assert _eval(result) == 3 * self.val

    def test_add_stays_in_class(self):
        result = self._g() + 4
        assert isinstance(result, GosperGeneric)
        assert _eval(result) == self.val + 4

    def test_sub_stays_in_class(self):
        result = self._g() - 1
        assert isinstance(result, GosperGeneric)
        assert _eval(result) == self.val - 1

    def test_rsub_stays_in_class(self):
        result = 10 - self._g()
        assert isinstance(result, GosperGeneric)
        assert _eval(result) == 10 - self.val

    def test_neg_stays_in_class(self):
        result = -self._g()
        assert isinstance(result, GosperGeneric)
        assert _eval(result) == -self.val

    def test_truediv_stays_in_class(self):
        result = self._g() / 2
        assert isinstance(result, GosperGeneric)
        assert _eval(result) == self.val / 2

    def test_rtruediv_stays_in_class(self):
        result = 1 / self._g()
        assert isinstance(result, GosperGeneric)
        assert _eval(result) == 1 / self.val

    def test_truediv_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            self._g() / 0

    def test_fraction_mul_stays_in_class(self):
        result = self._g() * Fraction(2, 5)
        assert isinstance(result, GosperGeneric)
        assert _eval(result) == self.val * Fraction(2, 5)

    def test_fraction_add_stays_in_class(self):
        result = self._g() + Fraction(1, 3)
        assert isinstance(result, GosperGeneric)
        assert _eval(result) == self.val + Fraction(1, 3)


class TestGosperGenericRmatmul:
    def test_gospergeneric_eval(self):
        x = CF.from_fraction(3, 7)
        y = CF.from_fraction(5, 11)
        g = GosperGeneric([x, y], [0, 1, 1, 0], [1, 0, 0, 0])  # x+y
        assert _eval(g) == Fraction(3, 7) + Fraction(5, 11)
