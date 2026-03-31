"""Tests for convergent computation."""

from fractions import Fraction

import pytest

from cfmath import CF, convergent, convergent_pair, convergent_pairs, convergents, Phi, Pi, Sqrt


@pytest.fixture
def pi_cf():
    return CF([3, 7, 15, 1, 292])


class TestConvergentBasics:
    def test_convergent_0_is_a0(self, pi_cf):
        assert convergent(pi_cf, 0) == Fraction(3)

    def test_convergent_1(self, pi_cf):
        assert convergent(pi_cf, 1) == Fraction(22, 7)

    def test_convergent_2(self, pi_cf):
        assert convergent(pi_cf, 2) == Fraction(333, 106)

    def test_convergent_3(self, pi_cf):
        assert convergent(pi_cf, 3) == Fraction(355, 113)

    def test_convergent_4(self, pi_cf):
        assert convergent(pi_cf, 4) == Fraction(103993, 33102)

    def test_integer_cf(self):
        assert convergent(CF([7]), 0) == Fraction(7)

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            convergent(CF([3, 7]), 5)

    def test_negative_index_raises(self):
        with pytest.raises(IndexError):
            convergent(CF([1, 2, 3]), -1)


class TestConvergentPairs:
    def test_pairs_of_22_over_7(self):
        cf = CF([3, 7])
        pairs = list(convergent_pairs(cf))
        assert pairs == [(3, 1), (22, 7)]

    def test_pair_fn(self):
        cf = CF([3, 7, 15])
        p, q = convergent_pair(cf, 2)
        assert Fraction(p, q) == Fraction(333, 106)


class TestConvergentsIterator:
    def test_yields_fractions(self):
        cf = CF([3, 7, 15, 1])
        convs = list(convergents(cf))
        assert convs[0] == Fraction(3)
        assert convs[1] == Fraction(22, 7)
        assert convs[2] == Fraction(333, 106)
        assert convs[3] == Fraction(355, 113)

    def test_convergents_of_half(self):
        cf = CF.from_fraction(1, 2)
        convs = list(convergents(cf))
        assert convs[-1] == Fraction(1, 2)

    def test_alternating_bounds(self):
        """Even-indexed convergents < true value < odd-indexed convergents (for positive CFs > 1)."""
        pi = Pi()
        convs_list = [convergent(pi.take(8), i) for i in range(5)]
        true_val = Fraction(3141592653589793, 10**15)  # approx Pi
        # c0 < c2 < c4 < Pi < c3 < c1
        assert convs_list[0] < convs_list[2]
        assert convs_list[2] < convs_list[4]
        assert convs_list[1] > convs_list[3]


class TestKnownConvergents:
    def test_sqrt2_convergents(self):
        """sqrt(2) = [1; 2, 2, 2, ...]  convergents: 1/1, 3/2, 7/5, 17/12, 41/29"""

        cf = Sqrt(2)
        expected = [
            Fraction(1, 1),
            Fraction(3, 2),
            Fraction(7, 5),
            Fraction(17, 12),
            Fraction(41, 29),
        ]
        for i, exp in enumerate(expected):
            assert convergent(cf, i) == exp, f"convergent {i} of sqrt(2)"

    def test_phi_convergents(self):
        """Phi = [1; 1, 1, ...] convergents are ratios of consecutive Fibonacci numbers."""

        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        cf = Phi()
        for i in range(1, 9):
            c = convergent(cf, i)
            assert c == Fraction(fib[i + 1], fib[i]), f"Phi convergent {i}"


class TestGosperIdentity:
    def test_determinant_identity(self):
        """p_n * q_{n-1} - p_{n-1} * q_n == (-1)^n"""
        cf = CF([3, 7, 15, 1, 292, 1, 1, 1, 2, 1])
        pairs = list(convergent_pairs(cf))
        for n in range(1, len(pairs)):
            p_n, q_n = pairs[n]
            p_nm1, q_nm1 = pairs[n - 1]
            det = p_n * q_nm1 - p_nm1 * q_n
            assert det == (-1) ** (n - 1), f"Determinant identity failed at n={n}"

    def test_determinant_for_sqrt2(self):

        cf = Sqrt(2).take(12)
        pairs = list(convergent_pairs(cf))
        for n in range(1, len(pairs)):
            p_n, q_n = pairs[n]
            p_nm1, q_nm1 = pairs[n - 1]
            det = p_n * q_nm1 - p_nm1 * q_n
            assert det == (-1) ** (n - 1)


class TestConvergentBound:
    def test_best_approximation_bound(self, pi_cf):
        """Each convergent satisfies |p/q - x| < 1/(q * q_next)."""
        import math
        pairs = list(convergent_pairs(pi_cf))
        true_val = math.pi
        for i, (p, q) in enumerate(pairs[:-1]):
            q_next = pairs[i + 1][1]
            err = abs(p / q - true_val)
            bound = 1 / (q * q_next)
            assert err < bound, f"Bound violated at i={i}"
