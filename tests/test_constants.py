"""Tests for named mathematical constants."""

import math
from fractions import Fraction

import pytest

from cfmath import CF, Phi, E, Pi, Tau, EulerGamma, Catalan, Apery, Plastic, Khinchin, convergent


class TestPhi:
    def test_phi_terms(self):
        phi = Phi()
        assert phi.terms == [1]
        assert phi.repeating == [1]

    def test_phi_is_periodic(self):
        assert Phi().is_periodic()

    def test_phi_value(self):
        phi = Phi()
        val = float(convergent(phi.take(30), 29))
        expected = (1 + math.sqrt(5)) / 2
        assert abs(val - expected) < 1e-12

    def test_phi_fibonacci_convergents(self):
        """Convergents of Phi are Fibonacci ratios."""
        phi = Phi()
        fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        for i in range(1, 9):
            c = convergent(phi, i)
            assert c == Fraction(fibs[i + 1], fibs[i])


class TestE:
    def test_e_first_terms(self):
        e = E()
        first_10 = list(e.take(10))
        assert first_10 == [2, 1, 2, 1, 1, 4, 1, 1, 6, 1]

    def test_e_more_terms(self):
        e = E()
        terms = list(e.take(13))
        assert terms[11] == 8

    def test_e_value(self):
        e = E()
        val = float(convergent(e.take(15), 14))
        assert abs(val - math.e) < 1e-10

    def test_e_not_periodic(self):
        assert not E().is_periodic()

    def test_e_pattern(self):
        """The pattern [2; 1, 2k, 1] for k=1,2,3,..."""
        e = E()
        terms = list(e.take(20))
        assert terms[0] == 2
        k = 1
        i = 1
        while i + 2 < len(terms):
            assert terms[i] == 1, f"terms[{i}] should be 1"
            assert terms[i + 1] == 2 * k, f"terms[{i+1}] should be {2*k}"
            assert terms[i + 2] == 1, f"terms[{i+2}] should be 1"
            i += 3
            k += 1


class TestPi:
    def test_pi_first_terms(self):
        pi = Pi()
        assert list(pi.take(8)) == [3, 7, 15, 1, 292, 1, 1, 1]

    def test_pi_term_0(self):
        pi = Pi()
        assert next(iter(pi)) == 3

    def test_pi_value(self):
        pi = Pi()
        val = float(convergent(pi.take(10), 9))
        assert abs(val - math.pi) < 1e-10

    def test_pi_22_over_7(self):
        pi = Pi()
        assert convergent(pi, 1) == Fraction(22, 7)

    def test_pi_355_over_113(self):
        pi = Pi()
        assert convergent(pi, 3) == Fraction(355, 113)

    def test_pi_not_periodic(self):
        assert not Pi().is_periodic()

    def test_pi_independent_iters(self):
        """Two iterators from the same Pi instance stay independent."""
        pi = Pi()
        it1 = pi._iter_from(0)
        it2 = pi._iter_from(0)
        for _ in range(5):
            next(it1)
        assert next(it2) == 3


class TestTau:
    def test_tau_first_terms(self):
        tau = Tau()
        assert list(tau.take(7)) == [6, 3, 1, 1, 7, 2, 146]

    def test_tau_value(self):
        tau = Tau()
        val = float(convergent(tau.take(12), 11))
        assert abs(val - 2 * math.pi) < 1e-8

    def test_tau_not_periodic(self):
        assert not Tau().is_periodic()


class TestEulerGamma:
    def test_euler_gamma_value(self):
        import mpmath
        mpmath.mp.dps = 30
        expected = float(mpmath.euler)
        val = float(convergent(EulerGamma().take(20), 19))
        assert abs(val - expected) < 1e-12

    def test_euler_gamma_first_terms(self):
        # γ ≈ 0.5772... → [0; 1, 1, 2, 1, 2, 1, 4, 3, 13, ...]
        assert list(EulerGamma().take(4)) == [0, 1, 1, 2]


class TestCatalan:
    def test_catalan_first_terms(self):
        assert list(Catalan().take(8)) == [0, 1, 10, 1, 8, 1, 88, 4]

    def test_catalan_value(self):
        import mpmath
        mpmath.mp.dps = 30
        expected = float(mpmath.catalan)
        val = float(convergent(Catalan().take(20), 19))
        assert abs(val - expected) < 1e-12

    def test_catalan_decimal_matches_mpmath(self):
        from cfmath.constants import _catalan_terms_from_decimal, _catalan_terms_mpmath
        assert _catalan_terms_from_decimal(30) == _catalan_terms_mpmath(30)


class TestApery:
    def test_apery_first_terms(self):
        assert list(Apery().take(8)) == [1, 4, 1, 18, 1, 1, 1, 4]

    def test_apery_value(self):
        import mpmath
        mpmath.mp.dps = 30
        expected = float(mpmath.apery)
        val = float(convergent(Apery().take(20), 19))
        assert abs(val - expected) < 1e-12

    def test_apery_decimal_matches_mpmath(self):
        from cfmath.constants import _apery_terms_from_decimal, _apery_terms_mpmath
        assert _apery_terms_from_decimal(30) == _apery_terms_mpmath(30)


class TestPlastic:
    def test_plastic_first_terms(self):
        assert list(Plastic().take(8)) == [1, 3, 12, 1, 1, 3, 2, 3]

    def test_plastic_value(self):
        import mpmath
        mpmath.mp.dps = 30
        rho = mpmath.findroot(lambda x: x**3 - x - 1, 1.3)
        expected = float(rho)
        val = float(convergent(Plastic().take(20), 19))
        assert abs(val - expected) < 1e-12

    def test_plastic_satisfies_polynomial(self):
        """ρ³ - ρ - 1 = 0: verify via CF convergents."""
        p = convergent(Plastic().take(30), 29)
        residual = abs(float(p**3 - p - 1))
        assert residual < 1e-10

    def test_plastic_decimal_matches_mpmath(self):
        from cfmath.constants import _plastic_terms, _plastic_terms_mpmath
        assert _plastic_terms(30) == _plastic_terms_mpmath(30)


class TestKhinchin:
    def test_khinchin_first_terms(self):
        assert list(Khinchin().take(8)) == [2, 1, 2, 5, 1, 1, 2, 1]

    def test_khinchin_value(self):
        import mpmath
        mpmath.mp.dps = 30
        expected = float(mpmath.khinchin)
        val = float(convergent(Khinchin().take(20), 19))
        assert abs(val - expected) < 1e-12
