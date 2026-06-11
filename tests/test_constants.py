"""Tests for named mathematical constants."""

import math
from fractions import Fraction

from cfmath import (
    Apery,
    Catalan,
    CF,
    E,
    EulerGamma,
    Gamma,
    Khinchin,
    Phi,
    Pi,
    Plastic,
    Tau,
    Zeta,
    convergent,
)


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
            assert terms[i + 1] == 2 * k, f"terms[{i + 1}] should be {2 * k}"
            assert terms[i + 2] == 1, f"terms[{i + 2}] should be 1"
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
        import mpmath

        from cfmath._backend import _extract_cf_terms
        from cfmath.constants import _catalan_terms_from_decimal

        mpmath.mp.dps = 200
        reference = _extract_cf_terms(mpmath.catalan)[:30]
        assert _catalan_terms_from_decimal(30) == reference


class TestExtractCfTerms:
    def test_huge_partial_quotient_not_truncated_by_float_underflow(self):
        """A partial quotient large enough that frac underflows a double must
        still be extracted, since mpmath retains the precision to continue."""
        import mpmath

        from cfmath._backend import _extract_cf_terms

        mpmath.mp.dps = 700
        # 5 + 1e-330: next partial quotient ~10^330; frac=1e-330 underflows a
        # double to 0.0, but mpmath at 700 dps can still resolve it.
        assert float(mpmath.mpf(10) ** -330) == 0.0  # the underflow that bit
        terms = _extract_cf_terms(mpmath.mpf(5) + mpmath.mpf(10) ** -330)
        assert terms[0] == 5
        assert len(str(terms[1])) in (330, 331)  # the ~10^330 term survives


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
        import mpmath

        from cfmath._backend import _extract_cf_terms
        from cfmath.constants import _apery_terms_from_decimal

        mpmath.mp.dps = 200
        reference = _extract_cf_terms(mpmath.apery)[:30]
        assert _apery_terms_from_decimal(30) == reference


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
        import mpmath

        from cfmath._backend import _extract_cf_terms
        from cfmath.constants import _plastic_terms

        mpmath.mp.dps = 200
        rho = mpmath.findroot(lambda z: z**3 - z - 1, mpmath.mpf("1.3"))
        reference = _extract_cf_terms(rho)[:30]
        assert _plastic_terms(30) == reference


class TestKhinchin:
    def test_khinchin_first_terms(self):
        assert list(Khinchin().take(8)) == [2, 1, 2, 5, 1, 1, 2, 1]

    def test_khinchin_value(self):
        import mpmath

        mpmath.mp.dps = 30
        expected = float(mpmath.khinchin)
        val = float(convergent(Khinchin().take(20), 19))
        assert abs(val - expected) < 1e-12


class TestZeta:
    def test_bernoulli_numbers_are_exact(self):
        from cfmath.special import _bernoulli

        assert _bernoulli(0) == Fraction(1)
        assert _bernoulli(1) == Fraction(-1, 2)
        assert _bernoulli(6) == Fraction(1, 42)

    def test_zeta_bad_argument_raises(self):
        for s in (1, 0, -1, Fraction(5, 2)):
            try:
                Zeta(s)  # type: ignore[arg-type]
            except ValueError:
                pass
            else:
                raise AssertionError(f"Zeta({s!r}) should raise ValueError")

    def test_zeta_2_value(self):
        val = float(convergent(Zeta(2).take(15), 14))
        assert abs(val - math.pi**2 / 6) < 1e-8

    def test_zeta_3_is_apery(self):
        assert Zeta(3).take(12) == Apery().take(12)

    def test_zeta_4_value(self):
        val = float(convergent(Zeta(4).take(15), 14))
        assert abs(val - math.pi**4 / 90) < 1e-8

    def test_zeta_5_matches_mpmath_backend(self):
        from cfmath.special import _zeta_odd_terms_mpmath

        assert list(Zeta(5).take(12)) == _zeta_odd_terms_mpmath(5, 12)

    def test_zeta_decimal_backend_matches_mpmath(self):
        from cfmath.special import _zeta_odd_terms_from_decimal, _zeta_odd_terms_mpmath

        assert _zeta_odd_terms_from_decimal(5, 12) == _zeta_odd_terms_mpmath(5, 12)


class TestGamma:
    def test_gamma_integer_exact(self):
        assert Gamma(1) == CF.from_int(1)
        assert Gamma(5) == CF.from_int(24)

    def test_gamma_bad_type_raises(self):
        try:
            Gamma(1.5)  # type: ignore[arg-type]
        except TypeError:
            pass
        else:
            raise AssertionError("Gamma(1.5) should raise TypeError")

    def test_gamma_nonpositive_raises(self):
        for x in (0, -1, Fraction(-1, 2)):
            try:
                Gamma(x)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Gamma({x!r}) should raise ValueError")

    def test_gamma_half_value(self):
        val = float(convergent(Gamma(Fraction(1, 2)).take(20), 19))
        assert abs(val - math.sqrt(math.pi)) < 1e-8

    def test_gamma_three_halves_value(self):
        val = float(convergent(Gamma(Fraction(3, 2)).take(20), 19))
        assert abs(val - math.sqrt(math.pi) / 2) < 1e-8

    def test_gamma_backend_stops_on_exact_integer_value(self):
        from cfmath.special import _gamma_terms_mpmath

        assert _gamma_terms_mpmath(1, 1, 5) == [1]
