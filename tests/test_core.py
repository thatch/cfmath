"""Tests for the CF core class."""

import math
from fractions import Fraction
from itertools import islice

import pytest

from cfmath import CF, CountingIterator, digits_with_debug
from cfmath._backend import _cf_terms_from_interval_approximator

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFromInt:
    def test_empty_cf_rejected(self):
        with pytest.raises(ValueError):
            CF([])

    def test_positive(self):
        cf = CF.from_int(5)
        assert cf.terms == [5]
        assert cf.repeating == []

    def test_zero(self):
        cf = CF.from_int(0)
        assert cf.terms == [0]

    def test_negative(self):
        cf = CF.from_int(-3)
        assert cf.terms == [-3]

    def test_large(self):
        cf = CF.from_int(10**9)
        assert cf.terms == [10**9]


class TestFromFraction:
    def test_half(self):
        cf = CF.from_fraction(1, 2)
        assert cf.terms == [0, 2]

    def test_22_over_7(self):
        cf = CF.from_fraction(22, 7)
        assert cf.terms == [3, 7]

    def test_integer_result(self):
        cf = CF.from_fraction(6, 3)
        assert cf.to_fraction() == Fraction(2)

    def test_355_over_113(self):
        cf = CF.from_fraction(355, 113)
        assert cf.to_fraction() == Fraction(355, 113)

    def test_negative_fraction(self):
        cf = CF.from_fraction(-1, 2)
        assert cf.to_fraction() == Fraction(-1, 2)

    def test_negative_improper_fraction(self):
        cf = CF.from_fraction(-7, 3)
        assert cf.to_fraction() == Fraction(-7, 3)

    def test_negative_both(self):
        cf = CF.from_fraction(-3, -4)
        assert cf.to_fraction() == Fraction(3, 4)

    def test_zero_numerator(self):
        cf = CF.from_fraction(0, 5)
        assert cf.to_fraction() == Fraction(0)

    def test_zero_denominator_raises(self):
        with pytest.raises(ZeroDivisionError):
            CF.from_fraction(1, 0)

    def test_roundtrip_various(self):
        fractions = [
            Fraction(1, 3),
            Fraction(5, 7),
            Fraction(100, 99),
            Fraction(355, 113),
            Fraction(1, 1000),
        ]
        for f in fractions:
            cf = CF.from_rational(f)
            assert cf.to_fraction() == f, f"Roundtrip failed for {f}"


class TestFromFloat:
    def test_pi_approx(self):
        cf = CF.from_float(math.pi, max_terms=5)
        assert cf.terms[0] == 3
        assert cf.terms[1] == 7

    def test_integer(self):
        cf = CF.from_float(5.0, max_terms=10)
        assert cf.terms[0] == 5

    def test_exact_integer_stops_after_one_term(self):
        assert CF.from_float(5.0, max_terms=10).terms == [5]

    def test_half(self):
        cf = CF.from_float(0.5, max_terms=10)
        assert cf.to_fraction() == Fraction(1, 2)

    def test_third(self):
        cf = CF.from_float(1 / 3, max_terms=15)
        # Should be approximately [0; 3]
        assert cf.terms[0] == 0
        assert cf.terms[1] == 3


class TestIntervalApproximator:
    def test_exact_singleton(self):
        terms = _cf_terms_from_interval_approximator(lambda _: (Fraction(22, 7), Fraction(22, 7)), 2)
        assert terms == [3, 7]

    def test_doubles_until_term_is_pinned(self):
        calls = []

        def interval(precision: int) -> tuple[Fraction, Fraction]:
            calls.append(precision)
            if precision < 32:
                return Fraction(0), Fraction(2)
            return Fraction(3, 2), Fraction(8, 5)

        terms = _cf_terms_from_interval_approximator(interval, 1, initial=8)

        assert terms == [1]
        assert calls == [8, 16, 32]

    def test_negative_interval(self):
        terms = _cf_terms_from_interval_approximator(
            lambda _: (Fraction(-3, 2), Fraction(-7, 5)),
            1,
        )

        assert terms == [-2]

    def test_zero_straddling_interval_refines(self):
        calls = []

        def interval(precision: int) -> tuple[Fraction, Fraction]:
            calls.append(precision)
            if precision < 16:
                return Fraction(-1, 10), Fraction(1, 10)
            return Fraction(17, 50), Fraction(7, 20)

        terms = _cf_terms_from_interval_approximator(interval, 2, initial=8)

        assert terms == [0, 2]
        assert calls == [8, 16]

    def test_unpinned_interval_raises_at_max_precision(self):
        with pytest.raises(ValueError):
            _cf_terms_from_interval_approximator(
                lambda _: (Fraction(0), Fraction(2)),
                1,
                initial=8,
                max_precision=8,
            )


class TestDebugUtilities:
    def test_counting_iterator_counts_consumed_items(self):
        counter = CountingIterator(iter([10, 20, 30]))

        assert iter(counter) is counter
        assert next(counter) == 10
        assert counter.count == 1
        assert next(counter) == 20
        assert counter.count == 2

    def test_digits_with_debug_reports_digit_and_terms_consumed(self):
        digits = list(islice(digits_with_debug(CF.from_rational(Fraction(355, 113))), 3))

        assert [digit for digit, _ in digits] == [3, 1, 4]
        assert all(cost >= 0 for _, cost in digits)
        assert sum(cost for _, cost in digits) > 0


class TestFromTerms:
    def test_finite(self):
        cf = CF.from_terms([1, 2, 3])
        assert cf.terms == [1, 2, 3]
        assert cf.repeating == []

    def test_periodic(self):
        cf = CF.from_terms([1], repeating=[2])
        assert cf.terms == [1]
        assert cf.repeating == [2]


# ---------------------------------------------------------------------------
# Iteration and take
# ---------------------------------------------------------------------------


class TestIteration:
    def test_finite_iter(self):
        cf = CF([3, 7, 15, 1])
        assert list(cf) == [3, 7, 15, 1]

    def test_periodic_iter(self):
        cf = CF([1], repeating=[2])
        it = iter(cf)
        assert next(it) == 1
        assert next(it) == 2
        assert next(it) == 2
        assert next(it) == 2

    def test_take(self):
        cf = CF([1], repeating=[2, 3])
        t = cf.take(5)
        assert list(t) == [1, 2, 3, 2, 3]

    def test_take_finite(self):
        cf = CF([3, 7, 15, 1, 292])
        assert list(cf.take(3)) == [3, 7, 15]

    def test_independent_iters(self):
        """Two iterators over the same CF must be independent."""
        cf = CF([1], repeating=[1])
        it1 = cf._iter_from(0)
        it2 = cf._iter_from(0)
        for _ in range(5):
            next(it1)
        v2 = next(it2)
        assert v2 == 1  # it2 unaffected by it1's advancement

    def test_lazy_source_exhausts(self):
        cf = CF([], _source=iter([1, 2]))
        assert list(cf) == [1, 2]
        assert cf.is_finite()

    def test_take_from_empty_lazy_source_raises(self):
        with pytest.raises(ValueError):
            CF([], _source=iter([])).take(1)

    def test_to_fraction_from_exhausted_empty_source_raises(self):
        cf = CF([], _source=iter([]))
        assert list(cf) == []
        with pytest.raises(ValueError, match="empty CF"):
            cf.to_fraction()

    def test_digits_reject_invalid_later_term(self):
        with pytest.raises(ValueError, match="invalid CF term"):
            list(CF([1, 0]).digits())


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


class TestPredicates:
    def test_finite(self):
        assert CF([1, 2, 3]).is_finite()

    def test_not_finite_if_repeating(self):
        assert not CF([1], repeating=[2]).is_finite()

    def test_periodic(self):
        assert CF([1], repeating=[2]).is_periodic()

    def test_not_periodic_finite(self):
        assert not CF([1, 2]).is_periodic()


class TestErrorEstimate:
    def test_single_term_error_estimate(self):
        assert CF.from_int(3).err_estimate == Fraction(1)

    def test_multi_term_error_estimate(self):
        assert CF([3, 7]).err_estimate == abs(Fraction(22, 7) - Fraction(3))

    def test_infinite_error_estimate_raises(self):
        with pytest.raises(ValueError):
            CF([1], repeating=[2]).err_estimate


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


class TestRepr:
    def test_simple(self):
        r = repr(CF([3]))
        assert r.startswith("[3]") and "= 3" in r

    def test_with_tail(self):
        r = repr(CF([3, 7, 15]))
        assert r.startswith("[3; 7, 15]") and "3.141" in r

    def test_periodic(self):
        r = repr(CF([1], repeating=[2]))
        assert "(2)" in r

    def test_infinite_hint(self):
        from cfmath import Pi

        r = repr(Pi())
        assert "..." in r or ";" in r

    def test_lazy_repr_peeks_at_source(self):
        r = repr(CF([], _source=iter([1, 2, 3, 4])))
        assert r.startswith("[1; 2, 3")

    def test_exhausted_lazy_repr(self):
        assert repr(CF([], _source=iter([]))) == "CF([])"

    def test_exact_terminating(self):
        # 1/4 = 0.25 terminates in base 10 → "="
        assert "= 0.25" in repr(CF.from_fraction(1, 4))

    def test_approx_irrational(self):
        # irrational → "≈"
        from cfmath import Sqrt

        assert "≈" in repr(Sqrt(2))


# ---------------------------------------------------------------------------
# to_fraction
# ---------------------------------------------------------------------------


class TestToFraction:
    def test_integer(self):
        assert CF([5]).to_fraction() == Fraction(5)

    def test_22_over_7(self):
        assert CF([3, 7]).to_fraction() == Fraction(22, 7)

    def test_finite_cf(self):
        cf = CF.from_fraction(355, 113)
        assert cf.to_fraction() == Fraction(355, 113)

    def test_periodic_raises(self):
        with pytest.raises(ValueError):
            CF([1], repeating=[2]).to_fraction()


# ---------------------------------------------------------------------------
# Equality and ordering
# ---------------------------------------------------------------------------


class TestEquality:
    def test_equal_fractions(self):
        a = CF.from_fraction(1, 2)
        b = CF.from_fraction(1, 2)
        assert a == b

    def test_unequal_fractions(self):
        a = CF.from_fraction(1, 2)
        b = CF.from_fraction(1, 3)
        assert a != b

    def test_ordering(self):
        a = CF.from_fraction(1, 3)
        b = CF.from_fraction(1, 2)
        assert a < b
        assert b > a
        assert a <= b
        assert b >= a

    def test_equal_different_reps(self):
        # [3, 7] and [3, 6, 1] both = 22/7
        ta = [3, 7]
        tb = [3, 6, 1]
        a0 = CF(ta)
        b0 = CF(tb)
        assert a0 == b0
        a1 = CF([], _source=iter(ta))
        b1 = CF([], _source=iter(tb))
        assert b0 == a1 == b1 == a0
        a2 = 3 + 1 / CF.from_int(7)
        b2 = 3 + 1 / (CF.from_int(6) + 1)
        assert b1 == a2 == b2 == a1

    def test_equality_accepts_int_rhs(self):
        assert CF.from_int(1) == 1
        assert CF([0, 1]) == 1
        assert CF.from_fraction(1, 2) != 0

    def test_equality_with_unsupported_type_is_not_implemented(self):
        assert CF.from_int(1).__eq__(object()) is NotImplemented

    def test_hash_matches_equal_finite_value(self):
        assert hash(CF([3, 7])) == hash(CF([3, 6, 1]))

    def test_hash_for_infinite_uses_prefix(self):
        assert isinstance(hash(CF([1], repeating=[2])), int)

    def test_interval_for_finite_cf_collapses(self):
        assert CF.from_rational(Fraction(3, 2)).interval(3) == (Fraction(3, 2), Fraction(3, 2))

    def test_interval_orders_odd_and_even_depths(self):
        cf = CF([1], repeating=[2])
        lo1, hi1 = cf.interval(1)
        lo2, hi2 = cf.interval(2)

        assert lo1 == Fraction(1)
        assert hi1 == Fraction(2)
        assert lo2 == Fraction(4, 3)
        assert hi2 == Fraction(3, 2)

    def test_interval_reuses_cached_convergents(self):
        cf = CF([1], repeating=[2])
        assert cf.interval(4) == (Fraction(24, 17), Fraction(17, 12))
        cached = list(cf._convergent_cache)

        assert cf.interval(2) == (Fraction(4, 3), Fraction(3, 2))
        assert cf._convergent_cache == cached


class TestOperators:
    def test_unsupported_binary_operands_return_not_implemented(self):
        cf = CF.from_int(1)

        assert cf.__add__(object()) is NotImplemented
        assert cf.__sub__(object()) is NotImplemented
        assert cf.__mul__(object()) is NotImplemented
        assert cf.__truediv__(object()) is NotImplemented

    def test_fraction_exponent_dispatches_to_pow(self):
        result = CF.from_int(4).__pow__(Fraction(1, 2))
        assert result == CF.from_int(2)

    def test_reciprocal(self):
        assert CF.from_int(4).reciprocal().take(4).to_fraction() == Fraction(1, 4)

    def test_normalize_absorbs_zero_term(self):
        assert CF([1, 0, 2]).normalize().take(4).to_fraction() == Fraction(3)


# ---------------------------------------------------------------------------
# from_digits — inverse of digits()
#
# digits() converts CF → base-B digit stream by pushing CF terms in and
# pulling digits out via a forward Möbius state.  from_digits() inverts this:
# it maintains an inverse Möbius state (p,q,r,s) representing
#
#     x_remaining = (p·y + q) / (r·y + s)
#
# where y is the decimal value being consumed.  Each incoming digit narrows
# the rational interval [y_lo, y_lo + step); a CF term is emitted whenever
# both interval endpoints floor to the same integer.  The base only affects
# the step size 1/Bᵏ, so the algorithm works identically for any base ≥ 2.
# ---------------------------------------------------------------------------


class TestFromDigits:
    # ---- exact rational values -----------------------------------------
    #
    # CFs created by from_digits() have a lazy _source, so is_finite() returns
    # False until the source is consumed.  We use == against a known CF (term-by-
    # term comparison) or take() to drain into a finite CF before calling
    # to_fraction().

    def test_integer_only(self):
        """A single-digit input gives the corresponding integer CF."""
        assert CF.from_digits([7]) == CF.from_int(7)

    def test_empty_digit_stream_raises_on_take(self):
        with pytest.raises(ValueError):
            CF.from_digits([]).take(1)

    def test_zero(self):
        assert CF.from_digits([0]) == CF.from_int(0)

    def test_three_quarters_decimal(self):
        """0.75 in decimal = 3/4."""
        assert CF.from_digits([0, 7, 5]) == CF.from_fraction(3, 4)

    def test_three_quarters_binary(self):
        """0.11 in base 2 = 1/2 + 1/4 = 3/4.  CF of 3/4 = [0; 1, 3]."""
        cf = CF.from_digits([0, 1, 1], base=2)
        assert list(cf.take(3)) == [0, 1, 3]  # shows the actual terms produced
        assert cf == CF.from_fraction(3, 4)  # confirms the rational value

    def test_one_and_half_hex(self):
        """1.8 in base 16 = 1 + 8/16 = 3/2.

        Note: the algorithm produces [1; 1, 1] rather than canonical [1; 2],
        because the open interval [1.5, 1.5625) contains values whose CF
        starts with 1 rather than 2.  The algorithm can only commit to a₁=2
        once it knows y is *exactly* 1.5 — which only becomes clear at digit
        exhaustion.  By then the state has already advanced past that point.

        [1; 1, 1] and [1; 2] are the two representations of 3/2; they compare
        equal as finite CFs (both have to_fraction() == 3/2).  Drain the
        lazy source with take() first so __eq__ uses the to_fraction() path.
        """
        cf = CF.from_digits([1, 8], base=16)
        assert cf.take(10) == CF.from_fraction(3, 2)

    def test_one_third_base3(self):
        """0.1 in base 3 = 1/3 exactly (terminates in base 3)."""
        assert CF.from_digits([0, 1], base=3) == CF.from_fraction(1, 3)

    # ---- Pi approximation: CF terms are pinned progressively -----------

    def test_pi_terms_from_15_digits(self):
        """15 decimal digits of π pin [3; 7, 15, 1] as the first four CF terms.

        Digit 0 (integer part 3) pins a₀ = 3 immediately.
        Digits up to 3.141 pin a₁ = 7 (the 1/7 level of the CF tower).
        Digits up to 3.141592 pin a₂ = 15.
        Digits up to 3.14159265... distinguish π from 355/113 and pin a₃ = 1.
        The term a₄ = 292 (the famous large Gosper term) is also pinned here.
        """
        pi_digits = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
        cf = CF.from_digits(pi_digits)
        assert list(cf.take(4)) == [3, 7, 15, 1]

    def test_few_pi_digits_still_get_first_term(self):
        """Even a single digit gives a₀ = 3."""
        assert CF.from_digits([3]) == CF.from_int(3)

    def test_four_pi_digits_pin_a1(self):
        """3.141 is enough to confirm a₁ = 7 (since 3.141 < 22/7 < 3.142)."""
        cf = CF.from_digits([3, 1, 4, 1])
        assert list(cf.take(2)) == [3, 7]

    # ---- roundtrip: Pi.digits() → from_digits() → same CF terms -------

    def test_roundtrip_pi_digits(self):
        """20 decimal digits of Pi fed back through from_digits() recover [3;7,15,1].

        This tests the full loop: CF → digit stream → CF.
        The base-10 digits are ground truth; from_digits() inverts the process.
        """
        from itertools import islice

        from cfmath import Pi

        pi_digits = list(islice(Pi().digits(10), 20))
        cf = CF.from_digits(pi_digits)
        assert list(cf.take(4)) == [3, 7, 15, 1]
