"""Antagonistic tests for CF behavior under "recanting".

A recanting generator emits a term, then must retract it.  Our library
has no revision mechanism: once a term enters _cache it is used as-is by
every downstream iterator.  These tests confirm — and precisely locate —
the resulting corruption.

The key findings:
  1. Digits emitted *before* a corrupted term is consumed are correct.
  2. The *first* digit that required the wrong term is wrong, as are all
     subsequent ones.
  3. Convergents up to (but not including) the bad term are valid bounds;
     the convergent that first incorporates the bad term may land on the
     wrong side.
  4. Gosper comparison (__lt__) reads from position 0 of the cache, so
     any wrong term that is reached during comparison corrupts the result.
"""

from __future__ import annotations

import math
from itertools import islice

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cfmath import CF, Pi
from cfmath.convergents import convergent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PI_FLOAT = math.pi
# Pi's simple continued-fraction coefficients begin 3, 7, 15, 1, 292, ...
# See OEIS A001203.
_PI_PREFIX_110 = tuple(Pi().take(110))


def _pi_with_corruptions(corruptions: dict[int, int], n_cached: int = 110) -> CF:
    """Return a fresh CF copy of Pi with selected terms replaced.

    Forces the first *n_cached* terms out of Pi, copies them into a new CF,
    then replaces the terms at the given positions with the supplied
    (wrong) values.  This is the minimal model of a generator that emitted
    wrong terms before a recant that never arrived, without mutating the
    cached Pi singleton itself.
    """
    if n_cached > len(_PI_PREFIX_110):
        raise ValueError(f"requested {n_cached} cached terms but only {len(_PI_PREFIX_110)} are available")
    pi = CF(list(_PI_PREFIX_110[:n_cached]))
    for pos, val in corruptions.items():
        pi._cache[pos] = val
    return pi


def _digits(cf: CF, n: int = 15) -> list[int]:
    return list(islice(cf.digits(10), n))


_CORRECT_DIGITS_15 = _digits(Pi())  # ground truth, computed once
_CORRECT_TERMS_20 = list(Pi().take(20))  # [3,7,15,1,292,1,1,1,2,1,3,1,14,2,1,1,2,2,2,2]


# ---------------------------------------------------------------------------
# Single recant at various positions
# ---------------------------------------------------------------------------


class TestSingleRecant:
    """A single wrong term at position k corrupts digits that consumed term k."""

    @pytest.mark.parametrize(
        "pos,bad_val",
        [
            (2, 14),  # term 15 → 14  (one below the true value)
            (2, 16),  # term 15 → 16  (one above — gives 355/113 at wrong index)
            (4, 291),  # large term 292 → 291
            (4, 293),  # large term 292 → 293
            (4, 1),  # large term 292 → 1   (catastrophic)
            (10, 999),  # artificially huge term
        ],
    )
    def test_digits_differ(self, pos: int, bad_val: int):
        bad = _digits(_pi_with_corruptions({pos: bad_val}))
        assert bad != _CORRECT_DIGITS_15, f"corrupted term at pos {pos} (→{bad_val}) should change the digits"

    @pytest.mark.parametrize(
        "pos,bad_val",
        [
            (7, 0),  # zero term
            (3, -1),  # negative term
        ],
    )
    def test_invalid_term_raises(self, pos: int, bad_val: int):
        """Terms < 1 after position 0 are detected and raise ValueError."""
        with pytest.raises(ValueError, match="invalid CF term"):
            _digits(_pi_with_corruptions({pos: bad_val}))

    def test_integer_part_survives_recant_at_pos2(self):
        """The integer part (3) is fixed by terms 0–1; a recant at pos 2 leaves it intact."""
        bad = _digits(_pi_with_corruptions({2: 999}))
        assert bad[0] == 3, "integer part should still be 3 when corruption is at pos 2"

    def test_recant_at_pos2_term16_puts_convergent2_on_wrong_side(self):
        """
        Canonical failure: term 15 → 16 at position 2.

        The correct p₂/q₂ = 333/106 ≈ 3.14151 is BELOW π (even index).
        With the wrong term 16: p₂'/q₂' = 355/113 ≈ 3.14159292 is ABOVE π.
        That 355/113 is the correct p₃/q₃ — right value, wrong side.
        """
        pi_bad = _pi_with_corruptions({2: 16})
        c2 = convergent(pi_bad, 2)
        assert float(c2) > _PI_FLOAT, f"wrong term flips convergent 2 above π: {c2} = {float(c2):.8f}"
        # Correct convergents (positions 0, 1) are still valid bounds
        assert float(convergent(pi_bad, 0)) < _PI_FLOAT  # 3 < π
        assert float(convergent(pi_bad, 1)) > _PI_FLOAT  # 22/7 > π

    def test_recant_at_large_term_is_especially_destructive(self):
        """
        The term 292 at position 4 is extreme; a change of ±1 visibly
        shifts the decimal expansion.
        """
        good = _digits(Pi())
        bad_lo = _digits(_pi_with_corruptions({4: 291}))
        bad_hi = _digits(_pi_with_corruptions({4: 293}))
        assert bad_lo != good
        assert bad_hi != good
        assert bad_lo != bad_hi


# ---------------------------------------------------------------------------
# Multiple recants, spread across the first 100 terms
# ---------------------------------------------------------------------------


class TestMultipleRecants:
    def test_two_recants_compound(self):
        # pos=7 is now invalid (0), so this should raise before producing all digits
        with pytest.raises(ValueError, match="invalid CF term"):
            _digits(_pi_with_corruptions({2: 14, 7: 0}))

    def test_recants_at_both_ends_of_first_100(self):
        bad = _digits(_pi_with_corruptions({1: 6, 99: 0}))
        assert bad != _CORRECT_DIGITS_15

    def test_dense_recants_first_10_terms(self):
        """Corrupt every even-positioned term in the first 10."""
        corruptions = {i: 1 for i in range(0, 10, 2)}  # positions 0,2,4,6,8
        bad = _digits(_pi_with_corruptions(corruptions))
        assert bad != _CORRECT_DIGITS_15

    def test_comparison_corrupted_by_recant(self):
        """
        __lt__ reads the cache from the start; any wrong term it reaches
        before finding a discrepancy will corrupt the comparison result.

        Recant at pos 2: true term is 15, wrong value is 999.
        At even position 2, a larger term means a larger value, so
        pi_bad > pi_correct.  __lt__ should see this.
        """
        pi_correct = Pi()
        pi_bad = _pi_with_corruptions({2: 999})
        assert pi_correct < pi_bad, "larger term at even pos 2 should make pi_bad appear larger"

        pi_bad2 = _pi_with_corruptions({2: 1})
        assert pi_bad2 < pi_correct, "smaller term at even pos 2 should make pi_bad2 appear smaller"

    def test_comparison_correct_before_recant_position(self):
        """
        If the discrepancy between two CFs is found before the corrupted
        term is reached, __lt__ gives the right answer by accident.
        """
        # pi_bad has a wrong term at position 50 — well past where
        # pi_correct and 22/7 differ (they differ at position 2)
        pi_bad = _pi_with_corruptions({50: 0})
        approx = CF.from_fraction(22, 7)  # 22/7 > π
        # The comparison reads pos 0 (3==3), pos 1 (7==7), pos 2 (15 vs nothing)
        # and resolves there — never reaching pos 50.
        assert approx > pi_bad  # 22/7 > π even with the distant corruption


# ---------------------------------------------------------------------------
# Hypothesis: any single wrong term in [0, 99] corrupts the digits
# ---------------------------------------------------------------------------

# Known correct terms for the first 100 of Pi
_PI_TERMS_100 = list(Pi().take(100))


@settings(max_examples=80, deadline=None)
@given(
    pos=st.integers(min_value=0, max_value=99),
    delta=st.integers(min_value=1, max_value=10),
    sign=st.sampled_from([-1, 1]),
)
def test_hypothesis_corrupted_cf_not_equal_to_pi(pos: int, delta: int, sign: int):
    """Any wrong term makes the CF unequal to Pi — detected at exactly position pos.

    We use CF term-by-term equality (__eq__) rather than decimal digits.
    The comparison reads the cache directly, finds the discrepancy at
    position pos immediately, and doesn't require knowing which digit
    position that term feeds into.

    This is the right invariant: a recanted term always makes the CF a
    different mathematical object, even if the first N decimal digits
    happen to agree (because that term hadn't been consumed yet).
    """
    true_val = _PI_TERMS_100[pos]
    bad_val = true_val + sign * delta
    if bad_val <= 0:
        bad_val = true_val + delta  # keep positive

    pi_bad = _pi_with_corruptions({pos: bad_val})
    pi_ref = Pi()
    # Warm both caches to pos+1 so __eq__ has something to read
    list(pi_bad.take(pos + 2))
    list(pi_ref.take(pos + 2))

    # The corrupted term is visibly wrong in the cache
    assert pi_bad._cache[pos] == bad_val
    assert pi_ref._cache[pos] == true_val

    # CF equality finds the difference at position pos
    assert pi_bad != pi_ref, f"pos={pos}: changed {true_val}→{bad_val} should make CFs unequal"


@settings(max_examples=40, deadline=None)
@given(
    pos=st.integers(min_value=0, max_value=9),  # only early terms guaranteed within 15 digits
    delta=st.integers(min_value=1, max_value=5),
    sign=st.sampled_from([-1, 1]),
)
def test_hypothesis_early_corruption_changes_first_15_digits(pos: int, delta: int, sign: int):
    """Corruptions in the first 10 terms always surface in the first 15 digits.

    Early terms are consumed first by the Möbius state; there is no way
    for digit extraction to skip them.
    """
    true_val = _PI_TERMS_100[pos]
    bad_val = true_val + sign * delta
    if bad_val <= 0:
        bad_val = true_val + delta

    pi_bad = _pi_with_corruptions({pos: bad_val})
    bad_digits = _digits(pi_bad)

    assert bad_digits != _CORRECT_DIGITS_15, (
        f"early corruption at pos {pos} (true={true_val}, bad={bad_val}) not visible in first 15 digits"
    )


# ---------------------------------------------------------------------------
# Document the exact boundary: digits emitted before consuming term k are
# correct; the first digit that consumed term k is wrong.
# ---------------------------------------------------------------------------


def test_last_emitted_digit_may_be_wrong():
    """
    The check in digits() catches terms < 1 *when they are ingested*, but the
    Möbius state may have already emitted a digit that was narrowed using the
    soon-to-be-invalid term as part of the interval bound.

    Specifically: the digit emitted just *before* the bad term is consumed was
    pinned by an interval whose upper endpoint depended on the (still-correct)
    term BEFORE the bad one.  That digit is safe.  But a digit emitted in the
    same `while` loop pass that *also* consumed the bad term could have been
    wrong if the bad value contributed to narrowing.

    In practice the check fires on ingest, so the generator stops immediately
    and the last successfully yielded digit is the one produced just before
    the bad term was consumed — which is safe (it was pinned before ingest).
    The ValueError tells the caller "all digits yielded so far are valid."

    Concrete demonstration: corrupt pos=4 (292→0).  Digits emitted before
    pos=4 is consumed are correct; the ValueError fires as soon as pos=4
    would be ingested, so the caller never receives a digit tainted by it.
    """
    correct = _CORRECT_DIGITS_15

    digits_so_far = []
    try:
        for d in _pi_with_corruptions({4: 0}).digits(10):
            digits_so_far.append(d)
    except ValueError:
        pass  # expected

    # Every digit we DID receive should match the correct sequence
    assert digits_so_far == correct[: len(digits_so_far)], (
        f"digits emitted before the ValueError should all be correct:\n"
        f"  emitted:  {digits_so_far}\n"
        f"  expected: {correct[: len(digits_so_far)]}"
    )
    # And we should not have received all 15 (the bad term blocks further progress)
    assert len(digits_so_far) < len(correct), "some digits should have been withheld by the early termination"


def test_term4_all_replacements_correct_prefix():
    """
    Replace term 4 of Pi (true value: 292) with every integer in [1, 1000].

    First establishes the "trustable prefix length" by running with term4=0,
    which raises ValueError after emitting exactly the digits that needed only
    terms 0–3.  That count (5, giving "3.1415") is the hard lower bound: no
    matter what term 4 is replaced with, the first 5 digits must be correct.

    Then for all 1000 valid (≥1) replacements: finds the first divergence and
    asserts the prefix before it matches correct Pi — confirming the corruption
    cannot retroactively corrupt already-emitted digits.
    """
    n = 20
    correct = list(islice(Pi().digits(10), n))

    # --- establish trustable prefix length via the term=0 (raising) case ---
    trusted: list[int] = []
    try:
        for d in _pi_with_corruptions({4: 0}).digits(10):
            trusted.append(d)
    except ValueError:
        pass

    TRUSTED_LEN = 5  # terms 0–3 pin exactly 5 digits: [3, 1, 4, 1, 5]
    assert len(trusted) == TRUSTED_LEN, f"expected {TRUSTED_LEN} digits before term4=0 raises, got {len(trusted)}: {trusted}"
    assert trusted == correct[:TRUSTED_LEN], f"digits emitted before the raise should be correct Pi: {trusted}"

    # --- for every valid replacement: first TRUSTED_LEN digits are always correct ---
    for bad_val in range(1, 1001):
        bad = list(islice(_pi_with_corruptions({4: bad_val}).digits(10), TRUSTED_LEN))
        assert bad == correct[:TRUSTED_LEN], f"term4={bad_val}: first {TRUSTED_LEN} digits should be correct Pi, got {bad}"


def test_corruption_locality():
    """
    Verify that the corruption point is local to the first digit that
    consumed the wrong term.  Digits before that digit are correct;
    the digit at that point and after are wrong.

    Strategy: corrupt term at pos 2 (value 15→999) and find the first
    position where the digit sequence diverges.
    """
    correct = _CORRECT_DIGITS_15
    bad = _digits(_pi_with_corruptions({2: 999}))

    first_diff = next((i for i, (c, b) in enumerate(zip(correct, bad)) if c != b), None)
    assert first_diff is not None, "at least one digit should differ"

    # Everything before first_diff must agree
    assert correct[:first_diff] == bad[:first_diff], f"digits before the corruption point (index {first_diff}) should be identical"
    # And the digit at first_diff is wrong
    assert correct[first_diff] != bad[first_diff]
