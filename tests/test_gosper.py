"""Tests for Gosper HAKMEM arithmetic."""

from fractions import Fraction

import pytest

from cfmath import CF, Phi, Sqrt, convergent, convergents


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
        result = _eval(a + b)
        assert result == Fraction(7)

    def test_fraction_plus_fraction(self):
        a = CF.from_fraction(22, 7)
        b = CF.from_fraction(1, 3)
        result = _eval(a + b)
        assert result == Fraction(22, 7) + Fraction(1, 3)

    def test_add_zero(self):
        a = CF.from_fraction(5, 3)
        z = _ref(0)
        assert _eval(a + z) == Fraction(5, 3)

    def test_add_one(self):
        a = CF.from_fraction(355, 113)
        one = _ref(1)
        assert _eval(a + one) == Fraction(355, 113) + 1

    def test_commutativity(self):
        a = CF.from_fraction(3, 7)
        b = CF.from_fraction(5, 11)
        assert _eval(a + b) == _eval(b + a)

    def test_associativity(self):
        a = CF.from_fraction(1, 2)
        b = CF.from_fraction(1, 3)
        c = CF.from_fraction(1, 5)
        lhs = _eval((a + b) + c)
        rhs = _eval(a + (b + c))
        assert lhs == rhs

    def test_many_fractions(self):
        pairs = [(3, 5), (7, 11), (13, 17), (100, 101)]
        for p1, q1 in pairs:
            for p2, q2 in pairs:
                a = CF.from_fraction(p1, q1)
                b = CF.from_fraction(p2, q2)
                expected = Fraction(p1, q1) + Fraction(p2, q2)
                assert _eval(a + b) == expected


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------


class TestSubtraction:
    def test_simple_sub(self):
        a = CF.from_fraction(5, 3)
        b = CF.from_fraction(1, 3)
        assert _eval(a - b) == Fraction(4, 3)

    def test_sub_self(self):
        a = CF.from_fraction(7, 5)
        assert _eval(a - a) == Fraction(0)

    def test_sub_gives_negative(self):
        a = CF.from_fraction(1, 4)
        b = CF.from_fraction(1, 2)
        assert _eval(a - b) == Fraction(-1, 4)

    def test_sub_is_neg_add(self):
        a = CF.from_fraction(8, 3)
        b = CF.from_fraction(5, 7)
        assert _eval(a - b) == Fraction(8, 3) - Fraction(5, 7)


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


class TestMultiplication:
    def test_int_times_int(self):
        a = _ref(3)
        b = _ref(5)
        assert _eval(a * b) == Fraction(15)

    def test_fraction_times_fraction(self):
        a = CF.from_fraction(3, 7)
        b = CF.from_fraction(7, 3)
        assert _eval(a * b) == Fraction(1)

    def test_multiply_by_zero(self):
        a = CF.from_fraction(5, 3)
        z = _ref(0)
        assert _eval(a * z) == Fraction(0)

    def test_multiply_by_one(self):
        a = CF.from_fraction(22, 7)
        one = _ref(1)
        assert _eval(a * one) == Fraction(22, 7)

    def test_commutativity(self):
        a = CF.from_fraction(3, 11)
        b = CF.from_fraction(7, 5)
        assert _eval(a * b) == _eval(b * a)

    def test_many_pairs(self):
        pairs = [(2, 3), (5, 7), (11, 13)]
        for p1, q1 in pairs:
            for p2, q2 in pairs:
                a = CF.from_fraction(p1, q1)
                b = CF.from_fraction(p2, q2)
                expected = Fraction(p1, q1) * Fraction(p2, q2)
                assert _eval(a * b) == expected


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------


class TestDivision:
    def test_simple_div(self):
        a = CF.from_fraction(3, 4)
        b = CF.from_fraction(3, 4)
        assert _eval(a / b) == Fraction(1)

    def test_div_gives_fraction(self):
        a = CF.from_fraction(22, 7)
        b = CF.from_fraction(11, 7)
        assert _eval(a / b) == Fraction(2)

    def test_div_reciprocal(self):
        a = CF.from_fraction(5, 3)
        one = _ref(1)
        assert _eval(one / a) == Fraction(3, 5)

    def test_many_divs(self):
        pairs = [(3, 5), (7, 11), (2, 9)]
        for p1, q1 in pairs:
            for p2, q2 in pairs:
                a = CF.from_fraction(p1, q1)
                b = CF.from_fraction(p2, q2)
                expected = Fraction(p1, q1) / Fraction(p2, q2)
                assert _eval(a / b) == expected


# ---------------------------------------------------------------------------
# Powers
# ---------------------------------------------------------------------------


class TestPow:
    def test_square(self):
        a = CF.from_fraction(3, 2)
        assert _eval(a**2) == Fraction(9, 4)

    def test_cube(self):
        a = CF.from_fraction(2, 1)
        assert _eval(a**3) == Fraction(8)

    def test_power_zero(self):
        a = CF.from_fraction(5, 3)
        assert _eval(a**0) == Fraction(1)

    def test_power_one(self):
        a = CF.from_fraction(5, 3)
        assert _eval(a**1) == Fraction(5, 3)

    def test_negative_power(self):
        a = CF.from_fraction(2, 3)
        assert _eval(a**-1) == Fraction(3, 2)

    def test_power_4(self):
        a = CF.from_fraction(3, 2)
        assert _eval(a**4) == Fraction(81, 16)


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------


class TestNegation:
    def test_negate(self):
        a = CF.from_fraction(3, 4)
        neg = -a
        assert _eval(neg) == Fraction(-3, 4)

    def test_double_negate(self):
        a = CF.from_fraction(5, 7)
        assert _eval(-(-a)) == Fraction(5, 7)


# ---------------------------------------------------------------------------
# Mixed integer/CF arithmetic
# ---------------------------------------------------------------------------


class TestMixedArithmetic:
    def test_radd(self):
        cf = CF.from_fraction(1, 7)
        result = _eval(3 + cf)
        assert result == Fraction(22, 7)

    def test_rmul(self):
        cf = CF.from_fraction(1, 7)
        result = _eval(22 * cf)
        assert result == Fraction(22, 7)


# ---------------------------------------------------------------------------
# Phi identity: φ² = φ + 1
# ---------------------------------------------------------------------------


class TestPhiIdentity:
    def test_phi_squared_is_phi_plus_one(self):
        phi = Phi()
        phi_sq = phi**2
        phi_plus_1 = phi + CF.from_int(1)
        # Compare deep convergents
        depth = 15
        lhs = convergent(phi_sq.take(depth), depth - 1)
        rhs = convergent(phi_plus_1.take(depth), depth - 1)
        assert abs(float(lhs) - float(rhs)) < 1e-10


# ---------------------------------------------------------------------------
# sqrt(2) * sqrt(2) ≈ 2
# ---------------------------------------------------------------------------


class TestSqrt2Squared:
    def test_sqrt2_times_sqrt2(self):
        # Use a finite truncation of sqrt(2) to avoid the integer-boundary
        # divergence in Gosper's algorithm: exact sqrt(2)*sqrt(2)=2 causes
        # the convergent products to always straddle the boundary 2.0, so
        # the floor check never converges. With a finite approximation the
        # algorithm terminates when both inputs are exhausted.
        s2 = Sqrt(2).take(15)
        result = s2 * s2
        convs = list(convergents(result))
        val = float(convs[-1])
        assert abs(val - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# Homographic (unary) transform
# ---------------------------------------------------------------------------


class TestHomographic:
    def test_shift(self):
        from cfmath.gosper import cf_homographic

        # y = x + 2: (1*x + 2) / (0*x + 1)
        x = CF.from_fraction(3, 7)
        y = cf_homographic(x, 1, 2, 0, 1)
        assert _eval(y) == Fraction(3, 7) + 2

    def test_scale(self):
        from cfmath.gosper import cf_homographic

        # y = 3x: (3*x + 0) / (0*x + 1)
        x = CF.from_fraction(5, 11)
        y = cf_homographic(x, 3, 0, 0, 1)
        assert _eval(y) == Fraction(15, 11)

    def test_constant_map_terminates_on_infinite_input(self):
        """A degenerate (constant) map of an infinite input terminates.

        After emitting the constant the denominator becomes identically zero
        (value ∞ = CF ended).  Without that check the loop would ingest forever
        and now raise; it must return the finite CF instead.
        """
        from cfmath import Pi
        from cfmath.gosper import cf_homographic

        # (0*Pi + 5)/(0*Pi + 1) = 5
        assert cf_homographic(Pi(), 0, 5, 0, 1).take(3).terms == [5]
        # (0*Pi + 7)/(0*Pi + 2) = 7/2 = [3; 2]
        assert cf_homographic(Pi(), 0, 7, 0, 2).take(4).terms == [3, 2]


class TestLargeCornerOverflow:
    """A bihomographic corner can exceed the float range when a coefficient is
    huge (e.g. a CF with a large integer part).  The ingest-ordering heuristic
    converts corners to float, which used to raise OverflowError; it must fall
    back gracefully instead."""

    def test_huge_integer_part_does_not_overflow(self):
        big = CF([10**400])  # ~1e400, far past the float range
        # Completes (no OverflowError) and floors correctly: 10**400 + sqrt(2).
        assert (big + Sqrt(2)).take(2).terms[0] == 10**400 + 1
        # exact-rational result still resolves via the boundary handler
        assert ((big + 1) - big).take(3).terms == [1]


class TestBihomographicGimme:
    """An exact-rational result of +,-,*,/ on irrational inputs sits on an
    integer boundary the corner check can never confirm.  The bihomographic
    accepts it once the suppressed partial quotient would have at least
    _GIMME_MIN_TERM_DIGITS digits — the same digit-based heuristic and config as
    the metaCF gimme (replacing the old fixed term-count threshold).
    """

    def test_exact_rationals_resolve(self):
        from cfmath import Pi

        assert (Pi() - Pi()).take(3).terms == [0]
        assert (Pi() / Pi()).take(3).terms == [1]
        assert (Pi() * (2 / Pi())).take(3).terms == [2]
        assert ((Sqrt(2) - 1) * (Sqrt(2) + 1)).take(3).terms == [1]
        assert (Phi() * Phi() - Phi() - 1).take(3).terms == [0]

    def test_genuine_irrational_unaffected(self):
        """A non-rational result emits normally; the gimme never fires."""
        from cfmath import Pi

        # 2*pi = [6; 3, 1, 1, 7, 2, ...]
        assert (Pi() + Pi()).take(6).terms == [6, 3, 1, 1, 7, 2]

    def test_threshold_governs_resolution(self, monkeypatch):
        """The digit threshold drives the decision: an unreachably high value
        leaves the boundary unconfirmable within the refine cap, so it raises."""
        import cfmath.gosper as gosper
        from cfmath import Pi

        assert (Pi() / Pi()).take(3).terms == [1]  # default resolves
        # Unreachable threshold within a small refine budget -> raises (fast).
        monkeypatch.setattr(gosper, "_GIMME_MIN_TERM_DIGITS", 100000)
        monkeypatch.setattr(gosper, "_GIMME_REFINE_CAP", 30)
        with pytest.raises(ArithmeticError, match="bihomographic stalled"):
            (Pi() / Pi()).take(3)


class TestMetaCFFIngestionStall:
    """When the meta-CF F itself fails to determine an output term at the current
    x-bracket (F not converging), the metaCF now raises rather than silently
    truncating.  Distinct from the near-rational x-refinement gimme."""

    def test_nonconverging_F_raises(self, monkeypatch):
        import cfmath.gosper as gosper

        monkeypatch.setattr(gosper, "_METACF_STALL_LIMIT", 8)

        def bad_F():
            while True:
                yield [0, 1]  # term polynomial = x; the output never pins

        with pytest.raises(ArithmeticError, match=r"ingested \d+ terms of F"):
            gosper.cf_metaCF(Sqrt(2), bad_F()).take(2)
