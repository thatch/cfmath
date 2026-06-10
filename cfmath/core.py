"""Core CF class: the main data structure for working with continued fractions."""

from __future__ import annotations

import math
import os
from fractions import Fraction
from typing import Iterable, Iterator

_GCF_STALL_LIMIT = int(os.environ.get("CFRAC_GCF_STALL_LIMIT", "100"))


class CF:
    """
    A continued fraction — a way to represent any number as a sequence of integers.

    A continued fraction looks like:
        a0 + 1 / (a1 + 1 / (a2 + 1 / (a3 + ...)))
    and is written compactly as [a0; a1, a2, a3, ...].  Every real number has
    such a representation, and rational numbers (fractions) have finite ones.
    Square roots like sqrt(2) = [1; 2, 2, 2, ...] have a repeating pattern.

    Internally this stores:
      terms     — the non-repeating part  [a0, a1, ..., ak]
      repeating — the periodic block      (b1, b2, ..., bm), empty if none

    For infinite non-periodic values (like Pi or e), a _source iterator
    is consumed one term at a time; already-seen terms are cached so that
    multiple passes over the same CF (needed for arithmetic) stay in sync.
    """

    def __init__(
        self,
        terms: list[int],
        repeating: list[int] | None = None,
        _source: Iterator[int] | None = None,
    ) -> None:
        if not terms and not repeating and _source is None:
            raise ValueError("CF must have at least one term")
        self.terms: list[int] = list(terms)
        self.repeating: list[int] = list(repeating or [])
        self._source: Iterator[int] | None = _source
        self._debug_source: object | None = None
        # _cache grows lazily; starts populated with the static terms
        self._cache: list[int] = list(self.terms)
        self._convergent_cache: list[tuple[int, int]] = []
        self._float_convergent_cache: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Classmethods / constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_int(cls, n: int) -> CF:
        return cls([int(n)])

    @classmethod
    def from_fraction(cls, p: int, q: int) -> CF:
        """Convert p/q to a finite continued fraction via the Euclidean algorithm."""
        if q == 0:
            raise ZeroDivisionError("denominator is zero")
        # Handle negatives: ensure q > 0
        if q < 0:
            p, q = -p, -q
        terms: list[int] = []
        while q:
            a, r = divmod(p, q)
            # For negative remainders, adjust so remainder >= 0
            if r < 0:
                a -= 1
                r += q
            terms.append(a)
            p, q = q, r
        return cls(terms)

    @classmethod
    def from_rational(cls, f: Fraction) -> CF:
        return cls.from_fraction(f.numerator, f.denominator)

    @classmethod
    def from_float(cls, x: float, max_terms: int = 50) -> CF:
        """Approximate x as a continued fraction with up to max_terms terms."""
        terms: list[int] = []
        for _ in range(max_terms):
            a = math.floor(x)
            terms.append(a)
            frac = x - a
            if frac < 1e-10:
                break
            x = 1.0 / frac
        return cls(terms)

    @classmethod
    def from_terms(cls, terms: list[int], repeating: list[int] | None = None) -> CF:
        return cls(terms, repeating)

    @classmethod
    def from_digits(cls, digits: Iterable[int], base: int = 10) -> CF:
        """Convert a stream of base-B digits to a continued fraction.

        This is the inverse of digits(): where digits() pushes CF terms in and
        pulls base-B digits out via a forward Möbius state, from_digits() pushes
        base-B digits in and pulls CF terms out via an inverse Möbius state.

        The first element of *digits* is the integer part; subsequent elements
        are fractional digits in [0, base).  Any base ≥ 2 works — the algorithm
        is identical across bases; only the step size 1/Bᵏ changes.

        Algorithm — inverse Möbius state:
            Maintain (p, q, r, s) where x_remaining = (p·y + q)/(r·y + s) and
            y is the full decimal value being consumed.  Initially x = y (the
            identity state (1, 0, 0, 1)).

            On each incoming digit d:
              - Narrow the y-interval:  step /= base;  y_lo += d * step
              - Evaluate x at both y-interval endpoints
              - While both endpoints floor to the same integer a:
                  emit a;  update state (p,q,r,s) ← (r, s, p−a·r, q−a·s)

            On exhaustion: evaluate x at y_lo and extract remaining CF terms
            via the Euclidean algorithm.

        The state update (r, s, p−a·r, q−a·s) is the same Euclidean step that
        appears in the standard CF algorithm — it strips the leading integer a
        and takes the reciprocal of the fractional remainder.
        """

        def _gen() -> Iterator[int]:
            it = iter(digits)
            try:
                d0 = next(it)
            except StopIteration:
                return

            # Inverse Möbius state: x_remaining = (p·y + q) / (r·y + s)
            # Starts as the identity: x = y.
            p, q, r, s = Fraction(1), Fraction(0), Fraction(0), Fraction(1)

            y_lo = Fraction(d0)
            step = Fraction(1)  # width of the y-interval; starts at 1

            def emit_while_pinned() -> Iterator[int]:
                """Emit CF terms as long as the x-interval pins a single floor."""
                nonlocal p, q, r, s
                while True:
                    # Detect poles before evaluating: if the Möbius denominator
                    # (r·y + s) changes sign across [y_lo, y_lo + step), there is
                    # a pole inside the interval and we cannot determine a CF term.
                    den_lo = r * y_lo + s
                    den_hi = r * (y_lo + step) + s
                    if den_lo == 0 or den_hi == 0:
                        break  # endpoint is a pole; need more digits
                    if (den_lo > 0) != (den_hi > 0):
                        break  # pole strictly inside; need more digits

                    xv_lo = (p * y_lo + q) / den_lo
                    xv_hi = (p * (y_lo + step) + q) / den_hi
                    # Möbius transforms can be decreasing; normalise to xv_lo ≤ xv_hi.
                    if xv_lo > xv_hi:
                        xv_lo, xv_hi = xv_hi, xv_lo

                    a = math.floor(xv_lo)
                    # The upper bound is open (x < xv_hi), so its effective floor is
                    # floor(xv_hi) when xv_hi is not an integer, else xv_hi − 1.
                    hi_floor = math.floor(xv_hi) if xv_hi != math.floor(xv_hi) else math.floor(xv_hi) - 1
                    if a != hi_floor:
                        break  # interval straddles an integer; need more

                    yield a
                    # Strip term a from the state: x = a + 1/x' implies the new
                    # inverse state for x' is (r, s, p − a·r, q − a·s).
                    p, q, r, s = r, s, p - a * r, q - a * s

            yield from emit_while_pinned()
            for d in it:
                step = step / base
                y_lo = y_lo + d * step
                yield from emit_while_pinned()

            # Digits exhausted: y is the exact rational y_lo.
            # Evaluate the current state at y_lo to get the remaining CF value,
            # then extract its CF terms via the standard Euclidean algorithm.
            den = r * y_lo + s
            if den == 0:
                return  # pole at y_lo means the CF terminates here
            x = (p * y_lo + q) / den
            while True:
                a = math.floor(x)
                yield a
                frac = x - a
                if frac == 0:
                    break
                x = Fraction(1) / frac

        return cls([], _source=_gen())

    # ------------------------------------------------------------------
    # Iteration — index-stable so Gosper can hold two independent cursors
    # ------------------------------------------------------------------

    def _grow_cache(self) -> bool:
        """Try to extend _cache by one term. Returns True if a term was added."""
        if self.repeating:
            self._cache.extend(self.repeating)
            return True
        if self._source is not None:
            try:
                v = next(self._source)
                self._cache.append(v)
                return True
            except StopIteration:
                self._source = None
        return False

    def _iter_from(self, start: int = 0) -> Iterator[int]:
        """Yield terms starting at cache position `start`.

        Safe to call multiple times concurrently; each call carries its own
        cursor while sharing the underlying _cache.
        """
        i = start
        while True:
            if i < len(self._cache):
                yield self._cache[i]
                i += 1
            elif self._grow_cache():
                # new term(s) added to cache; loop will pick them up
                pass
            else:
                return  # exhausted

    def __iter__(self) -> Iterator[int]:
        return self._iter_from(0)

    def digits(self, base: int = 10) -> Iterator[int]:
        """Yield the base-B digits of this value, left to right.

        The first yielded value is the integer part (floor of self); it may
        exceed base for values like e^π in base 2.  All subsequent values are
        fractional digits in [0, base).

        This is Gosper's "multiply instead of reciprocate" algorithm.  The
        Möbius state (a,b,c,d) starts as the identity matrix — representing
        z = x, i.e. "output equals input unchanged" — and CF terms are consumed
        to narrow the interval [z(1), z(∞)] until each digit is pinned:

            CF output:      z ← 1/(z − t)      rows swap:   (c, d, a−tc, b−td)
            Base-B output:  z ← B·(z − t)      top scaled:  (Ba−tc, Bb−td, c, d)

        Same state, same corner check, different residual operation.
        """
        a, b, c, d = 1, 0, 0, 1
        scale = 1  # 1 for the integer part, then `base` for each fractional digit

        for index, term in enumerate(self._iter_from(0)):
            if index > 0 and term < 1:
                raise ValueError(
                    f"invalid CF term at position {index}: {term!r} "
                    f"(terms after the first must be ≥ 1; "
                    f"a zero or negative value indicates a recanted or corrupt generator)"
                )
            # Ingest: x = term + 1/x'  →  (a,b,c,d) = (a·term+b, a, c·term+d, c)
            a, b, c, d = a * term + b, a, c * term + d, c
            # Emit as many digits as this one narrowing allows
            while c != 0 and (c + d) != 0 and (c > 0) == (c + d > 0):
                t_inf = (scale * a) // c
                t_one = (scale * (a + b)) // (c + d)
                if t_inf != t_one:
                    break
                yield t_inf
                a, b, c, d = scale * a - t_inf * c, scale * b - t_inf * d, c, d
                scale = base

        # CF exhausted: remaining value is exactly a/c — long-divide the rest.
        if c == 0:
            return
        int_part, a = divmod(a, c)
        if scale == 1:
            yield int_part
            scale = base  # noqa: F841 (scale not used further, but mirrors the loop above)
        while a != 0:
            a *= base
            digit, a = divmod(a, c)
            yield digit

    def take(self, n: int) -> CF:
        """Return a new finite CF with the first n coefficients."""
        terms: list[int] = []
        for i, a in enumerate(self._iter_from(0)):
            if i >= n:
                break
            terms.append(a)
        return CF(terms)

    # ------------------------------------------------------------------
    # Value
    # ------------------------------------------------------------------

    def to_fraction(self) -> Fraction:
        """Return the exact rational value. Only valid for finite CFs."""
        if not self.is_finite():
            raise ValueError("to_fraction() only works for finite CFs; use convergent() for periodic/infinite ones")
        result = Fraction(0)
        # Walk backwards through the cache (finite, so drain first)
        terms = list(self._iter_from(0))
        if not terms:
            raise ValueError("empty CF")
        result = Fraction(terms[-1])
        for a in reversed(terms[:-1]):
            result = a + Fraction(1, result)
        return result

    def __float__(self) -> float:
        from .convergents import convergent

        # Use a deep convergent for a good float approximation
        c = convergent(self.take(40), 39 if self.take(40) else 0)
        return float(c)

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def is_finite(self) -> bool:
        return not self.repeating and self._source is None

    def is_periodic(self) -> bool:
        return bool(self.repeating)

    @property
    def err_estimate(self) -> Fraction:
        """Upper bound on the truncation error for a finite CF.

        For any infinite CF whose first n terms match self, the true value x
        satisfies:

            |x - self.to_fraction()| < |p_n/q_n - p_{n-1}/q_{n-1}|
                                      = 1 / (q_n * q_{n-1})

        because consecutive convergents always bracket x from opposite sides.
        The bound equals the difference of the last two convergents as an exact
        Fraction — no floating-point or mpmath required.

        The actual error is smaller by a factor of roughly q_n/q_{n-1} (the
        growth ratio of consecutive denominators), so this bound is tight when
        consecutive CF terms are similar in size.

        Only valid for finite CFs (i.e. results of take()); raises ValueError
        for infinite or periodic CFs.
        """
        if not self.is_finite():
            raise ValueError("err_estimate is only defined for finite CFs (use .take(n) first)")
        from .convergents import convergent_pairs as _pairs

        pairs = list(_pairs(self))
        if not pairs:
            raise ValueError("empty CF")
        if len(pairs) == 1:
            return Fraction(1)  # single-term CF: error could be up to 1
        p_n, q_n = pairs[-1]
        p_prev, q_prev = pairs[-2]
        return abs(Fraction(p_n, q_n) - Fraction(p_prev, q_prev))

    def _ensure_convergents(self, n_terms: int) -> None:
        """Extend cached convergents through the available prefix."""
        if n_terms < 1:
            raise ValueError("n_terms must be at least 1")

        while len(self._convergent_cache) < n_terms:
            i = len(self._convergent_cache)
            if i >= len(self._cache) and not self._grow_cache():
                return

            a = self._cache[i]
            if i == 0:
                p_prev2, q_prev2 = 0, 1
                p_prev1, q_prev1 = 1, 0
            elif i == 1:
                p_prev2, q_prev2 = 1, 0
                p_prev1, q_prev1 = self._convergent_cache[-1]
            else:
                p_prev2, q_prev2 = self._convergent_cache[-2]
                p_prev1, q_prev1 = self._convergent_cache[-1]
            self._convergent_cache.append((a * p_prev1 + p_prev2, a * q_prev1 + q_prev2))
            p_new, q_new = self._convergent_cache[-1]
            self._float_convergent_cache.append((float(p_new), float(q_new)))

    def interval(self, n_terms: int) -> tuple[Fraction, Fraction]:
        """Return rational bounds known to contain this CF after n_terms.

        The bound uses the nth convergent and the neighboring value obtained by
        increasing the last consumed term by one.  Consecutive depths flip order;
        the returned pair is always ``(lo, hi)``.  If the CF terminates before
        n_terms, both bounds are the exact finite value.
        """
        self._ensure_convergents(n_terms)
        if not self._convergent_cache:
            raise ValueError("empty CF")

        idx = min(n_terms, len(self._convergent_cache)) - 1
        p, q = self._convergent_cache[idx]
        q0 = Fraction(p, q)
        if len(self._convergent_cache) < n_terms:
            return q0, q0

        if idx == 0:
            p_prev, q_prev = 1, 0
        else:
            p_prev, q_prev = self._convergent_cache[idx - 1]
        q1 = Fraction(p + p_prev, q + q_prev)
        if q0 <= q1:
            return q0, q1
        return q1, q0

    def interval_ints(self, n_terms: int) -> tuple[int, int, int, int, bool]:
        """Return the interval endpoints as raw integer pairs, avoiding Fraction creation.

        Returns ``(p0, q0, p1, q1, same)`` where ``p0/q0`` and ``p1/q1`` are the
        low and high rational bounds (always p0/q0 <= p1/q1), and ``same`` is
        True when both endpoints are equal (CF terminated before n_terms).
        The returned pairs are in lowest terms only by the convergent recurrence,
        not via explicit GCD reduction, so callers must not reduce further.
        """
        self._ensure_convergents(n_terms)
        if not self._convergent_cache:
            raise ValueError("empty CF")

        idx = min(n_terms, len(self._convergent_cache)) - 1
        p, q = self._convergent_cache[idx]

        if len(self._convergent_cache) < n_terms:
            # CF terminated; both endpoints equal p/q.
            return p, q, p, q, True

        if idx == 0:
            p_prev, q_prev = 1, 0
        else:
            p_prev, q_prev = self._convergent_cache[idx - 1]

        # Perturbed convergent: (p + p_prev) / (q + q_prev)
        pp, qp = p + p_prev, q + q_prev

        # Even-depth convergents are lower bounds; odd are upper bounds.
        # Compare p/q vs pp/qp: p*qp vs pp*q (cross multiply, denominators > 0).
        # q_prev may be 0 at idx==0, but qp = q + q_prev = q + 0 = q which is > 0 for idx>0.
        # At idx==0: q=0 is not possible for valid CFs (first convergent is a0/1).
        # Actually p_prev=1, q_prev=0 at idx==0, so qp = q + 0 = q (=1 for a0=1-digit CF).
        if p * qp <= pp * q:
            return p, q, pp, qp, False
        return pp, qp, p, q, False

    def interval_float(self, n_terms: int) -> tuple[float, float]:
        """Like interval() but returns float bounds instead of Fraction.

        Uses the float convergent cache for speed.  Exact for convergent
        numerators below 2^53; above that, treat as an approximation only.
        Both bounds are the same float when the CF terminates before n_terms.
        """
        self._ensure_convergents(n_terms)
        if not self._float_convergent_cache:
            raise ValueError("empty CF")

        idx = min(n_terms, len(self._float_convergent_cache)) - 1
        fp, fq = self._float_convergent_cache[idx]
        fq0 = fp / fq
        if len(self._float_convergent_cache) < n_terms:
            return fq0, fq0

        if idx == 0:
            fp_prev, fq_prev = 1.0, 0.0
        else:
            fp_prev, fq_prev = self._float_convergent_cache[idx - 1]
        fq1 = (fp + fp_prev) / (fq + fq_prev)
        if fq0 <= fq1:
            return fq0, fq1
        return fq1, fq0

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    _REPR_DIGITS = 10  # decimal digits shown in repr approximation

    def __repr__(self) -> str:
        # Peek at cached terms (includes lazily-computed ones already seen)
        preview = list(self._cache)
        rep = list(self.repeating)
        finite = self.is_finite()

        if not preview:
            if self._source is not None:
                # Peek at a few terms so repr is informative, not just "CF([?])".
                for _ in self._iter_from(0):
                    preview = list(self._cache)
                    if len(preview) >= 3:
                        break
            if not preview:
                return "CF([])" if self._source is None else "CF([?])"

        a0 = str(preview[0])
        rest_parts = [str(x) for x in preview[1:]]

        if rep:
            rest_parts.append(f"({', '.join(str(x) for x in rep)})")
        elif not finite:
            rest_parts.append("...")

        if rest_parts:
            cf_str = f"[{a0}; {', '.join(rest_parts)}]"
        else:
            cf_str = f"[{a0}]"

        # Append decimal approximation; "=" for exact terminating, "≈" otherwise.
        # Request one extra digit beyond _REPR_DIGITS to detect whether it truncates.
        from itertools import islice as _islice

        raw = list(_islice(self.digits(10), self._REPR_DIGITS + 2))
        if not raw:
            return cf_str
        int_part = raw[0]
        frac = raw[1 : self._REPR_DIGITS + 1]  # at most _REPR_DIGITS fractional digits
        truncated = len(raw) > self._REPR_DIGITS + 1  # generator had more to give
        dec_str = str(int_part) + ("." + "".join(str(d) for d in frac) if frac else "")
        sym = "≈" if (not finite or truncated) else "="
        return f"{cf_str} {sym} {dec_str}"

    def __str__(self) -> str:
        return repr(self)

    # ------------------------------------------------------------------
    # Equality and ordering — Gosper term-by-term comparison
    #
    # CF comparison reduces to reading terms until the first discrepancy.
    # At position k where a_k != b_k:
    #   even k (a0, a2, ...): larger term → larger value  (normal order)
    #   odd  k (a1, a3, ...): larger term → smaller value (inverted, because
    #                         we're comparing denominators — a bigger a_k
    #                         makes 1/(a_k + …) smaller)
    # A CF that terminates at position k is treated as having a virtual +∞
    # there (the fractional tail vanishes, equivalent to "next denominator ∞").
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            if self.is_finite():
                return self.to_fraction() == other
            head = self.take(2)
            return len(head.terms) == 1 and head.terms[0] == other
        if not isinstance(other, CF):
            return NotImplemented
        # Finite CFs: exact rational comparison
        if self.is_finite() and other.is_finite():
            return self.to_fraction() == other.to_fraction()
        # Mixed or infinite: compare term by term, treating exhaustion as +∞
        xi = self._iter_from(0)
        yi = other._iter_from(0)
        for _ in range(self._CMP_DEPTH):
            av = next(xi, None)
            bv = next(yi, None)
            if av is None and bv is None:
                return True
            if av is None or bv is None:
                return False  # one terminated while the other still has non-equivalent terms
            if av != bv:
                # check for equivalent terms:
                #    [..., av, 1]    or     [..., bv + 1]
                # == [..., av + 1]       == [..., bv, 1]
                if bv == av + 1 and next(xi, None) == 1 and next(xi, None) is None and next(yi, None) is None:
                    return True  # equivalent terms: av, 1 == av + 1
                if av == bv + 1 and next(yi, None) == 1 and next(xi, None) is None and next(yi, None) is None:
                    return True  # equivalent terms: bv, 1 == bv + 1
                return False  # non-equivalent terms
        return True  # _CMP_DEPTH exhausted — treat as equal

    # Old convergent-based __eq__ (kept for reference):
    # def __eq__(self, other):
    #     if not isinstance(other, CF): return NotImplemented
    #     if self.is_finite() and other.is_finite():
    #         return self.to_fraction() == other.to_fraction()
    #     depth = 50
    #     from .convergents import convergents as _convs
    #     a_convs = list(_convs(self.take(depth)))
    #     b_convs = list(_convs(other.take(depth)))
    #     if not a_convs or not b_convs: return False
    #     return a_convs[-1] == b_convs[-1]

    def __hash__(self) -> int:
        if self.is_finite():
            return hash(self.to_fraction())
        return hash(tuple(self.take(20)._cache))

    _CMP_DEPTH = 300  # max terms to read before declaring equal

    def __lt__(self, other: CF) -> bool:
        xi = self._iter_from(0)
        yi = other._iter_from(0)
        for k in range(self._CMP_DEPTH):
            av = next(xi, None)
            bv = next(yi, None)
            if av is None and bv is None:
                return False  # equal
            if av is None:
                # self terminates; virtual +∞ at position k
                # even k: +∞ > bv → self > other → not less-than
                # odd  k: inverted order → +∞ means self < other
                return k % 2 == 1
            if bv is None:
                # other terminates; virtual +∞ at position k for other
                # even k: other's +∞ > self → self < other
                # odd  k: inverted → other < self → not less-than
                return k % 2 == 0
            if av != bv:
                return (av < bv) if k % 2 == 0 else (av > bv)
        return False  # depth exhausted — treat as equal

    # Old convergent-based __lt__ (kept for reference):
    # def __lt__(self, other):
    #     depth = 50
    #     from .convergents import convergents as _convs
    #     a_convs = list(_convs(self.take(depth)))
    #     b_convs = list(_convs(other.take(depth)))
    #     a = a_convs[-1] if a_convs else Fraction(0)
    #     b = b_convs[-1] if b_convs else Fraction(0)
    #     return a < b

    def __le__(self, other: CF) -> bool:
        return self == other or self < other

    def __gt__(self, other: CF) -> bool:
        return other < self

    def __ge__(self, other: CF) -> bool:
        return other <= self

    # ------------------------------------------------------------------
    # Arithmetic operators (delegate to gosper.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce(other: object) -> "CF | None":
        if isinstance(other, CF):
            return other
        if isinstance(other, (int, Fraction)):
            return CF.from_rational(Fraction(other))
        return None

    def __add__(self, other: object) -> "CF":
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        from .gosper import cf_add

        return cf_add(self, coerced)

    def __sub__(self, other: object) -> "CF":
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        from .gosper import cf_sub

        return cf_sub(self, coerced)

    def __mul__(self, other: object) -> "CF":
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        from .gosper import cf_mul

        return cf_mul(self, coerced)

    def __truediv__(self, other: object) -> "CF":
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        from .gosper import cf_div

        return cf_div(self, coerced)

    def __neg__(self) -> CF:
        from .gosper import cf_homographic

        return cf_homographic(self, -1, 0, 0, 1)

    def __pow__(self, n: int | Fraction | CF) -> CF:
        if isinstance(n, int):
            if n == 0:
                return CF.from_int(1)
            if n == 1:
                return self
            if n < 0:
                return CF.from_int(1) / (self ** (-n))
            # repeated squaring
            result = None
            base = self
            while n:
                if n & 1:
                    result = base if result is None else result * base
                base = base * base
                n >>= 1
            return result  # type: ignore[return-value]  # n>=2 guarantees result is set

        if isinstance(n, Fraction):
            from .power import Pow

            if self.is_finite():
                return Pow(self.to_fraction(), n)
            from .exponential import Exp
            from .logarithm import Ln

            return Exp(CF.from_rational(n) * Ln(self))

        if isinstance(n, CF):
            from .exponential import Exp
            from .logarithm import Ln

            return Exp(n * Ln(self))

        return NotImplemented

    def __rpow__(self, other: int | Fraction) -> CF:
        """Support int ** CF and Fraction ** CF (e.g. 2 ** Pi())."""
        if isinstance(other, int):
            other = Fraction(other)
        if isinstance(other, Fraction):
            if other <= 0:
                raise ValueError(f"base must be positive for CF exponent, got {other}")
            from .exponential import Exp
            from .logarithm import Ln

            return Exp(self * Ln(other))
        return NotImplemented

    def __floordiv__(self, other: object) -> "CF":
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        from .mod import cf_floordiv

        return cf_floordiv(self, coerced)

    def __mod__(self, other: object) -> "CF":
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        from .mod import cf_mod

        return cf_mod(self, coerced)

    def __divmod__(self, other: object) -> "tuple[CF, CF]":
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        from .gosper import cf_homographic, cf_sub
        from .mod import _floor_quotient

        n = _floor_quotient(self, coerced)
        return CF.from_int(n), cf_sub(self, cf_homographic(coerced, n, 0, 0, 1))

    def __floor__(self) -> int:
        return next(self._iter_from(0))

    def __ceil__(self) -> int:
        return -next((-self)._iter_from(0))

    def __trunc__(self) -> int:
        f = next(self._iter_from(0))
        # For negative values trunc = ceil (rounds toward zero)
        return f if f >= 0 else -next((-self)._iter_from(0))

    def __radd__(self, other: int | Fraction) -> CF:
        return CF.from_rational(Fraction(other)) + self

    def __rmul__(self, other: int | Fraction) -> CF:
        return CF.from_rational(Fraction(other)) * self

    def __rsub__(self, other: int | Fraction) -> CF:
        return CF.from_rational(Fraction(other)) - self

    def __rtruediv__(self, other: int | Fraction) -> CF:
        return CF.from_rational(Fraction(other)) / self

    def __rfloordiv__(self, other: int | Fraction) -> CF:
        return CF.from_rational(Fraction(other)) // self

    def __rmod__(self, other: int | Fraction) -> CF:
        return CF.from_rational(Fraction(other)) % self

    def __rdivmod__(self, other: int | Fraction) -> tuple[CF, CF]:
        return divmod(CF.from_rational(Fraction(other)), self)

    def reciprocal(self) -> CF:
        """Return 1/self via the Gosper homographic transform (0x+1)/(1x+0)."""
        from .gosper import cf_homographic

        return cf_homographic(self, 0, 1, 1, 0)

    def normalize(self) -> CF:
        """Return the canonical simple-CF form.

        Useful when self was built with 0 terms mid-sequence, repeated terms,
        or other non-canonical structure (e.g. from ``from_generalized_cf``).
        Pipes through the identity Gosper transform, which re-emits terms one
        at a time and absorbs any irregularities.
        """
        from .gosper import cf_homographic

        return cf_homographic(self, 1, 0, 0, 1)

    @classmethod
    def from_generalized_cf(cls, pairs: Iterable[tuple[int | Fraction, int | Fraction]]) -> CF:
        """Convert a generalized continued fraction to a simple CF, lazily.

        A generalized CF has the form:
            b0 + a1 / (b1 + a2 / (b2 + a3 / ...))

        ``pairs`` is an iterable of ``(b_n, a_{n+1})`` pairs — integers or
        Fractions.  Each pair consumed from the left adds one level of nesting.
        Simple CF terms are emitted as soon as the output is determined,
        without needing to read the whole input first.

        Internally, each level is a Möbius (linear fractional) substitution of
        the form (ax+b)/(cx+d) — a formula where x represents the rest of the
        fraction.
        The running state tracks how the accumulated layers map the tail,
        and emits a term whenever all possible tail values round to the same integer.

        **Negative a_{n+1} terms will stall.**  Each Möbius map with a < 0 is
        *order-reversing* on [1, ∞): it maps the interval to a bounded range
        and introduces a pole at x = −d/c inside [1, ∞).  After a few such
        maps the accumulated state has a genuine pole in the tracked range —
        the algorithm correctly refuses to emit because the output could be
        arbitrarily large near the pole, even though the *actual* tail value
        stays finite and produces a well-defined result.

        The theoretical fix is *even-part contraction*: consuming pairs two at
        a time composes two order-reversing maps into an order-preserving one.
        In practice that only pushes the pole further out — it resurfaces after
        a few more contracted pairs once the numerators grow large enough.
        The reliable workaround is to transform the GCF algebraically so that
        all a_{n+1} > 0 before calling this method.  For example:

            arctanh(x)  →  Ln((1+x)/(1−x)) / 2   (see ``Arctanh``)
            arcsinh(x)  →  Ln(x + sqrt(x²+1))      (see ``Arcsinh``)
        """
        from fractions import Fraction as _Frac

        def _terms() -> Iterator[int]:
            a = _Frac(1)
            b = _Frac(0)
            c = _Frac(0)
            d = _Frac(1)

            def _try_emit() -> Iterator[int]:
                # Emit simple CF terms while the output on x' ∈ [1,∞) is
                # determined: both corner denominators same sign, floors agree.
                nonlocal a, b, c, d
                while True:
                    den_inf = c
                    den_one = c + d
                    if den_inf == 0 or den_one == 0:
                        break
                    if (den_inf > 0) != (den_one > 0):
                        break  # pole in [1, ∞)
                    q_inf = int(a // c)
                    q_one = int((a + b) // (c + d))
                    if q_inf != q_one:
                        break
                    yield q_inf
                    a, b, c, d = c, d, a - q_inf * c, b - q_inf * d

            stall = 0
            for b_n, a_next in pairs:
                b_n = _Frac(b_n)
                a_next = _Frac(a_next)
                # Compose state with new Möbius transform [[b_n, a_next],[1,0]]
                a, b, c, d = (
                    a * b_n + b,
                    a * a_next,
                    c * b_n + d,
                    c * a_next,
                )
                n_emitted = 0
                for term in _try_emit():
                    yield term
                    n_emitted += 1
                if n_emitted == 0:
                    stall += 1
                    if stall >= _GCF_STALL_LIMIT:
                        raise ValueError(
                            f"from_generalized_cf stalled: no term emitted after consuming {_GCF_STALL_LIMIT} consecutive pairs"
                        )
                else:
                    stall = 0

            # Drain: both corners now converge to a/c
            if c != 0:
                yield from cls.from_rational(_Frac(a, c))

        return cls([], _source=_terms())
