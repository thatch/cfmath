"""HAKMEM 101C — trihomographic (three-input) continued fraction arithmetic.

== Background: the progression ==

  Homographic  (1 input): y = (ax + b) / (cx + d)                   — 4 coefficients
  Bihomographic (2 inputs): z = (axy+bx+cy+d) / (exy+fx+gy+h)      — 8 coefficients
  Trihomographic (3 inputs): z = N(w,x,y) / D(w,x,y)               — 16 coefficients

where N and D are multilinear polynomials in w, x, y (degree ≤ 1 in each variable).

== "Quadratic term": what Gosper means ==

In the bihomographic formula, the term `xy` is "quadratic" — a product of two
variables.  Item 101C extends this to three inputs, giving a trilinear formula
with cross-term `wxy`.

But the sharper motivation is *genuinely quadratic expressions in a single input*.
Bihomographic keeps each variable linear, so `x²` is out of reach.  The fix:
supply the same CF as **two independent iterators** (w = x = some_cf).  Then the
`wx` cross-term in the trihomographic formula becomes `x²`.  More generally,
`(ax² + bx + c) / (dx² + ex + f)` is a "quadratic fractional transformation" and
maps naturally onto a bihomographic formula — or onto a trihomographic when a
third input is also present.  This is the pattern 101C is pointing at: formulas
that are quadratic (or higher) in a CF input are handled by feeding that input
multiple times, with the n-homographic machinery tracking the interactions.

== Why Euler's algorithm specifically? ==

Euler's series-to-CF algorithm builds a CF from the series S = a₀ + a₁ + a₂ + …
via the step:

    tail_k  =  a_k / (a_k − a_{k+1} + a_{k+1} · tail_{k+1})

Setting  w = a_k,  x = a_{k+1},  y = tail_{k+1}:

    z  =  w / (w − x + x·y)

The denominator  w − x + xy  has a cross-term `xy` *and* depends on `w`
simultaneously — it cannot be factored into any chain of two-input steps without
materialising an intermediate lazy CF.  The trihomographic formula handles it in
one pass, allowing output terms to be emitted as soon as all three inputs agree.

== The state tensor ==

Two lists of 8 integers (numerator / denominator), indexed by bitmask:
  bit 2 = w-present,  bit 1 = x-present,  bit 0 = y-present

  index 7 = 0b111: wxy        index 3 = 0b011: xy
  index 6 = 0b110: wx         index 2 = 0b010: x
  index 5 = 0b101: wy         index 1 = 0b001: y
  index 4 = 0b100: w          index 0 = 0b000: 1 (constant)

Corner at C (bitmask of which variables → ∞):
  sum of coefficients at all indices i where (i & C) == C.
  All 8 corners must agree on a floor before emitting an output term.

== The "quadratic fractional transformation" and higher ==

The user asks: what would `z = (ax² + bx + c) / (dx² + ex + f)` be called?

It's a **quadratic fractional transformation** (or degree-2 rational map) — one
step above the Möbius/homographic (degree 1).  It is NOT directly implementable
as a single homographic in x.  But it slots neatly into the existing machinery:

  bihomographic(x, x, a, b/2, b/2, c,  d, e/2, e/2, f)   [split b and e evenly]

Setting both inputs to x supplies u = v = x, so the cross-term u·v becomes x².
More generally:

  Degree 1 in x (linear rational):  homographic with x once
  Degree 2 in x (quadratic rational): bihomographic with x twice
  Degree 3 in x (cubic rational):    trihomographic with x three times
  Degree n in x:                      n-homographic (the 2^(n+1)-coeff extension)

Is it useful?  Yes, for several things:
  - Computing a CF root of a known quadratic equation (e.g. √D = root of x²−D=0)
    by iterating the corresponding Möbius composition map
  - Computing the composition f∘f of a homographic f — one step of the
    quadratic convergence iteration used in, e.g., AGM
  - Verifying quadratic irrationals (any eventually-periodic CF satisfies a
    quadratic equation; plugging it back in should give the identity)

Run: ulimit -v 2097152 && python prototypes/hakmem101c_trihomographic.py
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction
from typing import Iterator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from cfmath import CF, Phi, Sqrt
from cfmath.gosper import _MAX_STALL, _bihomographic_terms, _homographic_terms

# ---------------------------------------------------------------------------
# Corner sums
# ---------------------------------------------------------------------------


def _corner_sum(coeffs: list[int], C: int) -> int:
    """Sum coefficients at corner C (bitmask of variables that → ∞)."""
    return sum(coeffs[i] for i in range(8) if (i & C) == C)


# ---------------------------------------------------------------------------
# Output check
# ---------------------------------------------------------------------------


def _tri_output(n: list[int], d: list[int]) -> int | None:
    """Return the next CF term if all 8 corners agree on a floor, else None."""
    sign = None
    floors = []
    for C in range(8):
        num = _corner_sum(n, C)
        den = _corner_sum(d, C)
        if den == 0:
            return None
        if sign is None:
            sign = den > 0
        elif (den > 0) != sign:
            return None
        floors.append(num // den)
    return floors[0] if len(set(floors)) == 1 else None


# ---------------------------------------------------------------------------
# Ingest: substitute variable → t + 1/variable'
# ---------------------------------------------------------------------------


def _tri_ingest(n: list[int], d: list[int], B: int, t: int) -> tuple[list[int], list[int]]:
    """Ingest term t from the variable whose bitmask is B (4=w, 2=x, 1=y).

    After substituting  var = t + 1/var'  and multiplying through by var':
      - Indices WITH bit B:    new[i] = old[i] * t + old[i & ~B]
        (old "with-var" coeff times t, plus its "without-var" partner)
      - Indices WITHOUT bit B: new[i] = old[i | B]
        (takes the old "with-var" coefficient directly)
    """
    new_n = list(n)
    new_d = list(d)
    for i in range(8):
        if i & B:
            j = i & ~B
            new_n[i] = n[i] * t + n[j]
            new_d[i] = d[i] * t + d[j]
        else:
            j = i | B
            new_n[i] = n[j]
            new_d[i] = d[j]
    return new_n, new_d


# ---------------------------------------------------------------------------
# Emit: z = q + 1/z'
# ---------------------------------------------------------------------------


def _tri_emit(n: list[int], d: list[int], q: int) -> tuple[list[int], list[int]]:
    new_n = list(d)
    new_d = [n[i] - q * d[i] for i in range(8)]
    return new_n, new_d


# ---------------------------------------------------------------------------
# Exhaustion reduction: fall through to the right lower-dim algorithm
# ---------------------------------------------------------------------------


def _reduce_exhausted(
    n: list[int],
    d: list[int],
    exhausted: int,
    w_iter: Iterator[int],
    x_iter: Iterator[int],
    y_iter: Iterator[int],
) -> Iterator[int]:
    """Yield remaining terms after variables in `exhausted` (bitmask) → ∞.

    When a variable is exhausted its tail → ∞, so only monomials containing
    that variable survive in the ratio.  After factoring it out, the remaining
    formula is lower-dimensional.

    General rule: for each index i in the reduced formula (a subset of the
    remaining-variable bits), the coefficient is  old[i | exhausted].

    Bit positions in the reduced formula are re-mapped from the remaining bits:
      bit pos 0 → first remaining bit, pos 1 → second, etc.
    """
    remaining = [b for b in (4, 2, 1) if not (exhausted & b)]
    k = len(remaining)

    # Build reduced coefficient arrays (size 2^k)
    size = 1 << k
    rn = [0] * size
    rd = [0] * size
    for new_i in range(size):
        old_i = exhausted
        for pos, bit in enumerate(remaining):
            if new_i & (1 << pos):
                old_i |= bit
        rn[new_i] = n[old_i]
        rd[new_i] = d[old_i]

    iters = {4: w_iter, 2: x_iter, 1: y_iter}

    if k == 0:
        # All three exhausted: value is n[7]/d[7]
        if rd[0]:
            yield from CF.from_fraction(rn[0], rd[0])

    elif k == 1:
        # One variable remains: homographic
        bit = remaining[0]
        yield from _homographic_terms(iters[bit], rn[1], rn[0], rd[1], rd[0])

    else:
        # Two variables remain: bihomographic
        # rn mapping: rn[0]=const, rn[1]=first-var, rn[2]=second-var, rn[3]=both
        b0, b1 = remaining
        yield from _bihomographic_terms(
            iters[b0],
            iters[b1],
            rn[3],
            rn[1],
            rn[2],
            rn[0],
            rd[3],
            rd[1],
            rd[2],
            rd[0],
        )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def _trihomographic_terms(
    w_iter: Iterator[int],
    x_iter: Iterator[int],
    y_iter: Iterator[int],
    n: list[int],
    d: list[int],
) -> Iterator[int]:
    """Yield CF terms of N(w,x,y)/D(w,x,y) given three input CF iterators."""
    w_done = x_done = y_done = False
    w_started = x_started = y_started = False
    stall = 0
    poll_idx = 0  # for round-robin input selection

    while True:
        if w_started and x_started and y_started:
            q = _tri_output(n, d)
        else:
            q = None

        if q is not None:
            yield q
            stall = 0
            n, d = _tri_emit(n, d, q)
            continue

        # Reduce to lower-dimensional algorithm when any inputs are exhausted
        exhausted = (4 if w_done else 0) | (2 if x_done else 0) | (1 if y_done else 0)
        if exhausted:
            yield from _reduce_exhausted(n, d, exhausted, w_iter, x_iter, y_iter)
            return

        # --- Choose which input to consume next ---
        # Phase 1: ensure all inputs have entered the [1, ∞) tail domain
        if not w_started:
            B = 4
        elif not x_started:
            B = 2
        elif not y_started:
            B = 1
        else:
            # Phase 2: round-robin to prevent input starvation (a pure spread
            # heuristic can starve a low-spread input, preventing discovery of
            # its StopIteration and thus blocking the exhaustion reduction).
            avail = [b for b, done in ((4, w_done), (2, x_done), (1, y_done)) if not done]
            B = avail[poll_idx % len(avail)]
            poll_idx += 1

        if B == 4:
            try:
                t = next(w_iter)
                w_started = True
                n, d = _tri_ingest(n, d, 4, t)
            except StopIteration:
                w_done = True
                w_started = True
        elif B == 2:
            try:
                t = next(x_iter)
                x_started = True
                n, d = _tri_ingest(n, d, 2, t)
            except StopIteration:
                x_done = True
                x_started = True
        else:
            try:
                t = next(y_iter)
                y_started = True
                n, d = _tri_ingest(n, d, 1, t)
            except StopIteration:
                y_done = True
                y_started = True

        stall += 1
        if stall >= _MAX_STALL:
            return  # give up (e.g. result is an exact integer — convergents loop)


def cf_trihomographic(
    w: CF,
    x: CF,
    y: CF,
    n: list[int],
    d: list[int],
) -> CF:
    """Compute N(w,x,y)/D(w,x,y) lazily as a continued fraction.

    n and d are each 8 integers indexed by bitmask
    (index 7=wxy, 6=wx, 5=wy, 4=w, 3=xy, 2=x, 1=y, 0=constant).
    """
    return CF(
        [],
        _source=_trihomographic_terms(
            w._iter_from(0),
            x._iter_from(0),
            y._iter_from(0),
            n,
            d,
        ),
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def cf_fma(w: CF, x: CF, y: CF) -> CF:
    """Fused multiply-add: w*x + y.

    The wx cross-term and y addend appear in the same trilinear formula.
    """
    #         [1, y, x, xy,  w, wy, wx, wxy]
    n = [0, 1, 0, 0, 0, 0, 1, 0]
    d = [1, 0, 0, 0, 0, 0, 0, 0]
    return cf_trihomographic(w, x, y, n, d)


def cf_euler_step(a_prev: CF, a_curr: CF, tail: CF) -> CF:
    """One step of Euler's series-to-CF algorithm.

    Advances the tail by one level:
        z  =  a_prev / (a_prev − a_curr + a_curr · tail)

    Setting w=a_prev, x=a_curr, y=tail:
      Numerator:   w               → n[4] = 1
      Denominator: w − x + x·y    → d[4]=1 (w), d[2]=−1 (x), d[3]=1 (xy)

    The xy cross-term in the denominator makes this genuinely trilinear: the
    denominator depends on all three inputs simultaneously.
    """
    #         [1, y, x,  xy, w, wy, wx, wxy]
    n = [0, 0, 0, 0, 1, 0, 0, 0]
    d = [0, 0, -1, 1, 1, 0, 0, 0]
    return cf_trihomographic(a_prev, a_curr, tail, n, d)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _approx(cf: CF, k: int = 10) -> float:
    from cfmath.convergents import convergent

    terms = list(cf.take(k + 1))
    if not terms:
        return float("nan")
    return float(convergent(cf, min(len(terms) - 1, k)))


def _show(label: str, cf: CF, n_terms: int = 10) -> None:
    terms = list(cf.take(n_terms))
    cf_str = "[%s; %s ...]" % (terms[0], ", ".join(str(t) for t in terms[1:]))
    approx = _approx(cf)
    print("  %-30s  %s  ≈ %.8f" % (label, cf_str, approx))


print("=" * 70)
print("HAKMEM 101C: trihomographic (three-input) CF arithmetic")
print("=" * 70)
print()

# ── Example 1: rational fma ─────────────────────────────────────────────────
print("── fused multiply-add: w*x + y ──")
print()

r = cf_fma(CF([2]), CF([3]), CF([5]))
assert list(r.take(2)) == [11], list(r.take(2))
_show("2*3 + 5 = 11", cf_fma(CF([2]), CF([3]), CF([5])))


half = CF.from_fraction(1, 2)
third = CF.from_fraction(1, 3)
qtr = CF.from_fraction(1, 4)
expected = Fraction(1, 2) * Fraction(1, 3) + Fraction(1, 4)  # = 5/12
_show("(1/2)*(1/3) + 1/4 = 5/12", cf_fma(half, third, qtr))
assert abs(_approx(cf_fma(half, third, qtr)) - float(expected)) < 1e-8

print()

# ── Example 2: x² via two copies — the "quadratic term" connection ──────────
print("── x² via two independent iterators (the 'quadratic term') ──")
print()
print("  Bihomographic is multilinear: each input appears at most once.")
print("  To compute x², supply x as two independent iterators (w = x = same CF).")
print("  The wx cross-term in the state then captures x² at runtime.")
print()
print("  NOTE: when the result happens to be an exact integer (e.g. √2·√2 = 2),")
print("  the algorithm loops — convergents oscillate above/below without all 8")
print("  corners ever agreeing on a floor.  This is a known CF arithmetic limit,")
print("  not specific to trihomographic (cf_mul(√2, √2) has the same issue).")
print()


# φ² = φ + 1 ≈ 2.618 (irrational — converges fine)
_show("φ²  = φ·φ + 0  ≈ 2.618 → [2;1,1,1,…]", cf_fma(Phi(), Phi(), CF.from_fraction(0, 1)))
assert list(cf_fma(Phi(), Phi(), CF.from_fraction(0, 1)).take(5)) == [2, 1, 1, 1, 1]

# √2 · √5 + 0 = √10 ≈ 3.162 (two distinct surds, irrational)
_show(
    "√2·√5 + 0 = √10 ≈ 3.162 → [3;6,6,6,…]",
    cf_fma(Sqrt(2), Sqrt(5), CF.from_fraction(0, 1)),
)
assert list(cf_fma(Sqrt(2), Sqrt(5), CF.from_fraction(0, 1)).take(4)) == [3, 6, 6, 6]

# φ²+φ = 2φ+1 ≈ 4.236 (three copies of φ: w·x + y with w=x=y=φ)
_show("φ·φ + φ = 2φ+1 ≈ 4.236", cf_fma(Phi(), Phi(), Phi()))
assert list(cf_fma(Phi(), Phi(), Phi()).take(4)) == [4, 4, 4, 4]

print()

# ── Example 3: Euler step on rationals ──────────────────────────────────────
print("── Euler's series-to-CF step ──")
print()
print("  z  =  a_prev / (a_prev − a_curr + a_curr·tail)")
print("  The denominator has cross-term a_curr·tail: genuinely trilinear.")
print()

# a_prev=3/2, a_curr=4/3, tail=2
# z = (3/2) / (3/2 − 4/3 + (4/3)·2) = (3/2) / (17/6) = 9/17
a_prev = CF.from_fraction(3, 2)
a_curr = CF.from_fraction(4, 3)
tail = CF([2])
step = cf_euler_step(a_prev, a_curr, tail)
_show("euler_step(3/2, 4/3, 2) = 9/17", step)
assert list(step.take(5)) == [0, 1, 1, 8], list(step.take(5))  # 9/17 = [0;1,1,8]

print()
print("=" * 70)
print("All assertions passed.")
