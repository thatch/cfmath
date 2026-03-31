"""Generalized n-input Gosper arithmetic on continued fractions.

The Gosper tensor
-----------------
A continued fraction is a sequence of integer terms a₀, a₁, a₂, ...  The
Gosper algorithm computes arithmetic on CFs directly in term-space, without
ever converting to floats or rationals.

For n CF inputs x₀, x₁, ..., x_{n-1} and one CF output z, the state is a
pair of multilinear polynomials (numerator and denominator):

    z = N(x₀, x₁, ..., x_{n-1}) / D(x₀, x₁, ..., x_{n-1})

A multilinear polynomial in n variables has 2ⁿ coefficients — one per
subset of {x₀, ..., x_{n-1}}.  We index coefficients by an integer i whose
binary representation is the subset: bit j is 1 iff xⱼ appears.

    n=1  →  4 entries total (2 num + 2 den):
        y = (ax + b) / (cx + d)
        num = [b, a]   (index 0=const, 1=x)
        den = [d, c]

    n=2  →  8 entries total (4 num + 4 den):
        z = (axy + bx + cy + d) / (exy + fx + gy + h)
        num = [d, b, c, a]   (index 00=const, 01=x, 10=y, 11=xy)
        den = [h, f, g, e]

    n=3  →  16 entries total (8 num + 8 den):
        Monomials: {1, x₀, x₁, x₀x₁, x₂, x₀x₂, x₁x₂, x₀x₁x₂}
        Indices:   {000, 001, 010, 011, 100, 101, 110, 111}
        num[i] = coefficient for the monomial with variable set i in the numerator
        den[i] = coefficient for the monomial with variable set i in the denominator

The total state for n inputs is 2^(n+1) integers.

Corner evaluation
-----------------
To check whether the next output term can be emitted, evaluate z at every
corner of the tail range, where each active input xⱼ' ∈ [1, ∞).

Corner c is an n-bit mask where bit j = 1 means xⱼ → ∞:
    num_c = Σ num[i]  for all i where (i & c) == c
    den_c = Σ den[i]  for all i where (i & c) == c

This sum picks out the "dominant" monomials as xⱼ → ∞ for all j in c.

For n=2 this reproduces the classic four corners:
    c=0b00 (1,1):   sum of all num          / sum of all den
    c=0b01 (∞,1):   num[01]+num[11]         / den[01]+den[11]   = (a+b)/(e+f)
    c=0b10 (1,∞):   num[10]+num[11]         / den[10]+den[11]   = (a+c)/(e+g)
    c=0b11 (∞,∞):   num[11]                 / den[11]           = a/e

If all relevant corners agree on the same integer floor, that is the next
output CF term.

When some inputs are exhausted (done), only corners where every done input
is at ∞ need to be checked — the exhausted tail is ∞, not a free variable.

Ingest and emit
---------------
Ingest term t from input j (substitute xⱼ = t + 1/xⱼ', then multiply
numerator and denominator through by xⱼ'):

    For each pair (i, i|(1<<j)) where bit j is 0 in i:
        new[i | (1<<j)] = old[i | (1<<j)] * t + old[i]
        new[i]          = old[i | (1<<j)]

Emit output term t (subtract t from z, take reciprocal: z = t + 1/z'):
    new_num    = old_den
    new_den[i] = old_num[i] - t * old_den[i]   for all i

Both updates touch exactly 2ⁿ entries in O(2ⁿ) time with no branching on n.

Input selection
---------------
Before the first term from input j is consumed, xⱼ may be less than 1 (the
full CF, not a tail), so corner evaluation is not yet valid.  We therefore
consume one term from every unstarted input before checking output.

Once all inputs are started, at each step we pick the input whose range of
corner values (as that input varies 1 → ∞) is widest.  Reducing the widest
range first converges fastest.

References
----------
Bill Gosper, HAKMEM, MIT AI Memo 239, 1972, items 101–101C.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .core import CF


_MAX_STALL = 1000


# ---------------------------------------------------------------------------
# Core tensor operations — O(2ⁿ), no branching on input count
# ---------------------------------------------------------------------------

def _corner_sum(coeffs: list[int], c: int) -> int:
    """Sum of coeffs[i] for all i where (i & c) == c."""
    return sum(v for i, v in enumerate(coeffs) if (i & c) == c)


def _check_output(num: list[int], den: list[int], done: int) -> int | None:
    """Return the next output CF term if all relevant corners agree, else None.

    Iterates over all 2ⁿ corners.  Corners where a done input is at 1 are
    skipped (done inputs have exhausted tails, i.e. their value is at ∞).
    All remaining corners must have non-zero same-sign denominators and share
    the same integer floor.
    """
    n = len(num).bit_length() - 1   # 2^n == len(num)
    floor_val: int | None = None
    sign: bool | None = None
    for c in range(1 << n):
        if (c & done) != done:
            continue
        cd = _corner_sum(den, c)
        if cd == 0:
            return None
        s = cd > 0
        if sign is None:
            sign = s
        elif s != sign:
            return None  # pole in range
        f = _corner_sum(num, c) // cd
        if floor_val is None:
            floor_val = f
        elif f != floor_val:
            return None
    return floor_val


def _ingest(num: list[int], den: list[int], n: int, j: int, t: int) -> tuple[list[int], list[int]]:
    """Ingest term t from input j: substitute xⱼ = t + 1/xⱼ' and multiply through by xⱼ'."""
    mask = 1 << j
    new_num = list(num)
    new_den = list(den)
    for i in range(1 << n):
        if not (i & mask):  # bit j is 0 — process pair (i, i|mask)
            new_num[i | mask] = num[i | mask] * t + num[i]
            new_num[i]        = num[i | mask]
            new_den[i | mask] = den[i | mask] * t + den[i]
            new_den[i]        = den[i | mask]
    return new_num, new_den


def _emit(num: list[int], den: list[int], t: int) -> tuple[list[int], list[int]]:
    """Emit output term t: subtract t from z, take reciprocal."""
    new_num = list(den)
    new_den = [num[i] - t * den[i] for i in range(len(num))]
    return new_num, new_den


# ---------------------------------------------------------------------------
# Input selection — pick which input to consume next
# ---------------------------------------------------------------------------

def _corner_val_float(num: list[int], den: list[int], c: int) -> float:
    cd = _corner_sum(den, c)
    if cd == 0:
        cn = _corner_sum(num, c)
        return float("inf") if cn >= 0 else float("-inf")
    return _corner_sum(num, c) / cd


def _input_spread(num: list[int], den: list[int], n: int, j: int, done: int) -> float:
    """Max variation of output corners as input j varies from 1 to ∞."""
    mask = 1 << j
    max_spread = 0.0
    for c in range(1 << n):
        if not (c & mask):
            continue  # need the j=∞ corner
        if (c & done) != done:
            continue  # done inputs must be at ∞
        v_inf = _corner_val_float(num, den, c)           # j at ∞
        v_one = _corner_val_float(num, den, c & ~mask)   # j at 1
        if abs(v_inf) == float("inf") or abs(v_one) == float("inf"):
            return float("inf")
        max_spread = max(max_spread, abs(v_inf - v_one))
    return max_spread


def _pick_input(num: list[int], den: list[int], n: int, started: int, done: int) -> int:
    """Return index of the next input to consume.

    Priority: unstarted inputs first (to enter the [1,∞) domain);
    among started non-done inputs, pick the one with the widest spread.
    """
    all_mask = (1 << n) - 1
    unstarted = all_mask & ~started & ~done
    if unstarted:
        for j in range(n):
            if unstarted & (1 << j):
                return j

    # All inputs started; find max-spread non-done input.
    best_j = -1
    best_spread = -1.0
    for j in range(n):
        if done & (1 << j):
            continue
        if best_j == -1:
            best_j = j  # ensure we always return a valid index
        s = _input_spread(num, den, n, j, done)
        if s > best_spread:
            best_spread = s
            best_j = j
    return best_j


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def _n_ary_terms(
    iters: list[Iterator[int]],
    num: list[int],
    den: list[int],
) -> Iterator[int]:
    """Yield CF terms of the n-ary Gosper formula given input iterators."""
    n = len(iters)
    all_mask = (1 << n) - 1
    started = 0
    done = 0
    stall = 0

    while True:
        # Try to emit an output term (only valid once all inputs are started).
        if started == all_mask:
            t = _check_output(num, den, done)
            if t is not None:
                yield t
                num, den = _emit(num, den, t)
                stall = 0
                continue

        # All inputs consumed: emit the remaining rational exactly.
        if done == all_mask:
            cn = _corner_sum(num, all_mask)
            cd = _corner_sum(den, all_mask)
            if cd != 0:
                from .core import CF as _CF
                yield from _CF.from_fraction(cn, cd)
            return

        # Pick next input and consume one term.
        j = _pick_input(num, den, n, started, done)
        if j < 0:
            return

        try:
            t_in = next(iters[j])
            started |= (1 << j)
            num, den = _ingest(num, den, n, j, t_in)
            stall += 1
        except StopIteration:
            started |= (1 << j)
            done |= (1 << j)
            stall = 0
            continue

        if stall >= _MAX_STALL:
            # Integer-boundary stall: emit max corner floor and terminate.
            # Mirrors the same fix in gosper.py — see comments there.
            n_bits = len(num).bit_length() - 1
            valid = []
            for c in range(1 << n_bits):
                if (c & done) != done:
                    continue
                cd = _corner_sum(den, c)
                if cd != 0:
                    valid.append((_corner_sum(num, c), cd))
            if valid and len({cd > 0 for _, cd in valid}) == 1:
                yield max(cn // cd for cn, cd in valid)
            return


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cf_n_ary(inputs: list[CF], num: list[int], den: list[int]) -> CF:
    """Compute a multilinear rational formula of n CF inputs.

    ``num`` and ``den`` are flat coefficient arrays of length 2ⁿ.  Index i
    is the monomial whose variable set is the binary representation of i:
    bit j = 1 iff input xⱼ appears.

    Example for n=2 (x+y):
        cf_n_ary([x, y], num=[0,1,1,0], den=[1,0,0,0])
    """
    from .core import CF as _CF
    iters = [x._iter_from(0) for x in inputs]
    return _CF([], _source=_n_ary_terms(iters, list(num), list(den)))


def cf_homographic(x: CF, a: int, b: int, c: int, d: int) -> CF:
    """Return the CF for (ax+b)/(cx+d)."""
    return cf_n_ary([x], num=[b, a], den=[d, c])


def cf_add(x: CF, y: CF) -> CF:
    """Return x + y."""
    return cf_n_ary([x, y], num=[0, 1, 1, 0], den=[1, 0, 0, 0])


def cf_sub(x: CF, y: CF) -> CF:
    """Return x - y."""
    if x is y:
        from .core import CF as _CF
        return _CF.from_int(0)
    return cf_n_ary([x, y], num=[0, 1, -1, 0], den=[1, 0, 0, 0])


def cf_mul(x: CF, y: CF) -> CF:
    """Return x * y."""
    if x is y:
        from .quadratic import _periodic_square
        result = _periodic_square(x)
        if result is not None:
            return result
    elif x.is_periodic() and y.is_periodic():
        from .quadratic import _periodic_mul
        result = _periodic_mul(x, y)
        if result is not None:
            return result
    return cf_n_ary([x, y], num=[0, 0, 0, 1], den=[1, 0, 0, 0])


def cf_div(x: CF, y: CF) -> CF:
    """Return x / y."""
    if x is y:
        from .core import CF as _CF
        return _CF.from_int(1)
    return cf_n_ary([x, y], num=[0, 1, 0, 0], den=[0, 0, 1, 0])


def cf_min(x: CF, y: CF) -> CF:
    """Return the smaller of x and y."""
    return x if x < y else y


def cf_max(x: CF, y: CF) -> CF:
    """Return the larger of x and y."""
    return x if x > y else y
