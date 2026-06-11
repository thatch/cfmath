"""Shared infrastructure for cfmath implementations.

Provides:
  _HAS_MPMATH       — True if mpmath is importable
  _lazy_cf(fn)      — wrap a batch-compute function into a lazy CF
  _extract_cf_terms — extract CF terms from an mpmath value until precision runs out
  _mpmath_cf(fn)    — precision-automatic lazy CF from an mpmath value function
  _coerce_trig_arg  — validate and coerce int/Fraction inputs
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable, Iterator

from .core import CF

try:
    import mpmath as _mpmath  # noqa: F401 — side-effect: validates import

    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False


def _lazy_cf(
    compute: Callable[[int], list[int]],
    initial: int = 60,
    batch: int = 50,
    debug_source: Any | None = None,
) -> CF:
    """Build a lazily-extending CF from a batch-compute function.

    ``compute(n)`` must return a list of at least *n* CF term integers.
    The first call uses *initial* terms; when those run out, it recomputes
    with *batch* more terms each time.
    """
    terms = compute(initial)
    static = terms[:10]
    rest = iter(terms[10:])

    def _more() -> Iterator[int]:
        yield from rest
        offset = len(terms)
        while True:
            more = compute(offset + batch)
            yield from more[offset:]
            offset = len(more)

    cf = CF(static, _source=_more())
    if debug_source is not None:
        cf._debug_source = debug_source
    return cf


def _annotate_cf(cf: CF, debug_source: Any | None) -> CF:
    """Attach debugging provenance to a CF and return it."""
    if debug_source is not None:
        cf._debug_source = debug_source
    return cf


def _extract_cf_terms(x: Any, guard_bits: int = 64) -> list[int]:
    """Extract CF terms from an mpmath value until precision is exhausted.

    After k extraction steps the effective precision is approximately
    ``mp.prec - 2*log2(q_k)`` bits, where q_k is the k-th convergent denominator.
    We stop before a step would push ``2*log2(q)`` past ``mp.prec - guard_bits``.

    We track ``log2(q)`` cheaply via the identity
    ``log2(q_{k+1}) ≈ log2(q_k) + log2(1/frac_k)``
    (since ``a_{k+1} = floor(1/frac_k) ≈ 1/frac_k``).

    This naturally accounts for large terms: π's term-4 value of 292 costs ~8 bits
    without any per-constant tuning, and also catches exhaustion by many
    moderate-sized terms (which a simple ``frac < threshold`` check misses).
    """
    import mpmath

    available = float(mpmath.mp.prec - guard_bits)
    log_q = 0.0  # tracks log2 of the convergent denominator
    terms: list[int] = []
    while True:
        a = int(mpmath.floor(x))
        frac = x - a
        terms.append(a)
        if frac == 0:
            break
        # Track log2 of the convergent denominator with mpmath, not float(frac).
        # A huge partial quotient makes frac underflow a double to 0.0, which
        # would stop extraction even when mpmath still has the precision to go on
        # (and escalating dps wouldn't help — float underflows at any dps).
        # mpmath.log of a tiny frac is a moderate number; the precision check
        # below — comparing the cost against available bits — is the real and
        # only stopping rule.
        log_q -= float(mpmath.log(frac, 2))  # log2(1/frac) = -log2(frac)
        if 2.0 * log_q > available:
            break
        x = 1 / frac
    return terms


def _mpmath_cf(
    value_fn: Callable[[], Any],
    initial_dps: int = 100,
    scale: float = 2.0,
    guard_bits: int = 64,
    debug_source: Any | None = None,
) -> CF:
    """Build a lazy CF from an mpmath value function with automatic precision management.

    ``value_fn()`` is called with ``mp.dps`` already set; it should return the
    mpmath value to convert (e.g. ``lambda: mpmath.euler``).

    Always maintains two consecutive precision levels (lo and hi = lo × scale).
    Only terms where lo and hi agree are emitted — the first disagreement is the
    empirical precision boundary, so no per-constant tuning is needed and marginal
    terms are automatically filtered.  When the consumer asks for more terms, hi
    becomes the new lo and a fresh hi is computed at the next level.
    """
    import mpmath

    def _get(dps: int) -> list[int]:
        mpmath.mp.dps = dps
        return _extract_cf_terms(value_fn(), guard_bits)

    def _agree_up_to(a: list[int], b: list[int]) -> int:
        """First index where a and b differ, or min(len(a), len(b)) if all agree."""
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        return min(len(a), len(b))

    # Bootstrap: compute at two consecutive levels, emit only agreed terms.
    lo = _get(initial_dps)
    hi = _get(int(initial_dps * scale))
    first_valid = _agree_up_to(lo, hi)

    static = hi[: min(10, first_valid)] if first_valid else hi[:1]

    def _source() -> Iterator[int]:
        cur_lo = hi
        cur_dps = int(initial_dps * scale)
        emitted = len(static)

        yield from cur_lo[emitted:first_valid]
        emitted = first_valid

        while True:
            cur_dps = int(cur_dps * scale)
            cur_hi = _get(cur_dps)
            new_valid = _agree_up_to(cur_lo, cur_hi)

            if new_valid < emitted:
                raise ArithmeticError(
                    f"CF term {new_valid} changed from {cur_lo[new_valid]} to "
                    f"{cur_hi[new_valid]} at {cur_dps} dps — previously emitted "
                    f"term is wrong"
                )

            new_terms = cur_hi[emitted:new_valid]
            if new_terms:
                yield from new_terms
                emitted = new_valid

            cur_lo = cur_hi

    cf = CF(static, _source=_source())
    if debug_source is not None:
        cf._debug_source = debug_source
    return cf

def _cf_terms_from_interval_approximator(
    interval_at_precision: Callable[[int], tuple[Fraction, Fraction]],
    n_terms: int,
    initial: int = 16,
    max_precision: int = 1 << 16,
) -> list[int]:
    """Return CF terms once rational intervals prove them.

    The caller supplies ``interval_at_precision(p)``, which must return
    rational bounds ``lo <= x <= hi``.  This helper extracts simple-CF terms
    only while both endpoints force the same next integer.  If the interval
    straddles an integer boundary or zero after subtracting a term, the helper
    doubles precision and starts over.
    """
    precision = initial
    while precision <= max_precision:
        lo, hi = interval_at_precision(precision)
        if lo > hi:
            lo, hi = hi, lo

        terms: list[int] = []
        while len(terms) < n_terms:
            if lo == hi:
                terms.extend(CF.from_rational(lo).terms)
                return terms[:n_terms]

            a_lo = lo.numerator // lo.denominator
            a_hi = hi.numerator // hi.denominator
            if a_lo != a_hi:
                break

            a = a_lo
            terms.append(a)
            lo -= a
            hi -= a

            if lo == hi == 0:
                return terms
            if lo <= 0 <= hi:
                break

            lo, hi = Fraction(1, hi), Fraction(1, lo)
            if lo > hi:
                lo, hi = hi, lo

        if len(terms) >= n_terms:
            return terms[:n_terms]
        precision *= 2

    raise ValueError(f"could not pin {n_terms} CF terms by precision {max_precision}")


def _coerce_trig_arg(x: int | Fraction) -> Fraction:
    """Validate and coerce a trig/hyperbolic function argument to Fraction."""
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, Fraction):
        return x
    raise TypeError(f"expected int or Fraction, got {type(x).__name__}")
