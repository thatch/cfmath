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

    return CF(static, _source=_more())


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
    import math

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
        f = float(frac)
        if f == 0.0:
            break  # underflows double — precision definitely gone
        log_q -= math.log2(f)  # log2(1/frac) = -log2(frac)
        if 2.0 * log_q > available:
            break
        x = 1 / frac
    return terms


def _mpmath_cf(
    value_fn: Callable[[], Any],
    initial_dps: int = 100,
    scale: float = 2.0,
    guard_bits: int = 64,
) -> CF:
    """Build a lazy CF from an mpmath value function with automatic precision management.

    ``value_fn()`` is called with ``mp.dps`` already set; it should return the
    mpmath value to convert (e.g. ``lambda: mpmath.euler``).

    Precision starts at *initial_dps* and doubles each time the consumer asks
    for more terms than the current precision supports.  On each doubling, the
    previously emitted terms are re-extracted and validated against the prior
    batch — catching any precision-related inconsistency before emitting new terms.
    """
    import mpmath

    mpmath.mp.dps = initial_dps
    first_batch = _extract_cf_terms(value_fn(), guard_bits)
    static = first_batch[:10]

    def _source() -> Iterator[int]:
        confirmed = list(first_batch)
        yield from confirmed[10:]

        dps = int(initial_dps * scale)
        while True:
            mpmath.mp.dps = dps
            batch = _extract_cf_terms(value_fn(), guard_bits)

            if batch[: len(confirmed)] != confirmed:
                bad = next(
                    i
                    for i, (a, b) in enumerate(zip(batch, confirmed))
                    if a != b
                )
                raise ArithmeticError(
                    f"CF term {bad} changed from {confirmed[bad]} to {batch[bad]} "
                    f"on precision increase to {dps} dps"
                )

            new = batch[len(confirmed) :]
            if not new:
                dps = int(dps * scale)
                continue

            yield from new
            confirmed = batch
            dps = int(dps * scale)

    return CF(static, _source=_source())


def _coerce_trig_arg(x) -> Fraction:
    """Validate and coerce a trig/hyperbolic function argument to Fraction."""
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, Fraction):
        return x
    raise TypeError(f"expected int or Fraction, got {type(x).__name__}")
