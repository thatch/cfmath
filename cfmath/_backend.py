"""Shared infrastructure for cfmath implementations.

Provides:
  _HAS_MPMATH       — True if mpmath is importable
  _lazy_cf(fn)      — wrap a batch-compute function into a lazy CF
  _coerce_trig_arg  — validate and coerce int/Fraction inputs
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Iterator

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


def _coerce_trig_arg(x) -> Fraction:
    """Validate and coerce a trig/hyperbolic function argument to Fraction."""
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, Fraction):
        return x
    raise TypeError(f"expected int or Fraction, got {type(x).__name__}")
