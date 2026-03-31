"""Debugging and instrumentation utilities for continued fractions."""

from __future__ import annotations

from typing import Iterator, TypeVar

T = TypeVar("T")


class CountingIterator:
    """Wraps any iterator and counts how many items have been consumed.

    >>> counter = CountingIterator(iter([10, 20, 30]))
    >>> next(counter)
    10
    >>> counter.count
    1
    """

    def __init__(self, it: Iterator) -> None:
        self._it = it
        self.count: int = 0

    def __iter__(self) -> CountingIterator:
        return self

    def __next__(self):
        val = next(self._it)
        self.count += 1
        return val


def digits_with_debug(cf, base: int = 10) -> Iterator[tuple[int, int]]:
    """Yield (digit, terms_consumed) pairs from a CF.

    Each pair is one base-B digit plus the count of CF terms consumed from
    the input since the previous digit was emitted.  Useful for understanding
    how 'hard' each digit is to pin down.

    Example::

        for digit, cost in digits_with_debug(Pi()):
            print(digit, cost)
        # 3 1       ← integer part, consumed 1 CF term
        # 1 1       ← first fractional digit, 1 more term
        # 4 2       ← ...
        # 1 1
        # 5 5       ← the '292' term in Pi's CF makes this cheap
        # ...
    """
    from .core import CF as _CF

    counter = CountingIterator(iter(cf))
    wrapper = _CF([], _source=counter)
    last = 0
    for digit in wrapper.digits(base):
        consumed = counter.count - last
        last = counter.count
        yield digit, consumed
