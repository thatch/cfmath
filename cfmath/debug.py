"""Debugging and instrumentation utilities for continued fractions."""

from __future__ import annotations

from typing import Any, Iterator

from .core import CF


class CountingIterator:
    """Wraps any iterator and counts how many items have been consumed.

    >>> counter = CountingIterator(iter([10, 20, 30]))
    >>> next(counter)
    10
    >>> counter.count
    1
    """

    def __init__(self, it: Iterator[Any]) -> None:
        self._it = it
        self.count: int = 0

    def __iter__(self) -> "CountingIterator":
        return self

    def __next__(self) -> Any:
        val = next(self._it)
        self.count += 1
        return val


def digits_with_debug(cf: CF, base: int = 10) -> Iterator[tuple[int, int]]:
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


def describe_source_tree(obj: Any, indent: int = 0) -> str:
    """Return a recursively indented tree of a transform and its sources.

    The helper understands the transform classes in this package by their
    private source fields, plus private `_debug_source` tags on CF values.
    Plain CF leaves are shown with their repr().
    """

    pad = "  " * indent

    if getattr(obj, "_debug_source", None) is not None:
        return _describe_debug_source(obj._debug_source, indent)

    if hasattr(obj, "_mono_mat") and hasattr(obj, "_source_cf"):
        a, b, c, d = obj._mono_mat
        lines = [f"{pad}{obj.__class__.__name__}({a}, {b}, {c}, {d})"]
        lines.append(describe_source_tree(obj._source_cf, indent + 1))
        return "\n".join(lines)

    if hasattr(obj, "_bi_mat") and hasattr(obj, "_source_x") and hasattr(obj, "_source_y"):
        lines = [f"{pad}{obj.__class__.__name__}({', '.join(map(str, obj._bi_mat))})"]
        lines.append(f"{pad}  x:")
        lines.append(describe_source_tree(obj._source_x, indent + 2))
        lines.append(f"{pad}  y:")
        lines.append(describe_source_tree(obj._source_y, indent + 2))
        return "\n".join(lines)

    if hasattr(obj, "_gen_num") and hasattr(obj, "_gen_den") and hasattr(obj, "_source_cfs"):
        lines = [f"{pad}{obj.__class__.__name__}(num={obj._gen_num}, den={obj._gen_den})"]
        for i, source in enumerate(obj._source_cfs):
            lines.append(f"{pad}  source[{i}]:")
            lines.append(describe_source_tree(source, indent + 2))
        return "\n".join(lines)

    if hasattr(obj, "_num") and hasattr(obj, "_den") and hasattr(obj, "_src_cf"):
        lines = [f"{pad}{obj.__class__.__name__}(num={obj._num}, den={obj._den})"]
        lines.append(f"{pad}  source:")
        lines.append(describe_source_tree(obj._src_cf, indent + 2))
        return "\n".join(lines)

    if hasattr(obj, "a") and hasattr(obj, "b") and hasattr(obj, "c") and hasattr(obj, "d"):
        return f"{pad}{obj.__class__.__name__}({obj.a}, {obj.b}, {obj.c}, {obj.d})"

    return f"{pad}{obj!r}"


def _is_tree_child(value: Any) -> bool:
    return (
        isinstance(value, CF)
        or getattr(value, "_debug_source", None) is not None
        or hasattr(value, "_mono_mat")
        or hasattr(value, "_bi_mat")
        or hasattr(value, "_gen_num")
        or hasattr(value, "_num")
        or hasattr(value, "a")
    )


def _describe_debug_source(source: Any, indent: int) -> str:
    pad = "  " * indent

    if isinstance(source, tuple) and source and isinstance(source[0], str):
        label = source[0]
        parts = list(source[1:])
        inline: list[str] = []
        children: list[tuple[int, Any]] = []
        for i, part in enumerate(parts):
            if _is_tree_child(part):
                children.append((i, part))
            else:
                inline.append(str(part))

        head = f"{pad}{label}"
        if inline:
            head += f"({', '.join(inline)})"
        if not children:
            return head

        lines = [head]
        for i, child in children:
            lines.append(f"{pad}  arg[{i}]:")
            lines.append(describe_source_tree(child, indent + 2))
        return "\n".join(lines)

    return f"{pad}{source!r}"
