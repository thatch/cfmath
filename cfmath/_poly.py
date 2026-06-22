"""Integer polynomial arithmetic over ascending-degree coefficient lists.

A polynomial is a list of int coefficients, lowest degree first: ``[c0, c1, c2]``
means ``c0 + c1*x + c2*x**2``.  The zero polynomial is ``[0]``.

Two callers share these primitives, and they pull in opposite directions.  The
``PolyTransform`` algebra in ``polyratio`` reduces by GCD, so it needs a *canonical*
form: no trailing zeros, leading coefficient nonzero, otherwise degree counts and
pseudo-remainders go wrong.  The meta-CF rebuild path in ``gosper`` only ever
evaluates the polynomials it builds, so trailing zeros are invisible to it but
the products are many and low-degree, where a scalar/monomial fast path pays off.

``add``/``sub``/``mul`` therefore strip their results (satisfying the first caller)
and ``mul`` keeps the fast paths (satisfying the second).  Stripping is
evaluation-invariant, so the rebuild path is unaffected by it.
"""

from __future__ import annotations

from math import gcd


def strip(p: list[int]) -> list[int]:
    """Drop trailing zero coefficients, keeping at least the constant term."""
    while len(p) > 1 and p[-1] == 0:
        p = p[:-1]
    return p


def content(p: list[int]) -> int:
    """Return the GCD of the coefficients (1 for the zero polynomial)."""
    g = 0
    for c in p:
        g = gcd(g, abs(c))
    return g or 1


def primitive(p: list[int]) -> list[int]:
    """Divide out the content, leaving a primitive polynomial."""
    g = content(p)
    return [c // g for c in p]


def add(p: list[int], q: list[int]) -> list[int]:
    """Add two polynomials."""
    n = max(len(p), len(q))
    return strip([(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)])


def sub(p: list[int], q: list[int]) -> list[int]:
    """Subtract q from p."""
    n = max(len(p), len(q))
    return strip([(p[i] if i < len(p) else 0) - (q[i] if i < len(q) else 0) for i in range(n)])


def mul(p: list[int], q: list[int]) -> list[int]:
    """Multiply two polynomials.

    Scalar and single-term (monomial) operands are common in the meta-CF rebuild
    path and get direct fast paths; the general case skips zero coefficients.
    """
    if len(p) == 1:
        k = p[0]
        if k == 0:
            return [0]
        if k == 1:
            return strip(list(q))
        return strip([k * c for c in q])
    if len(q) == 1:
        k = q[0]
        if k == 0:
            return [0]
        if k == 1:
            return strip(list(p))
        return strip([k * c for c in p])

    nonzero_p = [i for i, c in enumerate(p) if c != 0]
    if len(nonzero_p) == 1:
        shift = nonzero_p[0]
        k = p[shift]
        return strip([0] * shift + [k * c for c in q])

    nonzero_q = [i for i, c in enumerate(q) if c != 0]
    if len(nonzero_q) == 1:
        shift = nonzero_q[0]
        k = q[shift]
        return strip([0] * shift + [k * c for c in p])

    out = [0] * (len(p) + len(q) - 1)
    for i in nonzero_p:
        ci = p[i]
        for j in nonzero_q:
            out[i + j] += ci * q[j]
    return strip(out)
