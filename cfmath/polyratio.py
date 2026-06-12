"""PolyTransform: rational function P(x)/Q(x) of a single CF source.

Stores numerator and denominator as ascending-degree integer coefficient lists.
Every arithmetic operation multiplies polynomials through, then reduces by GCD,
so cancellation like Pi - Pi or x²/x → x falls out automatically.

The degree after normalization determines the generator:
  degree 0  →  constant rational, yields CF.from_fraction(num[0], den[0])
  degree 1  →  _homographic_terms  (same as GosperMono)
  degree n  →  _n_ary_terms with n copies of the source
"""

from __future__ import annotations

from math import gcd

from .core import CF
from .gosper import _homographic_terms
from .gosper_generalized import _n_ary_terms

# ---------------------------------------------------------------------------
# Integer polynomial arithmetic over ascending-degree lists
# ---------------------------------------------------------------------------


def _strip(p: list[int]) -> list[int]:
    while len(p) > 1 and p[-1] == 0:
        p = p[:-1]
    return p


def _content(p: list[int]) -> int:
    g = 0
    for c in p:
        g = gcd(g, abs(c))
    return g or 1


def _primitive(p: list[int]) -> list[int]:
    g = _content(p)
    return [c // g for c in p]


def _add(p: list[int], q: list[int]) -> list[int]:
    n = max(len(p), len(q))
    return _strip([(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)])


def _sub(p: list[int], q: list[int]) -> list[int]:
    n = max(len(p), len(q))
    return _strip([(p[i] if i < len(p) else 0) - (q[i] if i < len(q) else 0) for i in range(n)])


def _mul(p: list[int], q: list[int]) -> list[int]:
    result = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            result[i + j] += a * b
    return _strip(result)


def _pseudorem(p: list[int], q: list[int]) -> list[int]:
    """Pseudo-remainder of p divided by q in ℤ[x]."""
    p = list(p)
    lq = q[-1]
    dq = len(q) - 1
    while len(p) - 1 >= dq and p != [0]:
        k = len(p) - 1 - dq
        lp = p[-1]
        p = [c * lq for c in p]
        for i, c in enumerate(q):
            p[i + k] -= lp * c
        p = _strip(p)
    return p


def _poly_gcd(p: list[int], q: list[int]) -> list[int]:
    """GCD of two polynomials in ℤ[x], returned as a primitive polynomial."""
    p, q = _strip(list(p)), _strip(list(q))
    while q != [0]:
        r = _primitive(_pseudorem(p, q))
        p, q = q, r
    return _primitive(p)


def _divexact(p: list[int], g: list[int]) -> list[int]:
    """Exact division p / g in ℤ[x], assuming g divides p."""
    p = list(p)
    n = len(p) - len(g)
    result = []
    for i in range(n, -1, -1):
        qc = p[i + len(g) - 1] // g[-1]
        result.append(qc)
        for j in range(len(g)):
            p[i + j] -= qc * g[j]
    return _strip(list(reversed(result)))


def _normalize(num: list[int], den: list[int]) -> tuple[list[int], list[int]]:
    """Reduce num/den by integer content GCD then polynomial GCD."""
    num = _strip(list(num))
    den = _strip(list(den))
    if den == [0]:
        raise ZeroDivisionError("PolyTransform denominator is zero")
    g_int = gcd(_content(num), _content(den))
    if g_int > 1:
        num = [c // g_int for c in num]
        den = [c // g_int for c in den]
    g_poly = _poly_gcd(num, den)
    if g_poly != [1]:
        num = _divexact(num, g_poly)
        den = _divexact(den, g_poly)
    if den[-1] < 0:
        num = [-c for c in num]
        den = [-c for c in den]
    return num, den


def _reduce_by_minpoly(p: list[int], A: int, B: int, C: int) -> tuple[list[int], int]:
    """Reduce polynomial p modulo As²+Bs+C=0, returning (reduced, denom).

    In ℤ[s]/(As²+Bs+C), s² = -(Bs+C)/A.  Each degree-n term (n≥2) is replaced
    by multiplying everything by A and substituting: A·s^n = s^(n-2)·(A·s²)
    = s^(n-2)·(-Bs-C).  One pass per term drives degree down to ≤1.
    """
    p = list(p)
    denom = 1
    while len(p) > 2:
        an = p[-1]
        p = [c * A for c in p[:-1]]
        denom *= A
        p[-1] -= an * B
        p[-2] -= an * C
    return _strip(p), denom


def _poly_to_tensor(p: list[int], n: int) -> list[int]:
    """Encode polynomial p into the 2^n coefficient tensor for _n_ary_terms.

    Index i represents the monomial whose variable set is {j : bit j of i is 1}.
    With n identical sources, index i evaluates to source^popcount(i).  We
    concentrate each degree-k coefficient at index 2^k - 1 (k trailing ones),
    leaving all other same-popcount indices zero.
    """
    tensor = [0] * (1 << n)
    for k, c in enumerate(p):
        tensor[0 if k == 0 else (1 << k) - 1] = c
    return tensor


# ---------------------------------------------------------------------------
# PolyTransform
# ---------------------------------------------------------------------------


class PolyTransform(CF):
    """Rational function P(x)/Q(x) of a single CF source x.

    Arithmetic on two PolyTransforms over the same source stays in-class: the
    polynomials are multiplied through and the result is normalised by GCD,
    so cancellations like (f - f), (f * g / g), or any common factor reduce
    automatically.  Zero numerator produces the zero CF.

    The generator is chosen by degree after normalisation:
      0  constant  →  CF.from_fraction
      1  linear    →  _homographic_terms  (Möbius)
      n  degree n  →  _n_ary_terms with n copies of source
    """

    def __init__(self, source: CF, num: list[int], den: list[int]) -> None:
        num, den = _normalize(num, den)
        deg = max(len(num) - 1, len(den) - 1)
        if deg >= 2 and source.is_periodic():
            from .quadratic import _minimal_poly

            poly = _minimal_poly(source)
            if poly is not None:
                A, B, C = poly
                num_r, dn = _reduce_by_minpoly(num, A, B, C)
                den_r, dd = _reduce_by_minpoly(den, A, B, C)
                num = [c * dd for c in num_r]
                den = [c * dn for c in den_r]
                num, den = _normalize(num, den)
        self._src_cf = source
        self._num = num
        self._den = den
        deg = max(len(num) - 1, len(den) - 1)
        if deg == 0:
            gen = iter(CF.from_fraction(num[0], den[0]))
        elif deg == 1:
            a1 = num[1] if len(num) > 1 else 0
            b1 = den[1] if len(den) > 1 else 0
            gen = _homographic_terms(source._iter_from(0), a1, num[0], b1, den[0])
        else:
            tn = _poly_to_tensor(num, deg)
            td = _poly_to_tensor(den, deg)
            gen = _n_ary_terms([source._iter_from(0)] * deg, tn, td)
        super().__init__([], _source=gen)

    def _new(self, num: list[int], den: list[int]) -> PolyTransform:
        return PolyTransform(self._src_cf, num, den)

    # ------------------------------------------------------------------
    # Arithmetic — multiply through, normalise in __init__
    # ------------------------------------------------------------------

    def __add__(self, other: object) -> CF:
        if isinstance(other, PolyTransform) and other._src_cf is self._src_cf:
            return self._new(
                _add(_mul(self._num, other._den), _mul(other._num, self._den)),
                _mul(self._den, other._den),
            )
        if isinstance(other, int):
            return self._new(_add(self._num, _mul([other], self._den)), self._den)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__add__(self, coerced)

    def __radd__(self, other: object) -> CF:
        if isinstance(other, int):
            return self.__add__(other)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__add__(coerced, self)

    def __sub__(self, other: object) -> CF:
        if isinstance(other, PolyTransform) and other._src_cf is self._src_cf:
            return self._new(
                _sub(_mul(self._num, other._den), _mul(other._num, self._den)),
                _mul(self._den, other._den),
            )
        if isinstance(other, int):
            return self._new(_sub(self._num, _mul([other], self._den)), self._den)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__sub__(self, coerced)

    def __rsub__(self, other: object) -> CF:
        if isinstance(other, int):
            return self._new(_sub(_mul([other], self._den), self._num), self._den)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__sub__(coerced, self)

    def __mul__(self, other: object) -> CF:
        if isinstance(other, PolyTransform) and other._src_cf is self._src_cf:
            return self._new(_mul(self._num, other._num), _mul(self._den, other._den))
        if isinstance(other, int):
            return self._new([c * other for c in self._num], self._den)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__mul__(self, coerced)

    def __rmul__(self, other: object) -> CF:
        if isinstance(other, int):
            return self._new([c * other for c in self._num], self._den)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__mul__(coerced, self)

    def __truediv__(self, other: object) -> CF:
        if isinstance(other, PolyTransform) and other._src_cf is self._src_cf:
            return self._new(_mul(self._num, other._den), _mul(self._den, other._num))
        if isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("PolyTransform / 0")
            return self._new(self._num, [c * other for c in self._den])
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__truediv__(self, coerced)

    def __rtruediv__(self, other: object) -> CF:
        if isinstance(other, int):
            return self._new(_mul([other], self._den), self._num)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__truediv__(coerced, self)

    def __neg__(self) -> PolyTransform:
        return self._new([-c for c in self._num], self._den)

    def __repr__(self) -> str:
        return f"PolyTransform(source, num={self._num}, den={self._den})"
