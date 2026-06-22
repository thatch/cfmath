"""CF subclasses that carry their Gosper matrix and source CFs.

GosperMono, GosperBi, and GosperGeneric are CF subclasses — they work
anywhere a CF is expected.  The extra invariant: each instance stores the
coefficient matrix and its source CF(s), so integer and Fraction scalar
arithmetic pre-multiplies the matrix instead of wrapping another generator
layer.

Scalar arithmetic returns the same class:

    g = GosperMono(x, a, b, c, d)
    (3 + g * 2)(x)   →  two matrix left-multiplies, single generator

Same-source GosperMono binary ops return GosperBi:

    g1 + g2  (same underlying CF)  →  GosperBi(x, x, ...)

Composition with @ (GosperMono only):

    (g2 @ g1)(x)     →  matrix product M_g2·M_g1, one generator
    Requires same underlying source or g2._source_cf is g1.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from .core import CF
from .gosper import _bihomographic_terms, _homographic_terms
from .gosper_generalized import _n_ary_terms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_scalar(k: object) -> tuple[int, int] | None:
    """Return (numerator, denominator) for int or Fraction k, or None."""
    if isinstance(k, int):
        return k, 1
    if isinstance(k, Fraction):
        return k.numerator, k.denominator
    return None


def _apply_sym(
    num: list[int],
    den: list[int],
    sa: int,
    sb: int,
    sc: int,
    sd: int,
) -> tuple[list[int], list[int]]:
    """Apply symbolic Gosper (sa,sb,sc,sd) to every entry of a coefficient tensor.

    Left-multiplies the output by [[sa,sb],[sc,sd]]:
        new_num[i] = sa * num[i] + sb * den[i]
        new_den[i] = sc * num[i] + sd * den[i]
    """
    new_num = [sa * n + sb * d for n, d in zip(num, den)]
    new_den = [sc * n + sd * d for n, d in zip(num, den)]
    return new_num, new_den


# ---------------------------------------------------------------------------
# GosperMono: (ax+b)/(cx+d) with one CF source
# ---------------------------------------------------------------------------


class GosperMono(CF):
    """Homographic transform (ax+b)/(cx+d) applied to a single CF source.

    Is-a CF: iterate it, pass it to any function that expects a CF, use it as
    a source for another GosperMono.  The extra feature: int/Fraction scalar
    arithmetic pre-multiplies the 2×2 coefficient matrix and returns a new
    GosperMono, so a chain of scalar transforms stays a single generator layer.

    When both operands are GosperMono with the same source, binary ops (+, -,
    *, /) return a GosperBi over that shared source — no extra generator layer.

    Compose two GosperMono transforms with @:

        g2 @ g1  →  GosperMono(g1._source_cf, M_g2 · M_g1)

    Requires g1 and g2 to share a root: either they have the same _source_cf,
    or g2._source_cf is g1 (g2 was built on top of g1).
    """

    def __init__(self, source: CF, a: int, b: int, c: int, d: int) -> None:
        _gcd = gcd(abs(a), abs(b), abs(c), abs(d))
        if _gcd > 1:
            a, b, c, d = a // _gcd, b // _gcd, c // _gcd, d // _gcd
        self._source_cf: CF = source
        self._mono_mat: tuple[int, int, int, int] = (a, b, c, d)
        super().__init__([], _source=_homographic_terms(source._iter_from(0), a, b, c, d))

    # ------------------------------------------------------------------
    # Scalar output arithmetic — pre-multiply the 2×2 matrix
    # For k = p/q:
    #   G*k  → (pa, pb, qc, qd)
    #   G+k  → (qa+pc, qb+pd, qc, qd)
    #   G-k  → (qa-pc, qb-pd, qc, qd)
    #   k-G  → (pc-qa, pd-qb, qc, qd)
    #   G/k  → (qa, qb, pc, pd)
    #   k/G  → (pc, pd, qa, qb)
    # ------------------------------------------------------------------

    def __mul__(self, other: object) -> CF:
        if isinstance(other, GosperMono) and other._source_cf is self._source_cf:
            a1, b1, c1, d1 = self._mono_mat
            a2, b2, c2, d2 = other._mono_mat
            return GosperBi(
                self._source_cf,
                self._source_cf,
                a1 * a2,
                a1 * b2,
                b1 * a2,
                b1 * b2,
                c1 * c2,
                c1 * d2,
                d1 * c2,
                d1 * d2,
            )
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d = self._mono_mat
            return GosperMono(self._source_cf, p * a, p * b, q * c, q * d)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__mul__(self, coerced)

    def __rmul__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d = self._mono_mat
            return GosperMono(self._source_cf, p * a, p * b, q * c, q * d)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__mul__(coerced, self)

    def __add__(self, other: object) -> CF:
        if isinstance(other, GosperMono) and other._source_cf is self._source_cf:
            a1, b1, c1, d1 = self._mono_mat
            a2, b2, c2, d2 = other._mono_mat
            return GosperBi(
                self._source_cf,
                self._source_cf,
                a1 * c2 + a2 * c1,
                a1 * d2 + a2 * d1,
                b1 * c2 + b2 * c1,
                b1 * d2 + b2 * d1,
                c1 * c2,
                c1 * d2,
                d1 * c2,
                d1 * d2,
            )
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d = self._mono_mat
            return GosperMono(self._source_cf, q * a + p * c, q * b + p * d, q * c, q * d)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__add__(self, coerced)

    def __radd__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d = self._mono_mat
            return GosperMono(self._source_cf, q * a + p * c, q * b + p * d, q * c, q * d)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__add__(coerced, self)

    def __sub__(self, other: object) -> CF:
        if isinstance(other, GosperMono) and other._source_cf is self._source_cf:
            a1, b1, c1, d1 = self._mono_mat
            a2, b2, c2, d2 = other._mono_mat
            return GosperBi(
                self._source_cf,
                self._source_cf,
                a1 * c2 - a2 * c1,
                a1 * d2 - a2 * d1,
                b1 * c2 - b2 * c1,
                b1 * d2 - b2 * d1,
                c1 * c2,
                c1 * d2,
                d1 * c2,
                d1 * d2,
            )
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d = self._mono_mat
            return GosperMono(self._source_cf, q * a - p * c, q * b - p * d, q * c, q * d)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__sub__(self, coerced)

    def __rsub__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d = self._mono_mat
            return GosperMono(self._source_cf, p * c - q * a, p * d - q * b, q * c, q * d)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__sub__(coerced, self)

    def __neg__(self) -> CF:
        a, b, c, d = self._mono_mat
        return GosperMono(self._source_cf, -a, -b, c, d)

    def __truediv__(self, other: object) -> CF:
        if isinstance(other, GosperMono) and other._source_cf is self._source_cf:
            a1, b1, c1, d1 = self._mono_mat
            a2, b2, c2, d2 = other._mono_mat
            return GosperBi(
                self._source_cf,
                self._source_cf,
                a1 * c2,
                a1 * d2,
                b1 * c2,
                b1 * d2,
                c1 * a2,
                c1 * b2,
                d1 * a2,
                d1 * b2,
            )
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            if p == 0:
                raise ZeroDivisionError("GosperMono / 0")
            a, b, c, d = self._mono_mat
            return GosperMono(self._source_cf, q * a, q * b, p * c, p * d)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__truediv__(self, coerced)

    def __rtruediv__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d = self._mono_mat
            return GosperMono(self._source_cf, p * c, p * d, q * a, q * b)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__truediv__(coerced, self)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def __matmul__(self, other: object) -> CF:
        """Compose: self(other(x)).

        GosperMono @ GosperMono  →  matrix product, right operand's source.
            Requires same underlying source or self._source_cf is other
            (self was built on top of other).
        GosperMono @ CF          →  wrap CF as new source, keep matrix.
        """
        if isinstance(other, GosperMono):
            if not (self._source_cf is other._source_cf or self._source_cf is other):
                raise ValueError("cannot compose GosperMono with mismatched sources; ")
            a2, b2, c2, d2 = self._mono_mat
            a1, b1, c1, d1 = other._mono_mat
            return GosperMono(
                other._source_cf,
                a2 * a1 + b2 * c1,
                a2 * b1 + b2 * d1,
                c2 * a1 + d2 * c1,
                c2 * b1 + d2 * d1,
            )
        if isinstance(other, CF):
            return GosperMono(other, *self._mono_mat)
        return NotImplemented

    def __rmatmul__(self, other: object) -> CF:
        """Gosper @ self  →  sym.mat · self.mat, keep self's source."""
        return NotImplemented

    def __repr__(self) -> str:
        a, b, c, d = self._mono_mat
        return f"GosperMono(source, {a}, {b}, {c}, {d})"


# ---------------------------------------------------------------------------
# GosperBi: (axy+bx+cy+d)/(exy+fx+gy+h) with two CF sources
# ---------------------------------------------------------------------------


class GosperBi(CF):
    """Bihomographic transform over two CF sources x and y.

    Is-a CF.  Int/Fraction scalar arithmetic pre-updates the 8 coefficients
    and returns a new GosperBi.  Gosper @ GosperBi applies a symbolic Möbius
    transform to the output via __rmatmul__.
    """

    def __init__(
        self,
        x: CF,
        y: CF,
        a: int,
        b: int,
        c: int,
        d: int,
        e: int,
        f: int,
        g: int,
        h: int,
    ) -> None:
        _gcd = gcd(abs(a), abs(b), abs(c), abs(d), abs(e), abs(f), abs(g), abs(h))
        if _gcd > 1:
            a, b, c, d = a // _gcd, b // _gcd, c // _gcd, d // _gcd
            e, f, g, h = e // _gcd, f // _gcd, g // _gcd, h // _gcd
        self._source_x: CF = x
        self._source_y: CF = y
        self._bi_mat: tuple[int, int, int, int, int, int, int, int] = (a, b, c, d, e, f, g, h)
        super().__init__(
            [],
            _source=_bihomographic_terms(x._iter_from(0), y._iter_from(0), a, b, c, d, e, f, g, h),
        )

    def _new(self, *coeffs: int) -> GosperBi:
        return GosperBi(self._source_x, self._source_y, *coeffs)

    # ------------------------------------------------------------------
    # Scalar output arithmetic
    # For k = p/q:
    #   G*k  → (p*num, q*den)
    #   G+k  → (q*num + p*den, q*den)
    #   G/k  → (q*num, p*den)
    #   k/G  → (p*den, q*num)
    # ------------------------------------------------------------------

    def __mul__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d, e, f, g, h = self._bi_mat
            return self._new(p * a, p * b, p * c, p * d, q * e, q * f, q * g, q * h)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__mul__(self, coerced)

    def __rmul__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            return self.__mul__(other)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__mul__(coerced, self)

    def __add__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d, e, f, g, h = self._bi_mat
            return self._new(
                q * a + p * e,
                q * b + p * f,
                q * c + p * g,
                q * d + p * h,
                q * e,
                q * f,
                q * g,
                q * h,
            )
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__add__(self, coerced)

    def __radd__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            return self.__add__(other)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__add__(coerced, self)

    def __sub__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d, e, f, g, h = self._bi_mat
            return self._new(
                q * a - p * e,
                q * b - p * f,
                q * c - p * g,
                q * d - p * h,
                q * e,
                q * f,
                q * g,
                q * h,
            )
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__sub__(self, coerced)

    def __rsub__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d, e, f, g, h = self._bi_mat
            return self._new(
                p * e - q * a,
                p * f - q * b,
                p * g - q * c,
                p * h - q * d,
                q * e,
                q * f,
                q * g,
                q * h,
            )
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__sub__(coerced, self)

    def __neg__(self) -> CF:
        a, b, c, d, e, f, g, h = self._bi_mat
        return self._new(-a, -b, -c, -d, e, f, g, h)

    def __truediv__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            if p == 0:
                raise ZeroDivisionError("GosperBi / 0")
            a, b, c, d, e, f, g, h = self._bi_mat
            return self._new(q * a, q * b, q * c, q * d, p * e, p * f, p * g, p * h)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__truediv__(self, coerced)

    def __rtruediv__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            a, b, c, d, e, f, g, h = self._bi_mat
            return self._new(p * e, p * f, p * g, p * h, q * a, q * b, q * c, q * d)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__truediv__(coerced, self)

    # ------------------------------------------------------------------
    # Outer composition: Gosper @ GosperBi
    # ------------------------------------------------------------------

    def __rmatmul__(self, other: object) -> CF:
        """sym @ self  →  apply sym to this bihomographic's output."""
        return NotImplemented

    def __repr__(self) -> str:
        return f"GosperBi(source_x, source_y, {', '.join(map(str, self._bi_mat))})"


# ---------------------------------------------------------------------------
# GosperGeneric: n-ary Gosper with 2^n coefficient arrays
# ---------------------------------------------------------------------------


class GosperGeneric(CF):
    """n-ary Gosper formula over n CF sources.

    Coefficient arrays num and den each have length 2^n.  Index i encodes
    the monomial whose variable set is {j : bit j of i is 1}.  See
    gosper_generalized.py for the full description.

    Is-a CF.  Int/Fraction scalar arithmetic updates all 2^n numerator (or
    denominator) entries and returns a new GosperGeneric.
    Gosper @ GosperGeneric is supported via __rmatmul__.
    """

    def __init__(self, sources: list[CF], num: list[int], den: list[int]) -> None:
        n = len(sources)
        if len(num) != 1 << n or len(den) != 1 << n:
            raise ValueError(f"For {n} sources, num and den must each have length {1 << n}")
        _gcd = gcd(*[abs(v) for v in num], *[abs(v) for v in den])
        if _gcd > 1:
            num = [v // _gcd for v in num]
            den = [v // _gcd for v in den]
        self._source_cfs: list[CF] = list(sources)
        self._gen_num: list[int] = list(num)
        self._gen_den: list[int] = list(den)
        super().__init__(
            [],
            _source=_n_ary_terms(
                [s._iter_from(0) for s in sources],
                list(num),
                list(den),
            ),
        )

    def _new(self, num: list[int], den: list[int]) -> GosperGeneric:
        return GosperGeneric(self._source_cfs, num, den)

    # ------------------------------------------------------------------
    # Scalar output arithmetic (all O(2^n))
    # ------------------------------------------------------------------

    def __mul__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            return self._new(
                [p * v for v in self._gen_num],
                [q * v for v in self._gen_den],
            )
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__mul__(self, coerced)

    def __rmul__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            return self.__mul__(other)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__mul__(coerced, self)

    def __add__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            num = [q * n + p * d for n, d in zip(self._gen_num, self._gen_den)]
            den = [q * d for d in self._gen_den]
            return self._new(num, den)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__add__(self, coerced)

    def __radd__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            return self.__add__(other)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__add__(coerced, self)

    def __sub__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            num = [q * n - p * d for n, d in zip(self._gen_num, self._gen_den)]
            den = [q * d for d in self._gen_den]
            return self._new(num, den)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__sub__(self, coerced)

    def __rsub__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            num = [p * d - q * n for n, d in zip(self._gen_num, self._gen_den)]
            den = [q * d for d in self._gen_den]
            return self._new(num, den)
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__sub__(coerced, self)

    def __neg__(self) -> CF:
        return self._new([-v for v in self._gen_num], list(self._gen_den))

    def __truediv__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            if p == 0:
                raise ZeroDivisionError("GosperGeneric / 0")
            return self._new(
                [q * v for v in self._gen_num],
                [p * v for v in self._gen_den],
            )
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__truediv__(self, coerced)

    def __rtruediv__(self, other: object) -> CF:
        pq = _coerce_scalar(other)
        if pq is not None:
            p, q = pq
            return self._new(
                [p * v for v in self._gen_den],
                [q * v for v in self._gen_num],
            )
        coerced = CF._coerce(other)
        if coerced is None:
            return NotImplemented
        return CF.__truediv__(coerced, self)

    # ------------------------------------------------------------------
    # Outer composition: Gosper @ GosperGeneric
    # ------------------------------------------------------------------

    def __rmatmul__(self, other: object) -> CF:
        """sym @ self  →  apply sym to this n-ary formula's output."""
        return NotImplemented

    def __repr__(self) -> str:
        n = len(self._source_cfs)
        return f"GosperGeneric({n} sources, num={self._gen_num}, den={self._gen_den})"
