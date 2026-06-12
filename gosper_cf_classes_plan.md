Plan: GosperMono / GosperBi / GosperGeneric

The three classes subclass CF so they can be used directly anywhere a CF is
accepted.  The extra invariant: each instance tracks its source CFs and its
coefficient matrix, so integer scalar arithmetic pre-multiplies the matrix
rather than wrapping another generator layer.

---

New file: cfmath/gosper_cf.py

---

GosperMono(CF)
    Storage: _source_cf: CF,  _mono_mat: (a, b, c, d)
    Generates: _homographic_terms(source._iter_from(0), a, b, c, d)

    Scalar ops (all return GosperMono, all pre-multiply the 2×2 matrix):
      G * k       -> (ka, kb, c, d)          [[k,0],[0,1]]·M
      G + k       -> (a+kc, b+kd, c, d)      [[1,k],[0,1]]·M
      G - k       -> (a-kc, b-kd, c, d)      [[1,-k],[0,1]]·M
      k - G       -> (kc-a, kd-b, c, d)      [[-1,k],[0,1]]·M
      -G          -> (-a, -b, c, d)           [[-1,0],[0,1]]·M
      G / k       -> (a, b, kc, kd)          [[1,0],[0,k]]·M   (scale denom)
      k / G       -> (kc, kd, a, b)          [[0,k],[1,0]]·M   (k×reciprocal)
    All return super() for non-int, non-Fraction operands.

    Composition via @:
      GosperMono @ GosperMono:
        Result: GosperMono(other._source_cf, M_self · M_other)
        Semantics: self(other(x)).  Always composes regardless of whether
        _source_cf objects match; the right operand's source is used.

      GosperMono @ Gosper (symbolic, no source):
        Result: GosperMono(self._source_cf, M_self · M_gosper)
        Semantics: apply Gosper as inner transform to self's source.

      Gosper @ GosperMono (via GosperMono.__rmatmul__):
        Result: GosperMono(self._source_cf, M_gosper · M_self)
        Semantics: apply self first, then the outer Gosper.

      GosperMono @ CF (plain CF, not a GosperMono):
        Result: GosperMono(other, self._mono_mat)
        Semantics: wrap CF `other` as the new source, re-apply self's matrix.


GosperBi(CF)
    Storage: _source_x: CF, _source_y: CF, _bi_mat: (a,b,c,d,e,f,g,h)
    Formula:  (axy + bx + cy + d) / (exy + fx + gy + h)
    Generates: _bihomographic_terms(x._iter_from(0), y._iter_from(0), a..h)

    Scalar ops update (a..d) while (e..h) is the denominator:
      G * k   -> (ka,kb,kc,kd, e,f,g,h)
      G + k   -> (a+ke,b+kf,c+kg,d+kh, e,f,g,h)
      G - k   -> (a-ke,b-kf,c-kg,d-kh, e,f,g,h)
      k - G   -> (ke-a,kf-b,kg-c,kh-d, e,f,g,h)
      -G      -> (-a,-b,-c,-d, e,f,g,h)
      G / k   -> (a,b,c,d, ke,kf,kg,kh)
      k / G   -> (ke,kf,kg,kh, a,b,c,d)
    All return super() for non-int operands.

    __rmatmul__(Gosper sym):
      Semantics: sym @ self — apply sym to the output of this bihomographic.
      new_num[i] = sym.a * num[i] + sym.b * den[i]   for each i in 0..3
      new_den[i] = sym.c * num[i] + sym.d * den[i]
      (Uses the generalized-tensor representation internally for the formula,
      but maps back to the (a,b,c,d,e,f,g,h) storage.)

    No __matmul__: GosperBi @ X is ambiguous (which of the two inputs does X
    substitute?).


GosperGeneric(CF)
    Storage: _source_cfs: list[CF], _gen_num: list[int], _gen_den: list[int]
    Both _gen_num and _gen_den have length 2^n.
    Generates: _n_ary_terms([s._iter_from(0) for s in sources], num, den)

    The coefficient indexing is the generalized-tensor one from
    gosper_generalized.py: index i has bit j set iff source j appears.

    Scalar ops (all O(2^n)):
      G * k   -> new_num[i] = k * num[i],              den unchanged
      G + k   -> new_num[i] = num[i] + k * den[i],     den unchanged
      G - k   -> new_num[i] = num[i] - k * den[i],     den unchanged
      k - G   -> new_num[i] = k * den[i] - num[i],     den unchanged
      -G      -> new_num[i] = -num[i],                  den unchanged
      G / k   -> num unchanged, new_den[i] = k * den[i]
      k / G   -> new_num = [k*d for d in den], new_den = list(num)

    __rmatmul__(Gosper sym):
      new_num[i] = sym.a * gen_num[i] + sym.b * gen_den[i]
      new_den[i] = sym.c * gen_num[i] + sym.d * gen_den[i]

    No __matmul__: same ambiguity as GosperBi.


Change to existing gosper.py:
    Gosper.__matmul__ currently uses other.a/.b/.c/.d directly and will
    AttributeError if other is a GosperMono.  Add isinstance(other, Gosper)
    check and return NotImplemented otherwise, so __rmatmul__ on the CF
    subclass can handle Gosper @ GosperMono/Bi/Generic.


Changes to cfmath/__init__.py:
    Export GosperMono, GosperBi, GosperGeneric from gosper_cf.


Tests: tests/test_gosper_cf.py
    TestGosperMono:
      - construction and _iter_from usage (is a CF, yields correct terms)
      - each scalar op stays in-class and produces correct value
      - non-int scalar falls to CF (returns plain CF, not GosperMono)
      - GosperMono @ GosperMono matches nested cf_homographic calls
      - GosperMono @ Gosper  and  Gosper @ GosperMono both work
      - GosperMono @ plain_cf wraps the plain CF

    TestGosperBi:
      - construction; yields same value as cf_add / cf_mul etc.
      - scalar ops stay in-class and produce correct values
      - Gosper @ GosperBi via __rmatmul__

    TestGosperGeneric:
      - construction from num/den arrays; yields correct value
      - scalar ops produce correct values
      - Gosper @ GosperGeneric via __rmatmul__


Usability gaps (to mention but not fix now):

1. Fraction scalars fall through to CF bihomographic.  Fix: scale the integer
   matrix by the denominator inside the operator.

2. GosperMono + GosperMono (same source) falls to cf_add bihomographic.
   Addition of two Möbius transforms over the same variable is degree-2/2;
   there is no GosperMono result in general.  The only safe special case
   (g + g == 2*g) could be detected by identity, but this is rarely useful.

3. GosperBi @ GosperMono / GosperGeneric @ GosperMono — substituting one
   input of an n-ary transform with a mono transform — is well-defined but
   not implemented.  It is ambiguous without an explicit input index.

4. __pow__ falls to CF's repeated-squaring path; the result is a plain CF
   (n≥2 produces a degree-n/n rational, not Möbius).

5. No GCD normalization of matrices.  Chains of scalar ops can cause
   coefficient growth.  The symbolic Gosper class normalises; the CF
   subclasses do not (re-deriving the generator on each op is cheap, but the
   integers in _mono_mat grow).  Add a .normalised() helper if needed.

6. GosperMono @ GosperMono always uses the right operand's _source_cf,
   ignoring any _source_cf stored in self.  If self and other have different
   intended domains, composing silently discards self's source.  Users who
   need domain-checked composition should verify sources match explicitly.
