# cfmath

A continued-fraction arithmetic library for Python.  Every function returns a
lazy `CF` object whose terms can be consumed one at a time, or in batches, to
any desired precision.

## Public API by module

The tables below list the public functions and the input types they accept.
✓ = supported, — = not applicable.

The `*GCF`, `*CF`, and `*MP` variant rows are implementation backends, not
separate exports.  A dispatcher such as `Sin` (or `Arctan`) picks one
automatically; to force a particular backend, pass the module's mode enum, e.g.
`Sin(x, TrigMode.MP)` or `Arctan(x, ArctrigMode.CF)`.

> **Updating this table:** regenerate it by reading `cfmath/__init__.py` for the
> exported names, then checking each function's signature (especially the first
> positional parameter) in the defining module listed here.  The column split
> (int / Fraction / CF) maps to `isinstance` checks in the function body; look
> for `_coerce_trig_arg` (int+Fraction only), `_coerce_meta_trig_arg`
> (int+Fraction+CF), `isinstance(x, CF)` guards, or the raw type annotation.
>
> The `*GCF`/`*CF`/`*MP` variants are *not* exported from `cfmath` — they are
> module-private (e.g. `cfmath.trig._SinCF`, `cfmath.arctrig._ArctanMP`) and
> selected through the function's mode enum, so they will not appear in
> `cfmath/__init__.py`.  Keep their rows; they document the backends the enum
> chooses between.

---

### `cfmath.core` — base type

| Symbol | int | Fraction | CF | Notes |
|--------|:---:|:--------:|:--:|-------|
| `CF`   | —   | —        | —  | The lazy continued-fraction type itself; construct via `CF.from_int`, `CF.from_rational`, `CF.from_fraction`, `CF.from_float` |

---

### `cfmath.constants` — mathematical constants (no input)

| Function      | int | Fraction | CF | Notes |
|---------------|:---:|:--------:|:--:|-------|
| `Pi`          | —   | —        | —  | π |
| `Tau`         | —   | —        | —  | 2π |
| `E`           | —   | —        | —  | Euler's number |
| `Phi`         | —   | —        | —  | Golden ratio |
| `EulerGamma`  | —   | —        | —  | Euler–Mascheroni constant |
| `Catalan`     | —   | —        | —  | Catalan's constant |
| `Apery`       | —   | —        | —  | Apéry's constant ζ(3) |
| `Khinchin`    | —   | —        | —  | Khinchin's constant |
| `Plastic`     | —   | —        | —  | Plastic constant |

---

### `cfmath.quadratic` — quadratic irrationals

| Function | int | Fraction | CF | Notes |
|----------|:---:|:--------:|:--:|-------|
| `Sqrt`   | ✓   | —        | —  | √n for integer n |

---

### `cfmath.power` — general exponentiation

`PowArg = int | Fraction | CF`

| Function         | int | Fraction | CF | Notes |
|------------------|:---:|:--------:|:--:|-------|
| `Pow`            | ✓   | ✓        | ✓  | `Pow(x, r)` — both args accept any PowArg |
| `PowCF`          | ✓   | ✓        | ✓  | meta-CF backend for `Pow` |
| `PowMP`          | ✓   | ✓        | ✓  | mpmath backend for `Pow` |
| `PowIntExponent` | ✓   | ✓        | ✓  | integer-exponent fast path |
| `PowInterval`    | ✓   | ✓        | ✓  | interval-arithmetic backend |
| `Nthroot`        | ✓   | ✓        | —  | `Nthroot(x, k)` — k-th root of rational x |
| `Cuberoot`       | ✓   | —        | —  | ∛n for integer n |

---

### `cfmath.exponential` — exponential function

| Function | int | Fraction | CF | Notes |
|----------|:---:|:--------:|:--:|-------|
| `Exp`    | ✓   | ✓        | ✓  | eˣ — alias for `ExpMP` |

---

### `cfmath.logarithm` — logarithms

| Function | int | Fraction | CF | Notes |
|----------|:---:|:--------:|:--:|-------|
| `Ln`     | ✓   | ✓        | ✓  | natural log |
| `Log`    | ✓   | ✓        | ✓  | `Log(x, base=None)` — base defaults to e; base is `int \| Fraction \| None` |
| `Log2`   | ✓   | ✓        | ✓  | log base 2 |
| `Log10`  | ✓   | ✓        | ✓  | log base 10 |

---

### `cfmath.trig` — trigonometric functions

Angles are in radians.  `*GCF` variants accept only rational inputs and use
exact generalized-CF arithmetic.  `*CF` variants accept any CF and use the
meta-GCF algorithm.  `*MP` variants use mpmath for verification.

| Function | int | Fraction | CF | Notes |
|----------|:---:|:--------:|:--:|-------|
| `Sin`    | ✓   | ✓        | ✓  | dispatches to `SinGCF` (rational) or `SinCF` (CF) |
| `SinGCF` | ✓   | ✓        | —  | exact Lambert GCF for rational x |
| `SinCF`  | ✓   | ✓        | ✓  | sin/x meta-GCF with z = x², reduces modulo 2π |
| `SinMP`  | ✓   | ✓        | ✓  | mpmath; for CF input uses dual-precision convergent approach |
| `Cos`    | ✓   | ✓        | ✓  | dispatches to `CosGCF` (rational) or `CosMP` (CF) |
| `CosGCF` | ✓   | ✓        | —  | exact Lambert GCF for rational x |
| `CosCF`  | ✓   | ✓        | ✓  | 1/cos meta-GCF (z = x²); ended up slower than expected, so AUTO sends CF to `CosMP` — request it explicitly with `TrigMode.CF` |
| `CosMP`  | ✓   | ✓        | ✓  | mpmath; for CF input uses the convergent approach |
| `Tan`    | ✓   | ✓        | ✓  | dispatches to `TanGCF` (rational) or `TanCF` (CF) |
| `TanGCF` | ✓   | ✓        | —  | exact Lambert GCF for rational x |
| `TanCF`  | ✓   | ✓        | ✓  | meta-GCF with argument reduction modulo π |
| `TanMP`  | ✓   | ✓        | ✓  | mpmath; for CF input uses the convergent approach |

---

### `cfmath.arctrig` — inverse trigonometric functions

| Function   | int | Fraction | CF | Notes |
|------------|:---:|:--------:|:--:|-------|
| `Arctan`   | ✓   | ✓        | ✓  | dispatches to `ArctanGCF` (rational) or `ArctanCF` (CF) |
| `ArctanGCF`| ✓   | ✓        | —  | Gauss GCF for rational x |
| `ArctanCF` | ✓   | ✓        | ✓  | meta-GCF path |
| `ArctanMP` | ✓   | ✓        | ✓  | mpmath; for CF input uses the convergent approach |
| `Arcsin`   | ✓   | ✓        | ✓  | rational uses Euler GCF; CF uses mpmath |
| `Arccos`   | ✓   | ✓        | ✓  | rational uses π/2 − arcsin; CF uses mpmath |

---

### `cfmath.hyperbolic` — hyperbolic functions

| Function | int | Fraction | CF | Notes |
|----------|:---:|:--------:|:--:|-------|
| `Sinh`   | ✓   | ✓        | ✓  | rational via mpmath/decimal; CF via `ExpCF` |
| `Cosh`   | ✓   | ✓        | ✓  | rational via mpmath/decimal; CF via `ExpCF` |
| `Tanh`   | ✓   | ✓        | ✓  | rational uses Lambert GCF; CF uses mpmath |

---

### `cfmath.archyperbolic` — inverse hyperbolic functions

| Function  | int | Fraction | CF | Notes |
|-----------|:---:|:--------:|:--:|-------|
| `Arcsinh` | ✓   | ✓        | ✓  | rational via mpmath/decimal; CF via mpmath |
| `Arccosh` | ✓   | ✓        | ✓  | rational via mpmath/decimal; CF via mpmath |
| `Arctanh` | ✓   | ✓        | ✓  | rational via `Ln`; CF via mpmath |

---

### `cfmath.special` — special functions

| Function | int | Fraction | CF | Notes |
|----------|:---:|:--------:|:--:|-------|
| `Gamma`  | ✓   | ✓        | —  | Γ(x) |
| `Zeta`   | ✓   | —        | —  | ζ(s) for integer s ≥ 2 |

---

### `cfmath.gosper` — Gosper arithmetic on CFs

All functions take `CF` inputs and return `CF`.

| Function        | Notes |
|-----------------|-------|
| `cf_add`        | x + y |
| `cf_sub`        | x − y |
| `cf_mul`        | x · y |
| `cf_div`        | x / y |
| `cf_homographic`| (ax + b) / (cx + d) |
| `cf_min`        | min(x, y) |
| `cf_max`        | max(x, y) |

---

### `cfmath.convergents` — rational approximations

All functions take a `CF` and work with its convergents p_n/q_n.

| Function          | Notes |
|-------------------|-------|
| `convergent`      | n-th convergent as `Fraction` |
| `convergent_pair` | n-th convergent as `(p, q)` |
| `convergent_pairs`| lazy iterator of `(p, q)` pairs |
| `convergents`     | lazy iterator of `Fraction` values |

---

## Version compatibility

Python 3.10+.  This library follows [meanver](https://meanver.org/).

## License

cfmath is copyright [Tim Hatch](https://timhatch.com/), and licensed under
the MIT license.  See the `LICENSE` file for details.
