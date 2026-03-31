# Implementation Methods Survey

Each CF-producing function, categorized by how it computes terms.

## Legend

- **exact** — constructs CF directly from known periodic/algebraic form (pure int arithmetic)
- **gen-CF** — uses a known generalized continued fraction (exact rational arithmetic on `Fraction`)
- **decimal** — Taylor/other series computed with Python's `decimal` module (high precision, no floats)
- **mpmath** — evaluated via mpmath to high precision, then CF terms extracted
- **Gosper** — derived by composing other CFs via Gosper arithmetic

A `✓` under decimal/mpmath means a fallback/alternative exists; `—` means only mpmath, no decimal fallback.

---

## By Module

### quadratic.py

| Function | exact | gen-CF | decimal | mpmath | Gosper |
|----------|:-----:|:------:|:-------:|:------:|:------:|
| `Sqrt(n)` | ✓ | | | | |

### constants.py

| Function | exact | gen-CF | decimal | mpmath | Gosper |
|----------|:-----:|:------:|:-------:|:------:|:------:|
| `Phi()` | ✓ | | | | |
| `E()` | ✓ | | | | |
| `Pi()` | | ✓ | | | |
| `Tau()` | | | | | ✓ (= 2·Pi) |
| `EulerGamma()` | | | — | ✓ | |
| `Catalan()` | | | ✓ | ✓ | |
| `Apery()` | | | ✓ | ✓ | |
| `Plastic()` | | | ✓ | ✓ | |
| `Khinchin()` | | | — | ✓ | |

### trig.py

| Function | exact | gen-CF | decimal | mpmath | Gosper |
|----------|:-----:|:------:|:-------:|:------:|:------:|
| `Sin(x)` | | ✓ | | (test only) | |
| `Cos(x)` | | ✓ | | (test only) | |
| `Tan(x)` | | ✓ | | (test only) | |

### arctrig.py

| Function | exact | gen-CF | decimal | mpmath | Gosper |
|----------|:-----:|:------:|:-------:|:------:|:------:|
| `Arctan(x)` | | ✓ | | (test only) | |
| `Arcsin(x)` | | ✓ | | (test only) | |
| `Arccos(x)` | | | | | ✓ (= π/2 − Arcsin) |

### hyperbolic.py

| Function | exact | gen-CF | decimal | mpmath | Gosper |
|----------|:-----:|:------:|:-------:|:------:|:------:|
| `Sinh(x)` | | | — | ✓ | |
| `Cosh(x)` | | | — | ✓ | |
| `Tanh(x)` | | ✓ | | (test only) | |

### archyperbolic.py

| Function | exact | gen-CF | decimal | mpmath | Gosper |
|----------|:-----:|:------:|:-------:|:------:|:------:|
| `Arctanh(x)` | | | | | ✓ (= Ln((1+x)/(1−x))/2) |
| `Arcsinh(x)` | | | ✓ | ✓ | |
| `Arccosh(x)` | | | ✓ | ✓ | |

### exponential.py

| Function | exact | gen-CF | decimal | mpmath | Gosper |
|----------|:-----:|:------:|:-------:|:------:|:------:|
| `Exp(x)` | | | — | ✓ | |

### logarithm.py

| Function | exact | gen-CF | decimal | mpmath | Gosper |
|----------|:-----:|:------:|:-------:|:------:|:------:|
| `Ln(x)` | | | ✓ | ✓ | |
| `Log(x, b)` | | | | | ✓ (= Ln(x)/Ln(b)) |
| `Log2(x)` | | | | | ✓ (= Ln(x)/Ln(2)) |
| `Log10(x)` | | | | | ✓ (= Ln(x)/Ln(10)) |

### power.py

| Function | exact | gen-CF | decimal | mpmath | Gosper |
|----------|:-----:|:------:|:-------:|:------:|:------:|
| `Cuberoot(n)` | | | ✓ | ✓ | |
| `Pow(x, n)` n∈ℤ | ✓ | | | | |
| `Pow(x, p/2)` | ✓ | | | | |
| `Pow(x, 1/3)` x∈ℤ | | | ✓ | ✓ | |
| `Pow(x, r)` general | | | | | ✓ (= Exp(r·Ln(x))) → needs mpmath |

### special.py

| Function | exact | gen-CF | decimal | mpmath | Gosper |
|----------|:-----:|:------:|:-------:|:------:|:------:|
| `Gamma(n)` n∈ℤ | ✓ | | | | |
| `Gamma(x)` x∈ℚ | | | — | ✓ | |
| `Zeta(2)` | | | | | ✓ (= π²/6) |
| `Zeta(2n)` | | | | | ✓ (rational·πⁿ) |
| `Zeta(3)` | | | ✓ | ✓ | |
| `Zeta(2n+1)` n≥2 | | | ✓ | ✓ | |

---

## Missing decimal fallbacks (mpmath required)

These currently have **no decimal fallback**. Rough difficulty to add one:

| Function | Approach | Difficulty |
|----------|----------|------------|
| `Exp(x)` | Taylor series: `Σ xⁿ/n!` with `decimal` | **easy** |
| `Sinh(x)` | Taylor series: `Σ x²ⁿ⁺¹/(2n+1)!` with `decimal` | **easy** |
| `Cosh(x)` | Taylor series: `Σ x²ⁿ/(2n)!` with `decimal` | **easy** |
| `EulerGamma()` | Brent-McMillan algorithm or alternating series | medium |
| `Khinchin()` | Sum over primes `Σ log(1 + 1/(k(k+2))) log(k)/log(2)` | medium |
| `Gamma(x)` x∈ℚ | Stirling + reflection + Lanczos, or via Exp/Ln | medium-hard |

## Possible gen-CF additions

| Function | Known formula | Notes |
|----------|--------------|-------|
| `Sinh(x)` | Lambert-style: same family as `Tanh` | `sinh(x) = x·cosh(x)/tanh⁻¹...` tricky |
| `Cosh(x)` | Related to `Tanh` via identity | Less clean than Taylor |
| `Exp(x)` | Euler: `eˣ = 1 + x/(1 − x/(2 + x/(3 − x/(2 + x/5 −...))))` | Works but mixes signs |

The Taylor-series-with-decimal route is the easiest path for `Exp`, `Sinh`, `Cosh` —
same pattern as the existing `Ln` decimal implementation.
