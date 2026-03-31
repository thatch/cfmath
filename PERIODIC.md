# Notes on periodicity of CF arithmetic

## Background: Lagrange's theorem

A continued fraction is *eventually periodic* if and only if the number it
represents is a **quadratic irrational** — a root of some `ax² + bx + c = 0`
with integer coefficients and irrational solution.

Equivalently: a number has a periodic CF iff it lies in `Q(√d)` for some
square-free positive integer `d`, but is not itself rational.

Rational numbers have *finite* CFs; quadratic irrationals have *periodic* CFs;
everything else (e, π, γ, …) has an aperiodic CF.

## The key question: does arithmetic preserve periodicity?

### Same quadratic field — YES

If both inputs are in `Q(√d)` for the **same** `d`, their sum, difference,
product, and quotient are all still in `Q(√d)` (or rational), so the result is
either finite or periodic.

Example: `Sqrt(2) + (1 + Sqrt(2))` = `1 + 2·Sqrt(2)` — still in `Q(√2)`,
still periodic.

### Different quadratic fields — NOT NECESSARILY

If the inputs live in different quadratic fields, the arithmetic result can be a
*degree-4* algebraic number, which is NOT a quadratic irrational, and therefore
does NOT have a periodic CF.

Example: `Sqrt(2) + Sqrt(3)` satisfies `x⁴ - 10x² + 1 = 0`, degree 4, so its
CF is aperiodic (even though both inputs are periodic).

This is confirmed empirically:

```
Sqrt(2):           [1; (2)]             period 1
Sqrt(3):           [1; (1, 2)]          period 2
Sqrt(2) + Sqrt(3): [3; 6, 1, 5, 7, 1, 1, 4, 1, 38, ...]   no period visible
```

## What determines the period length (when it exists)?

The period length of a quadratic irrational `(P + √D) / Q` (in standard
continued-fraction normal form) is determined by the **discriminant** of the
minimal polynomial and the **fundamental unit** of `Z[√d]`.  There is no
simple formula from the input periods.

Specifically, for `√n` the period length can be as large as `O(√n)` and
depends on the structure of the continued fraction of `√n` rather than any
obvious function of `n`.

### LCM intuition: approximately right, but not exact

Your intuition that operating on a period-2 and a period-7 CF might give
period `lcm(2,7) = 14` has *some* merit as an upper bound heuristic, but:

1. The minimal polynomial of the result may have smaller period than LCM.
2. For different-field inputs the result is aperiodic regardless.
3. For same-field inputs the period is governed by algebraic number theory,
   not the CF periods of the specific representations.

In practice, the period of the result (if in the same field) is typically
much smaller than LCM of the input periods.

## Approaches to study this computationally

### 1. Detect the quadratic field first

Before computing, check whether both CFs come from the same `Q(√d)`:
- For `Sqrt(n)` this is obvious from `n`.
- For a computed CF, derive its minimal polynomial from a sufficient number
  of convergents using the LLL lattice reduction algorithm or PSLQ.

If the minimal polynomial is degree 2 → periodic result guaranteed.
If degree > 2 → result is aperiodic.

### 2. Empirically find the period

Given a computed CF, look for a repeating block using a sliding-window comparison
over the emitted terms. A period-`k` CF satisfies `a[n] = a[n+k]` for all large
enough `n`. You can search up to period ~50 easily; longer periods require more
terms.

```python
def detect_period(cf, max_terms=500, max_period=50):
    terms = list(cf.take(max_terms))
    for p in range(1, max_period + 1):
        # Check if terms repeat with period p after some offset
        for offset in range(max_terms - 2 * p):
            if terms[offset:offset+p] == terms[offset+p:offset+2*p]:
                # Tentative hit; verify over more of the sequence
                if all(
                    terms[offset + i % p] == terms[offset + i]
                    for i in range(2 * p, min(4 * p, max_terms - offset))
                ):
                    return offset, p
    return None
```

### 3. Prove it algebraically (for same-field inputs)

For `Q(√d)` inputs, the arithmetic result is `(A + B√d) / C` for some integers
`A, B, C`.  The standard periodic CF algorithm then applies directly and gives
the exact period.

This is what `Sqrt(n)` does internally — the algorithm tracks `(m, d, a)` and
reads off the exact period.  A more general `QuadraticIrrational(A, B, C, d)`
constructor could do the same.

## Open questions / future work

- Implement `QuadraticIrrational(A, B, C, d)` that computes the exact periodic
  CF for `(A + B√d) / C`.
- Implement `detect_period(cf, ...)` as a utility.
- For a same-field arithmetic result, compute the minimal polynomial of the
  result and extract `A, B, C, d` to pass to the exact constructor.
