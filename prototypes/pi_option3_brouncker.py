"""Prototype: π as a continued fraction via Brouncker's generalized CF.

Brouncker's formula:
    π = 3 + 1²/(6 + 3²/(6 + 5²/(6 + 7²/(6 + ...))))

As (b_n, a_{n+1}) pairs fed to CF.from_generalized_cf:
    Pair 0: (3,  1)   ← b₀=3, a₁=1²=1
    Pair 1: (6,  9)   ← b₁=6, a₂=3²=9
    Pair 2: (6, 25)   ← b₂=6, a₃=5²=25
    Pair k (k≥1): (6, (2k+1)²)

The numerators 1², 3², 5², 7², ... grow as O(n²), which causes very slow
convergence from the generalized CF perspective: many GCF pairs must be
consumed before each simple CF term is determined.  The `from_generalized_cf`
machinery stalls after 100 consecutive pairs without emitting a term, so this
approach can struggle to produce the larger CF terms of π (e.g. 292).
"""

from __future__ import annotations

import sys
import os

# Ensure the library is importable when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fractions import Fraction
from cfmath.core import CF
from cfmath.convergents import convergent


# ---------------------------------------------------------------------------
# Pair generators
# ---------------------------------------------------------------------------

def _brouncker_pairs(counter: list[int]):
    """Yield (b_n, a_{n+1}) pairs for Brouncker's π formula.

    Pair 0: (3, 1)  →  b₀=3, a₁=1²
    Pair k≥1: (6, (2k+1)²)  →  odd-square numerators: 3², 5², 7², ...

    ``counter`` is a one-element list used as a mutable counter so callers
    can inspect how many pairs were consumed.
    """
    # Pair 0
    counter[0] += 1
    yield (3, 1)
    # Pairs 1, 2, 3, ...  with numerators 3², 5², 7², ...
    k = 1
    while True:
        counter[0] += 1
        yield (6, (2 * k + 1) ** 2)
        k += 1


# ---------------------------------------------------------------------------
# Main constructor
# ---------------------------------------------------------------------------

def Pi_brouncker() -> tuple[CF, list[int]]:
    """Return (CF for π, pair_counter) using Brouncker's generalized CF.

    The pair_counter is a mutable one-element list; after iterating the CF
    its value reflects how many GCF pairs were consumed.
    """
    counter = [0]
    cf = CF.from_generalized_cf(_brouncker_pairs(counter))
    return cf, counter


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    EXPECTED_TERMS = [3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1]
    N_TERMS = 15
    STALL_LIMIT = 100  # from_generalized_cf raises after this many dry pairs

    print("=" * 60)
    print("Brouncker's generalized CF for π")
    print("π = 3 + 1²/(6 + 3²/(6 + 5²/(6 + ...)))")
    print("=" * 60)
    print()

    pi_cf, counter = Pi_brouncker()

    # Collect up to N_TERMS, catching the stall ValueError
    terms: list[int] = []
    stalled_after: int | None = None
    try:
        for i, t in enumerate(pi_cf):
            if i >= N_TERMS:
                break
            terms.append(t)
    except ValueError as exc:
        stalled_after = len(terms)
        print(f"[WARNING] from_generalized_cf stalled after {stalled_after} terms.")
        print(f"  Detail: {exc}")
        print()

    pairs_consumed = counter[0]

    # ---- Report terms ----
    print(f"CF terms produced ({len(terms)} of {N_TERMS} requested):")
    print(f"  Got:      {terms}")
    if stalled_after is None:
        print(f"  Expected: {EXPECTED_TERMS}")
        match = terms == EXPECTED_TERMS
        print(f"  Match:    {'YES' if match else 'NO'}")
    else:
        print(f"  (Stalled before all {N_TERMS} terms could be produced.)")
    print()

    # ---- Convergent check ----
    print("Convergent check (convergent #3 should be 355/113):")
    if len(terms) >= 4:  # need indices 0..3
        partial_cf = CF(terms[:4])
        c3 = convergent(partial_cf, 3)
        expected_c3 = Fraction(355, 113)
        ok = c3 == expected_c3
        print(f"  convergent(Pi_brouncker(), 3) = {c3}")
        print(f"  Expected: {expected_c3}")
        print(f"  Match: {'YES' if ok else 'NO'}")
    else:
        print(f"  (Not enough terms to compute convergent #3; only {len(terms)} available.)")
    print()

    # ---- Pairs consumed ----
    print(f"GCF pairs consumed to produce {len(terms)} simple CF terms:")
    print(f"  {pairs_consumed} pairs")
    if len(terms) > 0:
        ratio = pairs_consumed / len(terms)
        print(f"  Ratio: ~{ratio:.1f} pairs per CF term")
    print()

    # ---- Convergence commentary ----
    print("Convergence rate:")
    print(
        "  Brouncker's GCF has numerators 1², 3², 5², ... growing as O(n²)."
    )
    print(
        "  The Möbius transform state must integrate many levels before the"
    )
    print(
        "  fractional output is pinned to a single integer floor — especially"
    )
    print(
        "  near large CF terms like 292 (the 5th term of π) where the tail"
    )
    print(
        "  must be known very precisely before that digit can be confirmed."
    )
    print(
        "  Convergence is therefore very slow: O(n²) pairs per output term,"
    )
    print(
        "  and the built-in stall guard (100 dry pairs) may trigger before"
    )
    print(
        "  large terms like 292 can be emitted."
    )
    if stalled_after is not None:
        print()
        print(
            f"  In this run it stalled after {stalled_after} terms / {pairs_consumed} pairs."
        )
        print(
            "  This is expected behaviour for Brouncker's series applied via"
        )
        print(
            "  the Gosper-style generalized-CF algorithm without augmentation."
        )
