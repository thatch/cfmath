"""Acceptance test: 5th convergent of Pi²."""

import math
from fractions import Fraction

import mpmath

from cfmath import CF, convergent, Pi


def test_pi_squared_5th_convergent():
    pi = Pi()
    pi2 = pi * pi  # bihomographic multiplication

    # 5th convergent (0-indexed: index 4, using terms a0..a4)
    c = convergent(pi2, 4)

    # Reference: compute Pi² via mpmath directly → CF → convergent
    mpmath.mp.dps = 50
    pi2_val = float(mpmath.pi ** 2)
    pi2_direct = CF.from_float(pi2_val, max_terms=20)
    c_ref = convergent(pi2_direct, 4)

    assert c == c_ref, f"Got {c}, expected {c_ref}"

    # Sanity check: approximates Pi² well
    pi_sq_approx = float(mpmath.pi ** 2)
    assert abs(float(c) - pi_sq_approx) < 0.01, (
        f"Convergent {c} = {float(c):.6f} too far from Pi² ≈ {pi_sq_approx:.6f}"
    )
