"""Acceptance test: 5th convergent of Pi²."""

import mpmath
import pytest
from fractions import Fraction

from cfmath import (
    Arccos,
    Arcsin,
    Arctan,
    Arccosh,
    Arcsinh,
    Arctanh,
    CF,
    Cos,
    Cosh,
    Exp,
    Gamma,
    Ln,
    Log,
    Log2,
    Log10,
    Pi,
    Pow,
    Sin,
    Sinh,
    Tan,
    Tanh,
    Zeta,
    convergent,
)
from cfmath.arctrig import ArctanCF
from cfmath.exponential import ExpCF
from cfmath.logarithm import LnCF, Log10CF, Log2CF, LogCF
from cfmath.trig import _CosCF as CosCF, _SinCF as SinCF, _TanCF as TanCF


def test_pi_squared_5th_convergent():
    pi = Pi()
    pi2 = pi * pi  # bihomographic multiplication

    # 5th convergent (0-indexed: index 4, using terms a0..a4)
    c = convergent(pi2, 4)

    # Reference: compute Pi² via mpmath directly → CF → convergent
    mpmath.mp.dps = 50
    pi2_val = float(mpmath.pi**2)
    pi2_direct = CF.from_float(pi2_val, max_terms=20)
    c_ref = convergent(pi2_direct, 4)

    assert c == c_ref, f"Got {c}, expected {c_ref}"

    # Sanity check: approximates Pi² well
    pi_sq_approx = float(mpmath.pi**2)
    assert abs(float(c) - pi_sq_approx) < 0.01, f"Convergent {c} = {float(c):.6f} too far from Pi² ≈ {pi_sq_approx:.6f}"


def test_float_inputs_are_rejected_by_public_math_functions():
    funcs = [
        Arccos,
        Arcsin,
        Arctan,
        ArctanCF,
        Arccosh,
        Arcsinh,
        Arctanh,
        Cos,
        CosCF,
        Cosh,
        Exp,
        ExpCF,
        Gamma,
        Ln,
        LnCF,
        Log2,
        Log2CF,
        Log10,
        Log10CF,
        Sin,
        SinCF,
        Sinh,
        Tan,
        TanCF,
        Tanh,
    ]
    for fn in funcs:
        with pytest.raises(TypeError):
            fn(1.5)  # type: ignore[arg-type]


def test_cf_inputs_are_accepted_by_cf_variants():
    x = CF.from_rational(Fraction(1, 2))
    positive = CF.from_int(2)

    for fn in (Exp, ExpCF, LnCF, Log2CF, Log10CF, Sin, SinCF, Cos, CosCF, Tan, TanCF):
        arg = positive if fn in (LnCF, Log2CF, Log10CF) else x
        assert isinstance(fn(arg), CF)

    assert isinstance(LogCF(positive, CF.from_int(10)), CF)
    assert Log2(CF.from_int(2)) == CF.from_int(1)
    assert isinstance(Pow(positive, CF.from_int(2)), CF)


def test_cf_inputs_are_rejected_by_rational_only_functions():
    # Gamma is the only remaining function that still requires int | Fraction.
    # The arc-trig, hyperbolic, and inverse-hyperbolic functions now accept CF.
    funcs = [
        Gamma,
    ]
    for fn in funcs:
        with pytest.raises(TypeError):
            fn(CF.from_int(2))  # type: ignore[arg-type]


def test_log_and_pow_reject_bad_secondary_argument_types():
    with pytest.raises(TypeError):
        Log(2, CF.from_int(2))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Pow(2, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Zeta(Fraction(5, 2))  # type: ignore[arg-type]
