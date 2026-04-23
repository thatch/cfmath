"""Test the relationship between CF periods and termination."""

import math
import signal
import sys

sys.path.insert(0, "src")
from cfmath import Sqrt
from cfmath.gosper import cf_mul


def alarm(s, f):
    raise TimeoutError()


signal.signal(signal.SIGALRM, alarm)


def period_of(cf):
    return len(cf.repeating)


def squarefree_part(n):
    """Return squarefree part of n (divide out all perfect-square factors)."""
    result = 1
    for p in range(2, int(n**0.5) + 2):
        while n % (p * p) == 0:
            n //= p * p
        if n % p == 0:
            result *= p
            n //= p
    result *= n
    return result


def try_mul(a, b, label):
    signal.alarm(2)
    try:
        r = cf_mul(Sqrt(a), Sqrt(b))
        terms = []
        for t in r._iter_from(0):
            terms.append(t)
            if len(terms) >= 4:
                break
        signal.alarm(0)
        if terms:
            result = "irrational (terminates)"
        else:
            result = "STALLED (rational result)"
    except TimeoutError:
        terms = None
        result = "LOOPS (rational result)"
    pa, pb = period_of(Sqrt(a)), period_of(Sqrt(b))
    sa, sb = squarefree_part(a), squarefree_part(b)
    same_field = sa == sb
    coprime_periods = math.gcd(pa, pb) == 1
    print(
        "  sqrt(%2d) x sqrt(%2d): periods (%d,%d) gcd=%d  same-field=%s  -> %s  %s"
        % (
            a,
            b,
            pa,
            pb,
            math.gcd(pa, pb),
            "YES" if same_field else "NO ",
            result,
            "(coprime, BUT same-field!)"
            if (coprime_periods and same_field and terms is None)
            else "(coprime, diff-field)"
            if (coprime_periods and not same_field)
            else "(equal periods, same-field)"
            if (not coprime_periods and same_field)
            else "",
        )
    )


print("=== Product of two Sqrt CFs ===")
print()
print("Hypothesis: coprime periods => better chance of terminating")
print()
pairs = [
    (2, 2),  # same field, period (1,1)  -> loops
    (2, 3),  # diff field, period (1,2)  -> terminates
    (2, 5),  # diff field, period (1,1)  -> terminates (periods NOT coprime!)
    (2, 8),  # SAME field (sqrt(8)=2sqrt(2)), period (1,2) coprime -> LOOPS
    (2, 18),  # same field (sqrt(18)=3sqrt(2)), period (1,2) coprime -> LOOPS
    (2, 32),  # same field (sqrt(32)=4sqrt(2)), coprime -> LOOPS
    (3, 3),  # same field, period (2,2) -> loops
    (3, 12),  # same field (sqrt(12)=2sqrt(3)), coprime? -> loops
    (3, 7),  # diff field, period (2,4) -> terminates
    (5, 5),  # same field, period (1,1) -> loops
    (5, 45),  # same field (sqrt(45)=3sqrt(5)), -> loops
    (5, 7),  # diff field, period (1,4) -> terminates
    (7, 7),  # same field, period (4,4) -> loops
    (2, 7),  # diff field, coprime (1,4) -> terminates
    (3, 5),  # diff field, coprime (2,1) -> terminates
]
for a, b in pairs:
    try_mul(a, b, "")

print()
print("=== Key finding ===")
print()
print("  Coprime periods FAIL for: sqrt(2) x sqrt(8), sqrt(2) x sqrt(18), etc.")
print("  These have coprime periods (1,2) but are in the SAME field Q(sqrt(2)).")
print("  The actual condition is: do the inputs have the SAME squarefree part?")
print("  - Same squarefree part => product may be rational => may loop")
print("  - Different squarefree parts => product always irrational => always terminates")
print()
print("  Coprime periods is a PARTIAL heuristic: elements from the same field")
print("  often have related (non-coprime) periods, so coprimality correlates")
print("  with being from different fields. But k*sqrt(d) has the same field")
print("  as sqrt(d) while potentially having a different (coprime) period.")
