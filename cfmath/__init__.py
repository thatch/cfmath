"""cfmath — continued fractions library."""

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "dev"

from .core import CF
from .convergents import convergent, convergent_pair, convergent_pairs, convergents
from .gosper import cf_add, cf_sub, cf_mul, cf_div, cf_homographic, cf_min, cf_max
from .quadratic import Sqrt
from .constants import Phi, E, Pi, Tau, EulerGamma, Catalan, Apery, Plastic, Khinchin
from .logarithm import Log2, Ln, Log, Log10
from .exponential import Exp
from .power import Pow, Cuberoot
from .trig import Sin, Cos, Tan
from .arctrig import Arcsin, Arccos, Arctan
from .hyperbolic import Sinh, Cosh, Tanh
from .archyperbolic import Arctanh, Arcsinh, Arccosh
from .special import Zeta, Gamma
from .debug import CountingIterator, digits_with_debug

__all__ = [
    "CF",
    "convergent",
    "convergent_pair",
    "convergent_pairs",
    "convergents",
    "cf_add",
    "cf_sub",
    "cf_mul",
    "cf_div",
    "cf_homographic",
    "cf_min",
    "cf_max",
    "Sqrt",
    "Phi",
    "E",
    "Pi",
    "Tau",
    "Cuberoot",
    "Log2",
    "Ln",
    "Sin",
    "Cos",
    "Tan",
    "Arcsin",
    "Arccos",
    "Arctan",
    "Sinh",
    "Cosh",
    "Tanh",
    "Arctanh",
    "Arcsinh",
    "Arccosh",
    "Exp",
    "EulerGamma",
    "Catalan",
    "Apery",
    "Log",
    "Log10",
    "Pow",
    "Gamma",
    "Zeta",
    "Plastic",
    "Khinchin",
    "CountingIterator",
    "digits_with_debug",
]
