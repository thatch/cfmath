"""cfmath — continued fractions library."""

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "dev"

from .archyperbolic import Arccosh, Arcsinh, Arctanh
from .arctrig import Arccos, Arcsin, Arctan, ArctanCF, ArctanGCF, ArctanMP, ArctrigMode
from .constants import Apery, Catalan, E, EulerGamma, Khinchin, Phi, Pi, Plastic, Tau
from .convergents import convergent, convergent_pair, convergent_pairs, convergents
from .core import CF
from .debug import CountingIterator, describe_source_tree, digits_with_debug
from .exponential import Exp
from .gosper import cf_add, cf_div, cf_homographic, cf_max, cf_min, cf_mul, cf_sub
from .gosper_cf import GosperBi, GosperGeneric, GosperMono
from .hyperbolic import Cosh, Sinh, Tanh
from .logarithm import Ln, Log, Log2, Log10
from .polyratio import PolyTransform
from .power import Cuberoot, Nthroot, Pow, PowCF, PowIntExponent, PowInterval, PowMode, PowMP
from .quadratic import Sqrt
from .special import Gamma, Zeta
from .trig import Cos, CosCF, CosGCF, CosMP, Sin, SinCF, SinGCF, SinMP, Tan, TanCF, TanGCF, TanMP, TrigMode

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
    "GosperMono",
    "GosperBi",
    "GosperGeneric",
    "PolyTransform",
    "cf_homographic",
    "cf_min",
    "cf_max",
    "Sqrt",
    "Phi",
    "E",
    "Pi",
    "Tau",
    "Cuberoot",
    "Nthroot",
    "Log2",
    "Ln",
    "Sin",
    "SinGCF",
    "SinCF",
    "SinMP",
    "Cos",
    "CosGCF",
    "CosCF",
    "CosMP",
    "Tan",
    "TanGCF",
    "TanCF",
    "TanMP",
    "TrigMode",
    "Arcsin",
    "Arccos",
    "Arctan",
    "ArctanGCF",
    "ArctanCF",
    "ArctanMP",
    "ArctrigMode",
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
    "PowMode",
    "PowIntExponent",
    "PowCF",
    "PowMP",
    "PowInterval",
    "Gamma",
    "Zeta",
    "Plastic",
    "Khinchin",
    "CountingIterator",
    "describe_source_tree",
    "digits_with_debug",
]
