"""l0MKL: sparse multiple-kernel learning with outer approximation."""

from .synthetic_data import make_copy_classification
from .estimator import L0MKLClassifier, L1MKLClassifier
from .kernels import FeatureLinearKernelizer, combine_kernels
from .model_selection import MKLGridSearchCV, parameter_sweep
from .oa import OADiagnostics, OASolution, solve_oa

__all__ = [
    "FeatureLinearKernelizer",
    "L0MKLClassifier",
    "L1MKLClassifier",
    "MKLGridSearchCV",
    "OADiagnostics",
    "OASolution",
    "combine_kernels",
    "make_copy_classification",
    "parameter_sweep",
    "solve_oa",
]

__version__ = "0.1.0"
