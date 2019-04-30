"""
Core module of the package. Contains the classes and methods used to the
basic usage of the functional data objects.
"""

# Abstact classes to implement custom Extrapolators and Interpolators
from ._evaluator import Evaluator, EvaluatorConstructor

# Generic extrapolators to be used with grids and basis
from .extrapolation import (PeriodicExtrapolation, BoundaryExtrapolation,
                            ExceptionExtrapolation, FillExtrapolation)

# Interpolators to be used with grids (could be used as extrapolators too)
from .interpolation import SplineInterpolator
