"""Operators applicable to functional data."""
from ._identity import Identity
from ._integral_transform import IntegralTransform
from ._linear_differential_operator import LinearDifferentialOperator
from ._operators import (
    MatrixOperator,
    Operator,
    gramian_matrix,
    gramian_matrix_optimization,
)
from ._srvf import SRSF
