"""Operators applicable to functional data."""
from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_identity": ["Identity"],
        "_integral_transform": ["IntegralTransform"],
        "_linear_differential_operator": ["LinearDifferentialOperator"],
        "_operators": [
            "MatrixOperator",
            "Operator",
            "gram_matrix",
            "gram_matrix_optimization",
        ],
        "_srvf": ["SRSF"],
    },
)

if TYPE_CHECKING:
    from ._identity import Identity as Identity
    from ._integral_transform import IntegralTransform as IntegralTransform
    from ._linear_differential_operator import (
        LinearDifferentialOperator as LinearDifferentialOperator
    )
    from ._operators import (
        MatrixOperator as MatrixOperator,
        Operator as Operator,
        gram_matrix as gram_matrix,
        gram_matrix_optimization as gram_matrix_optimization,
    )
    from ._srvf import SRSF as SRSF
