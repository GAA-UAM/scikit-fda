from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from ...representation import FDataGrid  # noqa: TC001
from ...representation.basis import Basis  # noqa: TC001
from ...typing._numpy import NDArrayFloat  # noqa: TC001
from ._operators import InputType, Operator, gram_matrix_optimization

T = TypeVar("T", bound=InputType)


class Identity(Operator[T, T]):
    """Identity operator.

    Linear operator that returns its input.

    .. math::
        Ix = x

    Can be applied to both functional and multivariate data.

    """

    def __call__(self, f: T) -> T:  # noqa: D102, RUF100
        return f


@gram_matrix_optimization.register
def basis_penalty_matrix_optimized(
    linear_operator: Identity[Any],  # noqa: ARG001
    basis: Basis,
) -> NDArrayFloat:
    """Optimized version of the penalty matrix for Basis."""
    return basis.gram_matrix()


@gram_matrix_optimization.register
def fdatagrid_penalty_matrix_optimized(
    linear_operator: Identity[Any],  # noqa: ARG001
    basis: FDataGrid,
) -> NDArrayFloat:
    """Optimized version of the penalty matrix for FDataGrid."""
    from ..metrics import l2_norm

    return np.diag(l2_norm(basis)**2)
