from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from ...representation import FDataGrid
from ...representation._typing import Vector
from ...representation.basis import Basis
from ._operators import Operator, gramian_matrix_optimization

T = TypeVar("T", bound=Vector)


class Identity(Operator[T, T]):
    """Identity operator.

    Linear operator that returns its input.

    .. math::
        Ix = x

    Can be applied to both functional and multivariate data.

    """

    def __call__(self, f: T) -> T:  # noqa: D102
        return f


@gramian_matrix_optimization.register
def basis_penalty_matrix_optimized(
    linear_operator: Identity[Any],
    basis: Basis,
) -> np.ndarray:
    """Optimized version of the penalty matrix for Basis."""
    return basis.gram_matrix()


@gramian_matrix_optimization.register
def fdatagrid_penalty_matrix_optimized(
    linear_operator: Identity[Any],
    basis: FDataGrid,
) -> np.ndarray:
    """Optimized version of the penalty matrix for FDataGrid."""
    from ..metrics import l2_norm

    return np.diag(l2_norm(basis)**2)
