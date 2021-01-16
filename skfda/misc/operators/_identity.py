import numpy as np

from ...representation import FDataGrid
from ...representation.basis import Basis
from ._operators import Operator, gramian_matrix_optimization


class Identity(Operator):
    """Identity operator.

    Linear operator that returns its input.

    .. math::
        Ix = x

    Can be applied to both functional and multivariate data.

    """

    def __call__(self, f):
        return f


@gramian_matrix_optimization.register
def basis_penalty_matrix_optimized(
        linear_operator: Identity,
        basis: Basis):

    return basis.gram_matrix()


@gramian_matrix_optimization.register
def fdatagrid_penalty_matrix_optimized(
        linear_operator: Identity,
        basis: FDataGrid):
    from ..metrics import l2_norm

    return np.diag(l2_norm(basis)**2)
