import numpy as np

from ...representation import FDataGrid
from ...representation.basis import Basis
from ._operators import Operator, gramian_matrix_optimization


class Identity(Operator):
    """Identity operator.

    .. math::
        Ix = x

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
    from ..metrics import norm_lp

    return np.diag(norm_lp(basis)**2)
