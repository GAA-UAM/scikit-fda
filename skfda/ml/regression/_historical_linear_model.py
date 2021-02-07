from typing import Tuple

import numpy as np
import scipy.integrate

from ..._utils import _pairwise_symmetric
from ...representation import FDataBasis, FDataGrid
from ...representation.basis._finite_element import FiniteElement


def _fem_inner_product_matrix(
    fem_basis: FiniteElement,
    fd: FDataGrid,
    limits: Tuple[float, float],
    y_val: float,
) -> np.ndarray:
    """
    Computes the matrix of inner products of an FEM basis with a functional
    data object over a range of x-values for a fixed y-value. The numerical
    integration uses Romberg integration with the trapezoidal rule.

    Arguments:
        fem_basis: an FEM basis defined by a triangulation within a
            rectangular domain. It is assumed that only the part of the mesh
            that is within the upper left triangular is of interest.
        fd: a regular functional data object.
        limits: limits of integration, as a tuple of form
            (lower limit, upper limit)
        y_val: the fixed y value.

    """

    fem_basis_fd = fem_basis.to_basis()
    grid = fd.grid_points[0]
    grid_index = (grid >= limits[0]) & (grid <= limits[1])
    grid = grid[grid_index]

    def _pairwise_fem_inner_product(
        fem_basis_fd: FDataBasis,
        fd: FDataGrid,
    ) -> np.ndarray:

        eval_grid_fem = np.concatenate(
            (
                grid[:, None],
                np.full(
                    shape=(len(grid), 1),
                    fill_value=y_val,
                )
            ),
            axis=1,
        )

        eval_fem = fem_basis_fd(eval_grid_fem)
        eval_fd = fd(grid)

        # Only for scalar valued functions for now
        assert eval_fem.shape[-1] == 1
        assert eval_fd.shape[-1] == 1

        prod = eval_fem[..., 0] * eval_fd[..., 0]

        return scipy.integrate.simps(prod, grid, axis=1)

    return _pairwise_symmetric(
        _pairwise_fem_inner_product,
        fem_basis_fd,
        fd,
    )


def _design_matrix(
    fem_basis: FiniteElement,
    fd: FDataGrid,
    pred_points: np.ndarray,
) -> np.ndarray:
    """
    Computes the indefinite integrals of the curves over s up to each t-value.

    Arguments:
        fem_basis: an FEM basis defined by a triangulation within a
            rectangular domain. It is assumed that only the part of the mesh
            that is within the upper left triangular is of interest.
        fd: a regular functional data object.
        pred_points: points where ``fd`` is evaluated.

    Returns:
        Design matrix.

    """

    matrix = np.array([
        _fem_inner_product_matrix(fem_basis, fd, limits=(0, t), y_val=t).T
        for t in pred_points
    ])

    return np.swapaxes(matrix, 0, 1)
