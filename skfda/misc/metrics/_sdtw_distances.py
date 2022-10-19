""""
Implementation of soft Dynamic-Time-Warping (sDTW) divergence.
"""
from __future__ import annotations

from typing import Dict, Optional, Union, Callable, Tuple

import numpy as np

from ...representation import FDataGrid
from ...representation._typing import NDArrayFloat

from skfda._utils._utils import _check_compatible_fdata
from skfda.misc.metrics import _sdtw_numba
from skfda.misc._math import inner_product_matrix, inner_product


def _check_shape_postive_cost_mat(
    cost: NDArrayFloat,
    expected_shape: Tuple[int, int]
) -> None:
    """check that matrix 'cost' is nonnegative and has expected shape"""
    n1, n2 = expected_shape
    # must have shape fdata1.grid_points[0], fdata1.grid_points[0]
    if not cost.shape == (n1, n2):
        raise ValueError(
            f"Cost matrix must have shape ({n1}, {n2})"
        )
    # non-negativity check
    if not np.all(cost >= 0):
        raise ValueError(
            "Cost matrix must contain non-negative entries"
        )


def _check_indiscernable(
    cost_XX: NDArrayFloat,
    cost_YY: NDArrayFloat
) -> None:
    """check that cost returns a symmetric matrix with zero diag"""
    # indiscernability: cost(x_t, x_t) = 0 for any t
    if (
        np.any(np.diag(cost_XX) != 0)
        or np.any(np.diag(cost_YY) != 0)
    ):
        raise ValueError(
            "The cost between two identical objects "
            "must be zero (i.e cost(X, X) must be zero-diagonal)"
        )


def _check_symmetry(
    cost_XY: NDArrayFloat,
    cost_YX: NDArrayFloat,
) -> None:
    """check that cost_XY==cost_YX.T"""

    if not np.allclose(cost_XY, cost_YX.T):
        raise ValueError(
            "The cost function does not return a symmetric matrix"
        )


def _check_input_fdata(
    fdata1,
    fdata2
) -> None:
    """check compatible, one sample and one-dimensional"""
    _check_compatible_fdata(fdata1, fdata2)

    if not isinstance(fdata1, FDataGrid):
        raise ValueError(
            "fdata1 and fdata2 must be FDataGrid objects"
        )

    # dim_codomain can be higher than one,
    # but dim_domain must be one
    if not fdata1.dim_domain == 1:
        raise ValueError(
            "fdata1 and fdata2 must have their domain dimension equal to one"
        )

    if not (fdata1.n_samples == 1 and fdata2.n_samples == 1):
        raise ValueError(
            "Both fdata1 and fdata2 must each contain a single sample"
        )


def half_sq_euclidean(
    arg1: NDArrayFloat,
    arg2: Optional[NDArrayFloat] = None,
) -> NDArrayFloat:
    """
    Return the half squared Euclidean row distance between
    two matrices of size (n1, d) and (n2, d).

    Args:
        arg1: First sample (n_samples_1, n_features).
        arg2: Second sample (n_samples_2, n_features).
            If it is ``None`` returns the half squared Euclidean distance
            between the rows of ``arg1``.

    The distance is taken w.r.t matrix rows.
    """
    if arg2 is not None:
        # sq_euclidean(X, Y) =
        # DotProduct(sigma_0_bounds=(10**-10, 10**10)) =
        # diag(np.dot(X, X.T)) + diag(np.dot(Y, Y.T)) - 2*np.dot(X, Y.T)
        # np.dot(X, Y.T)=inner_product_matrix(X, Y)
        cost_12 = -1 * inner_product_matrix(arg1, arg2)

        # np.sum(X ** 2, axis=1)=inner_product(X, X)
        cost_12 += (0.5 * inner_product(arg1, arg1))[:, np.newaxis]
        cost_12 += 0.5 * inner_product(arg2, arg2)

        return cost_12
    else:
        cost_11 = -1 * inner_product_matrix(arg1, arg2)
        # half Sum Of Squares on each row
        sos_row = 0.5 * inner_product(arg1, arg1)
        cost_11 += sos_row[:, np.newaxis]
        cost_11 += sos_row

        return cost_11


def _sdtw_divergence(
    fdata1: FDataGrid,
    fdata2: FDataGrid,
    *,
    gamma: float = 1.0,
    cost: Union[Tuple[NDArrayFloat], Callable[[
        NDArrayFloat, NDArrayFloat, NDArrayFloat]]] = None,
    check_cost: bool = False,
    **kwargs_cost: Dict
) -> float:

    # fdata1 and fdata2 are only required to be one-dimensional.
    # They can have different (one-dimensional) domain ranges and
    # different grid points size

    _check_input_fdata(fdata1, fdata2)

    n1, n2 = len(fdata1.grid_points[0]), len(fdata2.grid_points[0])

    if gamma <= 0:
        raise ValueError(
            f"gamma was set to {gamma} but must be positive"
        )

    # booleans to remember the input format of cost if given
    cost_was_tuple = isinstance(cost, tuple)
    cost_was_callable = isinstance(cost, Callable)

    if cost_was_tuple:
        if len(cost) != 3:
            raise ValueError(
                "When the alignment cost matrices are pre-computed, "
                "then 'cost' must be a tuple of three two-dimensional "
                "numpy arrays."
            )

    elif cost_was_callable:
        # pre-computation for the symmetry check
        # of the cross-product part of the cost
        if check_cost:
            cost_21 = cost(
                fdata2.data_matrix[0],
                fdata1.data_matrix[0],
                **kwargs_cost
            )

        # compute the cost matrices
        cost = (
            cost(
                fdata1.data_matrix[0],
                fdata2.data_matrix[0],
                **kwargs_cost
            ),
            cost(
                fdata1.data_matrix[0],
                fdata1.data_matrix[0],
                **kwargs_cost
            ),
            cost(
                fdata2.data_matrix[0],
                fdata2.data_matrix[0],
                **kwargs_cost
            )
        )

        # cost_12 = cost(
        #     fdata1.data_matrix[0],
        #     fdata2.data_matrix[0],
        #     **kwargs_cost
        # )
        # cost_11 = cost(
        #     fdata1.data_matrix[0],
        #     fdata1.data_matrix[0],
        #     **kwargs_cost
        # )
        # cost_22 = cost(
        #     fdata2.data_matrix[0],
        #     fdata2.data_matrix[0],
        #     **kwargs_cost
        # )

        # cost = (cost_12, cost_11, cost_22)

    elif cost is None:
        # default cost: 0.5 * squared Euclidean:
        # 0.5 * squared euclidean
        # or global kernel alignment (depends on gamma !=0)
        # 0.5 * diag(X@X.T) + 0.5 * diag(Y@Y.T) - X@Y.T

        cost = (
            half_sq_euclidean(
                fdata1.data_matrix[0],
                fdata2.data_matrix[0]
            ),
            half_sq_euclidean(
                fdata1.data_matrix[0],
                fdata1.data_matrix[0]
            ),
            half_sq_euclidean(
                fdata2.data_matrix[0],
                fdata2.data_matrix[0]
            )
        )

    else:
        raise ValueError(
            "Cost must be a tuple containing three numpy arrays "
            "or a Callable that returns a numpy array."
        )

    # now for any initial input format of cost, cost has become a 3-tuple
    if check_cost and (cost_was_callable or cost_was_tuple):

        expected_shapes = [(n1, n2), (n1, n1), (n2, n2)]
        for idx, c in enumerate(cost):
            _check_shape_postive_cost_mat(
                cost=c,
                expected_shape=expected_shapes[idx]
            )

        _check_indiscernable(
            cost_XX=cost[1],
            cost_YY=cost[2]
        )

        if cost_was_callable:
            _check_symmetry(
                cost_XY=cost[0],
                cost_YX=cost_21
            )

    # for nonnegativity, symmetry, unicity of div(X,Y)=0 at X=Y
    return(
        _sdtw_numba._sdtw(cost[0], gamma)
        - 0.5 * _sdtw_numba._sdtw(cost[1], gamma)
        - 0.5 * _sdtw_numba._sdtw(cost[2], gamma)
    )


class sdtwDivergence():
    r"""
    Compute the soft Dynamic-Time-Warping (DTW) divergence between
    two multivariate functional data whose domain is one-dimensional.
    The soft Dynamic-Time-Warping is defined as (:footcite:`Cuturi_2017_sdtw`):

    .. math::
        \text{SDTW}_{\gamma}(x, y) = - \gamma \log
            \sum_{A \in \mathcal{A}_{n_x,n_y}}
            \exp \big(- \text{Tr}(C^{\top} A) / \gamma \big)

    and the divergence is derived as:

    .. math::
        \text{SDTW}_{div, \gamma}(x, y) =
            \text{SDTW}_{\gamma}(x, y)
            - \frac{1}{2} \text{SDTW}_{\gamma}(x, x)
            - \frac{1}{2} \text{SDTW}_{\gamma}(y, y)

    with :math:`\gamma > 0, C \in \mathbf{R}^{n_x \times n_y}` is the cost
    matrix and
    :math:`\mathcal{A}_{n_x, n_y} \subset \{0, 1\}^{n_x \times n_y}`
    is an alignment matrix. If the cost is the half squared Euclidean
    distance,
    :math:`c_{ij} =  \frac{1}{2} \lVert x({t_i}) - y({t_j}) \rVert^2_2`.
    This a smooth version where the min operator has been replaced by
    the soft-min one, whose smoothness is controlled by :math:`\gamma`
    (the higher the smoother).

    Dynamic-Time-Warping is a distance metric defined as the minimal cost
    to align two time series X and Y. Typically, the cost is chosen as the
    squared Euclidean distance. However the classical DTW is not uniquely
    minimized and not differentiable w.r.t one argument given the other.
    Soft-DTW divergence is a differentiable version of DTW with unique
    minimizer, meaning that :math:`sdtw_div(X,Y)=0` if and only if
    :math:`X=Y`. It is symmetric and nonnegative but does not satisfy the
    triangle inequality. See :footcite:`Blondel_2021_sdtw_div`.

    Args:
        fdata1: First FDataGrid object.
        fdata2: Second FDataGrid object.
        gamma: smoothing parameter, must be positive.
        cost: Either a two-dimensional numpy array or a callable.
            If it is an array, then it must have shape ``(n1, n2)``
            where ``n1`` is the grid size of fdata1 and equivalently for
            ``n2``. If it is a callable, then it must take two
            two-dimensional numpy arrays (equivalent to the ``data_matirx``
            attribute of an FDataGrid object) as inputs and return a
            ``(n1, n2)`` numpy array.
        check_cost: Wheter to check the mathematical properties of
            ``cost`` (function or matrix).

    Returns:
        soft Dynamic-Time-Warping divergence value.

    References:
        .. footbibliography::

    """
    def __init__(
        self,
        gamma: float = 1.0,
        cost: Union[Tuple[NDArrayFloat], Callable[[
            NDArrayFloat, NDArrayFloat, NDArrayFloat]]] = None,
    ) -> None:

        self.gamma = gamma
        self.cost = cost

    def __call__(
        self,
        fdata1: FDataGrid,
        fdata2: FDataGrid,
        *,
        check_cost: Optional[bool] = False
    ) -> NDArrayFloat:
        """Compute the soft-DTW divergence."""
        return _sdtw_divergence(
            fdata1,
            fdata2,
            gamma=self.gamma,
            cost=self.cost,
            check_cost=check_cost,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}()"
        )


# @pairwise_metric_optimization.register
# def _pairwise_metric_optimization_sdtw(
#     metric: sdtwDivergence,
#     elem1: FData,
#     elem2: FData,
# ) -> NDArrayFloat:

#     return metric(elem1, elem2)
