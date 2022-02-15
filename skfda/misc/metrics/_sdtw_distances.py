""""
Implementation of soft Dynamic-Time-Warping (sDTW) divergence.
"""
from __future__ import annotations

from typing import Optional, Union, Callable, List, Tuple
from numba.core.types.scalars import Boolean

import numpy as np
from typing_extensions import Final

from ...representation import FData, FDataBasis
from ...representation._typing import NDArrayFloat, GridPointsLike, ArrayLike
# from ._utils import pairwise_metric_optimization

from skfda._utils._utils import _check_compatible_fdata
from skfda.misc.metrics import _sdtw_numba
from skfda.misc._math import inner_product_matrix, inner_product


def _check_shape_postive_cost_mat(cost, expected_shape, shape_only=False):
    """check that matrix 'cost' is nonnegative and has expected shape"""
    n1, n2 = expected_shape
    # must have shape fdata1.grid_points[0], fdata1.grid_points[0]
    if not cost.shape == (n1, n2):
        raise ValueError(
            "Cost matrix must have shape"
            " ({}, {})".format(n1, n2)
        )
    # non-negativity
    if not(shape_only):
        if not np.all(cost >= 0):
            raise ValueError(
                "Cost matrix must contain non-negative entries"
            )


def _check_indiscernable_sym(cost, X, Y):
    """check cost returns a symmetric matrix with zero diag"""
    # indiscernability: cost(x_t, x_t) = 0 for any t
    if (
        np.all(np.diag(cost(X, X)) != 0)
        or np.all(np.diag(cost(Y, Y)) != 0)
    ):
        raise ValueError(
            "The cost between two identical objects "
            " must be identically (null diagonal)"
        )

    # symmetry: cost(X1, X2) = cost(X2, X1).T
    if not np.allclose(cost(X, Y), cost(Y, X).T):
        raise ValueError(
            "The cost function does not return a symmetric matrix"
        )


def _check_input_evalpoints(eval_points):
    """check eval_points contains one or two one-dim array"""
    if isinstance(eval_points, Tuple) or isinstance(eval_points, List):
        if not len(eval_points) <= 2:
            raise ValueError(
                "eval_points must be a tuple or a list containing one or"
                " two elements, each one being a one-dimensional numpy array"
            )

        for idx, ep_i in enumerate(eval_points):
            if (
                not (isinstance(ep_i, NDArrayFloat) and ep_i.ndim == 1)
            ):
                raise ValueError(
                    "{}-th element of the tuple must be a one-dimensional"
                    " numpy array.".format(idx)
                )

            if not np.all(np.diff(ep_i) > 0):
                raise ValueError(
                    "Values in eval_points must be arranged increasingly"
                )

    else:
        raise ValueError(
            "eval_points must be a tuple containing either one"
            " or two one-dimensional numpy array(s)."
        )


def _check_input_fdata(fdata1, fdata2):
    """check compatible and one sample per object"""
    _check_compatible_fdata(fdata1, fdata2)

    same_fdata_format = type(fdata1) == type(fdata2)

    if not same_fdata_format:
        raise ValueError(
            "fdata1 and fdata2 should be either two FDataBasis objects,"
            " either two FDataGrid objects"
        )

    # if not (fdata1.n_samples == 1 and fdata2.n_samples == 1):
    #     raise ValueError(
    #         "Both fdata1 and fdata2 must contain a single sample"
    #     )


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
        # np.dot(X, Y.T)=inner_product_matrix(X, Y) but np ~1.6 faster
        # X.shape is (n_X, d) and Y.shape is (n_Y, d)
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
    fdata1: FData,
    fdata2: FData,
    *,
    gamma: 1.0,
    cost: None,
    check_cost: False,
    eval_points: None
) -> NDArrayFloat:

    # For developers and PR:
    # final format of fdata1 and fdata2 is required to be FDataGrid,
    # sharing the same domain(1D) and codomain dimensions(1D...nD).
    # No need for _check_compatible and _cast_to_grid:
    # => would raise an error in the following situation:
    # if fdata1.domain_range[0] != fdata1.grid_points[0].domain_range[0]
    # (sdtw_divergence can deal with different domain_range and grid_points)

    # check whether fdata1 and fdata2 are equivalent
    _check_input_fdata(fdata1, fdata2)

    if isinstance(fdata1, FDataBasis):
        # check eval_points
        # must be a tuple of one or two one-dim grid
        # maybe an existing skfda function does the job ?
        _check_input_evalpoints(eval_points)

        if len(eval_points) == 1:
            fdata1 = fdata1.to_grid(eval_points[0])
            fdata2 = fdata2.to_grid(eval_points[0])

        elif len(eval_points) == 2:
            fdata1 = fdata1.to_grid(eval_points[0])
            fdata2 = fdata2.to_grid(eval_points[1])

    n1, n2 = fdata1.grid_points[0].size, fdata2.grid_points[0].size

    if gamma <= 0:
        raise ValueError(
            "gamma was set to {} but must be positive".format(gamma)
        )

    # cost must satisfy:
    # shape=(n1, n2),
    # nonnegativity,
    # indiscernability and symmetry
    if cost is not None:
        if isinstance(cost, List):
            if not len(cost) == 3:
                raise ValueError(
                    "If the alignment cost matrices are pre-computed,"
                    " then 'cost' must be a list of three two-dimensional"
                    " numpy arrays"
                )
            else:
                # shape of cost_12, cost_11 and cost_22
                expected_shapes = [(n1, n2), (n1, n1), (n2, n2)]
                for idx, c in enumerate(cost):
                    if not (isinstance(c, NDArrayFloat) and c.ndim == 2):
                        raise ValueError(
                            "The elements of 'cost' must all be "
                            "two-dimensional numpy arrays. "
                            "The {}-th element does not satisfy the expected "
                            "format".format(idx)
                        )

                    if not c.shape == (n1, n2):
                        raise ValueError(
                            "Cost matrix must have shape"
                            " ({}, {})".format(n1, n2)
                        )

                    _check_shape_postive_cost_mat(
                        cost=c,
                        expected_shape=expected_shapes[idx],
                        shape_only=not(check_cost)
                    )

            cost_12, cost_11, cost_22 = cost[0], cost[1], cost[2]

        elif isinstance(cost, Callable):
            cost_12 = cost(
                fdata1.data_matrix[0, :, :],
                fdata2.data_matrix[0, :, :]
            )

            # check nonnegativity and/or shape
            _check_shape_postive_cost_mat(
                cost_12,
                expected_shape=(n1, n2),
                shape_only=not(check_cost)
            )

            if check_cost:
                # check indiscernability and symmetry
                _check_indiscernable_sym(
                    cost,
                    fdata1.data_matrix[0, :, :],
                    fdata2.data_matrix[0, :, :]
                )

            cost_11 = cost(
                fdata1.data_matrix[0, :, :],
                fdata1.data_matrix[0, :, :]
            )
            cost_22 = cost(
                fdata2.data_matrix[0, :, :],
                fdata2.data_matrix[0, :, :]
            )

        else:
            raise ValueError(
                "Cost must be either "
                " a list of three numpy arrays"
                " or a callable that returns a numpy array"
            )

    else:  # default cost: 0.5 * squared Euclidean

        # 0.5 * squared euclidean
        # or global kernel alignment (depends on gamma !=0)
        # 0.5 * diag(X@X.T) + 0.5 * diag(Y@Y.T) - X@Y.T
        # <=> sq_euclidean = DotProduct(sigma_0_bounds=(10**-10, 10**10))
        # <=> diag(np.dot(X, X.T)) + diag(np.dot(Y, Y.T)) - 2*np.dot(X, Y.T)
        cost_11 = half_sq_euclidean(fdata1.data_matrix[0, :, :])
        cost_22 = half_sq_euclidean(fdata2.data_matrix[0, :, :])
        cost_12 = half_sq_euclidean(
            fdata1.data_matrix[0, :, :],
            fdata2.data_matrix[0, :, :]
        )

    # for nonnegativity, symmetry, unicity of div(X,Y)=0 at X=Y
    return(
        _sdtw_numba._sdtw(cost_12, gamma)
        - 0.5 * _sdtw_numba._sdtw(cost_11, gamma)
        - 0.5 * _sdtw_numba._sdtw(cost_22, gamma)
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
        fdata1: First FData object.
        fdata2: Second FData object.
        gamma: smoothing parameter, must be positive.
        cost: Either a two-dimensional numpy array or a callable.
            If it is an array, then it must have shape ``(n1, n2)``
            where ``n1`` is the grid size of fdata1 and equivalently for
            ``n2``. When fdata1 and fdata2 are FDataBasis, then ``cost``
            must be shaped according to the grid size(s) of ``eval_points``.
            If it is a callable, then it must take two two-dimensional numpy
            arrays (equivalent to the ``data_matirx`` attribute of an
            FDataGrid object) as inputs and return a ``(n1, n2)`` numpy array.
        check_cost: Wheter to check the mathematical properties of
            ``cost`` (function or matrix).
        eval_points: Tuple of two one-dimensional numpy arrays with points of
            evaluation if fdata1 and fdata2 are FDataBasis.
            If the tuple contains a single array, it is used to convert
            both fdata1 and fdata2 into FDataGrid objects.

    Returns:
        soft Dynamic-Time-Warping divergence value.

    References:
        .. footbibliography::

    """
    def __init__(
        self,
        gamma: float = 1.0,
        cost: Optional[Union[List, ArrayLike]] = None,
    ) -> None:

        self.gamma = gamma
        self.cost = cost

    def __call__(
        self,
        fdata1: FData,
        fdata2: FData,
        *,
        eval_points: Optional[Union[Tuple, GridPointsLike]] = None,
        check_cost: Optional[Boolean] = False
    ) -> NDArrayFloat:
        """Compute the soft-DTW divergence."""
        return _sdtw_divergence(
            fdata1,
            fdata2,
            gamma=self.gamma,
            cost=self.cost,
            check_cost=check_cost,
            eval_points=eval_points
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}()"
        )


sdtw_divergence: Final = sdtwDivergence()

# @pairwise_metric_optimization.register
# def _pairwise_metric_optimization_sdtw(
#     metric: sdtwDivergence,
#     elem1: FData,
#     elem2: FData,
# ) -> NDArrayFloat:

#     return metric(elem1, elem2)
