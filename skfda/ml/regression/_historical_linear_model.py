from __future__ import annotations

import math
from math import ceil
from typing import Tuple

import numpy as np
import scipy.integrate
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ..._utils import _cartesian_product, _pairwise_symmetric
from ...representation import FDataBasis, FDataGrid
from ...representation.basis import Basis, FiniteElement


def _inner_product_matrix(
    basis: Basis,
    fd: FDataGrid,
    limits: Tuple[float, float],
    y_val: float,
) -> np.ndarray:
    """
    Computes the matrix of inner products of an FEM basis with a functional
    data object over a range of x-values for a fixed y-value. The numerical
    integration uses Romberg integration with the trapezoidal rule.

    Arguments:
        basis: typically a FEM basis defined by a triangulation within a
            rectangular domain. It is assumed that only the part of the mesh
            that is within the upper left triangular is of interest.
        fd: a regular functional data object.
        limits: limits of integration, as a tuple of form
            (lower limit, upper limit)
        y_val: the fixed y value.

    """

    basis_fd = basis.to_basis()
    grid = fd.grid_points[0]
    grid_index = (grid >= limits[0]) & (grid <= limits[1])
    grid = grid[grid_index]

    def _pairwise_fem_inner_product(
        basis_fd: FDataBasis,
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

        eval_fem = basis_fd(eval_grid_fem)
        eval_fd = fd(grid)

        # Only for scalar valued functions for now
        assert eval_fem.shape[-1] == 1
        assert eval_fd.shape[-1] == 1

        prod = eval_fem[..., 0] * eval_fd[..., 0]

        return scipy.integrate.simps(prod, grid, axis=1)

    return _pairwise_symmetric(
        _pairwise_fem_inner_product,
        basis_fd,
        fd,
    )


def _design_matrix(
    basis: Basis,
    fd: FDataGrid,
    pred_points: np.ndarray,
) -> np.ndarray:
    """
    Computes the indefinite integrals of the curves over s up to each t-value.

    Arguments:
        basis: typically a FEM basis defined by a triangulation within a
            rectangular domain. It is assumed that only the part of the mesh
            that is within the upper left triangular is of interest.
        fd: a regular functional data object.
        pred_points: points where ``fd`` is evaluated.

    Returns:
        Design matrix.

    """

    matrix = np.array([
        _inner_product_matrix(basis, fd, limits=(0, t), y_val=t).T
        for t in pred_points
    ])

    return np.swapaxes(matrix, 0, 1)


def _get_valid_points(
    interval_len: float,
    n_intervals: int,
    lag: float,
) -> np.ndarray:
    """Return the valid points as integer tuples."""
    interval_points = np.arange(n_intervals + 1)
    full_grid_points = _cartesian_product((interval_points, interval_points))

    past_points = full_grid_points[
        full_grid_points[:, 0] <= full_grid_points[:, 1]
    ]

    discrete_lag = np.inf if lag == np.inf else ceil(lag / interval_len)

    valid_points = past_points[
        past_points[:, 1] - past_points[:, 0] <= discrete_lag
    ]

    return valid_points


def _get_triangles(
    n_intervals: int,
    valid_points: np.ndarray,
) -> np.ndarray:
    """Construct the triangle grid given the valid points."""
    # A matrix where the (integer) coords of a point match
    # to its index or to -1 if it does not exist.
    indexes_matrix = np.full(
        shape=(n_intervals + 1, n_intervals + 1),
        fill_value=-1,
        dtype=np.int_,
    )

    indexes_matrix[
        valid_points[:, 0],
        valid_points[:, 1],
    ] = np.arange(len(valid_points))

    interval_without_end = np.arange(n_intervals)

    pts_coords = _cartesian_product(
        (interval_without_end, interval_without_end),
    )

    down_triangles = np.stack(
        (
            indexes_matrix[pts_coords[:, 0], pts_coords[:, 1]],
            indexes_matrix[pts_coords[:, 0] + 1, pts_coords[:, 1]],
            indexes_matrix[pts_coords[:, 0] + 1, pts_coords[:, 1] + 1],
        ),
        axis=1,
    )

    up_triangles = np.stack(
        (
            indexes_matrix[pts_coords[:, 0], pts_coords[:, 1]],
            indexes_matrix[pts_coords[:, 0], pts_coords[:, 1] + 1],
            indexes_matrix[pts_coords[:, 0] + 1, pts_coords[:, 1] + 1],
        ),
        axis=1,
    )

    triangles = np.concatenate((down_triangles, up_triangles))
    has_wrong_index = np.any(triangles < 0, axis=1)

    triangles = triangles[~has_wrong_index]

    return triangles


def _create_fem_basis(
    start: float,
    stop: float,
    n_intervals: int,
    lag: float,
) -> FiniteElement:

    interval_len = (stop - start) / n_intervals

    valid_points = _get_valid_points(
        interval_len=interval_len,
        n_intervals=n_intervals,
        lag=lag,
    )

    final_points = valid_points * interval_len + start

    triangles = _get_triangles(
        n_intervals=n_intervals,
        valid_points=valid_points,
    )

    return FiniteElement(
        vertices=final_points,
        cells=triangles,
        domain_range=(start, stop),
    )


class HistoricalLinearRegression(
        BaseEstimator,  # type: ignore
        RegressorMixin,  # type: ignore
):

    def __init__(self, *, n_intervals: int, lag: float=math.inf) -> None:
        self.n_intervals = n_intervals
        self.lag = lag

    def _fit_and_return_matrix(self, X: FDataGrid, y: FDataGrid) -> np.ndarray:

        self._pred_points = y.grid_points[0]
        self._pred_domain_range = y.domain_range[0]

        self._basis = _create_fem_basis(
            start=X.domain_range[0][0],
            stop=X.domain_range[0][1],
            n_intervals=self.n_intervals,
            lag=self.lag,
        )

        design_matrix = _design_matrix(
            self._basis,
            X,
            pred_points=self._pred_points,
        )
        design_matrix = design_matrix.reshape(-1, design_matrix.shape[-1])

        self.discretized_coef_ = np.linalg.lstsq(
            design_matrix,
            y.data_matrix[:, ..., 0].ravel(),
            rcond=None,
        )[0]

        return design_matrix

    def _prediction_from_matrix(self, design_matrix: np.ndarray) -> FDataGrid:

        points = (design_matrix @ self.discretized_coef_).reshape(
            -1,
            len(self._pred_points),
        )

        return FDataGrid(
            points,
            grid_points=self._pred_points,
            domain_range=self._pred_domain_range,
        )

    def fit(self, X: FDataGrid, y: FDataGrid) -> HistoricalLinearRegression:

        self._fit_and_return_matrix(X, y)
        return self

    def fit_predict(self, X: FDataGrid, y: FDataGrid) -> FDataGrid:

        design_matrix = self._fit_and_return_matrix(X, y)
        return self._prediction_from_matrix(design_matrix)

    def predict(self, X: FDataGrid) -> FDataGrid:

        check_is_fitted(self)

        design_matrix = _design_matrix(
            self._basis,
            X,
            pred_points=self._pred_points,
        )
        design_matrix = design_matrix.reshape(-1, design_matrix.shape[-1])

        return self._prediction_from_matrix(design_matrix)
