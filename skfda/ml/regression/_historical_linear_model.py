from __future__ import annotations

import math
from typing import Tuple, Union

import numpy as np
import scipy.integrate
from sklearn.utils.validation import check_is_fitted

from ..._utils import _cartesian_product, _pairwise_symmetric
from ..._utils._sklearn_adapter import BaseEstimator, RegressorMixin
from ...representation import FData, FDataBasis, FDataGrid
from ...representation.basis import (
    Basis,
    FiniteElementBasis,
    VectorValuedBasis,
)
from ...typing._numpy import NDArrayFloat

_MeanType = Union[FDataGrid, float]


def _pairwise_fem_inner_product(
    basis_fd: FData,
    fd: FData,
    y_val: float,
    grid: NDArrayFloat,
) -> NDArrayFloat:

    eval_grid_fem = np.concatenate(
        (
            grid[:, None],
            np.full(
                shape=(len(grid), 1),
                fill_value=y_val,
            ),
        ),
        axis=1,
    )

    eval_fem = basis_fd(eval_grid_fem)
    eval_fd = fd(grid)

    prod = eval_fem * eval_fd
    integral = scipy.integrate.simps(prod, grid, axis=1)
    return np.sum(integral, axis=-1)  # type: ignore[no-any-return]


def _inner_product_matrix(
    basis: Basis,
    fd: FData,
    limits: Tuple[float, float],
    y_val: float,
) -> NDArrayFloat:
    """
    Compute inner products with the FEM basis.

    Compute the matrix of inner products of an FEM basis with a functional
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

    Returns:
        Matrix of inner products.

    """
    basis_fd: FData = basis.to_basis()
    grid = fd.grid_points[0]
    grid_index = (grid >= limits[0]) & (grid <= limits[1])
    grid = grid[grid_index]

    return _pairwise_symmetric(
        _pairwise_fem_inner_product,  # type: ignore[arg-type]
        basis_fd,
        fd,
        y_val=y_val,  # type: ignore[arg-type]
        grid=grid,
    )


def _design_matrix(
    basis: Basis,
    fd: FDataGrid,
    pred_points: NDArrayFloat,
) -> NDArrayFloat:
    """
    Compute the indefinite integrals of the curves over s up to each t-value.

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
) -> NDArrayFloat:
    """Return the valid points as integer tuples."""
    interval_points = np.arange(n_intervals + 1)
    full_grid_points = _cartesian_product((interval_points, interval_points))

    past_points = full_grid_points[
        full_grid_points[:, 0] <= full_grid_points[:, 1]
    ]

    discrete_lag = np.inf if lag == np.inf else math.ceil(lag / interval_len)

    return past_points[  # type: ignore[no-any-return]
        past_points[:, 1] - past_points[:, 0] <= discrete_lag
    ]


def _get_triangles(
    n_intervals: int,
    valid_points: NDArrayFloat,
) -> NDArrayFloat:
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

    pts_coords_x = pts_coords[:, 0]
    pts_coords_y = pts_coords[:, 1]

    down_triangles = np.stack(
        (
            indexes_matrix[pts_coords_x, pts_coords_y],
            indexes_matrix[pts_coords_x + 1, pts_coords_y],
            indexes_matrix[pts_coords_x + 1, pts_coords_y + 1],
        ),
        axis=1,
    )

    up_triangles = np.stack(
        (
            indexes_matrix[pts_coords_x, pts_coords_y],
            indexes_matrix[pts_coords_x, pts_coords_y + 1],
            indexes_matrix[pts_coords_x + 1, pts_coords_y + 1],
        ),
        axis=1,
    )

    triangles = np.concatenate((down_triangles, up_triangles))
    has_wrong_index = np.any(triangles < 0, axis=1)

    return triangles[~has_wrong_index]  # type: ignore[no-any-return]


def _create_fem_basis(
    start: float,
    stop: float,
    n_intervals: int,
    lag: float,
) -> FiniteElementBasis:

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

    return FiniteElementBasis(
        vertices=final_points,
        cells=triangles,
        domain_range=((start, stop),) * 2,
    )


class HistoricalLinearRegression(
    BaseEstimator,
    RegressorMixin[FDataGrid, FDataGrid],
):
    r"""Historical functional linear regression.

    This is a linear regression method where the covariate and the response are
    both functions :math:`\mathbb{R}` to :math:`\mathbb{R}` with the same
    domain. In order to predict the value of the response function at point
    :math:`t`, only the information of the covariate at points :math:`s < t` is
    used. Is thus an "historical" model in the sense that, if the domain
    represents time, only the data from the past, or historical data, is used
    to predict a given point.

    The model assumed by this method is:

    .. math::
        y_i = \alpha(t) + \int_{s_0(t)}^t x_i(s) \beta(s, t) ds

    where :math:`s_0(t) = \max(0, t - \delta)` and :math:`\delta` is a
    predefined time lag that can be specified so that points far in the past
    do not affect the predicted value.

    Args:
        n_intervals: Number of intervals used to create the basis of the
            coefficients. This will be a bidimensional
            :class:`~skfda.representation.basis.FiniteElement` basis, and
            this parameter indirectly specifies the number of
            elements of that basis, and thus the granularity.
        fit_intercept:  Whether to calculate the intercept for this
            model. If set to False, no intercept will be used in calculations
            (i.e. data is expected to be centered).
        lag: The maximum time lag at which points in the past can still
            influence the prediction.

    Attributes:
        basis_coef\_: The fitted coefficient function as a FDataBasis.
        coef\_: The fitted coefficient function as a FDataGrid.
        intercept\_: Independent term in the linear model. Set to the constant
            function 0 if `fit_intercept = False`.

    Examples:
        The following example test a case that conforms to this model.

        >>> from skfda import FDataGrid
        >>> from skfda.ml.regression import HistoricalLinearRegression
        >>> import numpy as np
        >>> import scipy.integrate

        >>> random_state = np.random.RandomState(0)
        >>> data_matrix = random_state.choice(10, size=(8, 6))
        >>> data_matrix
        array([[5, 0, 3, 3, 7, 9],
               [3, 5, 2, 4, 7, 6],
               [8, 8, 1, 6, 7, 7],
               [8, 1, 5, 9, 8, 9],
               [4, 3, 0, 3, 5, 0],
               [2, 3, 8, 1, 3, 3],
               [3, 7, 0, 1, 9, 9],
               [0, 4, 7, 3, 2, 7]])
        >>> intercept = random_state.choice(10, size=(1, 6))
        >>> intercept
        array([[2, 0, 0, 4, 5, 5]])
        >>> y_data = scipy.integrate.cumtrapz(
        ...              data_matrix,
        ...              initial=0,
        ...              axis=1,
        ...          ) + intercept
        >>> y_data
        array([[  2. ,   2.5,   4. ,  11. ,  17. ,  25. ],
               [  2. ,   4. ,   7.5,  14.5,  21. ,  27.5],
               [  2. ,   8. ,  12.5,  20. ,  27.5,  34.5],
               [  2. ,   4.5,   7.5,  18.5,  28. ,  36.5],
               [  2. ,   3.5,   5. ,  10.5,  15.5,  18. ],
               [  2. ,   2.5,   8. ,  16.5,  19.5,  22.5],
               [  2. ,   5. ,   8.5,  13. ,  19. ,  28. ],
               [  2. ,   2. ,   7.5,  16.5,  20. ,  24.5]])
        >>> X = FDataGrid(data_matrix)
        >>> y = FDataGrid(y_data)
        >>> hist = HistoricalLinearRegression(n_intervals=8)
        >>> _ = hist.fit(X, y)
        >>> hist.predict(X).data_matrix[..., 0].round(1)
        array([[  2. ,   2.5,   4. ,  11. ,  17. ,  25. ],
               [  2. ,   4. ,   7.5,  14.5,  21. ,  27.5],
               [  2. ,   8. ,  12.5,  20. ,  27.5,  34.5],
               [  2. ,   4.5,   7.5,  18.5,  28. ,  36.5],
               [  2. ,   3.5,   5. ,  10.5,  15.5,  18. ],
               [  2. ,   2.5,   8. ,  16.5,  19.5,  22.5],
               [  2. ,   5. ,   8.5,  13. ,  19. ,  28. ],
               [  2. ,   2. ,   7.5,  16.5,  20. ,  24.5]])
        >>> abs(hist.intercept_.data_matrix[..., 0].round())
        array([[ 2.,  0.,  0.,  4.,  5.,  5.]])

    References:
        Malfait, N., & Ramsay, J. O. (2003). The historical functional linear
        model. Canadian Journal of Statistics, 31(2), 115-128.

    """

    def __init__(
        self,
        *,
        n_intervals: int,
        fit_intercept: bool = True,
        lag: float = math.inf,
    ) -> None:
        self.n_intervals = n_intervals
        self.fit_intercept = fit_intercept
        self.lag = lag

    def _center_X_y(
        self,
        X: FDataGrid,
        y: FDataGrid,
    ) -> Tuple[FDataGrid, FDataGrid, _MeanType, _MeanType]:

        X_mean: Union[FDataGrid, float] = (
            X.mean() if self.fit_intercept else 0
        )
        X_centered = X - X_mean
        y_mean: Union[FDataGrid, float] = (
            y.mean() if self.fit_intercept else 0
        )
        y_centered = y - y_mean

        return X_centered, y_centered, X_mean, y_mean

    def _fit_and_return_centered_matrix(
        self,
        X: FDataGrid,
        y: FDataGrid,
    ) -> Tuple[NDArrayFloat, _MeanType]:

        X_centered, y_centered, X_mean, y_mean = self._center_X_y(X, y)

        self._pred_points = y_centered.grid_points[0]
        self._pred_domain_range = y_centered.domain_range[0]

        fem_basis = _create_fem_basis(
            start=X_centered.domain_range[0][0],
            stop=X_centered.domain_range[0][1],
            n_intervals=self.n_intervals,
            lag=self.lag,
        )

        self._basis = VectorValuedBasis(
            [fem_basis] * X_centered.dim_codomain,
        )

        design_matrix = _design_matrix(
            self._basis,
            X_centered,
            pred_points=self._pred_points,
        )
        design_matrix = design_matrix.reshape(-1, design_matrix.shape[-1])

        self._coef_coefs = np.linalg.lstsq(
            design_matrix,
            y_centered.data_matrix[:, ..., 0].ravel(),
            rcond=None,
        )[0]

        self.basis_coef_ = FDataBasis(
            basis=self._basis,
            coefficients=self._coef_coefs,
        )

        self.coef_ = self.basis_coef_.to_grid(
            grid_points=[X.grid_points[0]] * 2,
        )

        if self.fit_intercept:
            assert isinstance(X_mean, FDataGrid)
            self.intercept_ = (
                y_mean - self._predict_no_intercept(X_mean)
            )
        else:
            self.intercept_ = y.copy(
                data_matrix=np.zeros_like(y.data_matrix[0]),
            )

        return design_matrix, y_mean

    def _prediction_from_matrix(
        self,
        design_matrix: NDArrayFloat,
    ) -> FDataGrid:

        points = (design_matrix @ self._coef_coefs).reshape(
            -1,
            len(self._pred_points),
        )

        return FDataGrid(
            points,
            grid_points=self._pred_points,
            domain_range=self._pred_domain_range,
        )

    def fit(  # noqa: D102
        self,
        X: FDataGrid,
        y: FDataGrid,
    ) -> HistoricalLinearRegression:

        self._fit_and_return_centered_matrix(X, y)
        return self

    def fit_predict(  # noqa: D102
        self,
        X: FDataGrid,
        y: FDataGrid,
    ) -> FDataGrid:

        design_matrix, y_mean = self._fit_and_return_centered_matrix(X, y)
        return (
            self._prediction_from_matrix(design_matrix)
            + y_mean
        )

    def _predict_no_intercept(self, X: FDataGrid) -> FDataGrid:

        design_matrix = _design_matrix(
            self._basis,
            X,
            pred_points=self._pred_points,
        )
        design_matrix = design_matrix.reshape(-1, design_matrix.shape[-1])

        return self._prediction_from_matrix(design_matrix)

    def predict(self, X: FDataGrid) -> FDataGrid:  # noqa: D102

        check_is_fitted(self)

        return self._predict_no_intercept(X) + self.intercept_
