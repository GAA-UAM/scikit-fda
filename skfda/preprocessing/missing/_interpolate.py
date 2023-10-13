from typing import Any, TypeVar

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate.interpnd import LinearNDInterpolator

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...representation import FDataGrid
from ...typing._base import GridPoints
from ...typing._numpy import NDArrayFloat, NDArrayInt

T = TypeVar("T", bound=FDataGrid)


def _coords_from_indices(
    coord_indices: NDArrayInt,
    grid_points: GridPoints,
) -> NDArrayFloat:
    return np.stack([
        grid_points[i][coord_index]
        for i, coord_index in enumerate(coord_indices.T)
    ]).T


def _interpolate_nans(
    fdatagrid: T,
) -> T:

    data_matrix = fdatagrid.data_matrix.copy()

    for n_sample in range(fdatagrid.n_samples):
        for n_coord in range(fdatagrid.dim_codomain):

            data_points = data_matrix[n_sample, ..., n_coord]
            nan_pos = np.isnan(data_points)
            valid_pos = ~nan_pos
            coord_indices = np.argwhere(valid_pos)
            desired_coord_indices = np.argwhere(nan_pos)
            coords = _coords_from_indices(
                coord_indices,
                fdatagrid.grid_points,
            )
            desired_coords = _coords_from_indices(
                desired_coord_indices,
                fdatagrid.grid_points,
            )
            values = data_points[valid_pos]

            if fdatagrid.dim_domain == 1:
                interpolation = InterpolatedUnivariateSpline(
                    coords,
                    values,
                    k=1,
                    ext=3,
                )
            else:
                interpolation = LinearNDInterpolator(
                    coords,
                    values,
                )

            new_values = interpolation(
                desired_coords,
            )

            data_matrix[n_sample, nan_pos, n_coord] = new_values.ravel()

    return fdatagrid.copy(data_matrix=data_matrix)


class MissingValuesInterpolation(
    BaseEstimator,
    InductiveTransformerMixin[T, T, Any],
):
    """
    Class to interpolate missing values.

    Missing values are represented as NaNs.
    They are interpolated from nearby values with valid data.
    Note that this may be a poor choice if there are large contiguous portions
    of the function with missing values, as some of them would be inferred from
    very far away points.

    Examples:
        It is possible to interpolate NaNs scalar-valued univariate functions:

        >>> from skfda import FDataGrid
        >>> from skfda.preprocessing.missing import MissingValuesInterpolation
        >>> import numpy as np

        >>> X = FDataGrid([
        ...     [1, 2, np.nan, 4],
        ...     [5, np.nan, 7, 8],
        ...     [9, 10, np.nan, 12],
        ... ])
        >>> nan_interp = MissingValuesInterpolation()
        >>> X_transformed = nan_interp.fit_transform(X)
        >>> X_transformed.data_matrix[..., 0]
        array([[ 1.,  2.,  3.,  4.],
               [ 5.,  6.,  7.,  8.],
               [ 9., 10., 11., 12.]])

        For vector-valued functions each coordinate is interpolated
        independently:

        >>> X = FDataGrid(
        ...     [
        ...         [
        ...             (1, 5),
        ...             (2, np.nan),
        ...             (np.nan, 7),
        ...             (4, 8),
        ...         ],
        ...         [
        ...             (9, 13),
        ...             (10, np.nan),
        ...             (np.nan, np.nan),
        ...             (12, 16),
        ...         ],
        ...     ],
        ...     grid_points=np.linspace(0, 1, 4)
        ... )
        >>> nan_interp = MissingValuesInterpolation()
        >>> X_transformed = nan_interp.fit_transform(X)
        >>> X_transformed.data_matrix # doctest: +NORMALIZE_WHITESPACE
        array([[[  1.,  5.],
                [  2.,  6.],
                [  3.,  7.],
                [  4.,  8.]],
               [[  9., 13.],
                [ 10., 14.],
                [ 11., 15.],
                [ 12., 16.]]])

        For multivariate functions, such as surfaces all dimensions are
        considered. This is currently done using
        :external:class:`~scipy.interpolate.LinearNDInterpolator`, which
        triangulates the space and performs linear barycentric interpolation:

        >>> X = FDataGrid(
        ...     [
        ...         [
        ...             [1, 2, 3, 4],
        ...             [5, np.nan, 7, 8],
        ...             [10, 10, np.nan, 10],
        ...             [13, 14, 15, 16],
        ...         ],
        ...     ],
        ...     grid_points=(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
        ... )
        >>> nan_interp = MissingValuesInterpolation()
        >>> X_transformed = nan_interp.fit_transform(X)
        >>> X_transformed.data_matrix[..., 0]
        array([[[  1.,   2.,   3.,   4.],
                [  5.,   6.,   7.,   8.],
                [ 10.,  10.,  11.,  10.],
                [ 13.,  14.,  15.,  16.]]])
    """

    def transform(
        self,
        X: T,
    ) -> T:
        return _interpolate_nans(X)
