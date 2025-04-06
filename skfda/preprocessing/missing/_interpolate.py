from typing import Any, TypeVar

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...representation import FDataGrid
from ...typing._base import GridPoints
from ...typing._numpy import NDArrayFloat, NDArrayInt

T = TypeVar("T", bound=FDataGrid)


def _left_non_nan_index(
    array: NDArrayFloat,
    axis: int = -1,
) -> NDArrayInt:
    """
    Return the index of the first non-NaN on the left.

    Adapted from https://stackoverflow.com/a/41191127/2455333
    """
    mask = np.isnan(array)
    shape = [1] * array.ndim
    shape[axis] = -1
    indexes = np.reshape(np.arange(mask.shape[axis]), shape)
    idx = np.where(~mask, indexes, -1)
    return np.maximum.accumulate(idx, axis=axis)


def _right_non_nan_index(
    array: NDArrayFloat,
    axis: int = -1,
) -> NDArrayInt:
    """Return the index of the first non-NaN on the right."""
    return array.shape[axis] - 1 - np.flip(  # type: ignore[no-any-return]
        _left_non_nan_index(
            np.flip(array, axis=axis),
            axis=axis,
        ),
        axis=axis,
    )


def interpolate_nans_1d(
    x: NDArrayFloat,
    y: NDArrayFloat,
    axis: int = -1,
) -> NDArrayFloat:
    """
    Interpolates NaN along a dimension of the array.

    The input arrays are considered to represent a set of functions of the
    form ``y_i = f(x)``.
    This function attempts to remove internal NaNs by linearly interpolating
    with the nearest non-nan values at each side.

    Args:
        x: Array of common points where the function is evaluated.
        y: Value(s) of the function at these points.
        axis: Axis in the ``y`` array corresponding to the ``x``-coordinate
            values.

    Returns:
        A copy of ``y`` where all NaN entries have been replaced with their
        linear interpolation using the nearest non-NaN values at each side.

    Examples:
        Interpolating using equispaced points. Each NaN in the middle is
        interpolated linearly. The NaN at the extremes are extrapolated with
        a constant, as they only have a nearest non-NaN value at one side.
        Note also that if a row contains only NaN, the row is returned
        unchanged.

        >>> import numpy as np
        >>>
        >>> x = np.linspace(0, 1, 5)
        >>> y = np.array([
        ...     [np.nan, 2, 3, np.nan, 5],
        ...     [1, np.nan, np.nan, 4, 5],
        ...     [1, np.nan, 3, 4, np.nan],
        ...     [np.nan, np.nan, np.nan, np.nan, np.nan],
        ... ])
        >>>
        >>> interpolate_nans_1d(x, y)
        array([[  2.,  2.,  3.,  4.,  5.],
               [  1.,  2.,  3.,  4.,  5.],
               [  1.,  2.,  3.,  4.,  4.],
               [ nan, nan, nan, nan, nan]])

        It is possible to change the axis used in the interpolation:

        >>> interpolate_nans_1d(x, y[..., None], axis=1)
        array([[[  2.],
                [  2.],
                [  3.],
                [  4.],
                [  5.]],
        <BLANKLINE>
               [[  1.],
                [  2.],
                [  3.],
                [  4.],
                [  5.]],
        <BLANKLINE>
               [[  1.],
                [  2.],
                [  3.],
                [  4.],
                [  4.]],
        <BLANKLINE>
               [[ nan],
                [ nan],
                [ nan],
                [ nan],
                [ nan]]])

        Non-equispaced points are also allowed:

        >>> x = np.array([0, 0.2, 0.5, 0.6, 1])
        >>> interpolate_nans_1d(x, y)
        array([[ 2. , 2. , 3. , 3.4, 5. ],
               [ 1. , 2. , 3.5, 4. , 5. ],
               [ 1. , 1.8, 3. , 4. , 4. ],
               [ nan, nan, nan, nan, nan]])

    """
    left_idx = _left_non_nan_index(y, axis=axis)
    right_idx = _right_non_nan_index(y, axis=axis)

    # Add constant extrapolation on the sides
    invalid_left_idx = left_idx < 0
    invalid_right_idx = right_idx >= len(x)

    # The order of the following two lines is important: in case that
    # all values are NaN, the final index is -1 for all.
    right_idx[invalid_right_idx] = left_idx[invalid_right_idx]
    left_idx[invalid_left_idx] = right_idx[invalid_left_idx]

    point_left = x[left_idx]
    point_right = x[right_idx]
    value_left = np.take_along_axis(y, left_idx, axis=axis)
    value_right = np.take_along_axis(y, right_idx, axis=axis)

    shape = [1] * y.ndim
    shape[axis] = len(x)
    reshaped_x = np.reshape(x, shape)
    denom = (point_right - point_left)
    # If the denominator is 0, the numerator is also 0.
    # Prevent the generation of NaN or Inf.
    denom[denom == 0] = 1
    delta_t = (reshaped_x - point_left) / denom

    return (1 - delta_t) * value_left + delta_t * value_right


def _coords_from_indices(
    coord_indices: NDArrayInt,
    grid_points: GridPoints,
) -> NDArrayFloat:
    return np.stack([  # type: ignore[no-any-return]
        grid_points[i][coord_index]
        for i, coord_index in enumerate(coord_indices.T)
    ]).T


def _interpolate_nans(
    fdatagrid: T,
) -> T:

    data_matrix = fdatagrid.data_matrix.copy()

    if fdatagrid.dim_domain == 1:
        data_matrix = interpolate_nans_1d(
            fdatagrid.grid_points[0],
            fdatagrid.data_matrix,
            axis=1,
        )
    else:
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
    InductiveTransformerMixin[T, T, Any],
    BaseEstimator,
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
