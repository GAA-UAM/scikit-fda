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

    def transform(
        self,
        X: T,
    ) -> T:
        return _interpolate_nans(X)
