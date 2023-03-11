from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from ...representation import FDataGrid
from ...typing._numpy import NDArrayBool, NDArrayFloat, NDArrayInt


def compute_region(
    fdatagrid: FDataGrid,
    indices_descending_depth: NDArrayInt,
    prob: float,
) -> FDataGrid:
    """Compute central region of a given quantile."""
    indices_samples = indices_descending_depth[
        :math.ceil(fdatagrid.n_samples * prob)
    ]
    return fdatagrid[indices_samples]


def compute_envelope(region: FDataGrid) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Compute curves comprising a region."""
    max_envelope = np.max(region.data_matrix, axis=0)
    min_envelope = np.min(region.data_matrix, axis=0)

    return min_envelope, max_envelope


def predict_outliers(
    fdatagrid: FDataGrid,
    non_outlying_threshold: Tuple[NDArrayFloat, NDArrayFloat],
) -> NDArrayBool:
    """
    Predict outliers given a threshold.

    A functional datum is considered an outlier if it has ANY point
    in ANY dimension outside the envelope for inliers.

    """
    min_threshold, max_threshold = non_outlying_threshold

    or_axes = tuple(i for i in range(1, fdatagrid.data_matrix.ndim))

    below_outliers: NDArrayBool = np.any(
        fdatagrid.data_matrix < min_threshold,
        axis=or_axes,
    )
    above_outliers: NDArrayBool = np.any(
        fdatagrid.data_matrix > max_threshold,
        axis=or_axes,
    )

    return below_outliers | above_outliers


def non_outlying_threshold(
    central_envelope: Tuple[NDArrayFloat, NDArrayFloat],
    factor: float,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Compute a non outlying threshold."""
    iqr = central_envelope[1] - central_envelope[0]
    non_outlying_threshold_max = central_envelope[1] + iqr * factor
    non_outlying_threshold_min = central_envelope[0] - iqr * factor
    return (
        non_outlying_threshold_min,
        non_outlying_threshold_max,
    )
