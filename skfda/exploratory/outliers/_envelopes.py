import math

import numpy as np


def _compute_region(fdatagrid,
                    indices_descending_depth,
                    prob):
    indices_samples = indices_descending_depth[
        :math.ceil(fdatagrid.n_samples * prob)]
    return fdatagrid[indices_samples]


def _compute_envelope(region):
    max_envelope = np.max(region.data_matrix, axis=0)
    min_envelope = np.min(region.data_matrix, axis=0)

    return min_envelope, max_envelope


def _predict_outliers(fdatagrid, non_outlying_threshold):
    # A functional datum is considered an outlier if it has ANY point
    # in ANY dimension outside the envelope for inliers

    min_threshold, max_threshold = non_outlying_threshold

    or_axes = tuple(i for i in range(1, fdatagrid.data_matrix.ndim))

    below_outliers = np.any(fdatagrid.data_matrix <
                            min_threshold, axis=or_axes)
    above_outliers = np.any(fdatagrid.data_matrix >
                            max_threshold, axis=or_axes)

    return below_outliers | above_outliers


def _non_outlying_threshold(central_envelope, factor):
    iqr = central_envelope[1] - central_envelope[0]
    non_outlying_threshold_max = central_envelope[1] + iqr * factor
    non_outlying_threshold_min = central_envelope[0] - iqr * factor
    non_outlying_threshold = (non_outlying_threshold_min,
                              non_outlying_threshold_max)

    return non_outlying_threshold
