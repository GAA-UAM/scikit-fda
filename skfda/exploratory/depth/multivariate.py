import scipy.stats

import numpy as np

from . import outlyingness_to_depth


def _stagel_donoho_outlyingness(X, *, pointwise=False):

    if pointwise is False:
        raise NotImplementedError("Only implemented pointwise")

    if X.dim_codomain == 1:
        # Special case, can be computed exactly
        m = X.data_matrix[..., 0]

        return (np.abs(m - np.median(m, axis=0)) /
                scipy.stats.median_absolute_deviation(m, axis=0))

    else:
        raise NotImplementedError("Only implemented for one dimension")


def projection_depth(X, *, pointwise=False):
    """Returns the projection depth.

    The projection depth is the depth function associated with the
    Stagel-Donoho outlyingness.
    """

    depth = outlyingness_to_depth(_stagel_donoho_outlyingness)

    return depth(X, pointwise=pointwise)
