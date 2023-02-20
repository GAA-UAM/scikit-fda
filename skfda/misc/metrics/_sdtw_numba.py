import numpy as np
from numba import njit
from ...typing._numpy import NDArrayFloat


@njit(cache=True, fastmath=True)
def _soft_min_argmin(
    x: float,
    y: float,
    z: float
) -> float:
    """soft-min of three real numbers"""

    min_xyz = np.minimum(x, np.minimum(y, z))
    nn = np.exp(min_xyz - x)
    nn += np.exp(min_xyz - y)
    nn += np.exp(min_xyz - z)

    return min_xyz - np.log(nn)


@njit(cache=True)
def _sdtw(
    cost_mat: NDArrayFloat,
    gamma: float = 1.0
) -> float:
    """soft-dtw divergence with dynamic recursion"""

    len_x = cost_mat.shape[0]
    len_y = cost_mat.shape[1]

    V = np.zeros(shape=(len_x + 1, len_y + 1))
    V[0, 1:] = np.inf
    V[1:, 0] = np.inf

    for i in range(1, len_x + 1):
        for j in range(1, len_y + 1):

            V[i, j] = (cost_mat[i - 1, j - 1] / gamma) \
                + _soft_min_argmin(V[i, j - 1], V[i - 1, j - 1], V[i - 1, j])

    return gamma * V[len_x, len_y]
