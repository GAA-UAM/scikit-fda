"""Functional Transformers Module."""

from __future__ import annotations

from functools import reduce

import numpy as np

from ...representation import FDataGrid


def moments_of_norm(
    data: FDataGrid,
) -> np.ndarray:
    r"""
    Calculate the moments of the norm of the process of a FDataGrid.

    It performs the following map:
    :math:`f_1(X)=\mathbb{E}(||X||),\dots,f_p(X)=\mathbb{E}(||X||^p)`.

        Args:
            data: FDataGrid where we want to calculate
            the moments of the norm of the process.
        Returns:
            ndarray of shape (n_dimensions, n_samples)\
            with the values of the moments.

    Example:

    We import the Canadian Weather dataset
    >>> from skfda.datasets import fetch_weather
    >>> X = fetch_weather(return_X_y=True)[0]

    Then we call the function with the dataset.
    >>> import numpy as np
    >>> from skfda.exploratory.stats import moments_of_norm
    >>> np.around(moments_of_norm(X), decimals=2)
        array([[  4.69,   4.06],
               [  6.15,   3.99],
               [  5.51,   4.04],
               [  6.81,   3.46],
               [  5.23,   3.29],
               [  5.26,   3.09],
               [ -5.06,   2.2 ],
               [  3.1 ,   2.46],
               [  2.25,   2.55],
               [  4.08,   3.31],
               [  4.12,   3.04],
               [  6.13,   2.58],
               [  5.81,   2.5 ],
               [  7.27,   2.14],
               [  7.31,   2.62],
               [  2.46,   1.93],
               [  2.47,   1.4 ],
               [ -0.15,   1.23],
               [ -7.09,   1.12],
               [  2.75,   1.02],
               [  0.68,   1.11],
               [ -3.41,   0.99],
               [  2.26,   1.27],
               [  3.99,   1.1 ],
               [  8.75,   0.74],
               [  9.96,   3.16],
               [  9.62,   2.33],
               [  3.84,   1.67],
               [  7.  ,   7.1 ],
               [ -0.85,   0.74],
               [ -4.79,   0.9 ],
               [ -5.02,   0.73],
               [ -9.65,   1.14],
               [ -9.24,   0.71],
               [-16.52,   0.39]])
    """
    curves = data.data_matrix
    norms = np.empty((0, curves.shape[2]))
    for c in curves:  # noqa: WPS426
        x, y = c.shape
        curve_norms = []
        for i in range(0, y):  # noqa: WPS426
            curve_norms = curve_norms + [
                reduce(lambda a, b: a + b[i], c, 0) / x,
            ]
        norms = np.concatenate((norms, [curve_norms]))
    return np.array(norms)


def moments_of_process(
    data: FDataGrid,
) -> np.ndarray:
    r"""
    Calculate the moments of the process of a FDataGrid.

    It performs the following map:
    .. math::
        f_1(X)=\int_a^b X(t,\omega)dP(\omega),\dots,f_p(X)=\int_a^b \\
        X^p(t,\omega)dP(\omega).

        Args:
            data: FDataGrid where we want to calculate
            the moments of the process.
        Returns:
            ndarray of shape (n_dimensions, n_samples)\
            with the values of the moments.

    Example:

    We import the Canadian Weather dataset
    >>> from skfda.datasets import fetch_weather
    >>> X = fetch_weather(return_X_y=True)[0]

    Then we call the function with the dataset.
    >>> import numpy as np
    >>> from skfda.exploratory.stats import moments_of_process
    >>> np.around(moments_of_process(X), decimals=2)
        array([[ 5.2430e+01,  2.2500e+00],
               [ 7.7800e+01,  2.9200e+00],
               [ 7.2150e+01,  2.6900e+00],
               [ 5.1250e+01,  2.1200e+00],
               [ 8.9310e+01,  1.6000e+00],
               [ 1.0504e+02,  1.5300e+00],
               [ 1.6209e+02,  1.5000e+00],
               [ 1.4143e+02,  1.4500e+00],
               [ 1.4322e+02,  1.1500e+00],
               [ 1.2593e+02,  1.6500e+00],
               [ 1.1139e+02,  1.4000e+00],
               [ 1.2215e+02,  1.1600e+00],
               [ 1.2593e+02,  1.0500e+00],
               [ 9.3560e+01,  1.0300e+00],
               [ 9.3080e+01,  1.2200e+00],
               [ 1.2836e+02,  1.2700e+00],
               [ 1.8069e+02,  1.2000e+00],
               [ 1.8907e+02,  9.2000e-01],
               [ 1.9644e+02,  5.6000e-01],
               [ 1.5856e+02,  9.4000e-01],
               [ 1.7739e+02,  8.1000e-01],
               [ 2.3041e+02,  4.5000e-01],
               [ 1.1928e+02,  1.2700e+00],
               [ 8.4900e+01,  1.0200e+00],
               [ 7.9470e+01,  2.0000e-01],
               [ 2.6700e+01,  3.5200e+00],
               [ 2.1680e+01,  3.0900e+00],
               [ 7.7390e+01,  4.9000e-01],
               [ 1.8950e+01,  9.9500e+00],
               [ 1.2783e+02,  2.5000e-01],
               [ 2.5206e+02,  2.9000e-01],
               [ 2.5201e+02,  2.7000e-01],
               [ 1.6401e+02,  4.9000e-01],
               [ 2.5490e+02,  2.0000e-01],
               [ 1.8772e+02,  1.2000e-01]])
    """
    norm = moments_of_norm(data)
    curves = data.data_matrix
    moments = np.empty((0, curves.shape[2]))
    for i, c in enumerate(curves):  # noqa: WPS426
        x, y = c.shape
        curve_moments = []
        for j in range(0, y):  # noqa: WPS426
            curve_moments = curve_moments + [
                reduce(lambda a, b: a + ((b[j] - norm[i][j]) ** 2), c, 0) / x,
            ]
        moments = np.concatenate((moments, [curve_moments]))

    return moments
