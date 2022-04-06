"""Functional Transformers Module."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ...representation import FDataGrid


def moments_of_norm(
    data: FDataGrid,
    p: int,
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
    >>> np.around(moments_of_norm(X, 1), decimals=2)
        array([[  7.05,   4.06],
               [  8.98,   3.99],
               [  8.38,   4.04],
               [  8.13,   3.46],
               [  9.17,   3.29],
               [  9.95,   3.09],
               [ 11.55,   2.2 ],
               [ 10.92,   2.46],
               [ 10.84,   2.55],
               [ 10.53,   3.31],
               [ 10.02,   3.04],
               [ 10.97,   2.58],
               [ 11.01,   2.5 ],
               [ 10.15,   2.14],
               [ 10.18,   2.62],
               [ 10.36,   1.93],
               [ 12.36,   1.4 ],
               [ 12.28,   1.23],
               [ 13.13,   1.12],
               [ 11.62,   1.02],
               [ 12.05,   1.11],
               [ 13.5 ,   0.99],
               [ 10.16,   1.27],
               [  8.83,   1.1 ],
               [ 10.32,   0.74],
               [  9.96,   3.16],
               [  9.62,   2.33],
               [  8.4 ,   1.67],
               [  7.02,   7.1 ],
               [ 10.06,   0.74],
               [ 14.29,   0.9 ],
               [ 14.47,   0.73],
               [ 13.05,   1.14],
               [ 15.97,   0.71],
               [ 17.65,   0.39]])
    """
    return moments(data, lambda x: pow(np.abs(x), 1))


def moments(
    data: FDataGrid,
    f: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Calculate the moments of a FDataGrid.

        Args:
            data: FDataGrid where we want to calculate
            the moments.
            f: function that specifies the moments to be calculated.
        Returns:
            ndarray of shape (n_dimensions, n_samples)\
            with the values of the moments.

    Example:
    We will use this funtion to calculate the moments of the
    norm of a FDataGrid.
    We will first import the Canadian Weather dataset.
    >>> from skfda.datasets import fetch_weather
    >>> X = fetch_weather(return_X_y=True)[0]

    We will define a function that calculates the moments of the norm.
    >>> f = lambda x: pow(np.abs(x), 1)

    Then we call the function with the dataset and the function.
    >>> import numpy as np
    >>> from skfda.exploratory.stats import moments

    >>> np.around(moments(X, f), decimals=2)
    array([[  7.05,   4.06],
           [  8.98,   3.99],
           [  8.38,   4.04],
           [  8.13,   3.46],
           [  9.17,   3.29],
           [  9.95,   3.09],
           [ 11.55,   2.2 ],
           [ 10.92,   2.46],
           [ 10.84,   2.55],
           [ 10.53,   3.31],
           [ 10.02,   3.04],
           [ 10.97,   2.58],
           [ 11.01,   2.5 ],
           [ 10.15,   2.14],
           [ 10.18,   2.62],
           [ 10.36,   1.93],
           [ 12.36,   1.4 ],
           [ 12.28,   1.23],
           [ 13.13,   1.12],
           [ 11.62,   1.02],
           [ 12.05,   1.11],
           [ 13.5 ,   0.99],
           [ 10.16,   1.27],
           [  8.83,   1.1 ],
           [ 10.32,   0.74],
           [  9.96,   3.16],
           [  9.62,   2.33],
           [  8.4 ,   1.67],
           [  7.02,   7.1 ],
           [ 10.06,   0.74],
           [ 14.29,   0.9 ],
           [ 14.47,   0.73],
           [ 13.05,   1.14],
           [ 15.97,   0.71],
           [ 17.65,   0.39]])

    """
    return np.mean(f(data.data_matrix), axis=1)
