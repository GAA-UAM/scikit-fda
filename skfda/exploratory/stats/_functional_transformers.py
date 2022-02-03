"""Functional Transformers Module."""

from __future__ import annotations

import numpy as np

from ...representation import FDataGrid


def number_up_crossings(
    data: FDataGrid,
    levels: np.ndarray,
) -> np.ndarray:
    r"""
    Calculate the number of up crossings to a level of a FDataGrid.

    Let f_1(X) = N_i, where N_i is the number of up crossings of X
    to a level c_i \in \mathbb{R}, i = 1,\dots,p.

    Recall that the process X(t) is said to have an up crossing of c
    at :math:`t_0 > 0` if for some :math:`\epsilon >0`, X(t) $\leq$
    c if t :math:'\in (t_0 - \epsilon, t_0) and X(t) $\geq$ c if
    :math:`t\in (t_0, t_0+\epsilon)`.

    If the trajectories are differentiable, then
    :math:`N_i = card\{t \in[a,b]: X(t) = c_i, X' (t) > 0\}.`

        Args:
            data: FDataGrid where we want to calculate
            the number of up crossings.
            levels: sequence of numbers including the levels
            we want to consider for the crossings.
        Returns:
            ndarray of shape (n_levels, n_samples)\
            with the values of the counters.

    Example:

    We import the Medflies dataset and for simplicity we use
    the first 50 samples.
    >>> from skfda.datasets import fetch_medflies
    >>> dataset = fetch_medflies()
    >>> X = dataset['data'][:50]

    Then we decide the level we want to consider (in our case 40)
    and call the function with the dataset. The output will be the number of
    times each curve cross the level 40 growing.
    >>> from skfda.exploratory.stats import number_up_crossings
    >>> import numpy as np
    >>> number_up_crossings(X, np.asarray([40]))
    array([[[6],
            [3],
            [7],
            [7],
            [3],
            [4],
            [5],
            [7],
            [4],
            [6],
            [4],
            [4],
            [5],
            [6],
            [0],
            [5],
            [1],
            [6],
            [0],
            [7],
            [0],
            [6],
            [2],
            [5],
            [6],
            [5],
            [8],
            [4],
            [3],
            [7],
            [1],
            [3],
            [0],
            [5],
            [2],
            [7],
            [2],
            [5],
            [5],
            [5],
            [4],
            [4],
            [1],
            [2],
            [3],
            [5],
            [3],
            [3],
            [5],
            [2]]])
    """
    curves = data.data_matrix

    distances = np.asarray([
        level - curves
        for level in levels
    ])

    points_greater = distances >= 0
    points_smaller = distances <= 0
    points_smaller_rotated = np.roll(points_smaller, -1, axis=2)

    return np.sum(
        points_greater & points_smaller_rotated,
        axis=2,
    )
