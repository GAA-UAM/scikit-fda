"""Functional Transformers Module."""

from __future__ import annotations

from typing import Union

import numpy as np

from ...representation import FDataGrid


def local_averages(
    data: Union[FDataGrid, FDataBasis],
    n_intervals: int,
) -> np.ndarray:
    r"""
    Calculate the local averages of a given data.

    Take functional data as a grid or a basis and performs
    the following map:

    .. math::
        f_1(X) = \frac{1}{|T_1|} \int_{T_1} X(t) dt,\dots, \\
        f_p(X) = \frac{1}{|T_p|} \int_{T_p} X(t) dt

    where {T_1,\dots,T_p} are disjoint intervals of the interval [a,b]

    It is calculated for a given number of intervals,
    which are of equal sizes.
    Args:
        data: FDataGrid or FDataBasis where we want to
        calculate the local averages.
        n_intervals: number of intervals we want to consider.
    Returns:
        ndarray of shape (n_intervals, n_samples, n_dimensions)
        with the transformed data for FDataBasis and (n_intervals, n_samples)
        for FDataGrid.

    Example:

        We import the Berkeley Growth Study dataset.
        We will use only the first 30 samples to make the
        example easy.
        >>> from skfda.datasets import fetch_growth
        >>> dataset = fetch_growth(return_X_y=True)[0]
        >>> X = dataset[:30]

        Then we decide how many intervals we want to consider (in our case 2)
        and call the function with the dataset.
        >>> import numpy as np
        >>> from skfda.exploratory.stats import local_averages
        >>> np.around(local_averages(X, 2), decimals=2)
        array([[ 116.94,  111.86,  107.29,  111.35,  104.39,  109.43,  109.16,
                 112.91,  109.19,  117.95,  112.14,  114.3 ,  111.48,  114.85,
                 116.25,  114.6 ,  111.02,  113.57,  108.88,  109.6 ,  109.7 ,
                 108.54,  109.18,  106.92,  109.44,  109.84,  115.32,  108.16,
                 119.29,  110.62],
               [ 177.26,  157.62,  154.97,  163.83,  156.66,  157.67,  155.31,
                 169.02,  154.18,  174.43,  161.33,  170.14,  164.1 ,  170.1 ,
                 166.65,  168.72,  166.85,  167.22,  159.4 ,  162.76,  155.7 ,
                 158.01,  160.1 ,  155.95,  157.95,  163.53,  162.29,  153.1 ,
                 178.48,  161.75]])
    """
    left, right = data.domain_range[0]

    intervals, step = np.linspace(
        left,
        right,
        num=n_intervals + 1,
        retstep=True,
    )

    integrated_data = [
        data.integrate(interval=((intervals[i], intervals[i + 1]))) / step
        for i in range(n_intervals)
    ]
    return np.asarray(integrated_data)

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
