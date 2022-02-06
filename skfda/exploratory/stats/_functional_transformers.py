"""Functional Transformers Module."""

from __future__ import annotations

from typing import Union

import numpy as np

from ...representation import FDataBasis, FDataGrid


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
        for i in np.arange(0, n_intervals)
    ]
    return np.asarray(integrated_data)
