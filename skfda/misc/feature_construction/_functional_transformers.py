"""Functional Transformers Module."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

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
        ndarray of shape (n_intervals, n_samples, n_dimensions)\
        with the transformed data.
    """
    domain_range = data.domain_range

    x, y = domain_range[0]
    interval_size = (y - x) / n_intervals

    integrated_data = [[]]
    for i in np.arange(x, y, interval_size):
        interval = (i, i + interval_size)
        if isinstance(data, FDataGrid):
            data_grid = data.restrict(interval)
            integrated_data = integrated_data + [
                data_grid.integrate(),
            ]
        else:
            integrated_data = integrated_data + [
                data.integrate(interval),
            ]
    return np.asarray(integrated_data[1:])


def occupation_measure(
    data: FDataGrid,
    intervals: Sequence[Tuple],
) -> np.ndarray:
    r"""
    Take functional data as a grid and calculate the occupation measure.

    It performs the following map.
        ..math:
            :math:`f_1(X) = |t: X(t)\in T_p|,\dots,|t: X(t)\in T_p|`

        where :math:`{T_1,\dots,T_p}` are disjoint intervals in
        :math:`\mathbb{R}` and | | stands for the Lebesgue measure.

    Args:
            data: FDataGrid where we want to calculate
            the occupation measure.
            intervals: sequence of tuples containing the
            intervals we want to consider.
    Returns:
            ndarray of shape (n_dimensions, n_samples)\
            with the transformed data.
    """
    curves = data.data_matrix
    grid = data.grid_points[0]

    transformed_intervals = []
    for i, interval in enumerate(intervals):
        a, b = interval
        curves_intervals = []
        for c in curves:
            first = None
            last = None
            for index, point in enumerate(c):
                if a <= point[i] <= b and first is None:
                    first = grid[index]
                elif a <= point[i] <= b:
                    last = grid[index]

            if first is None:
                t = ()
            elif last is None:
                t = (first, first)
            else:
                t = (first, last)

            curves_intervals = curves_intervals + [t]

        transformed_intervals = transformed_intervals + [curves_intervals]

    return np.asarray(transformed_intervals, dtype=list)
