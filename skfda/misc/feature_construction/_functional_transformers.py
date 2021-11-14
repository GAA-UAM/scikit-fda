"""Functional Transformers Module."""

from __future__ import annotations

from functools import reduce
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
                data.integrate(interval=interval),
            ]
    return np.asarray(integrated_data[1:])


def occupation_measure(
    data: FDataGrid,
    intervals: Sequence[Tuple],
) -> np.ndarray:
    r"""
    Calculate the occupation measure of a grid.

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
        if b < a:
            a, b = b, a
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


def number_up_crossings(  # noqa: WPS231
    data: FDataGrid,
    intervals: Sequence[Tuple],
) -> np.ndarray:
    r"""
    Calculate the number of up crossings to a level of a FDataGrid.

        Args:
            data: FDataGrid where we want to calculate
            the number of up crossings.
            intervals: sequence of tuples containing the
            intervals we want to consider for the crossings.
        Returns:
            ndarray of shape (n_dimensions, n_samples)\
            with the values of the counters.
    """
    curves = data.data_matrix
    transformed_counters = []
    for index, interval in enumerate(intervals):
        a, b = interval
        if b < a:
            a, b = b, a
        curves_counters = []
        for c in curves:
            counter = 0
            inside_interval = False
            size_curve = c.shape[0]
            for i in range(0, size_curve - 1):
                if (
                    # Check that the chunk of function grows
                    c[i][index] < c[i + 1][index]  # noqa: WPS408
                    and (
                        # First point <= a, second >= a
                        (c[i][index] <= a and c[i + 1][index] >= a)
                        # First point inside interval, second >=a
                        or (a <= c[i][index] <= b and c[i + 1][index] >= a)
                    )
                ):
                    # Last pair of points where not inside interval
                    if inside_interval is False:
                        counter += 1
                        inside_interval = True
                else:
                    inside_interval = False

            curves_counters = curves_counters + [counter]

        transformed_counters = transformed_counters + [curves_counters]

    return np.asarray(transformed_counters, dtype=list)


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
    """
    curves = data.data_matrix
    norms = []
    for c in curves:  # noqa: WPS426
        x, y = c.shape
        curve_norms = []
        for i in range(0, y):  # noqa: WPS426
            curve_norms = curve_norms + [
                reduce(lambda a, b: a + b[i], c, 0) / x,
            ]
        norms = norms + [curve_norms]
    return np.asarray(norms, dtype=list)


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
    """
    norm = moments_of_norm(data)
    curves = data.data_matrix
    moments = []
    for i, c in enumerate(curves):  # noqa: WPS426
        x, y = c.shape
        curve_moments = []
        for j in range(0, y):  # noqa: WPS426
            curve_moments = curve_moments + [
                reduce(lambda a, b: a + ((b[j] - norm[i][j]) ** 2), c, 0) / x,
            ]
        moments = moments + [curve_moments]
    return np.asarray(moments, dtype=list)
