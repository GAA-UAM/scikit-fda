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

    Example:

    We import the Berkeley Growth Study dataset.
    >>> from skfda.datasets import fetch_growth
    >>> X = fetch_growth(return_X_y=True)[0]

    Then we decide how many intervals we want to consider (in our case 4)
    and call the function with the dataset.
    >>> from skfda.misc.feature_construction import local_averages
    >>> local_averages(X, 4)
        [[[400.97],
          [384.42],
           ...
          [399.46],
          [389.384]],
         [[474.30],
          [450.67],
          ...
          [472.78],
          [467.61]],
         [[645.68],
          [583.68],
          ...
          [629.6],
          [629.83]],
         [[769.],
          [678.53],
          ...
          [670.5],
          [673.43]]]
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


def _calculate_time_on_interval(
    curves: Sequence[Sequence[Sequence[float]]],
    interval_dim: int,
    a: float,
    b: float,
    grid: Sequence,
) -> np.ndarray:
    for j, c in enumerate(curves):
        # TODO: Considerar que la curva pasa varias veces por el intervalo
        total_time = 0
        points_a = _get_interpolation_points(c, a, interval_dim)
        x, y = points_a
        if y is None:
            interp1x = None
        elif x is None:
            index2, y2 = y
            interp1x = _calculate_interpolation(
                (grid[index2], y2),
                (grid[0], c[0][interval_dim]),
                a,
            )
        else:
            index1, y1 = x
            index2, y2 = y
            first_grid = grid[index1]
            if y1 == y2:
                interp1x = first_grid
            else:
                interp1x = _calculate_interpolation(
                    (first_grid, y1),
                    (grid[index2], y2),
                    a,
                )
        points_b = _get_interpolation_points(c, b, interval_dim)
        x, y = points_b
        if x is None:
            interp2x = None
        elif y is None:
            index1, y1 = x
            interp2x = _calculate_interpolation(
                (grid[index1], y1),
                (grid[0], c[0][interval_dim]),
                b,
            )
        else:
            index1, y1 = x
            index2, y2 = y
            first_grid = grid[index1]
            if y1 == y2:
                interp2x = first_grid
            else:
                interp2x = _calculate_interpolation(
                    (first_grid, y1),
                    (grid[index2], y2),
                    b,
                )
        if interp1x is not None and interp2x is not None:
            total_time += (interp2x - interp1x)
        if j == 0:
            curves_time = np.array([[total_time]])
        else:
            curves_time = np.concatenate(
                (curves_time, np.array([[total_time]])),
            )
    return curves_time


def _get_interpolation_points(
    curve: Sequence[Sequence[float]],
    y: float,
    dimension: int,
) -> Tuple:
    less = None
    greater = None
    for i, p in enumerate(curve):
        value = p[dimension]
        if value < y:
            less = (i, value)
        elif value > y:
            greater = (i, value)
        else:
            return ((i, value), (i, value))
        if less is not None and greater is not None:
            return (less, greater)
    return (less, greater)


def _calculate_interpolation(p1: Tuple, p2: Tuple, y: float) -> float:
    """
    Calculate the linear interpolation following this formula.

    x = x_1 + (x_2 – x_1)(y_0 – y_1) / (y_2 – y_1).
    """
    x1, y1 = p1
    x2, y2 = p2
    if y2 == y1:
        return 0

    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)


def occupation_measure(
    data: FDataGrid,
    intervals: Sequence[Sequence[Tuple]],
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
            intervals: sequence of sequence of tuples containing the
            intervals we want to consider. The shape should be
            (n_sequences,n_dimensions, 2)
        Returns:
            ndarray of shape (n_intervals, n_samples, n_dimensions)
            with the transformed data.
    """
    curves = data.data_matrix
    grid = data.grid_points[0]

    transformed_times = []
    for interval in intervals:
        for i, dimension_interval in enumerate(interval):
            a, b = dimension_interval
            if b < a:
                a, b = b, a
            time = np.array(_calculate_time_on_interval(curves, i, a, b, grid))
            if i == 0:
                curves_time = time
            else:
                curves_time = np.hstack((curves_time, time))
        transformed_times = transformed_times + [curves_time]

    return np.asarray(transformed_times, dtype=list)


def number_up_crossings(  # noqa: WPS231
    data: FDataGrid,
    intervals: Sequence[Tuple],
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
            intervals: sequence of tuples containing the
            intervals we want to consider for the crossings.
        Returns:
            ndarray of shape (n_dimensions, n_samples)\
            with the values of the counters.

    Example:

    We import the Phoneme dataset and for simplicity we use
    the first 200 samples.
    >>> from skfda.datasets import fetch_phoneme
    >>> dataset = fetch_phoneme()
    >>> X = dataset['data'][:200]

    Then we decide the interval we want to consider (in our case (5.0,7.5))
    and call the function with the dataset. The output will be the number of
    times each curve cross the interval (5.0,7.5) growing.
    >>> from skfda.misc.feature_construction import number_up_crossings
    >>> number_up_crossings(X, [(5.0,7.5)])
        [[1, 20, 69, 64, 42, 33, 14, 3, 35, 31, 0, 4, 67, 6, 12, 16, 22, 1,
          25, 30, 2, 27, 61, 0, 11, 20, 3, 36, 28, 1, 67, 36, 12, 29, 2,
          16, 25, 1, 24, 57, 65, 26, 20, 18, 43, 0, 35, 40, 0, 2, 56, 4,
          21, 28, 1, 0, 19, 24, 1, 2, 8, 63, 0, 2, 3, 3, 0, 8, 3, 2, 10,
          62, 72, 19, 36, 46, 0, 1, 2, 18, 1, 10, 67, 60, 20, 21, 23, 12,
          3, 30, 21, 1, 57, 64, 15, 4, 4, 17, 0, 2, 31, 0, 5, 24, 56, 8,
          11, 14, 17, 1, 25, 1, 3, 61, 10, 33, 17, 1, 12, 18, 0, 2, 57, 4,
          6, 5, 2, 0, 7, 17, 4, 23, 60, 62, 2, 19, 21, 0, 42, 28, 0, 10,
          29, 74, 34, 29, 7, 0, 25, 23, 0, 15, 19, 1, 43, 1, 11, 9, 4, 0,
          0, 2, 1, 54, 55, 14, 14, 6, 1, 24, 20, 2, 27, 55, 62, 32, 26, 24,
          37, 0, 26, 28, 1, 3, 41, 64, 8, 6, 27, 12, 1, 5, 16, 0, 0, 61,
          62, 1, 3, 7]]
    """
    curves = data.data_matrix
    if curves.shape[2] != len(intervals):
        raise ValueError(
            "Sequence of intervals should have the "
            + "same number of dimensions as the data samples",
        )
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
                p1 = c[i][index]
                p2 = c[i + 1][index]
                if (
                    # Check that the chunk of function grows
                    p1 < p2  # noqa: WPS408
                    and (
                        # First point <= a, second >= a
                        (p1 <= a and p2 >= a)
                        # First point inside interval, second >=a
                        or (a <= p1 <= b and p2 >= a)
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

    Example:

    We import the Canadian Weather dataset
    >>> from skfda.datasets import fetch_weather
    >>> X = fetch_weather(return_X_y=True)[0]
    Then we call the function with the dataset.
    >>> from skfda.misc.feature_construction import moments_of_norm
    >>> moments_of_norm(X)
        [[4.69, 4.06],
         [6.15, 3.99],
         [5.51, 4.04],
         [6.81, 3.46],
         [5.23, 3.29],
         [5.26, 3.09],
         [-5.06, 2.20],
         [3.10, 2.46],
         [2.25, 2.55],
         [4.08, 3.31],
         [4.12, 3.04],
         [6.13, 2.58],
         [5.81, 2.50],
         [7.27, 2.14],
         [7.31, 2.62],
         [2.46, 1.93],
         [2.47, 1.40],
         [-0.15, 1.23],
         [-7.09, 1.20],
         [2.75, 1.02],
         [0.68, 1.11],
         [-3.41, 0.99],
         [2.26, 1.27],
         [3.99, 1.10],
         [8.75, 0.74],
         [9.96, 3.16],
         [9.62, 2.33],
         [3.84, 1.67],
         [7.00, 7.10],
         [-0.85, 0.74],
         [-4.79, 0.90],
         [-5.02, 0.73],
         [-9.65, 1.14],
         [-9.24, 0.71],
         [-16.52, 0.39]]
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

    Example:

    We import the Canadian Weather dataset
    >>> from skfda.datasets import fetch_weather
    >>> X = fetch_weather(return_X_y=True)[0]

    Then we call the function with the dataset.
    >>> from skfda.misc.feature_construction import moments_of_process
    >>> moments_of_process(X)
        [[52.43, 2.25],
         [77.80, 2.92],
         [72.15, 2.69],
         [51.25, 2.12],
         [89.31, 1.59],
         [105.04, 1.53],
         [162.09, 1.49],
         [141.43, 1.45],
         [143.21, 1.15],
         [125.92, 1.64],
         [111.39, 1.39],
         [122.14, 1.16],
         [125.92, 1.04],
         [93.56, 1.02],
         [93.08, 1.21],
         [128.36, 1.26],
         [180.68, 1.19],
         [189.07, 0.92],
         [196.43, 0.55],
         [158.55, 0.94],
         [177.39, 0.81],
         [230.41, 0.44],
         [119.27, 1.26],
         [84.89, 1.01],
         [79.47, 0.20],
         [26.70, 3.52],
         [21.67, 3.09],
         [77.38, 0.49],
         [18.94, 9.94],
         [127.83, 0.25],
         [252.06, 0.29],
         [252.01, 0.27],
         [164.01, 0.49],
         [254.90, 0.20],
         [187.72, 0.12]]
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
